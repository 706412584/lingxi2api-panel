package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/lingxi2api/lingxi2api-panel/internal/auth"
	"github.com/lingxi2api/lingxi2api-panel/internal/upstream"
)

// newLXMockUpstream 模拟灵犀三步骤握手：
//   POST /api/aioffice/v1/cowork/sessions     → 返回 session_id
//   POST /api/public/v1/credits/runs          → 返回 run_id
//   POST /api/public/v1/agent-runtime/chat/completions → 返回 chatHandler 决定的内容
//
// lxBases 是包级变量（upstream.lxBases），server 包测试改不了；改为通过
// 自定义 RoundTripper 把 lingxi.kdocs.cn / lingxi.wps.cn 的请求重写到 httptest server。
func newLXMockUpstream(t *testing.T, chatHandler http.HandlerFunc) *upstream.Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/aioffice/v1/cowork/sessions":
			_, _ = w.Write([]byte(`{"result":"Success","data":{"session_id":"sess_mock"}}`))
		case "/api/public/v1/credits/runs":
			_, _ = w.Write([]byte(`{"result":"Success","data":{"run_id":"run_mock"}}`))
		case "/api/public/v1/agent-runtime/chat/completions":
			chatHandler(w, r)
		default:
			// 静默其他辅助端点（models、execution_status、balance 等）。
			_, _ = w.Write([]byte(`{"result":"Success","data":{}}`))
		}
	}))
	t.Cleanup(srv.Close)
	target, _ := url.Parse(srv.URL)
	rewrite := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		// 仅改写灵犀域名的请求；其他走默认 transport。
		if strings.Contains(r.URL.Host, "lingxi.") {
			r = r.Clone(r.Context())
			r.URL.Scheme = target.Scheme
			r.URL.Host = target.Host
			r.Host = target.Host
		}
		return http.DefaultTransport.RoundTrip(r)
	})}
	return &upstream.Client{HTTP: rewrite, ChatBaseCN: srv.URL, BillingBaseCN: srv.URL}
}

// ── 请求转换 ──────────────────────────────────────────────────────────────

func TestAnthropicToOpenAI_Basic(t *testing.T) {
	in := `{"model":"kimi-k3.1-pro","max_tokens":64,"system":"You are helpful.","messages":[{"role":"user","content":"hi"}]}`
	out, err := anthropicToOpenAI([]byte(in), newToolNameMap())
	if err != nil {
		t.Fatal(err)
	}
	var obj map[string]any
	_ = json.Unmarshal(out, &obj)
	if obj["model"] != "kimi-k3.1-pro" {
		t.Errorf("model=%v", obj["model"])
	}
	if intFromAny(obj["max_tokens"], 0) != 64 {
		t.Errorf("max_tokens=%v", obj["max_tokens"])
	}
	msgs, _ := obj["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages len=%d want 2", len(msgs))
	}
	sys, _ := msgs[0].(map[string]any)
	if sys["role"] != "system" || sys["content"] != "You are helpful." {
		t.Errorf("sys=%v", sys)
	}
}

func TestAnthropicToOpenAI_SystemBlocksFilterBilling(t *testing.T) {
	in := `{"model":"m","messages":[{"role":"user","content":"hi"}],"system":[{"type":"text","text":"Line A\nx-anthropic-billing-header: secret\nLine B"}]}`
	out, err := anthropicToOpenAI([]byte(in), newToolNameMap())
	if err != nil {
		t.Fatal(err)
	}
	var obj map[string]any
	_ = json.Unmarshal(out, &obj)
	msgs, _ := obj["messages"].([]any)
	sys, _ := msgs[0].(map[string]any)
	text, _ := sys["content"].(string)
	if strings.Contains(text, "x-anthropic-billing-header") {
		t.Errorf("billing header 未过滤: %q", text)
	}
	if !strings.Contains(text, "Line A") || !strings.Contains(text, "Line B") {
		t.Errorf("原文缺失: %q", text)
	}
}

func TestAnthropicToOpenAI_ToolUseAndToolResult(t *testing.T) {
	in := `{"model":"m","messages":[
		{"role":"user","content":"what is 2+2?"},
		{"role":"assistant","content":[
			{"type":"thinking","thinking":"Should calc"},
			{"type":"tool_use","id":"call_abc","name":"calc","input":{"a":2,"b":2}}
		]},
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"call_abc","content":"4"}
		]}
	]}`
	out, err := anthropicToOpenAI([]byte(in), newToolNameMap())
	if err != nil {
		t.Fatal(err)
	}
	var obj map[string]any
	_ = json.Unmarshal(out, &obj)
	msgs, _ := obj["messages"].([]any)

	// assistant 消息应带 tool_calls + reasoning_content（DeepSeek 思考模式要求）
	var asst map[string]any
	for _, m := range msgs {
		mm, _ := m.(map[string]any)
		if mm["role"] == "assistant" {
			asst = mm
			break
		}
	}
	if asst == nil {
		t.Fatal("没有 assistant 消息")
	}
	calls, _ := asst["tool_calls"].([]any)
	if len(calls) != 1 {
		t.Fatalf("tool_calls=%v", asst["tool_calls"])
	}
	call, _ := calls[0].(map[string]any)
	if call["id"] != "call_abc" {
		t.Errorf("call id=%v", call["id"])
	}
	fn, _ := call["function"].(map[string]any)
	if fn["name"] != "calc" {
		t.Errorf("fn name=%v", fn["name"])
	}
	if !strings.Contains(fn["arguments"].(string), `"a":2`) {
		t.Errorf("args=%v", fn["arguments"])
	}
	if asst["reasoning_content"] != "Should calc" {
		t.Errorf("reasoning_content=%v", asst["reasoning_content"])
	}

	// tool_result 应被拆成独立 tool 消息
	var toolMsg map[string]any
	for _, m := range msgs {
		mm, _ := m.(map[string]any)
		if mm["role"] == "tool" {
			toolMsg = mm
			break
		}
	}
	if toolMsg == nil {
		t.Fatal("没有 tool 消息")
	}
	if toolMsg["tool_call_id"] != "call_abc" || toolMsg["content"] != "4" {
		t.Errorf("toolMsg=%v", toolMsg)
	}
}

func TestAnthropicToOpenAI_ToolsAndToolChoice(t *testing.T) {
	in := `{"model":"m","messages":[{"role":"user","content":"hi"}],
		"tools":[{"name":"calc","description":"add","input_schema":{"type":"object","properties":{"a":{"type":"number"}}}}],
		"tool_choice":{"type":"tool","name":"calc"}}`
	out, err := anthropicToOpenAI([]byte(in), newToolNameMap())
	if err != nil {
		t.Fatal(err)
	}
	var obj map[string]any
	_ = json.Unmarshal(out, &obj)
	tools, _ := obj["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools=%v", obj["tools"])
	}
	tool, _ := tools[0].(map[string]any)
	fn, _ := tool["function"].(map[string]any)
	if fn["name"] != "calc" || fn["description"] != "add" {
		t.Errorf("fn=%v", fn)
	}
	params, _ := fn["parameters"].(map[string]any)
	if params["type"] != "object" {
		t.Errorf("params=%v", params)
	}
	tc, _ := obj["tool_choice"].(map[string]any)
	if tc["type"] != "function" {
		t.Errorf("tool_choice=%v", tc)
	}
}

// ── 非流式响应转换 ────────────────────────────────────────────────────────

func TestOpenAIToAnthropic_TextAndReasoning(t *testing.T) {
	in := map[string]any{
		"id": "chatcmpl-1",
		"choices": []any{map[string]any{
			"message": map[string]any{
				"role":              "assistant",
				"content":           "hello",
				"reasoning_content": "thinking...",
			},
			"finish_reason": "stop",
		}},
		"usage": map[string]any{"prompt_tokens": float64(10), "completion_tokens": float64(5)},
	}
	out := openAIToAnthropic(in, "kimi-k3.1-pro", newToolNameMap())
	if out["type"] != "message" || out["role"] != "assistant" {
		t.Errorf("out type=%v role=%v", out["type"], out["role"])
	}
	if out["id"] != "chatcmpl-1" {
		t.Errorf("id=%v", out["id"])
	}
	content, _ := out["content"].([]map[string]any)
	if len(content) != 2 {
		t.Fatalf("content len=%d", len(content))
	}
	if content[0]["type"] != "thinking" || content[0]["thinking"] != "thinking..." {
		t.Errorf("thinking block=%v", content[0])
	}
	if content[1]["type"] != "text" || content[1]["text"] != "hello" {
		t.Errorf("text block=%v", content[1])
	}
	if out["stop_reason"] != "end_turn" {
		t.Errorf("stop_reason=%v", out["stop_reason"])
	}
	usage, _ := out["usage"].(map[string]any)
	if usage["input_tokens"] != 10 || usage["output_tokens"] != 5 {
		t.Errorf("usage=%v", usage)
	}
}

func TestOpenAIToAnthropic_ToolCalls(t *testing.T) {
	in := map[string]any{
		"id": "x",
		"choices": []any{map[string]any{
			"message": map[string]any{
				"tool_calls": []any{map[string]any{
					"id":   "call_1",
					"type": "function",
					"function": map[string]any{
						"name":      "calc",
						"arguments": `{"a":1,"b":2}`,
					},
				}},
			},
			"finish_reason": "tool_calls",
		}},
	}
	out := openAIToAnthropic(in, "m", newToolNameMap())
	content, _ := out["content"].([]map[string]any)
	if len(content) != 1 || content[0]["type"] != "tool_use" {
		t.Fatalf("content=%v", content)
	}
	if content[0]["id"] != "call_1" || content[0]["name"] != "calc" {
		t.Errorf("tool_use=%v", content[0])
	}
	input, _ := content[0]["input"].(map[string]any)
	if intFromAny(input["a"], 0) != 1 {
		t.Errorf("input=%v", input)
	}
	if out["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason=%v", out["stop_reason"])
	}
}

// ── 流式状态机 ────────────────────────────────────────────────────────────

func TestAnthropicStream_TextAndThinking(t *testing.T) {
	st := newAnthropicStream("m", newToolNameMap())
	events := st.start()
	if events[0].event != "message_start" {
		t.Fatalf("start event=%v", events[0].event)
	}

	// 第一帧：reasoning_content
	evs := st.feed(map[string]any{
		"choices": []any{map[string]any{
			"delta": map[string]any{"reasoning_content": "Should "},
		}},
	})
	// 应输出 content_block_start(thinking) + content_block_delta(thinking_delta)
	if len(evs) != 2 {
		t.Fatalf("thinking feed events=%d want 2", len(evs))
	}
	if evs[0].event != "content_block_start" {
		t.Errorf("evs[0]=%v", evs[0])
	}
	if evs[1].event != "content_block_delta" {
		t.Errorf("evs[1]=%v", evs[1])
	}
	delta, _ := evs[1].data["delta"].(map[string]any)
	if delta["type"] != "thinking_delta" || delta["thinking"] != "Should " {
		t.Errorf("delta=%v", delta)
	}

	// 第二帧：切换到 content，应先 closeThink
	evs = st.feed(map[string]any{
		"choices": []any{map[string]any{
			"delta": map[string]any{"content": "hi"},
		}},
	})
	// 期望：signature_delta + content_block_stop(think) + content_block_start(text) + content_block_delta(text_delta)
	var kinds []string
	for _, e := range evs {
		kinds = append(kinds, e.event)
	}
	joined := strings.Join(kinds, ",")
	if !strings.Contains(joined, "content_block_stop") ||
		!strings.Contains(joined, "content_block_start") ||
		!strings.Contains(joined, "content_block_delta") {
		t.Errorf("events=%v", joined)
	}

	// 收尾
	fin := st.finishEvents()
	if fin[len(fin)-1].event != "message_stop" {
		t.Errorf("last event=%v", fin[len(fin)-1].event)
	}
}

func TestAnthropicStream_ToolCallFragments(t *testing.T) {
	st := newAnthropicStream("m", newToolNameMap())
	_ = st.start()

	// 第一片：id + name + 空 arguments
	evs := st.feed(map[string]any{
		"choices": []any{map[string]any{
			"delta": map[string]any{
				"tool_calls": []any{map[string]any{
					"index":    float64(0),
					"id":       "call_xyz",
					"type":     "function",
					"function": map[string]any{"name": "calc", "arguments": ""},
				}},
			},
		}},
	})
	if len(evs) == 0 || evs[0].event != "content_block_start" {
		t.Fatalf("first fragment events=%v", evs)
	}
	cb, _ := evs[0].data["content_block"].(map[string]any)
	if cb["type"] != "tool_use" || cb["id"] != "call_xyz" || cb["name"] != "calc" {
		t.Errorf("cb=%v", cb)
	}

	// 第二片：arguments 分片
	evs = st.feed(map[string]any{
		"choices": []any{map[string]any{
			"delta": map[string]any{
				"tool_calls": []any{map[string]any{
					"index":    float64(0),
					"function": map[string]any{"arguments": `{"a":`},
				}},
			},
		}},
	})
	if len(evs) != 1 || evs[0].event != "content_block_delta" {
		t.Fatalf("args fragment events=%v", evs)
	}
	delta, _ := evs[0].data["delta"].(map[string]any)
	if delta["type"] != "input_json_delta" || delta["partial_json"] != `{"a":` {
		t.Errorf("delta=%v", delta)
	}

	// finish 时若 finish_reason 是 tool_calls，stop_reason 应为 tool_use
	st.feed(map[string]any{
		"choices": []any{map[string]any{
			"delta":         map[string]any{},
			"finish_reason": "tool_calls",
		}},
	})
	fin := st.finishEvents()
	var mdEvent anthropicEvent
	for _, e := range fin {
		if e.event == "message_delta" {
			mdEvent = e
		}
	}
	delta2, _ := mdEvent.data["delta"].(map[string]any)
	if delta2["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason=%v", delta2["stop_reason"])
	}
}

// ── 端到端路由 ────────────────────────────────────────────────────────────

func TestAnthropicRouteNonStream(t *testing.T) {
	up := newLXMockUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// 灵犀 wire：上游总是 SSE。非流式由 handler 内部聚合。
		_, _ = io.WriteString(w, "data: {\"code\":\"Success\",\"choices\":[{\"text\":\"world\",\"reasoning_content\":\"think\"}],\"finish_reason\":\"stop\",\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n")
	})
	h := NewHandler(Config{Pool: testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999}), Upstream: up})

	reqBody := `{"model":"kimi-k3.1-pro","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader([]byte(reqBody)))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("非 JSON: %s", rec.Body.String())
	}
	if out["type"] != "message" {
		t.Errorf("type=%v", out["type"])
	}
	content, _ := out["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("content len=%d body=%s", len(content), rec.Body.String())
	}
	first, _ := content[0].(map[string]any)
	if first["type"] != "thinking" {
		t.Errorf("first block=%v", first)
	}
	second, _ := content[1].(map[string]any)
	if second["type"] != "text" || second["text"] != "world" {
		t.Errorf("second block=%v", second)
	}
}

func TestAnthropicRouteStream(t *testing.T) {
	up := newLXMockUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"code\":\"Success\",\"choices\":[{\"text\":\"你好\"}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"code\":\"Success\",\"choices\":[],\"finish_reason\":\"stop\",\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n")
	})
	h := NewHandler(Config{Pool: testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999}), Upstream: up})

	reqBody := `{"model":"kimi-k3.1-pro","max_tokens":32,"stream":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader([]byte(reqBody)))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	for _, want := range []string{"event: message_start", "event: content_block_start", "event: content_block_delta", "event: message_delta", "event: message_stop"} {
		if !strings.Contains(body, want) {
			t.Errorf("缺少事件 %q; body=\n%s", want, body)
		}
	}
	if !strings.Contains(body, `"type":"text_delta"`) || !strings.Contains(body, `"text":"你好"`) {
		t.Errorf("缺少 text_delta; body=\n%s", body)
	}
}

func TestAnthropicRouteErrorPassthrough(t *testing.T) {
	// 模拟选不到号：Pool 里没账号 → chatCompletions 走 no_healthy_account 错误。
	// 验证错误被转成 Anthropic 错误格式 {"type":"error","error":{...}}。
	up := newFakeUpstream(t, func(authz string) (int, string, bool) {
		return http.StatusInternalServerError, `{"error":"unexpected"}`, false
	})
	h := NewHandler(Config{Pool: testPoolWith(), Upstream: up})

	req := httptest.NewRequest(http.MethodPost, "/v1/messages",
		bytes.NewReader([]byte(`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code < 400 {
		t.Fatalf("code=%d 应为 4xx/5xx", rec.Code)
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("非 JSON: %s", rec.Body.String())
	}
	if out["type"] != "error" {
		t.Errorf("type=%v 应为 error", out["type"])
	}
	errObj, _ := out["error"].(map[string]any)
	if errObj == nil || errObj["message"] == "" {
		t.Errorf("error.message 缺失: %v", out)
	}
}
