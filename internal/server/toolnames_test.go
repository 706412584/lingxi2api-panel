package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lingxi2api/lingxi2api-panel/internal/auth"
)

// ── sanitizeToolName ────────────────────────────────────────────────────────

func TestSanitizeToolName(t *testing.T) {
	cases := []struct {
		in       string
		want     string // 空 = 不要求精确，只验证合法性 + changed 标志
		changed  bool
		validate func(string) bool
	}{
		// 合法名原样返回，不改。
		{in: "calc", want: "calc", changed: false},
		{in: "get_weather", want: "get_weather", changed: false},
		{in: "mcp__server__tool", want: "mcp__server__tool", changed: false},
		{in: "tool-with-dash", want: "tool-with-dash", changed: false},
		{in: "_private", want: "_private", changed: false},

		// 非法字符折叠。
		{in: "some.tool", changed: true, validate: func(s string) bool { return s == "some_tool" }},
		{in: "some tool", changed: true, validate: func(s string) bool { return s == "some_tool" }},
		{in: "some/tool", changed: true, validate: func(s string) bool { return s == "some_tool" }},
		{in: "some...tool", changed: true, validate: func(s string) bool { return s == "some_tool" }},
		{in: "   padded   ", changed: true, validate: func(s string) bool { return s == "padded" }},

		// 中文 / 全非法字符 → 合成 hash 名。
		{in: "计算器", changed: true, validate: func(s string) bool {
			return strings.HasPrefix(s, "tool_") && validToolName(s)
		}},
		{in: "???", changed: true, validate: func(s string) bool {
			return strings.HasPrefix(s, "tool_") && validToolName(s)
		}},

		// 数字/连字符开头在 OpenAI 规范下合法，不修复（与 validToolName 一致）。
		{in: "3d-print", want: "3d-print", changed: false},
		{in: "-lead-dash", want: "-lead-dash", changed: false},

		// 超长按 hash 截断。
		{in: strings.Repeat("a", 100), changed: true, validate: func(s string) bool {
			return len(s) <= 64 && validToolName(s)
		}},
	}
	for _, c := range cases {
		got, changed := sanitizeToolName(c.in)
		if changed != c.changed {
			t.Errorf("sanitize(%q).changed=%v want %v (got=%q)", c.in, changed, c.changed, got)
		}
		if !validToolName(got) {
			t.Errorf("sanitize(%q)=%q 不是合法 OpenAI 工具名", c.in, got)
		}
		if c.want != "" && got != c.want {
			t.Errorf("sanitize(%q)=%q want %q", c.in, got, c.want)
		}
		if c.validate != nil && !c.validate(got) {
			t.Errorf("sanitize(%q)=%q 校验失败", c.in, got)
		}
	}
}

// 同输入必得同输出（重试/分片场景下不能漂移）。
func TestSanitizeToolName_Deterministic(t *testing.T) {
	for _, name := range []string{"计算器", "some.tool", strings.Repeat("x", 80), "???", "3d-print"} {
		a, _ := sanitizeToolName(name)
		b, _ := sanitizeToolName(name)
		if a != b {
			t.Errorf("sanitize(%q) 不确定: %q vs %q", name, a, b)
		}
	}
}

// ── toolNameMap 双向映射 ────────────────────────────────────────────────────

func TestToolNameMap_ForwardAndReverse(t *testing.T) {
	m := newToolNameMap()
	// 合法名不进入映射。
	if got := m.sanitize("calc"); got != "calc" {
		t.Errorf("sanitize(calc)=%q", got)
	}
	if m.changed() != 0 {
		t.Errorf("合法名不应登记: changed=%d", m.changed())
	}
	// 非法名登记 + 还原。
	san := m.sanitize("some.tool")
	if san != "some_tool" {
		t.Errorf("sanitize(some.tool)=%q", san)
	}
	if m.changed() != 1 {
		t.Errorf("changed=%d", m.changed())
	}
	if got := m.restore(san); got != "some.tool" {
		t.Errorf("restore(%q)=%q want some.tool", san, got)
	}
	// 同名重复 sanitize 幂等。
	if got := m.sanitize("some.tool"); got != san {
		t.Errorf("重复 sanitize 漂移: %q vs %q", got, san)
	}
	if m.changed() != 1 {
		t.Errorf("重复登记不应增加映射: changed=%d", m.changed())
	}
	// 未登记的 restore 原样返回。
	if got := m.restore("unknown"); got != "unknown" {
		t.Errorf("restore(unknown)=%q", got)
	}
}

// 两个原名映射到同一 sanitized 时，reverse 保留第一个，避免串号。
func TestToolNameMap_ReverseConflictKeepsFirst(t *testing.T) {
	m := newToolNameMap()
	a := m.sanitize("foo.bar")
	b := m.sanitize("foo bar")
	if a != b {
		t.Skipf("本用例要求冲突: %q vs %q", a, b)
	}
	if got := m.restore(a); got != "foo.bar" {
		t.Errorf("restore 应保留第一个原名: %q", got)
	}
}

// ── 端到端：请求侧修复 + 响应侧还原 ────────────────────────────────────────

// 客户端声明带点的工具名 some.tool，请求侧应改名为 some_tool，响应侧返回给
// 客户端的 tool_use block 名字必须还是 some.tool（客户端用它匹配 tool_result）。
func TestAnthropicRouteToolNameRoundTrip(t *testing.T) {
	up := newLXMockUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		// 读取出站请求，验证工具名已被修复。
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		tools, _ := req["tools"].([]any)
		if len(tools) == 0 {
			t.Errorf("上游应收到 tools")
		} else {
			tool, _ := tools[0].(map[string]any)
			fn, _ := tool["function"].(map[string]any)
			if fn["name"] != "some_tool" {
				t.Errorf("出站工具名应为 some_tool，实际 %v", fn["name"])
			}
		}
		// 返回带修复后名字的 tool_call。
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"code":"Success","choices":[{"text":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"some_tool","arguments":"{\"a\":1}"}}]}],"finish_reason":"tool_calls","usage":{"prompt_tokens":1,"completion_tokens":1}}`+"\n\n")
	})
	h := NewHandler(Config{Pool: testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999}), Upstream: up})

	body := `{"model":"kimi-k3.1-pro","max_tokens":64,"messages":[{"role":"user","content":"hi"}],
		"tools":[{"name":"some.tool","description":"x","input_schema":{"type":"object","properties":{}}}],
		"tool_choice":{"type":"tool","name":"some.tool"}}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader([]byte(body)))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("非 JSON: %s", rec.Body.String())
	}
	content, _ := out["content"].([]any)
	if len(content) == 0 {
		t.Fatalf("没有 content block: %s", rec.Body.String())
	}
	var toolUse map[string]any
	for _, c := range content {
		if cc, _ := c.(map[string]any); cc["type"] == "tool_use" {
			toolUse = cc
			break
		}
	}
	if toolUse == nil {
		t.Fatalf("没有 tool_use block: %s", rec.Body.String())
	}
	// 客户端看到的必须是它原本声明的 some.tool，不是 some_tool。
	if toolUse["name"] != "some.tool" {
		t.Errorf("响应工具名应还原为 some.tool，实际 %v", toolUse["name"])
	}
}

// 流式 tool_use 名字也要还原。
func TestAnthropicRouteStreamToolNameRoundTrip(t *testing.T) {
	up := newLXMockUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// 流式分片：第一片给 id+name，第二片给 args。
		_, _ = io.WriteString(w, `data: {"code":"Success","choices":[{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"some_tool","arguments":""}}]}]}`+"\n\n")
		_, _ = io.WriteString(w, `data: {"code":"Success","choices":[{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":1}"}}]}],"finish_reason":"tool_calls"}`+"\n\n")
	})
	h := NewHandler(Config{Pool: testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999}), Upstream: up})

	body := `{"model":"kimi-k3.1-pro","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}],
		"tools":[{"name":"some.tool","description":"x","input_schema":{"type":"object","properties":{}}}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader([]byte(body)))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body.String())
	}
	text := rec.Body.String()
	if !strings.Contains(text, `"name":"some.tool"`) {
		t.Errorf("流式 tool_use block 名字应还原为 some.tool; body=\n%s", text)
	}
	if strings.Contains(text, `"name":"some_tool"`) {
		t.Errorf("流式不应泄漏修复后的名字; body=\n%s", text)
	}
}

// tool_choice.type="tool" 的 name 也要修复。
func TestAnthropicRouteToolChoiceNameRepair(t *testing.T) {
	var sawToolChoice map[string]any
	up := newLXMockUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		sawToolChoice, _ = req["tool_choice"].(map[string]any)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"code":"Success","choices":[{"text":"ok"}],"finish_reason":"stop","usage":{"prompt_tokens":1,"completion_tokens":1}}`+"\n\n")
	})
	h := NewHandler(Config{Pool: testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999}), Upstream: up})

	body := `{"model":"kimi-k3.1-pro","max_tokens":32,"messages":[{"role":"user","content":"hi"}],
		"tools":[{"name":"get weather","description":"x","input_schema":{"type":"object","properties":{}}}],
		"tool_choice":{"type":"tool","name":"get weather"}}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader([]byte(body)))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body.String())
	}
	fn, _ := sawToolChoice["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("tool_choice.name 应修复为 get_weather，实际 %v", fn["name"])
	}
}

// 历史 tool_use（在 messages 里）也要修复。
func TestAnthropicRouteHistoryToolNameRepair(t *testing.T) {
	up := newLXMockUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		msgs, _ := req["messages"].([]any)
		for _, m := range msgs {
			mm, _ := m.(map[string]any)
			if mm["role"] != "assistant" {
				continue
			}
			calls, _ := mm["tool_calls"].([]any)
			if len(calls) == 0 {
				continue
			}
			call, _ := calls[0].(map[string]any)
			fn, _ := call["function"].(map[string]any)
			if fn["name"] != "some_tool" {
				t.Errorf("历史 tool_use 名字应修复为 some_tool，实际 %v", fn["name"])
			}
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"code":"Success","choices":[{"text":"ok"}],"finish_reason":"stop","usage":{"prompt_tokens":1,"completion_tokens":1}}`+"\n\n")
	})
	h := NewHandler(Config{Pool: testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999}), Upstream: up})

	body := `{"model":"kimi-k3.1-pro","max_tokens":32,"messages":[
		{"role":"user","content":"hi"},
		{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"some.tool","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":"done"}]}
	]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader([]byte(body)))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body.String())
	}
}

// ── 工具 schema 形状修复 ────────────────────────────────────────────────────

// 缺 description / 缺 input_schema / 缺 properties：都要补成合法形状。
func TestAnthropicToOpenAI_ToolSchemaDefaults(t *testing.T) {
	in := `{"model":"m","messages":[{"role":"user","content":"hi"}],
		"tools":[{"name":"calc"},{"name":"weather","description":"x"}]}`
	out, err := anthropicToOpenAI([]byte(in), newToolNameMap())
	if err != nil {
		t.Fatal(err)
	}
	var obj map[string]any
	_ = json.Unmarshal(out, &obj)
	tools, _ := obj["tools"].([]any)
	if len(tools) != 2 {
		t.Fatalf("tools=%v", tools)
	}
	for i, rawT := range tools {
		tool, _ := rawT.(map[string]any)
		fn, _ := tool["function"].(map[string]any)
		if _, ok := fn["description"].(string); !ok {
			t.Errorf("tools[%d] 缺 description", i)
		}
		params, _ := fn["parameters"].(map[string]any)
		if params["type"] != "object" {
			t.Errorf("tools[%d] parameters.type=%v 应为 object", i, params["type"])
		}
		if _, ok := params["properties"].(map[string]any); !ok {
			t.Errorf("tools[%d] parameters.properties 缺失", i)
		}
	}
}

// tool_choice 各形态：auto/any/none/tool。
func TestAnthropicToOpenAI_ToolChoiceVariants(t *testing.T) {
	cases := []struct {
		in   string
		want any
	}{
		{in: `{"type":"auto"}`, want: "auto"},
		{in: `{"type":"any"}`, want: "required"},
		{in: `{"type":"none"}`, want: "none"},
	}
	for _, c := range cases {
		in := `{"model":"m","messages":[{"role":"user","content":"hi"}],"tool_choice":` + c.in + `}`
		out, err := anthropicToOpenAI([]byte(in), newToolNameMap())
		if err != nil {
			t.Fatal(err)
		}
		var obj map[string]any
		_ = json.Unmarshal(out, &obj)
		if obj["tool_choice"] != c.want {
			t.Errorf("tool_choice %s → %v want %v", c.in, obj["tool_choice"], c.want)
		}
	}
}

// 并行工具调用（delta 里 index=0 和 index=1 同时流）。
func TestAnthropicStream_ParallelToolCalls(t *testing.T) {
	st := newAnthropicStream("m", newToolNameMap())
	_ = st.start()
	evs := st.feed(map[string]any{
		"choices": []any{map[string]any{
			"delta": map[string]any{
				"tool_calls": []any{
					map[string]any{"index": float64(0), "id": "c1", "type": "function",
						"function": map[string]any{"name": "a", "arguments": ""}},
					map[string]any{"index": float64(1), "id": "c2", "type": "function",
						"function": map[string]any{"name": "b", "arguments": ""}},
				},
			},
		}},
	})
	// 应有 2 个 content_block_start
	starts := 0
	for _, e := range evs {
		if e.event == "content_block_start" {
			starts++
		}
	}
	if starts != 2 {
		t.Errorf("并行工具应产生 2 个 content_block_start，实际 %d", starts)
	}
}

// 无参数工具：finish 时自动补 {}。
func TestAnthropicStream_NoArgsToolGetsEmptyObject(t *testing.T) {
	st := newAnthropicStream("m", newToolNameMap())
	_ = st.start()
	st.feed(map[string]any{
		"choices": []any{map[string]any{
			"delta": map[string]any{
				"tool_calls": []any{map[string]any{
					"index": float64(0), "id": "c1", "type": "function",
					"function": map[string]any{"name": "ping", "arguments": ""},
				}},
			},
			"finish_reason": "tool_calls",
		}},
	})
	fin := st.finishEvents()
	// 应有一个 input_json_delta partial_json="{}"
	found := false
	for _, e := range fin {
		if e.event != "content_block_delta" {
			continue
		}
		delta, _ := e.data["delta"].(map[string]any)
		if delta["type"] == "input_json_delta" && delta["partial_json"] == "{}" {
			found = true
		}
	}
	if !found {
		t.Errorf("无参数工具 finish 应补 {} 的 input_json_delta")
	}
}
