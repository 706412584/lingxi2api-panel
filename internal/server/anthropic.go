// Package server Anthropic Messages API 适配器。
//
// 把 Anthropic 的 /v1/messages 请求/响应翻译到既有的 OpenAI Chat Completions
// 通路（h.chatCompletions）。整体形态对齐 loomy2api/anthropic.py 的行为：
//   - 请求：system/messages/tools 完整翻译；tool_use/tool_result 双向还原；
//     Claude Code 的 billing header 行过滤；thinking 块在 assistant 消息上
//     透传为 reasoning_content（DeepSeek 思考模式要求重放）。
//   - 响应：非流式把 reasoning_content → thinking block，content → text，
//     tool_calls → tool_use；流式按 content_block_index 状态机输出
//     message_start/content_block_*/message_delta/message_stop 序列。
//
// 适配器与 chatCompletions 的关系是 ResponseWriter 包装：请求改写后调用
// 既有的 chatCompletions，从而复用选号/粘性/冷却/统计/审核识别全部逻辑。
package server

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// ── 请求转换：Anthropic → OpenAI ────────────────────────────────────────────

// anthropicToOpenAI 把 Anthropic /v1/messages 请求体翻译为 OpenAI Chat Completions 请求体。
// 行为对齐 loomy2api/anthropic.py::anthropic_to_openai，并额外做工具名兼容修复：
// 上游灵犀对工具名遵循 OpenAI 规范 ^[a-zA-Z0-9_-]{1,64}$，客户端声明的名字
// 若带空格/中文/点/斜杠会被上游 AIError 拒绝。修复是双向的：请求侧改名，
// 响应侧（openAIToAnthropic / 流式 translator）通过同一个 names 映射还原。
func anthropicToOpenAI(body []byte, names *toolNameMap) ([]byte, error) {
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("parse anthropic body: %w", err)
	}

	messages := make([]map[string]any, 0, 8)

	// system：str 或 [{type:"text",...}]。Claude Code 在 system 里夹带
	// x-anthropic-billing-header 行，过滤后透传。
	if sys, ok := req["system"]; ok {
		text := blocksToText(sys)
		var kept []string
		for _, line := range strings.Split(text, "\n") {
			if strings.Contains(line, "x-anthropic-billing-header") {
				continue
			}
			kept = append(kept, line)
		}
		text = strings.TrimSpace(strings.Join(kept, "\n"))
		if text != "" {
			messages = append(messages, map[string]any{"role": "system", "content": text})
		}
	}

	rawTurns, _ := req["messages"].([]any)
	for _, rawTurn := range rawTurns {
		turn, _ := rawTurn.(map[string]any)
		role, _ := turn["role"].(string)
		content := turn["content"]
		if s, ok := content.(string); ok {
			messages = append(messages, map[string]any{"role": role, "content": s})
			continue
		}
		blocks, _ := content.([]any)

		var texts []string
		var parts []map[string]any // 有序的 text/image parts
		var images []map[string]any
		var toolCalls []map[string]any
		var thinkings []string
		var toolMessages []map[string]any

		for _, rawBlock := range blocks {
			block, _ := rawBlock.(map[string]any)
			kind, _ := block["type"].(string)
			switch kind {
			case "text":
				if t, _ := block["text"].(string); t != "" {
					texts = append(texts, t)
					parts = append(parts, map[string]any{"type": "text", "text": t})
				}
			case "image":
				if part := anthropicImagePart(block); part != nil {
					images = append(images, part)
					parts = append(parts, part)
				}
			case "tool_use":
				id, _ := block["id"].(string)
				if id == "" {
					id = "call_" + randHex(12)
				}
				argsJSON, _ := json.Marshal(block["input"])
				rawName, _ := block["name"].(string)
				toolCalls = append(toolCalls, map[string]any{
					"id":   id,
					"type": "function",
					"function": map[string]any{
						"name":      names.sanitize(rawName),
						"arguments": string(argsJSON),
					},
				})
			case "tool_result":
				text := toolResultText(block)
				toolMessages = append(toolMessages, map[string]any{
					"role":         "tool",
					"tool_call_id": block["tool_use_id"],
					"content":      text,
				})
				// tool_result 里嵌的图片随 user 轮一起透传
				if inner, ok := block["content"].([]any); ok {
					for _, item := range inner {
						if ib, _ := item.(map[string]any); ib != nil && ib["type"] == "image" {
							if part := anthropicImagePart(ib); part != nil {
								images = append(images, part)
								parts = append(parts, part)
							}
						}
					}
				}
			case "thinking":
				if t, _ := block["thinking"].(string); t != "" {
					thinkings = append(thinkings, t)
				}
			}
		}

		if role == "assistant" {
			out := map[string]any{"role": "assistant"}
			if len(texts) > 0 {
				out["content"] = strings.Join(texts, "\n")
			} else {
				out["content"] = nil
			}
			if len(toolCalls) > 0 {
				out["tool_calls"] = toolCalls
				// DeepSeek 思考模式要求重放产生工具调用的 reasoning，否则拒 400。
				if len(thinkings) > 0 {
					out["reasoning_content"] = strings.Join(thinkings, "\n")
				}
			}
			if out["content"] != nil || len(toolCalls) > 0 {
				messages = append(messages, out)
			}
			continue
		}

		// tool_result 落在 user 轮里：拆成独立 tool 消息，剩余 text/images 跟随。
		messages = append(messages, toolMessages...)
		if len(texts) > 0 || len(images) > 0 {
			if len(images) > 0 {
				messages = append(messages, map[string]any{"role": role, "content": parts})
			} else {
				messages = append(messages, map[string]any{"role": role, "content": strings.Join(texts, "\n")})
			}
		} else if len(toolMessages) == 0 {
			messages = append(messages, map[string]any{"role": role, "content": ""})
		}
	}

	if len(messages) == 0 {
		messages = append(messages, map[string]any{"role": "user", "content": ""})
	}

	out := map[string]any{
		"model":    req["model"],
		"messages": messages,
		"stream":   req["stream"] == true,
	}
	for _, k := range []string{"max_tokens", "temperature", "top_p"} {
		if v, ok := req[k]; ok && v != nil {
			out[k] = v
		}
	}
	if ss, ok := req["stop_sequences"]; ok {
		out["stop"] = ss
	}
	if md, ok := req["metadata"].(map[string]any); ok {
		out["metadata"] = md
	}
	out["parallel_tool_calls"] = true

	if rawTools, ok := req["tools"].([]any); ok && len(rawTools) > 0 {
		tools := make([]map[string]any, 0, len(rawTools))
		for _, rawT := range rawTools {
			t, _ := rawT.(map[string]any)
			name, _ := t["name"].(string)
			if name == "" {
				continue
			}
			params := t["input_schema"]
			if params == nil {
				params = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			desc, _ := t["description"].(string)
			tools = append(tools, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        names.sanitize(name),
					"description": desc,
					"parameters":  params,
				},
			})
		}
		out["tools"] = tools
	}

	if choice, ok := req["tool_choice"].(map[string]any); ok {
		switch choice["type"] {
		case "auto":
			out["tool_choice"] = "auto"
		case "any":
			out["tool_choice"] = "required"
		case "none":
			out["tool_choice"] = "none"
		case "tool":
			if name, _ := choice["name"].(string); name != "" {
				out["tool_choice"] = map[string]any{
					"type":     "function",
					"function": map[string]any{"name": names.sanitize(name)},
				}
			}
		}
	}

	return json.Marshal(out)
}

// blocksToText 把 Anthropic system / tool_result.content 等多形态内容压成纯文本。
func blocksToText(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case []any:
		var parts []string
		for _, item := range t {
			switch b := item.(type) {
			case map[string]any:
				switch b["type"] {
				case "text":
					if s, _ := b["text"].(string); s != "" {
						parts = append(parts, s)
					}
				case "tool_result":
					if s := blocksToText(b["content"]); s != "" {
						parts = append(parts, s)
					}
				}
			case string:
				if b != "" {
					parts = append(parts, b)
				}
			}
		}
		return strings.Join(parts, "\n")
	case nil:
		return ""
	default:
		return fmt.Sprint(v)
	}
}

func toolResultText(block map[string]any) string {
	switch c := block["content"].(type) {
	case string:
		if c == "" {
			return "(empty)"
		}
		return c
	default:
		s := blocksToText(c)
		if s == "" {
			return "(empty)"
		}
		return s
	}
}

func anthropicImagePart(block map[string]any) map[string]any {
	source, _ := block["source"].(map[string]any)
	switch source["type"] {
	case "base64":
		media, _ := source["media_type"].(string)
		if media == "" {
			media = "image/png"
		}
		data, _ := source["data"].(string)
		if data == "" {
			return nil
		}
		return map[string]any{
			"type":      "image_url",
			"image_url": map[string]any{"url": "data:" + media + ";base64," + data},
		}
	case "url":
		u, _ := source["url"].(string)
		if u == "" {
			return nil
		}
		return map[string]any{"type": "image_url", "image_url": map[string]any{"url": u}}
	}
	return nil
}

// ── 非流式响应转换：OpenAI → Anthropic ────────────────────────────────────

func openAIToAnthropic(obj map[string]any, model string, names *toolNameMap) map[string]any {
	choices, _ := obj["choices"].([]any)
	var choice map[string]any
	if len(choices) > 0 {
		choice, _ = choices[0].(map[string]any)
	}
	if choice == nil {
		choice = map[string]any{}
	}
	message, _ := choice["message"].(map[string]any)
	if message == nil {
		message = map[string]any{}
	}

	content := make([]map[string]any, 0, 4)

	reasoning := stringField(message, "reasoning_content")
	if reasoning == "" {
		reasoning = stringField(message, "reasoning")
	}
	if reasoning != "" {
		content = append(content, map[string]any{
			"type": "thinking", "thinking": reasoning, "signature": "",
		})
	}
	if text := stringField(message, "content"); text != "" {
		content = append(content, map[string]any{"type": "text", "text": text})
	}
	if calls, ok := message["tool_calls"].([]any); ok {
		for _, rawCall := range calls {
			call, _ := rawCall.(map[string]any)
			fn, _ := call["function"].(map[string]any)
			argsText := stringField(fn, "arguments")
			var args any = map[string]any{}
			if argsText != "" {
				_ = json.Unmarshal([]byte(argsText), &args)
			}
			id := stringField(call, "id")
			if id == "" {
				id = "call_" + randHex(12)
			}
			content = append(content, map[string]any{
				"type":  "tool_use",
				"id":    id,
				"name":  names.restore(stringField(fn, "name")),
				"input": args,
			})
		}
	}

	finish := stringField(choice, "finish_reason")
	stopReason := "end_turn"
	switch finish {
	case "length":
		stopReason = "max_tokens"
	case "tool_calls":
		stopReason = "tool_use"
	}

	usage, _ := obj["usage"].(map[string]any)
	id := stringField(obj, "id")
	if id == "" {
		id = "msg_" + randHex(24)
	}
	return map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"model":         model,
		"content":       content,
		"stop_reason":   stopReason,
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":  intField(usage, "prompt_tokens"),
			"output_tokens": intField(usage, "completion_tokens"),
		},
	}
}

// ── 流式响应状态机：OpenAI SSE → Anthropic SSE ────────────────────────────

// anthropicStreamTranslator 把上游 OpenAI SSE chunk 序列翻译成 Anthropic SSE 事件序列。
//
// 用法：
//
//	st := newAnthropicStream(model)
//	events := st.start()                  // message_start
//	for chunk := range openaiChunks {
//	    events = st.feed(chunk)
//	    // 输出 events...
//	}
//	events = st.finish()                  // message_delta + message_stop
//
// 每个返回的事件是 (event, data) 二元组，调用方负责序列化成 SSE 行。
type anthropicStreamTranslator struct {
	model      string
	names      *toolNameMap
	messageID  string
	usage      map[string]any
	finish     string
	nextIndex  int
	thinkIndex int // -1 = 未打开
	thinkSig   string
	textIndex  int // -1 = 未打开
	tools      map[int]*anthropicToolBlock
	openTool   int
}

type anthropicToolBlock struct {
	index    int
	id       string
	name     string
	argsText strings.Builder
	argsSeen bool
}

type anthropicEvent struct {
	event string
	data  map[string]any
}

func newAnthropicStream(model string, names *toolNameMap) *anthropicStreamTranslator {
	return &anthropicStreamTranslator{
		model:      model,
		names:      names,
		messageID:  "msg_" + randHex(24),
		usage:      map[string]any{},
		thinkIndex: -1,
		textIndex:  -1,
		tools:      map[int]*anthropicToolBlock{},
		openTool:   -1,
	}
}

func (st *anthropicStreamTranslator) start() []anthropicEvent {
	return []anthropicEvent{{
		event: "message_start",
		data: map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id":            st.messageID,
				"type":          "message",
				"role":          "assistant",
				"model":         st.model,
				"content":       []any{},
				"stop_reason":   nil,
				"stop_sequence": nil,
				"usage":         map[string]any{"input_tokens": 0, "output_tokens": 0},
			},
		},
	}}
}

func (st *anthropicStreamTranslator) closeThink(out *[]anthropicEvent) {
	if st.thinkIndex < 0 {
		return
	}
	*out = append(*out, anthropicEvent{
		event: "content_block_delta",
		data: map[string]any{
			"type":  "content_block_delta",
			"index": st.thinkIndex,
			"delta": map[string]any{"type": "signature_delta", "signature": st.thinkSig},
		},
	})
	*out = append(*out, anthropicEvent{
		event: "content_block_stop",
		data:  map[string]any{"type": "content_block_stop", "index": st.thinkIndex},
	})
	st.thinkIndex = -1
	st.thinkSig = ""
}

func (st *anthropicStreamTranslator) closeText(out *[]anthropicEvent) {
	if st.textIndex < 0 {
		return
	}
	*out = append(*out, anthropicEvent{
		event: "content_block_stop",
		data:  map[string]any{"type": "content_block_stop", "index": st.textIndex},
	})
	st.textIndex = -1
}

func (st *anthropicStreamTranslator) feed(obj map[string]any) []anthropicEvent {
	var out []anthropicEvent
	if usage, ok := obj["usage"].(map[string]any); ok {
		for k, v := range usage {
			st.usage[k] = v
		}
	}
	choices, _ := obj["choices"].([]any)
	if len(choices) == 0 {
		return out
	}
	choice, _ := choices[0].(map[string]any)
	delta, _ := choice["delta"].(map[string]any)

	thinking := stringField(delta, "reasoning_content")
	if thinking == "" {
		thinking = stringField(delta, "reasoning")
	}
	if thinking != "" {
		if st.thinkIndex < 0 {
			st.closeText(&out)
			st.thinkIndex = st.nextIndex
			st.nextIndex++
			out = append(out, anthropicEvent{
				event: "content_block_start",
				data: map[string]any{
					"type":          "content_block_start",
					"index":         st.thinkIndex,
					"content_block": map[string]any{"type": "thinking", "thinking": ""},
				},
			})
		}
		out = append(out, anthropicEvent{
			event: "content_block_delta",
			data: map[string]any{
				"type":  "content_block_delta",
				"index": st.thinkIndex,
				"delta": map[string]any{"type": "thinking_delta", "thinking": thinking},
			},
		})
	}

	if text := stringField(delta, "content"); text != "" {
		st.closeThink(&out)
		if st.textIndex < 0 {
			st.textIndex = st.nextIndex
			st.nextIndex++
			out = append(out, anthropicEvent{
				event: "content_block_start",
				data: map[string]any{
					"type":          "content_block_start",
					"index":         st.textIndex,
					"content_block": map[string]any{"type": "text", "text": ""},
				},
			})
		}
		out = append(out, anthropicEvent{
			event: "content_block_delta",
			data: map[string]any{
				"type":  "content_block_delta",
				"index": st.textIndex,
				"delta": map[string]any{"type": "text_delta", "text": text},
			},
		})
	}

	if calls, ok := delta["tool_calls"].([]any); ok {
		for _, rawCall := range calls {
			call, _ := rawCall.(map[string]any)
			fn, _ := call["function"].(map[string]any)
			tidx := intFromAny(call["index"], len(st.tools))
			block := st.tools[tidx]
			if block == nil {
				st.closeThink(&out)
				st.closeText(&out)
				id := stringField(call, "id")
				if id == "" {
					id = "call_" + randHex(12)
				}
				block = &anthropicToolBlock{
					index: st.nextIndex,
					id:    id,
					name:  st.names.restore(stringField(fn, "name")),
				}
				st.nextIndex++
				st.tools[tidx] = block
				out = append(out, anthropicEvent{
					event: "content_block_start",
					data: map[string]any{
						"type":  "content_block_start",
						"index": block.index,
						"content_block": map[string]any{
							"type":  "tool_use",
							"id":    block.id,
							"name":  block.name,
							"input": map[string]any{},
						},
					},
				})
			}
			if block.id == "" {
				if id := stringField(call, "id"); id != "" {
					block.id = id
				}
			}
			if block.name == "" {
				if n := stringField(fn, "name"); n != "" {
					block.name = n
				}
			}
			if args := stringField(fn, "arguments"); args != "" {
				block.argsSeen = true
				block.argsText.WriteString(args)
				out = append(out, anthropicEvent{
					event: "content_block_delta",
					data: map[string]any{
						"type":  "content_block_delta",
						"index": block.index,
						"delta": map[string]any{"type": "input_json_delta", "partial_json": args},
					},
				})
			}
		}
	}

	if finish := stringField(choice, "finish_reason"); finish != "" {
		st.finish = finish
	}
	return out
}

func (st *anthropicStreamTranslator) finishEvents() []anthropicEvent {
	var out []anthropicEvent
	st.closeThink(&out)
	st.closeText(&out)
	hasTool := len(st.tools) > 0
	for _, block := range st.tools {
		if !block.argsSeen {
			out = append(out, anthropicEvent{
				event: "content_block_delta",
				data: map[string]any{
					"type":  "content_block_delta",
					"index": block.index,
					"delta": map[string]any{"type": "input_json_delta", "partial_json": "{}"},
				},
			})
		}
		out = append(out, anthropicEvent{
			event: "content_block_stop",
			data:  map[string]any{"type": "content_block_stop", "index": block.index},
		})
	}
	st.tools = map[int]*anthropicToolBlock{}

	stopReason := "end_turn"
	switch st.finish {
	case "length":
		stopReason = "max_tokens"
	case "tool_calls":
		stopReason = "tool_use"
	default:
		if hasTool {
			stopReason = "tool_use"
		}
	}
	out = append(out, anthropicEvent{
		event: "message_delta",
		data: map[string]any{
			"type":  "message_delta",
			"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil},
			"usage": map[string]any{"output_tokens": intField(st.usage, "completion_tokens")},
		},
	})
	out = append(out, anthropicEvent{
		event: "message_stop",
		data:  map[string]any{"type": "message_stop"},
	})
	return out
}

// ── ResponseWriter 包装 ────────────────────────────────────────────────────

// anthropicResponseWriter 把 chatCompletions 写到 w 的 OpenAI 响应翻译成 Anthropic。
//   - status >= 400：OpenAI 错误 JSON 转 Anthropic 错误 JSON。
//   - 非流式：缓冲整个 OpenAI JSON，最后一次性翻译。
//   - 流式：按 SSE chunk 行解析，每行翻译成若干 anthropic event 后立即 flush。
type anthropicResponseWriter struct {
	inner     http.ResponseWriter
	model     string
	streaming bool
	names     *toolNameMap

	status  int
	header  http.Header
	headers bool // 是否已写出 header（status 0 = 200）

	// 非流式：累积完整 OpenAI JSON
	nonStreamBuf bytes.Buffer

	// 流式：跨 Write 调用累积 SSE 行（OpenAI 一行可能被切成多次 Write）
	sseBuf   bytes.Buffer
	streamer *anthropicStreamTranslator
	started  bool // message_start 是否已写
	finished bool
}

func newAnthropicWriter(inner http.ResponseWriter, model string, streaming bool, names *toolNameMap) *anthropicResponseWriter {
	return &anthropicResponseWriter{
		inner:     inner,
		model:     model,
		streaming: streaming,
		names:     names,
		header:    make(http.Header),
	}
}

func (w *anthropicResponseWriter) Header() http.Header { return w.header }

func (w *anthropicResponseWriter) WriteHeader(status int) {
	w.status = status
}

// flushInnerHeaders 把累积的 header 写到 inner（按需覆盖 Content-Type）。
func (w *anthropicResponseWriter) flushInnerHeaders(contentType string) {
	if w.headers {
		return
	}
	w.headers = true
	for k, vs := range w.header {
		if strings.EqualFold(k, "Content-Type") || strings.EqualFold(k, "Content-Length") {
			continue
		}
		for _, v := range vs {
			w.inner.Header().Add(k, v)
		}
	}
	if contentType != "" {
		w.inner.Header().Set("Content-Type", contentType)
	}
	status := w.status
	if status == 0 {
		status = http.StatusOK
	}
	w.inner.WriteHeader(status)
}

func (w *anthropicResponseWriter) Write(p []byte) (int, error) {
	if w.status >= 400 {
		// 错误路径：缓冲，Close 时统一转错误 JSON
		return w.nonStreamBuf.Write(p)
	}
	if !w.streaming {
		return w.nonStreamBuf.Write(p)
	}
	// 流式：累积并按行处理
	w.sseBuf.Write(p)
	if !w.started {
		w.started = true
		w.flushInnerHeaders("text/event-stream")
		w.streamer = newAnthropicStream(w.model, w.names)
		for _, ev := range w.streamer.start() {
			w.emitEvent(ev)
		}
		w.flushNow()
	}
	w.processSSELines()
	return len(p), nil
}

// processSSELines 把 sseBuf 里完整的 SSE 行翻译成 anthropic 事件并 flush。
func (w *anthropicResponseWriter) processSSELines() {
	for {
		data := w.sseBuf.Bytes()
		idx := bytes.Index(data, []byte("\n\n"))
		if idx < 0 {
			break
		}
		chunk := string(data[:idx])
		w.sseBuf.Next(idx + 2)
		w.handleSSEChunk(chunk)
	}
	w.flushNow()
}

func (w *anthropicResponseWriter) handleSSEChunk(chunk string) {
	for _, line := range strings.Split(chunk, "\n") {
		line = strings.TrimRight(line, "\r")
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(line[5:])
		if payload == "[DONE]" {
			for _, ev := range w.streamer.finishEvents() {
				w.emitEvent(ev)
			}
			w.finished = true
			continue
		}
		var obj map[string]any
		if err := json.Unmarshal([]byte(payload), &obj); err != nil {
			continue
		}
		// 上游错误帧（罕见）：原样转 error 事件
		if errObj, ok := obj["error"].(map[string]any); ok && errObj != nil {
			msg := stringField(errObj, "message")
			w.emitEvent(anthropicEvent{
				event: "error",
				data: map[string]any{
					"type": "error",
					"error": map[string]any{
						"type":    "upstream_error",
						"message": msg,
					},
				},
			})
			continue
		}
		for _, ev := range w.streamer.feed(obj) {
			w.emitEvent(ev)
		}
	}
	w.flushNow()
}

func (w *anthropicResponseWriter) emitEvent(ev anthropicEvent) {
	data, _ := json.Marshal(ev.data)
	fmt.Fprintf(w.inner, "event: %s\ndata: %s\n\n", ev.event, data)
}

func (w *anthropicResponseWriter) flushNow() {
	if f, ok := w.inner.(http.Flusher); ok {
		f.Flush()
	}
}

// Close 完成响应：非流式转 JSON，流式 flush 残余。
// 必须在 chatCompletions 返回后调用。
func (w *anthropicResponseWriter) Close() {
	if w.status >= 400 {
		w.writeErrorJSON()
		return
	}
	if !w.streaming {
		w.writeNonStreamJSON()
		return
	}
	// 流式：处理残余 SSE 行 + 若未收到 [DONE] 也补 message_stop
	if w.sseBuf.Len() > 0 && w.streamer != nil {
		w.handleSSEChunk(w.sseBuf.String())
		w.sseBuf.Reset()
	}
	if w.started && !w.finished && w.streamer != nil {
		for _, ev := range w.streamer.finishEvents() {
			w.emitEvent(ev)
		}
		w.finished = true
		w.flushNow()
	}
}

func (w *anthropicResponseWriter) writeNonStreamJSON() {
	w.flushInnerHeaders("application/json; charset=utf-8")
	var obj map[string]any
	raw := w.nonStreamBuf.Bytes()
	if err := json.Unmarshal(raw, &obj); err != nil {
		_, _ = w.inner.Write(raw)
		return
	}
	out := openAIToAnthropic(obj, w.model, w.names)
	encoded, _ := json.Marshal(out)
	_, _ = w.inner.Write(encoded)
}

func (w *anthropicResponseWriter) writeErrorJSON() {
	w.flushInnerHeaders("application/json; charset=utf-8")
	raw := w.nonStreamBuf.Bytes()
	var obj map[string]any
	if err := json.Unmarshal(raw, &obj); err == nil {
		if errObj, ok := obj["error"].(map[string]any); ok {
			msg := stringField(errObj, "message")
			typ := stringField(errObj, "type")
			if typ == "" {
				typ = "api_error"
			}
			out := map[string]any{
				"type": "error",
				"error": map[string]any{
					"type":    typ,
					"message": msg,
				},
			}
			encoded, _ := json.Marshal(out)
			_, _ = w.inner.Write(encoded)
			return
		}
	}
	// 非标准错误体：包装成 Anthropic 错误
	out := map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    "api_error",
			"message": string(raw),
		},
	}
	encoded, _ := json.Marshal(out)
	_, _ = w.inner.Write(encoded)
}

// ── 工具函数 ──────────────────────────────────────────────────────────────

func stringField(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	s, _ := m[key].(string)
	return s
}

func intField(m map[string]any, key string) int {
	if m == nil {
		return 0
	}
	switch v := m[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	case int64:
		return int(v)
	case json.Number:
		i, _ := v.Int64()
		return int(i)
	}
	return 0
}

func intFromAny(v any, fallback int) int {
	switch t := v.(type) {
	case float64:
		return int(t)
	case int:
		return t
	case int64:
		return int(t)
	}
	return fallback
}

// randHex 返回 n 个十六进制字符（n/2 字节）。
func randHex(n int) string {
	if n <= 0 {
		return ""
	}
	buf := make([]byte, (n+1)/2)
	_, _ = rand.Read(buf)
	return hex.EncodeToString(buf)[:n]
}
