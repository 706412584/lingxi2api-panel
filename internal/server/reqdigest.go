// reqdigest.go 请求结构摘要：给排障用的一行日志，只描述请求的**形状**。
//
// 为什么只给摘要、不落正文：正文里可能有用户对话、代码、密钥。排查「上游一直
// 报 AI服务异常但看不出为什么」时，真正有用的是形状——消息条数、各条 role 与
// 长度、tools 数量与名字是否合法、有没有图、effort/max_tokens 取值、模型名。
// 这些足以区分上下文超限 / 工具声明丢失 / 参数不被上游接受 / 内容触发审核，
// 且不会把用户内容写进日志。
package server

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// digestMaxMsgs 摘要里逐条列出消息的最大条数；超出只给统计。
// 长会话几十轮时全列会淹没日志，而排查通常只需看首尾形状。
const digestMaxMsgs = 8

// digestPartLen 单条消息内容在摘要里保留的字符数（仅用于判断形态，
// 例如「空消息」「只有空白」「超长单条」）。不保留正文。
const digestPartLen = 24

// RequestDigest 生成请求结构摘要。永不返回正文内容。
func RequestDigest(body []byte) string {
	var req struct {
		Model               string            `json:"model"`
		Stream              *bool             `json:"stream"`
		MaxTokens           *int              `json:"max_tokens"`
		MaxCompletionTokens *int              `json:"max_completion_tokens"`
		Temperature         *float64          `json:"temperature"`
		TopP                *float64          `json:"top_p"`
		ReasoningEffort     string            `json:"reasoning_effort"`
		Thinking            any               `json:"thinking"`
		ToolChoice          any               `json:"tool_choice"`
		ParallelToolCalls   *bool             `json:"parallel_tool_calls"`
		Messages            []json.RawMessage `json:"messages"`
		Tools               []json.RawMessage `json:"tools"`
		System              json.RawMessage   `json:"system"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return fmt.Sprintf("body=%dB 解析失败: %v", len(body), err)
	}

	var b strings.Builder
	fmt.Fprintf(&b, "body=%dB model=%s", len(body), orDash(req.Model))
	if req.Stream != nil && *req.Stream {
		b.WriteString(" stream")
	}

	// 采样参数：上游对不认识的取值会直接拒绝，这里全列出来便于对照。
	if req.MaxTokens != nil {
		fmt.Fprintf(&b, " max_tokens=%d", *req.MaxTokens)
	}
	if req.MaxCompletionTokens != nil {
		fmt.Fprintf(&b, " max_completion_tokens=%d", *req.MaxCompletionTokens)
	}
	if req.Temperature != nil {
		fmt.Fprintf(&b, " temp=%g", *req.Temperature)
	}
	if req.TopP != nil {
		fmt.Fprintf(&b, " top_p=%g", *req.TopP)
	}
	if req.ReasoningEffort != "" {
		fmt.Fprintf(&b, " effort=%s", req.ReasoningEffort)
	}
	if req.Thinking != nil {
		fmt.Fprintf(&b, " thinking=%s", shortJSON(req.Thinking))
	}
	if req.ParallelToolCalls != nil {
		fmt.Fprintf(&b, " parallel_tools=%v", *req.ParallelToolCalls)
	}
	if req.ToolChoice != nil {
		fmt.Fprintf(&b, " tool_choice=%s", shortJSON(req.ToolChoice))
	}

	// system：Anthropic 风格顶层字段。网关需要把它转成 messages[0]，长度是关键形态。
	if len(req.System) > 0 {
		fmt.Fprintf(&b, " system=%s", shapeOfJSON(req.System))
	}

	// 消息：总计 + 逐条形状（role/长度/是否含 tool_calls/是否含图）。
	roles := map[string]int{}
	totalChars := 0
	hasImage := false
	hasToolCalls := false
	emptyMsgs := 0
	var lines []string
	for i, raw := range req.Messages {
		role, chars, img, tc, empty, parts := messageShape(raw)
		roles[role]++
		totalChars += chars
		hasImage = hasImage || img
		hasToolCalls = hasToolCalls || tc
		if empty {
			emptyMsgs++
		}
		if i < digestMaxMsgs {
			lines = append(lines, fmt.Sprintf("#%d %s len=%d parts=%d%s%s%s",
				i, orDash(role), chars, parts, flagIf(tc, " tool_calls"), flagIf(img, " image"), flagIf(empty, " EMPTY")))
		}
	}
	fmt.Fprintf(&b, " msgs=%d chars=%d", len(req.Messages), totalChars)
	var roleKeys []string
	for r := range roles {
		roleKeys = append(roleKeys, r)
	}
	sort.Strings(roleKeys)
	for _, r := range roleKeys {
		fmt.Fprintf(&b, " %s=%d", r, roles[r])
	}
	if emptyMsgs > 0 {
		fmt.Fprintf(&b, " EMPTY_MSGS=%d", emptyMsgs)
	}

	// tools：数量 + 名字 + 名字是否合法。工具名非法会让上游直接拒绝整个请求，
	// 而错误文案往往只有一句「AI服务异常」，看不出是这个原因。
	if len(req.Tools) > 0 {
		names, badNames := toolNames(req.Tools)
		fmt.Fprintf(&b, " tools=%d names=[%s]", len(req.Tools), strings.Join(names, ","))
		if len(badNames) > 0 {
			fmt.Fprintf(&b, " BAD_TOOL_NAMES=[%s]", strings.Join(badNames, ","))
		}
	}
	if hasToolCalls {
		b.WriteString(" history_has_tool_calls")
	}

	for _, l := range lines {
		b.WriteString(" | ")
		b.WriteString(l)
	}
	if len(req.Messages) > digestMaxMsgs {
		fmt.Fprintf(&b, " | ...(+%d 条)", len(req.Messages)-digestMaxMsgs)
	}
	return b.String()
}

// messageShape 提取一条消息的形态：role、字符数、是否含图/工具调用、是否空、
// 内容分片数。只读结构，不保留正文。
func messageShape(raw json.RawMessage) (role string, chars int, hasImage, hasToolCalls, empty bool, parts int) {
	var m struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
		Calls   []any           `json:"tool_calls"`
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		return "unparsed", 0, false, false, true, 0
	}
	role = m.Role
	hasToolCalls = len(m.Calls) > 0

	// content 可能是字符串，也可能是 parts 数组（OpenAI 多模态 / Anthropic 风格）
	var s string
	if json.Unmarshal(m.Content, &s) == nil {
		chars = len([]rune(s))
		parts = 1
		return role, chars, false, hasToolCalls, chars == 0, parts
	}
	var arr []struct {
		Type     string          `json:"type"`
		Text     string          `json:"text"`
		ImageURL json.RawMessage `json:"image_url"`
		Source   json.RawMessage `json:"source"`
	}
	if json.Unmarshal(m.Content, &arr) == nil {
		parts = len(arr)
		for _, p := range arr {
			chars += len([]rune(p.Text))
			if p.Type == "image_url" || p.Type == "image" || len(p.ImageURL) > 0 || len(p.Source) > 0 {
				hasImage = true
			}
		}
		return role, chars, hasImage, hasToolCalls, parts == 0, parts
	}
	return role, 0, false, hasToolCalls, true, 0
}

// toolNames 取 tools[].function.name，并挑出非法名字。
// OpenAI 规范要求 name 匹配 ^[a-zA-Z0-9_-]{1,64}$；空串/空格/中文都会让
// 上游拒绝整个请求（实测灵犀返回 code=AIError，message 为空）。
func toolNames(tools []json.RawMessage) (names, bad []string) {
	for i, raw := range tools {
		var t struct {
			Type     string `json:"type"`
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
			Name string `json:"name"` // 少数客户端把 name 放在顶层
		}
		if err := json.Unmarshal(raw, &t); err != nil {
			bad = append(bad, fmt.Sprintf("#%d<unparsed>", i))
			continue
		}
		name := t.Function.Name
		if name == "" {
			name = t.Name
		}
		if name == "" {
			names = append(names, fmt.Sprintf("#%d<empty>", i))
			bad = append(bad, fmt.Sprintf("#%d:empty", i))
			continue
		}
		names = append(names, name)
		if !validToolName(name) {
			bad = append(bad, fmt.Sprintf("#%d:%q", i, name))
		}
	}
	return names, bad
}

// validToolName OpenAI 工具名规范：字母数字下划线横线，1..64。
func validToolName(s string) bool {
	if len(s) == 0 || len(s) > 64 {
		return false
	}
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-':
		default:
			return false
		}
	}
	return true
}

// shapeOfJSON 给出 JSON 值的大小与类型，不返回内容。
func shapeOfJSON(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return fmt.Sprintf("str(%d)", len([]rune(s)))
	}
	var arr []json.RawMessage
	if json.Unmarshal(raw, &arr) == nil {
		return fmt.Sprintf("arr(%d)", len(arr))
	}
	return fmt.Sprintf("obj(%dB)", len(raw))
}

// shortJSON 把任意 JSON 值压成很短的一行（用于 thinking / tool_choice 这类小对象）。
func shortJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "?"
	}
	if len(b) > 60 {
		return string(b[:57]) + "..."
	}
	return string(b)
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func flagIf(on bool, label string) string {
	if on {
		return label
	}
	return ""
}