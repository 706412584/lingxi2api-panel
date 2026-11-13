package server

import (
	"strings"
	"testing"
)

// 摘要必须给出定位所需的结构信息，且绝不包含正文。
func TestRequestDigestShape(t *testing.T) {
	body := []byte(`{
		"model":"deepseek-v4.1-flash","stream":true,"max_tokens":32000,
		"temperature":0.7,"reasoning_effort":"max",
		"messages":[
			{"role":"system","content":"You are Claude Code, an AI coding assistant. SECRET_SYSTEM_TEXT"},
			{"role":"user","content":"解释这个仓库的结构 SECRET_USER_TEXT"},
			{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{}"}}]},
			{"role":"tool","content":"file contents SECRET_TOOL_TEXT"}
		],
		"tools":[
			{"type":"function","function":{"name":"read_file"}},
			{"type":"function","function":{"name":""}},
			{"type":"function","function":{"name":"bad name"}}
		]
	}`)

	got := RequestDigest(body)

	// 必须包含的形态信息
	for _, want := range []string{
		"deepseek-v4.1-flash", "max_tokens=32000", "effort=max", "stream",
		"msgs=4", "tools=3",
		"BAD_TOOL_NAMES", // 非法工具名要能被一眼看到
	} {
		if !strings.Contains(got, want) {
			t.Errorf("摘要缺少 %q:\n%s", want, got)
		}
	}

	// 绝不能泄露正文
	for _, secret := range []string{"SECRET_SYSTEM_TEXT", "SECRET_USER_TEXT", "SECRET_TOOL_TEXT"} {
		if strings.Contains(got, secret) {
			t.Errorf("摘要泄露了正文 %q:\n%s", secret, got)
		}
	}

	// 空 content 要标出来：Claude Code 的 assistant 占位消息就是这种形态
	if !strings.Contains(got, "EMPTY") {
		t.Errorf("未标出空消息:\n%s", got)
	}
}

// 图片消息要能被识别：带图请求在模型不支持时会整请求失败。
func TestRequestDigestDetectsImage(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":[
		{"type":"text","text":"看这张图"},
		{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}
	]}]}`)
	got := RequestDigest(body)
	if !strings.Contains(got, "image") {
		t.Errorf("未识别出图片:\n%s", got)
	}
}

// Anthropic 风格的顶层 system 是独立字段，长度形态需要单独给出。
func TestRequestDigestTopLevelSystem(t *testing.T) {
	body := []byte(`{"model":"m","system":"you are helpful","messages":[{"role":"user","content":"hi"}]}`)
	got := RequestDigest(body)
	if !strings.Contains(got, "system=str(15)") {
		t.Errorf("未给出顶层 system 形态:\n%s", got)
	}
}

// 畸形 body 不能 panic，要给出可读原因。
func TestRequestDigestMalformed(t *testing.T) {
	got := RequestDigest([]byte(`{not json`))
	if !strings.Contains(got, "解析失败") {
		t.Errorf("畸形 body 未给出原因: %s", got)
	}
}

// 工具名校验口径：空、空格、中文都不合法。
func TestValidToolName(t *testing.T) {
	cases := map[string]bool{
		"read_file": true, "ReadFile": true, "a-b_9": true,
		"": false, "bad name": false, "工具": false, strings.Repeat("a", 65): false,
	}
	for name, want := range cases {
		if got := validToolName(name); got != want {
			t.Errorf("validToolName(%q)=%v want %v", name, got, want)
		}
	}
}