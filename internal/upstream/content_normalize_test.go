package upstream

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestNormalizeTextContent 灵犀上游对数组形态 content 一律 400 + 空 body
// （2026-11-18 实机：同一文本 [{type:"text"}] → 400，字符串 → 200）。
// 纯文本数组必须折叠成字符串，含图片分片的数组保持原样。
func TestNormalizeTextContent(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want any
	}{
		{
			name: "单文本分片折叠",
			in:   []any{map[string]any{"type": "text", "text": "hi"}},
			want: "hi",
		},
		{
			name: "文本分片带 cache_control 剥离",
			in:   []any{map[string]any{"type": "text", "text": "hi", "cache_control": map[string]any{"type": "ephemeral"}}},
			want: "hi",
		},
		{
			name: "多文本分片按序拼接",
			in: []any{
				map[string]any{"type": "text", "text": "a"},
				map[string]any{"type": "text", "text": "b"},
			},
			want: "ab",
		},
		{
			name: "input_text 同样折叠",
			in:   []any{map[string]any{"type": "input_text", "text": "x"}},
			want: "x",
		},
		{
			name: "空数组折叠为空串",
			in:   []any{},
			want: "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			obj := map[string]any{"messages": []any{map[string]any{"role": "user", "content": tc.in}}}
			normalizeTextContent(obj)
			got := obj["messages"].([]any)[0].(map[string]any)["content"]
			if got != tc.want {
				t.Fatalf("content=%v want %v", got, tc.want)
			}
		})
	}

	// 含图片分片的数组不能折叠（会被压成字符串丢失图片）；cache_control 仍剥离。
	img := map[string]any{"messages": []any{map[string]any{"role": "user", "content": []any{
		map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://x/y.png"}, "cache_control": map[string]any{"type": "ephemeral"}},
		map[string]any{"type": "text", "text": "看这张图"},
	}}}}
	normalizeTextContent(img)
	parts, ok := img["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("图片数组必须保持数组形态: %v", img["messages"])
	}
	if _, has := parts[0].(map[string]any)["cache_control"]; has {
		t.Fatalf("cache_control 必须剥离: %v", parts[0])
	}

	// 字符串内容不受影响。
	str := map[string]any{"messages": []any{map[string]any{"role": "user", "content": "hi"}}}
	normalizeTextContent(str)
	if got := str["messages"].([]any)[0].(map[string]any)["content"]; got != "hi" {
		t.Fatalf("字符串内容被改动: %v", got)
	}
}

// TestLXRewriteBodyFlattensTextContent 灵犀真实出站路径必须折叠文本数组。
func TestLXRewriteBodyFlattensTextContent(t *testing.T) {
	body := []byte(`{"model":"GLM-5.3","messages":[
 {"role":"system","content":"sys"},
 {"role":"user","content":[{"type":"text","text":"你好","cache_control":{"type":"ephemeral"}}]}
]}`)
	out, _, err := lxRewriteBody(body)
	if err != nil {
		t.Fatalf("lxRewriteBody: %v", err)
	}
	var in map[string]any
	if err := json.Unmarshal(out, &in); err != nil {
		t.Fatal(err)
	}
	msg := in["messages"].([]any)[1].(map[string]any)
	if msg["content"] != "你好" {
		t.Fatalf("灵犀出站 content 仍是数组: %v", msg["content"])
	}
	if strings.Contains(string(out), "cache_control") {
		t.Fatalf("灵犀出站仍带 cache_control: %s", out)
	}
}