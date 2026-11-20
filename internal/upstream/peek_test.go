package upstream

import (
	"io"
	"strings"
	"testing"
)

// TestPeekFirstFrame 首帧缓冲判定：error 帧为第一帧时必须识别出来并把
// 已读字节完整回灌（调用方据此换号重试，客户端零感知）。
func TestPeekFirstFrame(t *testing.T) {
	// 形态一：第一帧就是 error 帧（2026-11-20 实机：AIError AI服务异常）。
	errFirst := "data: {\"error\":{\"code\":\"AIError\",\"message\":\"AI服务异常\",\"type\":\"upstream_error\"}}\n\n" +
		"data: [DONE]\n\n"
	rc := io.NopCloser(strings.NewReader(errFirst))
	rc2, isErr, payload := PeekFirstFrame(rc)
	if !isErr {
		t.Fatalf("首帧 error 未识别: payload=%q", payload)
	}
	if !strings.Contains(payload, "AIError") {
		t.Fatalf("payload=%q", payload)
	}
	// 回灌读取器必须能读出完整原始字节。
	got, err := io.ReadAll(rc2)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != errFirst {
		t.Fatalf("回灌字节不一致:\n got  %q\n want %q", got, errFirst)
	}

	// 形态二：正常内容帧开头（哪怕后面有 error 帧）→ 不算首帧错误。
	contentFirst := "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"index\":0}]}\n\n" +
		"data: {\"error\":{\"code\":\"AIError\",\"message\":\"x\"}}\n\n"
	rc = io.NopCloser(strings.NewReader(contentFirst))
	rc2, isErr, _ = PeekFirstFrame(rc)
	if isErr {
		t.Fatal("内容首帧被误判为 error")
	}
	got, err = io.ReadAll(rc2)
	if err != nil || string(got) != contentFirst {
		t.Fatalf("回灌字节不一致: %q err=%v", got, err)
	}

	// 形态三：SSE 注释行 + event 行在前，error 帧仍是首个 data 帧。
	commented := ": ping\n\nevent: message\n\n" +
		"data: {\"error\":{\"code\":\"Timeout\",\"message\":\"执行超时\"}}\n\n"
	rc = io.NopCloser(strings.NewReader(commented))
	rc2, isErr, payload = PeekFirstFrame(rc)
	if !isErr || !strings.Contains(payload, "Timeout") {
		t.Fatalf("注释前置的 error 首帧未识别: isErr=%v payload=%q", isErr, payload)
	}
	got, err = io.ReadAll(rc2)
	if err != nil || string(got) != commented {
		t.Fatalf("回灌字节不一致: %q err=%v", got, err)
	}

	// 形态四：空流（只有 [DONE]）→ 不是 error，交给既有空流兜底。
	rc = io.NopCloser(strings.NewReader("data: [DONE]\n\n"))
	_, isErr, _ = PeekFirstFrame(rc)
	if isErr {
		t.Fatal("[DONE] 被误判为 error 帧")
	}

	// 形态五：非 JSON data 帧 → 不算 error。
	rc = io.NopCloser(strings.NewReader("data: not-json\n\n"))
	_, isErr, _ = PeekFirstFrame(rc)
	if isErr {
		t.Fatal("非 JSON 帧被误判为 error")
	}
}
