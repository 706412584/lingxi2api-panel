package upstream

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
)

// PeekFirstFrame 从上游 SSE 流读出第一帧 data 行，判定它是「内容帧」还是
// 「error 帧」，并把读出的字节原样回灌到返回的读取器——调用方拿到的 rc2 与
// 原始流逐字节一致，可继续走既有透传路径。
//
// 用途（2026-11-20 实机事故）：灵犀上游间歇性在**第一帧**就返回
// {"error":{"code":"AIError","message":"AI服务异常"}}。此时网关还没向客户端
// 写出任何字节（SSE 头都未发），本可以像非流式 4xx 一样换号重试；但既有
// 流式路径拿到 rc 就直接 StreamHint 开流，错误帧作为首帧透传给客户端，
// 一次上游抖动直接变成客户端可见的失败。
//
// 判定口径与 sse.go 的 writeFrame 一致：帧 JSON 带 "error" 键 = error 帧。
// 只看第一帧：中途 error 帧（客户端已收到内容）仍按既有路径透传，不重试。
//
// 读取上限：一帧超过 8MB（与 StreamHint 的 scanner 缓冲同量级）视为内容帧
// 放行——超大首帧不可能是错误信封，别为探测把流掐死。
func PeekFirstFrame(rc io.ReadCloser) (rc2 io.ReadCloser, firstIsError bool, errPayload string) {
	br := bufio.NewReaderSize(rc, 64*1024)
	var held []byte
	for {
		line, err := br.ReadString('\n')
		held = append(held, line...)
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "data: ") {
			payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data: "))
			if payload == "[DONE]" {
				break // 空流 + DONE：交给既有空流兜底路径
			}
			if hasErrorFrameKey(payload) {
				return newPeekedReader(rc, br, held), true, payload
			}
			break
		}
		if trimmed != "" && !strings.HasPrefix(trimmed, ":") && !strings.HasPrefix(trimmed, "event:") {
			// 非 data 行（注释/event 头之外的杂项）：继续找第一帧 data。
			continue
		}
		if err != nil {
			break // EOF / 读错误：按内容帧放行，交给既有路径处理
		}
	}
	return newPeekedReader(rc, br, held), false, ""
}

// hasErrorFrameKey 判定 data 帧是否带 error 键（与 sse.go writeFrame 的
// `obj["error"]` 判定同口径）。非 JSON 帧不算 error 帧。
func hasErrorFrameKey(payload string) bool {
	var obj map[string]any
	if err := json.Unmarshal([]byte(payload), &obj); err != nil {
		return false
	}
	_, hasErr := obj["error"]
	return hasErr
}

// peekedReader 把已读出的字节先回放，再继续读底层 reader。
type peekedReader struct {
	src  io.ReadCloser
	buf  []byte // 待回放的已读字节
	rest *bufio.Reader
}

func newPeekedReader(src io.ReadCloser, br *bufio.Reader, held []byte) *peekedReader {
	return &peekedReader{src: src, buf: held, rest: br}
}

func (p *peekedReader) Read(b []byte) (int, error) {
	if len(p.buf) > 0 {
		n := copy(b, p.buf)
		p.buf = p.buf[n:]
		return n, nil
	}
	return p.rest.Read(b)
}

func (p *peekedReader) Close() error { return p.src.Close() }
