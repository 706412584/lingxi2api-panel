// reqdigest_sample.go 失败请求采样：仅在 logging.request_body_digest 打开时启用。
//
// 用途单一：上游只回一句「AI服务异常」（message 为空）时，日志里没有任何
// 可定位的信息。把失败请求的原始 body 存到本地，就能逐项删减复现。
//
// 边界：
//   - 只在本地落盘，不外传、不进归档、不进面板日志；
//   - 有硬上限（单文件 8MB、总数 20 个、总量 64MB），超限就跳过并只记一句；
//   - 目录 data/failed-requests/，文件名带时间戳，便于对照日志。
package server

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"time"
)

const (
	sampleDir          = "failed-requests"
	sampleMaxBytes     = 8 << 20  // 单个 body 上限
	sampleMaxFiles     = 20       // 保留文件数
	sampleMaxTotalByte = 64 << 20 // 目录总量上限
)

// sampleFailedRequest 把失败请求的 body 存到 <state 目录>/failed-requests/。
// 任何错误都只记日志，绝不影响请求本身。
func (h *Handler) sampleFailedRequest(body []byte, model, errFrame string) {
	if len(body) == 0 || len(body) > sampleMaxBytes {
		if len(body) > sampleMaxBytes {
			log.Printf("INFO: [server] failed-request sample skipped: body %dB exceeds %dB", len(body), sampleMaxBytes)
		}
		return
	}
	dir := h.sampleDir()
	if dir == "" {
		return
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		log.Printf("WARN: [server] failed-request sample: mkdir %s: %v", dir, err)
		return
	}
	// 预留 2 个位置：本次要写 body + .note
	pruneSamples(dir, sampleMaxFiles-2, sampleMaxTotalByte-int64(len(body)))

	stamp := time.Now().Format("20060102-150405.000")
	name := fmt.Sprintf("%s-%s.json", stamp, sanitizeFilePart(model))
	// 附一个旁注文件记录上游原话，body 本身保持原样（便于直接当请求体重放）
	note := fmt.Sprintf("model: %s\nupstream: %s\nbytes: %d\n", model, errFrame, len(body))
	_ = os.WriteFile(filepath.Join(dir, name+".note"), []byte(note), 0o644)
	if err := os.WriteFile(filepath.Join(dir, name), body, 0o600); err != nil {
		log.Printf("WARN: [server] failed-request sample: write: %v", err)
		return
	}
	log.Printf("INFO: [server] failed-request sampled -> %s (%dB)", filepath.Join(dir, name), len(body))
}

// sampleDir 采样目录：由 main 按 state 文件同目录推导后注入（同 request-logs 惯例）。
func (h *Handler) sampleDir() string {
	return h.cfg.SampleDir
}

// pruneSamples 按修改时间从旧到新删除，直到文件数 <= keepFiles 且总量 <= keepBytes。
// 任何一步失败都不阻止后续写入（宁可多存一个，也不要因为清理失败丢掉样本）。
func pruneSamples(dir string, keepFiles int, keepBytes int64) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	type fi struct {
		name string
		size int64
		mod  time.Time
	}
	var files []fi
	var total int64
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		files = append(files, fi{e.Name(), info.Size(), info.ModTime()})
		total += info.Size()
	}
	sort.Slice(files, func(i, j int) bool { return files[i].mod.Before(files[j].mod) })
	// 按**文件**计数，不假设 body/.note 成对：目录里可能只剩半对（写 note 后
	// body 写失败、或用户手工删了一个），按对数算会让保留量翻倍。
	for len(files) > 0 && (len(files) > keepFiles || total > keepBytes) {
		f := files[0]
		files = files[1:]
		total -= f.size
		_ = os.Remove(filepath.Join(dir, f.name))
	}
}

// sanitizeFilePart 让模型名能安全出现在文件名里（只留字母数字点横线）。
func sanitizeFilePart(s string) string {
	out := make([]rune, 0, len(s))
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '.', r == '-', r == '_':
			out = append(out, r)
		default:
			out = append(out, '_')
		}
	}
	if len(out) == 0 {
		return "model"
	}
	if len(out) > 80 {
		out = out[:80]
	}
	return string(out)
}