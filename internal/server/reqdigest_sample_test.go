package server

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// 采样要落盘 body + 旁注，且文件名里的模型名被清洗。
func TestSampleFailedRequestWritesBodyAndNote(t *testing.T) {
	dir := t.TempDir()
	h := &Handler{cfg: Config{SampleDir: dir}}

	body := []byte(`{"model":"m","messages":[]}`)
	h.sampleFailedRequest(body, "deepseek/v4.1:flash", `{"error":{"code":"AIError"}}`)

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read dir: %v", err)
	}
	if len(entries) != 2 { // body + .note
		t.Fatalf("期望 2 个文件，实际 %d", len(entries))
	}
	var gotBody, gotNote string
	for _, e := range entries {
		p := filepath.Join(dir, e.Name())
		b, _ := os.ReadFile(p)
		if strings.HasSuffix(e.Name(), ".note") {
			gotNote = string(b)
		} else {
			gotBody = string(b)
			if strings.ContainsAny(e.Name(), `/\:`) {
				t.Errorf("文件名未清洗: %s", e.Name())
			}
		}
	}
	if gotBody != string(body) {
		t.Errorf("body 未原样保存: %q", gotBody)
	}
	if !strings.Contains(gotNote, "AIError") {
		t.Errorf("旁注缺上游原话: %q", gotNote)
	}
}

// 超大 body 直接跳过，不落盘（避免一次请求写爆磁盘）。
func TestSampleFailedRequestSkipsOversize(t *testing.T) {
	dir := t.TempDir()
	h := &Handler{cfg: Config{SampleDir: dir}}

	h.sampleFailedRequest(make([]byte, sampleMaxBytes+1), "m", "x")

	if entries, _ := os.ReadDir(dir); len(entries) != 0 {
		t.Fatalf("超大 body 不该落盘，实际 %d 个文件", len(entries))
	}
}

// SampleDir 为空时静默跳过（未注入配置的裸用路径）。
func TestSampleFailedRequestNoDir(t *testing.T) {
	h := &Handler{cfg: Config{}}
	h.sampleFailedRequest([]byte(`{}`), "m", "x") // 不应 panic
}

// 清理按「文件数」生效：超过上限后旧文件被删掉。
func TestPruneSamplesByCount(t *testing.T) {
	dir := t.TempDir()
	// 造 10 个文件，保留 4 个
	for i := 0; i < 10; i++ {
		name := filepath.Join(dir, strings.Repeat("a", 1)+string(rune('a'+i))+".json")
		if err := os.WriteFile(name, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	pruneSamples(dir, 4, 1<<30) // 只按数量
	entries, _ := os.ReadDir(dir)
	if len(entries) != 4 {
		t.Fatalf("期望保留 4 个，实际 %d", len(entries))
	}
}

// 清理按「总量」生效：小上限时只留最近的文件。
func TestPruneSamplesByBytes(t *testing.T) {
	dir := t.TempDir()
	for i := 0; i < 5; i++ {
		name := filepath.Join(dir, string(rune('a'+i))+".json")
		if err := os.WriteFile(name, make([]byte, 100), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	pruneSamples(dir, 100, 250) // 只按总量：应留 2 个
	entries, _ := os.ReadDir(dir)
	if len(entries) != 2 {
		t.Fatalf("期望保留 2 个，实际 %d", len(entries))
	}
}