// Package server 工具名兼容转发。
//
// 灵犀上游对工具名遵循 OpenAI 规范 ^[a-zA-Z0-9_-]{1,64}$，但客户端（Cline /
// Claude Code / Codex）声明的工具名常常带空格、点、中文、命名空间分隔符
// （mcp__server__tool 是合法的；some.tool / some tool 是非法的）。直接透传
// 会被上游用 AIError 拒绝整个请求。
//
// 策略：
//   - 合法名字原样透传，不引入额外行为。
//   - 非法名字按字符级修复成合法名，并记录 originalName → sanitizedName
//     的映射；上游响应里出现的 sanitizedName 再还原为 originalName，保证
//     客户端看到的是自己声明的那个名字。
//   - 修复是确定性的（同输入必得同输出），避免重试/分片时映射漂移。
package server

import (
	"crypto/sha1"
	"encoding/hex"
	"regexp"
	"strings"
)

// toolNameRe 是合法工具名字符集（与 validToolName 同口径，但用 regexp 表达，
// 便于 ReplaceAllString 一次性把非法字符折叠为 '_'）。
var toolNameRe = regexp.MustCompile(`[^a-zA-Z0-9_-]+`)

// sanitizeToolName 把工具名修复到 OpenAI 规范（与 validToolName 同口径）。
// 返回 (修复后名, 是否被改动)。
//
// 规则：
//   - 已经是合法名（含 3d-print 这类数字/连字符开头）：原样返回，false。
//   - 把连续非法字符折叠为单个 '_'（"some.tool name" → "some_tool_name"）。
//   - 去掉首尾 '_'（"___foo___" → "foo"）。
//   - 空串/全非法：合成 "tool_<hash8>"。
//   - 超过 64 字符：截断到 55 字符 + '_' + hash8，仍满足总长 64。
//
// 用 SHA1 前 4 字节做 hash，是为了在重试/分片场景下同名工具始终映射到
// 同一个 sanitizedName——工具调用是分片流出的，名字漂移会让响应无法还原。
func sanitizeToolName(name string) (string, bool) {
	if validToolName(name) {
		return name, false
	}
	hash := ""
	hashOf := func(s string) string {
		if hash != "" {
			return hash
		}
		h := sha1.Sum([]byte(s))
		hash = hex.EncodeToString(h[:4])
		return hash
	}

	out := toolNameRe.ReplaceAllString(name, "_")
	out = strings.Trim(out, "_")
	if out == "" {
		return "tool_" + hashOf(name), true
	}
	if len(out) > 64 {
		out = out[:55] + "_" + hashOf(name)
	}
	// 截断后再 trim 一次，避免恰好切在 '_' 上留下尾巴。
	out = strings.TrimRight(out, "_")
	if out == "" {
		return "tool_" + hashOf(name), true
	}
	return out, true
}

// toolNameMap 记录 originalName ↔ sanitizedName 的双向映射。
// 用于请求侧修复 + 响应侧还原。
type toolNameMap struct {
	forward map[string]string // original → sanitized
	reverse map[string]string // sanitized → original
}

func newToolNameMap() *toolNameMap {
	return &toolNameMap{forward: map[string]string{}, reverse: map[string]string{}}
}

// sanitize 登记 originalName 并返回 sanitizedName。重复登记同一原名得到同一结果。
func (m *toolNameMap) sanitize(original string) string {
	if original == "" {
		return original
	}
	if v, ok := m.forward[original]; ok {
		return v
	}
	sanitized, changed := sanitizeToolName(original)
	if changed {
		m.forward[original] = sanitized
		// reverse 只在未冲突时写入：两个原名映射到同一 sanitized 时，
		// 保留第一个，避免响应侧把 A 工具的调用错归给 B 工具。
		if _, exists := m.reverse[sanitized]; !exists {
			m.reverse[sanitized] = original
		}
	}
	return sanitized
}

// restore 把上游返回的 sanitizedName 还原成 originalName。未登记的名字原样返回。
func (m *toolNameMap) restore(sanitized string) string {
	if m == nil {
		return sanitized
	}
	if v, ok := m.reverse[sanitized]; ok {
		return v
	}
	return sanitized
}

// changed 是否做过任何修复（用于打日志）。
func (m *toolNameMap) changed() int { return len(m.forward) }
