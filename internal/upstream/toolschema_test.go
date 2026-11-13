package upstream

import (
	"encoding/json"
	"testing"
)

// 上游拒收 properties 直接子级的 anyOf/oneOf/allOf（无 type）——
// 表现为 code=AIError、message 为空，整个请求的 tools 全部作废。
// 这些用例锁定归一化后的形状。
func TestNormalizeToolSchemaCombinators(t *testing.T) {
	cases := []struct {
		name  string
		param map[string]any
		want  string // 归一化后的 type
	}{
		{"anyOf string|null", map[string]any{"anyOf": []any{
			map[string]any{"type": "string"}, map[string]any{"type": "null"},
		}}, "string"},
		{"oneOf string|number", map[string]any{"oneOf": []any{
			map[string]any{"type": "string"}, map[string]any{"type": "number"},
		}}, "string"},
		{"allOf 单分支", map[string]any{"allOf": []any{
			map[string]any{"type": "object"},
		}}, "object"},
		{"只有 null 分支", map[string]any{"anyOf": []any{
			map[string]any{"type": "null"},
		}}, "string"}, // 退化：宁可宽松也不能让整轮失败
		{"无类型只有描述", map[string]any{"anyOf": []any{
			map[string]any{"description": "x"},
		}}, "string"},
		// 真实案例（Claude Code Workflow 工具的 args）：只有描述、连组合关键字都没有。
		// 上游同样拒收，必须补类型。
		{"只有描述无组合关键字", map[string]any{"description": "任意 JSON 值"}, "string"},
		{"只有 pattern", map[string]any{"description": "ID", "pattern": "^wf_[a-z]+$"}, "string"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			obj := wrapTool(c.param)
			normalizeToolSchema(obj)
			got := paramAt(obj, "a")
			if _, still := got["anyOf"]; still {
				t.Fatalf("组合关键字未展开: %#v", got)
			}
			if got["type"] != c.want {
				t.Fatalf("type=%v want %v (%#v)", got["type"], c.want, got)
			}
		})
	}
}

// 已有具体 type 的属性一律不动：上游接受，重写只会破坏客户端约束。
func TestNormalizeToolSchemaKeepsTyped(t *testing.T) {
	orig := map[string]any{
		"type":        "array",
		"items":       map[string]any{"type": "string"},
		"description": "列表",
	}
	obj := wrapTool(orig)
	normalizeToolSchema(obj)
	got := paramAt(obj, "a")
	if got["type"] != "array" {
		t.Fatalf("type 被改动: %#v", got)
	}
	if _, ok := got["items"]; !ok {
		t.Fatalf("items 丢失: %#v", got)
	}
}

// 深层 anyOf 上游接受，不能改写——否则会丢失嵌套约束。
func TestNormalizeToolSchemaLeavesDeepCombinators(t *testing.T) {
	deep := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"inner": map[string]any{"anyOf": []any{map[string]any{"type": "string"}}},
		},
	}
	obj := wrapTool(deep)
	normalizeToolSchema(obj)
	got := paramAt(obj, "a")
	inner := got["properties"].(map[string]any)["inner"].(map[string]any)
	if _, still := inner["anyOf"]; !still {
		t.Fatalf("深层 anyOf 不该被改写: %#v", inner)
	}
	if _, has := inner["type"]; has {
		t.Fatalf("深层不该被补 type: %#v", inner)
	}
}

// 分支自带的附加约束（enum/description）要合并进来，不能丢。
func TestNormalizeToolSchemaMergesBranchConstraints(t *testing.T) {
	obj := wrapTool(map[string]any{
		"anyOf": []any{map[string]any{"type": "string", "enum": []any{"a", "b"}}},
	})
	normalizeToolSchema(obj)
	got := paramAt(obj, "a")
	if got["type"] != "string" {
		t.Fatalf("type=%v", got["type"])
	}
	if _, ok := got["enum"]; !ok {
		t.Fatalf("enum 丢失: %#v", got)
	}
}

// 无 combinators 的普通工具必须逐字节不变（零副作用）。
func TestNormalizeToolSchemaNoOp(t *testing.T) {
	body := []byte(`{"model":"m","tools":[{"type":"function","function":{"name":"t","parameters":{"type":"object","properties":{"a":{"type":"string"}}}}}]}`)
	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		t.Fatal(err)
	}
	before, _ := json.Marshal(obj)
	normalizeToolSchema(obj)
	after, _ := json.Marshal(obj)
	if string(before) != string(after) {
		t.Fatalf("普通工具被改写:\nbefore=%s\nafter=%s", before, after)
	}
}

// ---- helpers ----

func wrapTool(propSchema map[string]any) map[string]any {
	return map[string]any{
		"tools": []any{
			map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": "t",
					"parameters": map[string]any{
						"type":       "object",
						"properties": map[string]any{"a": propSchema},
					},
				},
			},
		},
	}
}

func paramAt(obj map[string]any, name string) map[string]any {
	tools := obj["tools"].([]any)
	fn := tools[0].(map[string]any)["function"].(map[string]any)
	props := fn["parameters"].(map[string]any)["properties"].(map[string]any)
	return props[name].(map[string]any)
}