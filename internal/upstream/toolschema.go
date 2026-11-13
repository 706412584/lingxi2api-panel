// toolschema.go 工具参数 schema 归一化。
//
// 实测结论（2026-11-13，灵犀 cn 域）：
//
//	上游只接受「properties 直接子级」是普通类型（或带 type 的复合关键字）。
//	一旦某个属性的 schema **只有** anyOf / oneOf / allOf 而没有 type，
//	上游直接返回 code=AIError、message 为空（网关侧只看到「AI服务异常」），
//	整个请求失败——不是这个工具失败，是**全部 tools 作废、整轮对话挂掉**。
//
// 现象之所以难查：错误文案为空，日志里只有一句上游兜底话术；而失败与否
// 取决于调用方用哪个 MCP 客户端、注入了多少个工具（Claude Code 的 156 个
// 工具里恰好有 7 个用了 anyOf）。所以不能只报错，必须在出站前归一。
//
// 归一策略（按优先级取第一个可用的）：
//  1. 已有 type → 不动（例外：type 为 null 且 anyOf 里有具体类型时取具体类型）；
//  2. 组合关键字里有具体类型分支 → 取第一个非 null 分支的类型；
//  3. 有 enum → type 由枚举值推断（string/number）；
//  4. 都没有 → 退化成宽松的 string（宁可让上游接受，也不让整轮失败）。
//
// 只改写「properties 直接子级」这一层：更深层的 anyOf 实测上游接受，
// 无谓重写会破坏客户端对 schema 的期望（也可能丢掉约束）。
package upstream

import "strings"

// normalizeToolSchema 就地归一化 obj["tools"] 里各工具的 function.parameters。
// 无改动时不分配新结构（保持原 map，零副作用）。
func normalizeToolSchema(obj map[string]any) {
	tools, ok := obj["tools"].([]any)
	if !ok || len(tools) == 0 {
		return
	}
	for _, t := range tools {
		tm, ok := t.(map[string]any)
		if !ok {
			continue
		}
		fn, ok := tm["function"].(map[string]any)
		if !ok {
			continue
		}
		params, ok := fn["parameters"].(map[string]any)
		if !ok {
			continue
		}
		normalizeSchemaObject(params)
	}
}

// normalizeSchemaObject 处理一个 schema 对象的 properties 直接子级。
func normalizeSchemaObject(schema map[string]any) {
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		return
	}
	for name, raw := range props {
		pm, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if next, changed := flattenCombinator(pm); changed {
			props[name] = next
		}
	}
}

// flattenCombinator 把一个「没有具体类型」的属性 schema 补成具体类型。
//
// 两种形态都会让上游拒整个请求：
//   - 只有 anyOf / oneOf / allOf（组合关键字展开）；
//   - 连组合关键字都没有，只有 description/pattern 之类（如 Claude Code
//     Workflow 工具的 `args`：只有描述，表示「任意 JSON 值」）。
//
// 返回新的 map（不改原对象，避免影响同一请求里别处引用）。
func flattenCombinator(pm map[string]any) (map[string]any, bool) {
	curType, hasType := pm["type"]
	if hasType && curType != nil && curType != "null" {
		return pm, false // 已有具体类型：上游接受，不动
	}
	// 找出存在的组合关键字
	var branches []any
	found := false
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if arr, ok := pm[key].([]any); ok && len(arr) > 0 {
			branches = arr
			found = true
			break
		}
	}
	if !found {
		// 没有组合关键字：也必须有 type，否则上游同样拒绝。
		out := make(map[string]any, len(pm)+1)
		for k, v := range pm {
			out[k] = v
		}
		if enum, ok := out["enum"].([]any); ok && len(enum) > 0 {
			switch enum[0].(type) {
			case float64:
				out["type"] = "number"
			case bool:
				out["type"] = "boolean"
			default:
				out["type"] = "string"
			}
			return out, true
		}
		out["type"] = "string"
		return out, true
	}

	out := make(map[string]any, len(pm))
	for k, v := range pm {
		if k == "anyOf" || k == "oneOf" || k == "allOf" {
			continue // 展开后不再保留组合关键字
		}
		out[k] = v
	}

	// 分支里挑第一个具体类型；跳过 null / 空 / 嵌套组合。
	if t := firstConcreteType(branches); t != "" {
		out["type"] = t
		// 把该分支自己的约束（enum/description/items...）并进来，
		// 不覆盖已有的键（外层描述更权威）。
		if bm := firstBranchWithType(branches, t); bm != nil {
			for k, v := range bm {
				if k == "type" {
					continue
				}
				if _, exists := out[k]; !exists {
					out[k] = v
				}
			}
		}
		return out, true
	}

	// 没有具体类型分支：从 enum 推断，否则退化成宽松 string。
	if enum, ok := out["enum"].([]any); ok && len(enum) > 0 {
		switch enum[0].(type) {
		case float64:
			out["type"] = "number"
			return out, true
		case bool:
			out["type"] = "boolean"
			return out, true
		}
		out["type"] = "string"
		return out, true
	}
	out["type"] = "string"
	return out, true
}

// firstConcreteType 返回分支里第一个可用的 JSON 类型名（跳过 null）。
func firstConcreteType(branches []any) string {
	for _, b := range branches {
		bm, ok := b.(map[string]any)
		if !ok {
			continue
		}
		if t, ok := bm["type"].(string); ok && t != "" && t != "null" {
			return t
		}
		// 分支自己也是组合（少见）：递归再试一层
		for _, key := range []string{"anyOf", "oneOf", "allOf"} {
			if inner, ok := bm[key].([]any); ok && len(inner) > 0 {
				if t := firstConcreteType(inner); t != "" {
					return t
				}
			}
		}
	}
	return ""
}

// firstBranchWithType 取第一个类型匹配的分支对象（用于合并它的附加约束）。
func firstBranchWithType(branches []any, want string) map[string]any {
	for _, b := range branches {
		bm, ok := b.(map[string]any)
		if !ok {
			continue
		}
		if t, ok := bm["type"].(string); ok && strings.EqualFold(t, want) {
			return bm
		}
	}
	return nil
}