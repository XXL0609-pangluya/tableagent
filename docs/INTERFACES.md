# INTERFACES — 模块接口契约（活文档）

> 这是防止"接口漂移/越改越乱"的防线。**改任何模块前先看本文件对应小节；改完同步更新。**
> 数据结构的唯一真相源是 `src/schemas.py`；本文件描述各模块的**职责 + 公开 API + 不变量**。
> 规则：① 只通过这里列出的公开函数跨模块调用；② 改公开签名必须同步本文件 + 受影响的黄金测试。

---

## 进度勾选表

| Phase | 内容 | 状态 |
|---|---|---|
| 0 | 无模型地基（schemas/data/evaluator/trace/sandbox/tools.base/context_budget + smoke） | ✅ smoke PASS（评测器对齐=1.0，200/200 表加载） |
| 0.5 | config + llm.py(原生 FC 可用) + direct-answer baseline | ✅ baseline 0.70(14/20) on quick-set 抽样 |
| 1 | 单轨 function-calling agent（5 工具 + 循环 + pipeline + trace） | ✅ agent 0.75 vs baseline 0.70 (15 vs 14 / 20) |
| 2 | grounding / 严格 verify / 自一致性 / router + skill 库 | ⬜ |
| 3 | 错误归因驱动迭代（skill 从错误里长出来） | ⬜ |
| 4 | 对比实验 + 测试集最终分数 + 报告 | ⬜ |

---

## 数据契约（`src/schemas.py` — 唯一真相源）
`Example` / `TableContext` / `ToolCall` / `ToolResult` / `Observation` / `Budget` /
`AgentState` / `Prediction` / `TraceEvent`。详见源码 docstring。

**核心不变量**
- 表格一律以**字符串**读入 `TableContext.df`（不自动推断类型）。
- 工具**永不抛异常**：失败编码进 `ToolResult(ok=False, error=...)`。
- `run_python` 的产出放在 `ToolResult.structured`（`answer_items/evidence_*` 等）。
- `Prediction` 始终带 `evidence` 与 `trace_id`，可回溯。

---

## `src/data.py` — 数据加载
- `load_examples(split_basename, dataset_root=DEFAULT) -> list[Example]`
- `load_table(table_path, dataset_root=DEFAULT) -> TableContext`
- `sample_examples(examples, n, seed=13) -> list[Example]`（确定性"快集"）
- 常量：`DEFAULT_DATASET_ROOT`
- 不变量：`load_table` 返回的 `df` 全为字符串列；`schema_text`/`sample_rows` 已就绪。

## `src/evaluator.py` — 官方评测器(Py3 移植)
- `normalize(str) -> str`
- `to_value(...) / to_value_list(...) -> Value`
- `check_denotation(targets, preds) -> bool`
- `load_targets_from_tagged(path) -> dict[id, list[Value]]`（**权威**，用 targetCanon）
- `load_targets_from_tsv(path) -> dict[id, list[Value]]`（仅 targetValue，更宽松）
- `find_tagged_path(root, split) -> Optional[str]`
- `evaluate(predictions: dict[id,list[str]], targets) -> {accuracy,num_examples,num_correct,num_missing_predictions,per_example}`
- 不变量：评分语义与官方一致；gold 当预测应得 1.0。

## `src/context_budget.py` — token 预算/截断
- `estimate_tokens(text) -> int`（带 1.2 安全系数；有 tiktoken 用之，否则字符估算）
- `truncate_text(text, max_chars=8000, head_ratio=0.6) -> (str, truncated: bool)`（头尾保留）

## `src/trace.py` — 全链路留痕
- `Tracer(out_path, example_id)` → `.add(TraceEvent)` / `.flush(extra=None)`
- 不变量：每个 example 追加一条 JSON 记录（JSONL）；含 `trace_id`。

## `src/sandbox.py` — 受限代码执行
- `run_code(code, df, timeout_s=10.0) -> ExecResult{ok,stdout,answer,evidence,intermediate,error}`
- 约定：代码设置 `answer`（必要）和 `evidence`（可选）；白名单模块 + 超时；**永不抛异常**。
- 注意：软沙箱（受限 globals + 超时），非对抗性安全边界。

## `src/tools/base.py` — 工具契约 + 注册表
- `ToolSpec{name, description, input_schema}`
- `Tool`（抽象）：`spec`；`run(args, state) -> ToolResult`（不抛异常）；`available(state) -> Optional[str]`（None=可用）
- `ToolRegistry`：`register(tool)`（重名报错）/ `get(name)` / `all()` / `visible(state) -> (tools, hidden{name:reason})` / `specs(state) -> list[ToolSpec]`
- 不变量：新增工具 = 实现 `Tool` + `register` 一行，**不改 agent 主循环**。

---

## `src/config.py` — 配置（密钥来自 .env）
- `LLMConfig{base_url, api_key, model, temperature, max_tokens, timeout_s}`
- `load_llm_config(model=None, temperature=None) -> LLMConfig`（缺 base_url/key 抛错）

## `src/llm.py` — LLM 客户端（OpenAI 兼容）
- `LLMResponse{text, tool_calls: list[ToolCall], finish_reason, usage, raw}`
- `LLMClient(config=None)`：`chat(messages, tools=None, tool_choice=None, ...) -> LLMResponse`；`probe_native_tools() -> (bool, detail)`
- 实测：endpoint 外网 `.cn` 可用；**deepseek-v4-flash 原生支持 FC**。

## `src/formatter.py` — 答案解析/归一化
- `parse_answer_text(text) -> list[str]`（JSON 数组 / `|` 分隔 / 单值）
- `items_from_structured(structured) -> list[str]`（读 run_python 的 answer_items）

## `src/baselines.py` — 对照基线
- `run_direct_answer(example, table_context, client) -> (Prediction, usage)`（整表直答，不抛异常）

## `src/tools/wtq_tools.py` — 5 个 WTQ 工具 [1]
- `InspectTableTool / SearchColumnsTool / SearchCellsTool / RunPythonTool / SubmitAnswerTool`
- `build_registry(run_python_timeout_s=10.0) -> ToolRegistry`
- `coerce_items(value) -> list[str]`（pandas/标量 → 答案项）
- 约定：`run_python` 代码须设 `answer`；`submit_answer` 置 `terminate=True`。

## `src/harness.py` — 工具调用流水线 [1]
- `execute_tool(registry, name, args, state) -> ToolResult`（未知/不可用/缺参/崩溃都转 ToolResult，永不抛）

## `src/agent.py` — 单轨 agent 循环 [1]
- `Prompts{charter, general_skill}.system`；`load_prompts(dir) -> Prompts`
- `to_openai_tools(specs) -> list[dict]`
- `run_example(example, table_context, registry, client, prompts, budget=None, tracer=None) -> Prediction`
- 兜底顺序：submit_answer → 最后一次 run_python 的 answer_items → 最后文本 → 空。

## 待建模块（接口先占位，Phase 推进时填实现）
- `src/verifier.py` [1 宽松→2 严格]：`verify(state, answer) -> VerifyResult{ok, qtype, diagnostics}`（分类型）。
- `src/router.py` [2]：`route(utterance) -> list[skill_name]`。
- `src/consistency.py` [2]：`vote(trajectory_answers) -> (items, confidence)`（按 evaluator 归一化投票）。
- `src/run.py` [0.5+]：批量跑数据集 → 预测文件 + trace + 指标 + run manifest。
