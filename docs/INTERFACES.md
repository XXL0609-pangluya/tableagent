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

## Phase 3b — 生成器/验证器讨论协议 [3b]

### 设计思想
之前的问题：verifier 发一段文字说"可能有问题"→生成器因为没有更强的反驳手段而盲目认错（见 nt-5881 回归）。

新协议：**双方都必须拿代码上场，不能只靠说话改答案。**

1. verifier 的检查结果（`VerifyResult`）现在保存它自己跑过的 pandas 代码（`verifier_code`）和 stdout（`verifier_stdout`）。
2. 检查器 flag 时，如果有代码证据，agent 收到 `build_debate_prompt(vr, candidate)` ——这个 prompt 完整展示检查器的代码+结果，并明确要求生成器：
   - **只跑代码之后才能改答案**（defend=跑自己的代码证明正确，然后 resubmit 原答案；accept=跑代码确认检查器逻辑，然后 resubmit 修正值）。
   - 没有代码就不能改——这就堵住了"看到数字就认错"的盲目认错。
3. **讨论步数**：每次 verify 触发时，`effective_max_steps += budget.debate_extra_steps`（默认 3），保证生成器有足够的步数跑代码、核查、再提交。
4. 没有 compute 代码时（仅 A/B/C 文字反馈 / deterministic），退回到旧的简单 `build_verify_feedback`，不额外扩步数（开销更小）。

### 接口变化
- `VerifyResult`: 新增 `verifier_code: str`, `verifier_stdout: str`（`compute_check` 填充）。
- `verifier.py`: `build_debate_prompt(vr, solver_answer) -> str`（新增）；`build_verify_feedback` 保留作非计算型 flag 的兜底。
- `schemas.py`: `Budget.debate_extra_steps: int = 3`。
- `agent.py`: `for` 循环改为 `while step < effective_max_steps`；`effective_max_steps` 在讨论触发时动态扩展；`debate_rounds_granted` 防止重复扩步。

### 安全性
- 候选池（Phase 2b.1）和多数投票（Phase 3a）保留，仍是兜底保障。
- 讨论要求双方提供代码证据，相当于把"谁说了算"从"谁更权威（模型大小/角色）"变成"谁的代码更直接命中表格"。
- **设计原则**：A/B/C 维度审查是**always-on 主检查**，独立重算是**额外一路证据**。
  两模型答案一致 ≠ 没问题（可能共享盲点：都丢单位、都数行数而非去重），所以**即使重算
  一致也照样跑 A/B/C**。
- `verify(...)` 跑全部并合并：
  ① `deterministic_issues`（无模型，命中即短路）
  ② `llm_check` A/B/C 读表审查（**永远跑**）
  ③ `compute_check` 独立重算（有 df 时跑）——verifier 用**另一模型**从零写 pandas 跑沙箱，
     与主答案比对（`answers_match` 复用官方 evaluator）。`compute_match` = True 一致 /
     False 不一致 / None 弃权。CoT 模型 `max_tokens=1500`。
  任一路 flag → 整体 flag；`source` 形如 `llm` / `compute` / `llm+compute`。
- **投票兜底（agent.py `_majority_vote`）**：finalize 时对 {主模型各次提交} + {**不一致的**重算}
  做去噪聚类多数投票。关键：**只有"不一致"的重算才作为候选值入票**（提供真正的替代答案）；
  "一致"的重算只算背书、不入票，以免和 A/B/C 纠正打架（否则共享盲点的重算 + 原答案会把
  改对的答案投下去）。安全性质：**verifier 只有在某次主模型提交与它一致时才能改写答案**，
  孤立错误重算赢不了平票，平票取**最近一次主模型提交**。
- 实测：nt-647 重算出正确 16；nt-2354 verifier 重算乱码被投票挡掉；A/B/C 抓 `4x400 m`
  漏 `relay`、`tied`/`tie`、`17,693`/`17693` 这类两模型共享的形式盲点。
- 成本提醒：每次提交 2 次 verifier 调用（A/B/C + 重算），200 题整体耗时约 2-2.5x。

## `src/verifier.py` — Phase 2c table-aware + independent-model verify [2c]
- **独立模型**：verifier 用 `load_verifier_config()` 选的模型，刻意≠solver（默认
  solver=qwen → verifier=deepseek-v4-flash），避免同模型同源盲点。`run_example` 收
  `verifier_client`；脚本里只建一次。
- `verify(client, question, items, *, table_view, evidence_summary, use_llm=True) -> VerifyResult{ok, issues, fix_hint, source, axis, model}`
  - tier 1 `deterministic_issues`：anchor-inclusion（"other than/same as (X)"，含前缀子串匹配）、cardinality（单答却≥3项）。高精度、无 LLM。
  - tier 2 `llm_check(...)`：**带 `table_view`**（schema+前~30行），重新审题并带入答案查表，按三维判断——
    - **A 形式/精度**：是否逐字照抄单元格（不擅自四舍五入/加减单位/展开缩写/拆掉多段单元格）。
    - **B 意图**：答案类型对不对（名称 vs 数值；单值 vs 列表）。
    - **C 计数/聚合**：去重、空/"–"单元格、Total/汇总行的处理。
  - 仍**不直接断言替换值**，只指出维度+该复查什么（保留 2b 反向伤害修复）。
  - 注意：deepseek 等推理模型 CoT 会吃 token，`max_tokens=1024` 留足正文余量。
- `build_verify_feedback(vr) -> str` — 含 `Axis X` 提示的复查 prompt。
- Agent 在 `submit_answer` 后调用 verify；≤1 retry（`Budget.max_verify_retries`）。
## 争议题剔除 [2c]
- `eval/disputed.json`（tracked）：`{"disputed": {id: reason}}`，记录 gold 本身有歧义/与题不符的样本。
- `data.load_disputed() -> {id: reason}`。
- `evaluator.evaluate(preds, targets, exclude_ids=...)` 同时报 **raw** 与 **adjusted**（剔除争议题）准确率 + `num_excluded_disputed`；run 脚本两个数都打印、都存进 JSON。透明起见 raw 永远保留。
- 当前已标：nt-1470（队数 vs 场次）、nt-9465（名次 vs 数值）；独立 verifier 自检也把 nt-1470 的 gold 判为可疑，旁证标记合理。

## `src/verifier.py` 候选池
- **Candidate pool (agent.py)**: every submit is kept with its pass flag; finalize
  picks the last *passed* candidate, else the *first* submitted (so a noisy advisory
  can't replace a good answer). Recorded in `evidence.candidates` / run-row `candidates`.

## `src/analysis.py` — post-run error analysis [2b]
- `load_run_rows`, `load_traces_jsonl`, `compare_buckets`, `format_example_report`, `format_bucket_from_runs`
- Used by `scripts/diagnose.py`, `scripts/compare_runs.py`, `scripts/inspect_traces.py`, `scripts/analyze.py`

## 待建模块（接口先占位，Phase 推进时填实现）
- ~~`src/verifier.py`~~ ✅ Phase 2b lightweight verify
- ~~`src/router.py`~~ ✅ keyword skill router lives in `src/agent.py` (`select_skills`)
- `src/consistency.py` [2]：`vote(trajectory_answers) -> (items, confidence)`（按 evaluator 归一化投票）。
- `src/run.py` [0.5+]：批量跑数据集 → 预测文件 + trace + 指标 + run manifest。
