# WikiTableQuestions Table-QA Agent — 项目规划

> 目标：在 WikiTableQuestions (WTQ) 上，用 agent 方法尽可能提高问答准确率。
> 本文档是方案与迭代规划，**不含项目实现代码**。

---

## 0. 复盘：上一版为什么效果不好

| 上一版问题 | 根因 | 本版对策 |
|---|---|---|
| 所有东西塞进提示词，上下文爆炸 | 没有显式的上下文裁剪策略 | 每步只放「schema + 少量样例行 + 当前问题 + 上次报错」，其余一律丢弃 |
| 16 个工具，模型不会选 | 决策空间过大，工具粒度太细 | 收敛到 ~5 个高层工具；**保留工具调用 agent**，但用强契约 + step budget + 结构化观测 + verifier 约束自由度 |
| 模型不反思、不自检 | 自检只写在系统提示里，没有结构强制 | 把「执行结果 → 检查 → 重试」做成**代码层的状态循环**，不依赖模型自觉 |
| 没有专业评测/迭代流程 | 缺少可复现评测器 + 错误归因 | 移植官方 evaluator + 固定 dev 集 + 错误分类表 + 迭代曲线 |

---

## 1. 任务与数据事实（决策依据）

- **规模**：22033 题。`training.tsv`(14152) / `pristine-unseen-tables.tsv`(4344, **正式测试集**) / `pristine-seen-tables.tsv`(3537) / `random-split-{1..5}-{train,dev}` / `training-before300`。
- **样本格式**（TSV 4 列）：`id  utterance(问题)  context(csv路径)  targetValue(答案)`。
- **答案类型**：字符串 / 数字（可带千分位 `12,467`）/ 日期（`January 26, 1995`）/ **列表**（用 `|` 分隔，如 `A|B`）。
- **表格**：`csv/xxx-csv/yyy.csv`，第一行表头，规模小（几行~几十行），单元格常含脏值（`Did not qualify`、带括号注释、千分位等）。
- **评测**：**集合匹配**——预测集合与目标集合大小相等且逐项匹配。匹配规则（见 `evaluator.py`）：归一化字符串相等 / 数字相等 / 日期完全相等三选一。
- **⚠️ `evaluator.py` 是 Python 2 写的**，需移植到 Python 3（或子进程包装），作为唯一权威评分。

---

## 2. 架构决策（已定）

### 核心论点：胜负手是机制，不是"用没用 agent"
> 经验事实：纯 "codegen + 自纠错" 与纯 ReAct 在 WTQ 上都常常一般；已发表的重工程固定流程也只是中等水平。
> 我们要赢，靠的是三个**具体机制**，而它们都需要 function calling 才能自然落地：
> 1. **Grounding（接地）**——先用工具看表/搜单元格/看列取值，把代码建立在真实观测上，解决"问题词 ↔ 单元格值对不上"（实体链接）这个 WTQ 头号坑。
> 2. **Verification（验证）**——答完用第二种方式复核 / 检查答案是否真实出现在表里，再决定是否重答。
> 3. **Self-consistency（多轨投票）**——N 条独立轨迹用 evaluator 归一化规则投票，WTQ 上性价比最高的涨分手段。
>
> 设计原则：方案卖点钉死在这三条机制上；"步骤更多 / 更 agentic" 本身不是目标。

### 主路线：function-calling 工具调用 agent（codegen 作为其中一个工具）
保留真正的工具调用循环。失败的不是"用工具"，而是工具粒度乱、反馈差、上下文不裁剪、没有验证。对策：**工具少而精 + 强契约 + 高质量观测反馈 + 外层可靠性 harness。**

- **执行内核仍是 pandas**：表格统一按字符串读入 `DataFrame`，`run_python` 工具执行 pandas 代码做查询/聚合/比较/算术。
- pandas 优先于 SQL：单元格脏值多、类型混杂，清洗/类型转换更灵活（代码里按需 cast）。
- 不引入 LangGraph：自写最小 agent 循环 + 可靠性 harness，更可控、可解释、依赖少；但**借鉴框架的上下文管理思路**（见 §2.4.1）。

### 2.1 工具清单（少而精，每个都有强契约 + 好反馈）
| 工具 | 作用 | 关键反馈 |
|---|---|---|
| `inspect_table()` | 列名、dtype 推断、行数、每列样例/唯一值数 | 让模型先"认识"表 |
| `search_columns(query)` | 把问题用词对齐到表头（如 "attendance"→`Avg. Attendance`，"city"→`Venue`） | 返回疑似相关列 + 相似度 → **列 grounding** |
| `search_cells(query, col?)` | 模糊/精确搜单元格 | 返回命中单元格的"原始值+所在行列+所属列" → **值 grounding** |
| `run_python(code)` | 在持久 DataFrame 上执行 pandas | 见下方强契约（结构化 answer + evidence） |
| `submit_answer(items, evidence)` | 终止并提交答案（触发验证阶段） | 进入 verify，不直接结束 |

> 相比上一版 16 个：去掉所有"用 LLM 模拟解释器"的细粒度计算工具（计算交给 `run_python`），只保留**探索类**(inspect/search_columns/search_cells) + **执行类**(run_python) + **终止类**(submit)。决策空间小、几乎不会选错。
>
> **`run_python` 强契约（关键）**：要求生成的代码必须设置结构化变量，便于验证/留痕/归因：
> ```python
> answer = [...]          # 最终答案候选（list）
> evidence = {...}        # 来源：涉及的行索引/列名/中间量
> ```
> 工具返回 `{answer_items, evidence_rows, evidence_columns, intermediate, stdout, error, truncated}`；超长输出头尾截断。
> 好处：verifier 能校验"答案怎么来的"，而非只看一个孤立结果；trace 与错误归因也直接可用。

### 2.1.1 工具调用协议抽象（应对 API 是否原生支持 FC）
> 现实约束：公司 API 是 OpenAI 兼容，但**不确定是否支持原生 function calling**。
> 因此把"工具调用协议"抽象成一层接口 `ToolCallProtocol`，agent 主循环只依赖这层，不关心底层怎么实现：
> - **native 后端**：用 API 的 `tools` / `tool_calls` 字段。
> - **prompted 后端**：在 prompt 里描述工具，要求模型输出固定 JSON 调用，我们自己解析（任何 OpenAI 兼容 chat 模型可用）。
> - 配置切换 + 启动自检（探测目标模型是否支持 native，否则自动回退 prompted）。
>
> 收益：方案不被某个 API 是否支持 FC 卡死；同时这层抽象本身就是干净接口，便于后续换模型/换厂商。
> prompted 后端的健壮性由 harness 兜底：JSON 解析失败 → 回灌格式错误让模型重出；连续失败 → 走兜底答案。

### 2.1.2 Skill 库 + Router（渐进式披露，借鉴生产 agent 的 AGENT.md / SKILL.md 模式）
> 借鉴：生产 agent 常用 `AGENT.md`(总章程) + 大量 `SKILL.md`(每类问题一份) + 工具清单 JSON。
> 精髓是**渐进式披露**——默认上下文只有总章程+工具清单，**按问题类型动态加载对应 skill**，其余不进上下文。
> 这正是上一版"全塞进提示词"的解药。
>
> 在本项目的映射：
> - **总章程（顶层系统提示）**：agent 是谁、总体原则、何时调哪类工具、输出纪律。稳定不变。
> - **Skill 库**：按 WTQ 问题类型组织的"解题手册"——每份含：该类套路 + 2~3 个 `问题→工具调用/代码` 示例 + 常见坑（接地/格式/日期）。
>   - 候选类型：lookup / superlative(最值) / count-aggregate / arithmetic(差值求和) / ordinal(前一个后一个) / date-reasoning / multi-answer(列表)。
> - **Router**：先判问题类型 → 只把对应 skill 注入上下文（可多选）。判错也有兜底（回退通用 skill）。
> - **工具清单**：用轻量**工具注册表**实现（非 MCP——同进程任务，MCP 属过度工程，除非课题要求）。
>
> 纪律：**不预写一堆 skill**。Phase 1 先用 1 个通用 skill 跑通；之后**让 skill 从错误归因里长出来**（§4），每个新 skill 都要消融验证确实涨分。
>
> **统一 skill 模板（防止"渐进披露"退化成"分文件版大 prompt"）**：每个 `SKILL.md` 固定结构且**长度上限 ~1500~2500 tokens**：
> 1. When to use（命中条件）2. Required grounding steps（必做接地）3. Recommended tool sequence 4. Pandas pattern 5. Common mistakes 6. Answer formatting rules 7. ≤2 个 examples。

### 2.2 控制流（agentic，但有护栏）
```
load_table
  → 装配总章程 + skill（Phase 1: 固定 general skill / Phase 2: router 判类型选 skill）
  → [Phase 2, 可选] 简单 lookup? → 直接 run_python 快路径
  → agent loop: 模型自由调用 inspect/search_columns/search_cells/run_python (受 step budget 约束)
       每轮: 组装精简上下文 → 模型出 tool_call → 执行 → 结构化观测回灌
  → submit_answer 触发 → verify(分类型校验 answer+evidence，见 §5)
       ├─ 不通过且未超预算 → 带"为何失败"回到 loop
       └─ 通过 → 记录该轨迹答案
  → [Phase 2, self-consistency] 跑 N 条轨迹 → 归一化投票 → 最终答案
```
- **step budget**（如 ≤6 步）、单步超时、总重试上限，超限**必出兜底答案**（绝不空手而归）。
- **分阶段启用**：Phase 1 = 单轨 + general skill + 宽松 verifier（只诊断不硬拦）；router / 严格 verify / 多轨投票放 Phase 2（见 §7）。

### 2.3 答案格式化（隐形失分大户，独立模块）
- pandas 结果 → evaluator 形式：单值/列表统一成「字符串项集合」，数字去千分位，日期规范化。
- 复用移植后 evaluator 的归一化函数，保证投票口径与最终评分一致。

### 2.4 核心数据契约（契约先行，降低 AI 开发失控风险）
> 这是后续所有模块的"接缝"。先定死这些结构，再填实现。放 `src/schemas.py`。
```python
Example      = { id, utterance, table_path, target_value: list[str] }
TableContext = { df, columns, dtypes, n_rows, schema_text, sample_rows }
ToolCall     = { name, args }
ToolResult   = { ok, content_text, structured: dict, error, truncated, terminate }
Observation  = { tool_call, tool_result, step }           # 进 trace + 上下文装配
AgentState   = { example, table_context, observations, facts,
                 attempts, current_answer, evidence, budget(steps/tokens/retries) }
Prediction   = { id, items: list[str], evidence, confidence, trace_id }
TraceEvent   = { step, kind, prompt_hash, tool_call, observation, tokens, latency_ms }
```
- `ToolResult.structured` 承载 `run_python` 的 `answer_items/evidence_rows/...`，verifier 与 formatter 都读它。
- `Prediction` 始终带 `evidence` 与 `trace_id`，错误归因可一键回溯到具体轨迹。

### 2.4.1 从框架学到的上下文管理思路（落到自写 harness）
这部分是你上一版"上下文很差"的直接解药，把 LangGraph 等的理念吸收进来：
- **State vs Scratchpad 分离**：持久状态（表 schema、已确认的 grounding 事实、当前最佳答案）与本轮临时消息分开；临时消息可裁剪/丢弃，持久状态结构化保留。
- **Reducer 思维**：每步对状态做**显式归约**（"把这次观测压缩成一条事实"），而不是把原始输出无脑 append。
- **Per-step 上下文装配**：每一步只构造该步需要的上下文，不是把所有历史一股脑喂进去。
- **结构化观测**：工具返回结构化对象并截断超长内容，再格式化进 prompt；绝不丢原始大块文本。
- **消息窗口**：保留系统提示 + 最近 k 轮工具交互 + 状态摘要；更早的轮次压成摘要。

---

## 3. 模型策略

- **迭代期**：便宜但强的 OpenAI 兼容模型（DeepSeek-V3 / GPT-4o-mini / Qwen2.5-72B 等，按公司 API 实际可用项选）。**本方案是工具调用 agent**，但通过 `ToolCallProtocol` 抽象（§2.1.1）兼容"原生 FC"与"prompted-JSON"两种后端——**不强制 API 原生支持 FC**。选型时实测工具调用成功率，不稳的淘汰。
- **预算**（每日 ¥35）：200 题快集一轮 ≈ ¥3 以内；全量 dev(~2800) 一轮 ≈ ¥30。→ **日常在快集迭代，隔几天跑一次全量 dev**。
- **最终报告**：最强可用模型在测试集上**只跑一次**。
- 模型与 prompt 配置集中在一个 config，方便切换做对比。

---

## 4. 评测与迭代闭环（本版最重要的补强）

1. **数据分工**：
   - 主开发集：`random-split-1-dev`（调参只看它）。
   - 快集：从主开发集分层抽 200 题，用于分钟级快速迭代。
   - **测试集 `pristine-unseen-tables` 全程不碰**，最后报告才跑一次，防止过拟合。
2. **评测器**：移植 `evaluator.py` 到 Py3，封装 `evaluate(pred_tsv, gold_tsv) -> accuracy`。
3. **预测输出格式**：每行 `id \t item1 \t item2 ...`，无预测则只输出 `id`（与官方一致）。
4. **每条样本留痕**：`id / 问题 / 表路径 / 生成代码 / 执行结果 / 预测 / 标准答案 / 是否正确 / 错误类别 / 调用次数 / token`。
5. **错误归因分类**（每轮统计占比，按高频优先修）：
   - 解析/语法错（代码跑不起来）
   - 选错列 / 找错行
   - 聚合或比较逻辑错
   - **答案格式错**（多答案列表、数字千分位、日期格式）——WTQ 高频失分点
   - 超时/异常
   - 问题歧义 / 表格本身脏
6. **迭代记录**：每轮存 `准确率 + 各错误类别占比 + 平均调用次数/token` 到 CSV，画迭代曲线，支撑论文。

---

## 5. 可靠性 Harness（必不可少的护栏）

> 你的痛点之一：模型很笨、不按提示反思自检。对策是**把可靠性写进代码层，而不是寄希望于模型自觉**。

- **工具契约校验**：工具入参用 pydantic 校验；非法调用不执行，回灌结构化错误让模型改（而非静默失败）。
- **沙箱执行**：`run_python` 在受限环境执行——超时、内存上限、白名单模块（pandas/numpy/re/datetime）、禁网禁文件写；异常转成可读反馈。
- **观测治理**：所有工具输出截断/摘要（行数上限、字符上限），结构化后再进上下文。
- **预算护栏**：单题 turn budget、单步超时、总 token 上限；任何超限分支都走**兜底答案**（如取当前最可能值），保证每题都有输出。
- **验证层（Verifier，分类型，验证 answer+evidence 而非孤立答案）**：
  - lookup / multi-answer：答案应能在表中找到 provenance（evidence_rows 命中）。
  - arithmetic / count：答案**不要求**出现在表里，但必须能由 evidence 复现计算（核对中间量/参与行）。
  - superlative / ordinal：定位到的目标行必须存在且唯一/有序合理。
  - date-reasoning：允许标准化日期或推导结果，校验日期解析一致。
  - 数量/类型一致性：预测集合大小与问题期望一致（单值 vs 列表），类型与问题匹配。
  - **Phase 1 宽松**：只打分+诊断写进 trace，不硬拦；**Phase 2 收紧**：不过则带原因重试。
- **自一致性投票**：N 条独立轨迹（不同温度/采样）→ 用 evaluator 归一化规则投票，平票时回退到验证分最高的轨迹。
- **全链路 Trace**：每步记录 `(状态快照, prompt, tool_call, 观测, 耗时, token)`，结构化落盘——这是 AI 开发期定位 bug 的命脉。
- **确定性**：固定随机种子、记录模型版本与参数，保证实验可复现。

---

## 5.1 借鉴 openclaw 的工程护栏（采纳 / 简化 / 跳过）

> 已通读 `openclaw-agent-study/` 的 agent 核心（agent-loop / tools / compaction / skills / system-prompt 等 ~15k 行）。
> 原则：**采纳能直接提升单题 table-QA 可靠性的护栏；简化为我们窄场景所需；跳过为"长会话 / 多插件 / 多 provider"而生的重型设施。**

### A. 采纳（高价值，直接落进 harness/agent）
1. **工具调用流水线**：`查表→校验入参(pydantic)→before 钩子(可 block)→执行→after 钩子(可改写/置 terminate)`。任何环节失败都**转成结构化 tool_result 回灌模型**，绝不抛异常中断循环。（对应 openclaw `agent-loop.ts` prepare→validate→hook→finalize）
2. **显式 step budget**：openclaw 核心循环**竟然没有硬性最大轮数**（靠"无 tool_call 自然终止"）。我们**必须加** `max_tool_rounds`，超限走兜底答案。
3. **自然终止 + 显式 terminate**：`submit_answer` 返回 `terminate=true` 才结束 agent loop；一批工具全部 terminate 才停。（`types.ts` terminate 语义）
4. **工具可用性门控（简化版）**：声明式 `availability`——如 `run_python` 仅在表已加载时可见、`search_cells` 需合法列名；不可用工具**带诊断隐藏**而非静默丢。（`availability.ts` / `planner.ts` 的精简版：去掉 auth/plugin/env 信号，只留 "context 条件"）
5. **确定性工具计划**：注册时**断言工具名唯一**、可见工具必须有 executor，否则启动即报错。（`planner.ts` `buildToolPlan`）
6. **工具结果体量上限 + 头尾截断**：单结果上限（按上下文占比，如 ≤30% 或固定 8~16K 字符）+ **保留头和尾**（错误/汇总常在尾部）+ 中间省略标记。一张大表 dump 不能吃光上下文。（`tool-result-truncation.ts` `hasImportantTail`）
7. **请求前预算路由**：发 LLM 前估算 token（×1.2 安全系数），超预算先**截断工具结果**，仍超再考虑摘要。（`preemptive-compaction.ts` fits / truncate / compact 分流 + `SAFETY_MARGIN`）
8. **每次请求前的 `transformContext`**：上下文装配/裁剪在**每次 LLM 调用前**做（不是只在加载时），契约是"绝不抛异常、失败回退安全值"。正好落地我们 §2.4 的 per-step 装配。
9. **LLM 流不抛异常**：provider 报错编码成 `stopReason=error/aborted` 的消息，循环优雅退出 + 合成失败消息。（`types.ts` StreamFn 契约 / `agent.ts` pushLoopFailure）
10. **重试分类**：摘要/调用重试 3 次带 jitter，但 **abort/timeout 不重试**。（`compaction.ts` `shouldRetry`）
11. **provider 无关内核（依赖注入）**：核心只依赖注入的 `stream` 依赖——正是我们 `ToolCallProtocol`（§2.1.1）的同构思想。（`runtime-deps.ts`）
12. **系统提示三件套**：
    - 只列**实际注册启用**的工具（`system-prompt.ts` enabled-only）；
    - **execution-bias 段**：要求"用工具拿证据再回答、空结果就换方法、不要轻易放弃"——天然契合我们的 verify 机制（`system-prompt.ts` ~448）；
    - **稳定前缀 / 动态尾部分离**以利 prompt 缓存（稳定段哈希缓存）。
13. **bootstrap 上下文限额**：注入表 schema/样例行时做**单文件 + 总量字符上限**，避免大表文档喧宾夺主。（`bootstrap-files.ts`）
14. **Skill 渐进披露（精修我们的 §2.1.2）**：系统提示只放 `<available_skills>` 目录（name/description/location/**version**），命中才"读取"完整 skill；version 变了要重读；并设 **char/count 预算的三级降级**（full→compact(仅名+位置)→二分裁到放得下）。（`skill-contract.ts` `formatSkillsForPrompt` / `workspace.ts` `applySkillsPromptLimits`）
15. **Skill 可见性策略**：`disableModelInvocation` / `includeInAvailableSkillsPrompt` 等开关，区分"模型可调"与"仅人工/仅展示"。（`skills/types.ts`）
16. **工具用/结果配对完整性**：裁剪历史时不能把 `assistant 的 tool_use` 与其 `tool_result` 拆散，否则 provider 报 orphan 错。（`compaction-planning.ts` repair pairing）

### B. 简化（我们场景用不到完整版）
- **历史压缩/摘要（compaction）**：单题会话很短（turn budget ~6、表也小），基本不会撑爆上下文。→ **只保留"工具结果截断 + 请求前预算检查"两道便宜闸**，**不做** LLM 历史摘要这套重机器；真遇到超大表再按需启用截断。
- **可用性表达式**：只实现"context 条件"一种信号，不要 allOf/anyOf/auth/plugin/env 全家桶。
- **失败快停**：保留"表+问题即便截断仍放不下→优雅产出兜底答案/标记跳过"，但不用 openclaw 的 `FailoverError`→多 provider failover 链路。

### C. 跳过（为长会话/多插件/多 provider 而生，对我们是过度工程）
- worker 线程做 compaction planning
- append-only transcript DAG / 分支重写 / 写锁 / transcript 事件总线
- 多源 skill 发现（symlink、插件、深层目录扫描）、Skill Workshop 提案流程
- plugin / channel / MCP 工具计划与 inventory 分组
- steering / follow-up 消息队列、子 agent 编排
- 模型 fallback 链、handoff 摘要
- 跨 provider 的 context window 发现缓存
- bootstrap 续传 marker / 多模式注入

> 详细模式清单见本次代码勘探报告：[openclaw 护栏勘探](7280398f-36d9-4c03-84a9-df3c17b86f3e)。

---

## 6. 面向 AI 开发的工程化与接口（防止"越开发越乱"）

> 你担心全程 AI 开发到后期会乱套——对，所以**契约先行**：先定边界与接口，再填实现，让每次改动都发生在稳定接缝上。

- **契约先行**：先定义模块边界、核心 dataclass/pydantic 模型（`Example`、`TableContext`、`ToolCall`、`Observation`、`AgentState`、`Trace`、`Prediction`）与工具接口，再写逻辑。
- **工具注册表模式**：新增工具 = 实现统一 `Tool` 接口（`name/schema/run()`）+ 注册一行，**不改动 agent 主循环**。这从结构上杜绝"加工具就要动核心"的旧问题。
- **强模块边界**：`data / llm / tools / agent / harness / evaluator / formatter / run` 各司其职，依赖单向，互不串味；每个模块可独立单测。
- **黄金回归测试**：固定 ~50 条覆盖各题型的样例 + 期望通过情况，作为回归集；每次改动都跑，**分数掉了立刻发现**，避免 AI 改代码越改越崩。
- **配置驱动**：模型名、温度、turn budget、N、few-shot 等全进 config，无硬编码魔法值；每次实验存一份 run manifest（配置 + 数据 + 分数 + commit）。
- **小步提交**：每个 Phase / 每个机制独立提交，便于回滚和对比。

---

## 7. 分阶段路线图（含交付物）

> **MVP 收紧原则**：先把"无模型地基 → 单轨 agent + trace"跑通拿到 baseline，让真实 trace 暴露错误，再按数据加 router / 严格 verify / 多轨投票。不在没有 baseline 前堆机制。

**Phase 0 — 无模型地基（不依赖任何 LLM）**
- 项目骨架 + 依赖（pandas、python-dotenv 等）
- `schemas.py`：§2.4 核心数据契约
- 数据加载器：解析 TSV、按 context 读 CSV → `TableContext`（统一字符串读入）
- 评测器移植到 Py3 + `evaluate(pred_tsv, gold_tsv)`，用官方样例自测对齐
- trace 落盘格式 + tool registry 接口 + context_budget(截断) + sandbox 骨架
- 冒烟脚本：加载快集 → 跑通"加载+评测"链路（用占位/gold 预测验证 evaluator）
- 交付：无模型即可运行的地基 + 通过的评测器对齐自测

**Phase 0.5 — 模型接入 + direct-answer baseline**
- `llm.py` + `ToolCallProtocol`（native / prompted 自检）
- baseline：整表（截断）丢给 LLM 直接答 → 建立"不用 agent 的下限"对照
- 交付：快集上的 direct-answer 分数

**Phase 1 — 单轨 function-calling agent（核心 baseline）**
- 工具注册表（唯一名断言 + 可用性门控）+ `inspect_table`/`search_columns`/`search_cells`/`run_python`/`submit_answer`
- 工具调用流水线：校验→before 钩子→执行→after 钩子，错误转 tool_result 不抛异常
- 最小 agent 循环（**显式 step budget**、结构化观测、每步上下文装配）+ 沙箱 + 兜底
- 工具结果头尾截断 + 体量上限（context_budget）
- `AGENT.md` 总章程（含 execution-bias 段）+ **1 个通用 skill**（不做 router）
- 答案格式化模块（列表/数字/日期归一化）+ **宽松 verifier（仅诊断）**
- 交付：快集 + dev 上的 agent baseline 分数 + 全链路 trace

**Phase 2 — 加机制（grounding / verify / 自一致性 / router）**
- 强化 `search_cells` 接地；加入 Verifier 验证层；加入 N 轨投票
- 加入 router + skill 库脚手架（先按类型拆出几个核心 skill）
- 可选 lookup 快路径
- 交付：逐机制消融（baseline / +grounding / +verify / +voting / +skills）准确率对比

**Phase 3 — 系统化迭代（让 skill 从错误里长出来）**
- 错误归因表 → 高频错误类别**新增/打磨对应 skill** + 改工具反馈（重点攻接地失败、多答案、日期、数字格式）
- 每个新 skill 都做消融，确认涨分才保留
- 交付：迭代曲线 + 各错误类别下降情况 + skill 增长对照

**Phase 4 — 对比实验与报告**
- 补做纯 codegen 单步 / 纯 ReAct 等对照方案，与本方案对比
- 在测试集 `pristine-unseen-tables` 上跑最终分数（只跑一次）
- 交付：方法对比表 + 最终准确率 + 课题结论

---

## 8. 建议的项目结构（待实现时落地）

```
TableAgent/
├── WikiTableQuestions/        # 数据集（已下载）
├── PLAN.md                    # 本文档
├── src/
│   ├── config.py              # 模型/路径/超参集中配置 + run manifest
│   ├── schemas.py             # 核心 dataclass/pydantic：Example/TableContext/AgentState/Trace...
│   ├── data.py                # TSV/CSV 加载 → DataFrame（统一字符串读入）
│   ├── evaluator.py           # 移植自官方(Py3)，含归一化函数
│   ├── llm.py                 # 模型客户端封装 + ToolCallProtocol（native / prompted 两后端）
│   ├── tools/                 # 工具：base.Tool 接口 + registry(唯一名/可用性门控) + 各实现
│   ├── harness.py             # step budget/超时/兜底/trace/重试分类/校验流水线
│   ├── context_budget.py      # 请求前 token 估算 + 工具结果头尾截断 + 体量上限
│   ├── sandbox.py             # 安全执行 pandas 代码
│   ├── agent.py               # 最小 agent 循环 + transformContext(每步上下文装配)
│   ├── verifier.py            # 分类型验证层
│   ├── router.py              # [Phase 2] 问题类型判别 → 选 skill
│   ├── consistency.py         # [Phase 2] 多轨投票
│   ├── formatter.py           # 结果 → 答案集合 归一化
│   └── run.py                 # 跑数据集 → 预测文件 + trace + 指标
├── prompts/
│   ├── AGENT.md               # 总章程 / 顶层系统提示
│   └── skills/                # 各问题类型的 skill 手册（渐进披露，按需加载）
├── tests/                     # 单测 + 黄金回归集
├── results/                   # 预测、trace、迭代指标 CSV、错误分类、manifest
└── requirements.txt
```

---

## 9. 风险与注意点

- **评测器移植正确性**：移植后用官方 `*.examples`/已知样例自测，确保和原版打分一致。
- **沙箱安全**：执行 LLM 生成代码需超时 + 限制可用模块，防止死循环/危险调用。
- **答案格式是隐形失分大户**：务必把"集合匹配 + 数字/日期归一化"做扎实。
- **不要在测试集上调参**：纪律问题，直接决定结论是否可信。
- **成本看护**：每轮记录 token；默认快集，避免无意识跑全量烧额度。
- **function calling 稳定性 / 后端不确定**：公司 API 是否原生支持 FC 待确认；用 `ToolCallProtocol`（§2.1.1）抽象兼容原生与 prompted 两种后端，启动自检择优。先实测候选模型工具调用成功率，不稳的淘汰；harness 对非法/缺失/JSON 解析失败的 tool_call 要有降级处理。
- **"更 agentic ≠ 更准"**：每加一个机制都要用消融验证它真带来涨分，否则砍掉——避免堆复杂度而无收益。
- **自一致性成本**：N 轨投票成倍烧 token，先在快集确认收益再决定 N，必要时只对"低置信/验证失败"的题启用。
