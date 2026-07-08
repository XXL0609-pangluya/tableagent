# 论文级 Proposal：Compute-Optimal Reasoning Control for Table QA

> **性质**：动工前的论文级方向 proposal。整合了三位老师的建议（`docs/brainstorm.md` / `brainstorm2.md` / `brainstorm3.md`）与我们自己的数据发现（`docs/RESEARCH_DIRECTION.md`）。
> **关系**：`RESEARCH_DIRECTION.md`（数据发现 + 三杆流水线）是基础事实层，本文档是其**升维后的论文叙事 + 方法设计 + 完整想法档案**。`PLAN.md` 是工程规划，`HANDOFF.md` 是交接。
> **生成时间**：2026-07-02。

---

## 0. 一句话世界观（论文卖点）

> **Compute-Optimal Reasoning Control for Table QA**：LLM table agent 的性能瓶颈**不是生成能力**，而是**认识论不确定性的错误路由（misrouting of epistemic uncertainty）**——算力被浪费在简单题上，却对"虚假共识（false consensus / 漏检）"的硬题视而不见。我们不研究"如何推理得更好（reasoning generation）"，而研究"**如何管理推理（reasoning control）**"：用**扰动敏感度**识别虚假共识，把测试时算力精准投到真正需要"元认知干预"的那 ~20% 硬题上，在**相同算力预算**下超越全局多轨采样。

这条主线天然连接四条 2025-2026 活跃主线：**Test-Time Compute Scaling / Meta-Reasoning / Reasoning Control / Agentic AI**。

---

## 1. 核心动机：False Consensus（我们独有的数据护城河）

论文的第一张图应该是这个反直觉发现（来自 1000 题 trace 审计，`scripts/candidate_diversity.py`）：

> **业界都在卷 verifier 有多强、debate 有多深。但我们发现，agent 最大的敌人不是"吵不出结果"，而是"虚假的共识"——检查器自信地放行了错误答案。**

关键数据（详见 `RESEARCH_DIRECTION.md` §2）：

- **53%–87% 的错题，压根没触发 debate**（检查器一次放行，实则答错的 false negative）。
- 现有 verifier 有严重的**同源盲区**：它和 generator 犯同样的口径错误（`HANDOFF.md` §2.3）。
- 反过来，**79% 的题只有单一候选、且正确率高**（90.7% / 86.8%）——对它们做多轨/debate 是纯浪费。
- 纯选择的全局天花板只有 +2.3pt——**证明"换谁拍板"不能单独成篇，必须先解决"该对谁动用算力"（触发）**。

→ 因此本文不研究"如何让 debate 更激烈"，而研究两件事：**(1) 如何打破虚假共识（Trigger）；(2) 打破后如何建立基于证据的裁决（Adjudication）**。

---

## 2. 方法：两支柱串成的 Compute-Optimal Pipeline

用户已定：**两支柱一起做，串成完整 pipeline**。整体控制流：

```
                        ┌─────────────────────────────────────────────┐
   Question + Table ──▶ │  Fast Path: Generator 出初步答案 + 证据链      │
                        └───────────────────┬─────────────────────────┘
                                            │
                        ┌───────────────────▼─────────────────────────┐
   支柱1 (杆A/触发)      │  Semantic Perturbation Trigger               │
   打破虚假共识          │  微扰表格/问题 → 答案或代码AST是否漂移?         │
                        └───────┬───────────────────────┬─────────────┘
                          稳定  │                   漂移 │ (脆弱/高认识论不确定)
                        (可信放行,跳过省算力)             │
                                            ┌───────────▼─────────────┐
   支柱B (多样性,支撑)   │           Provenance-Diverse Generation     │
                        │  逻辑路径变体(先filter后groupby / SQL vs pandas) │
                        └───────────────────┬─────────────────────────┘
                                            │
   支柱2 (杆C/裁决)      ┌───────────────────▼─────────────────────────┐
   基于证据裁决          │  Evidentiary Adjudication                    │
                        │  Arbiter审"证据链/数据流是否排除合理怀疑",      │
                        │  而非判"38对不对"; 反事实查询区分候选           │
                        └───────────────────┬─────────────────────────┘
                                            ▼
                                        Final Answer
```

### 支柱 1 — Semantic Perturbation Trigger（杆A，主攻虚假共识）

**思想来源**：老师3 Story 2（控制论/物理的灵敏度分析、雅可比）。

- **痛点**：B 类漏检最难识别，因为它们正是检查器自认为"没问题"的题。我们原先设想"训一个 false-negative predictor"——老师3正确指出它**缺解释性、易过拟合**。
- **做法**：放弃训练黑盒。对**输入表格/问题**做语义微扰（例：`2023年`↔`2023 年`、`12467`↔`12468`、打乱无关行序、同义列名替换），让 generator 用同一逻辑重跑。
  - **代码 AST 突变** 或 **答案漂移** → 推理路径脆弱/高认识论不确定 → **触发深究（杆B+杆C）**。
  - 微扰后稳如泰山 → verifier 放行可信 → **跳过 debate，省算力**。
- **卖点**：Robustness-Aware Triggering via Semantic Perturbation。**零训练成本、有物理/逻辑解释性、可证伪**。
- **可选统计外衣**：用 semantic entropy / logits 熵量化认识论不确定性；Conformal Prediction 给触发一个覆盖率保证的阈值。

### 支柱 2 — Evidentiary Adjudication（杆C，主攻裁决质量）

**思想来源**：老师3 Story 1（法学的对抗性证据开示 + 排除合理怀疑）+ 老师2（Table-Critic 过程 Judge / PRM 过程验证）。

- **痛点**：现在 debate 太浅——"generator 说 38，verifier 说对/错"，结果级审查容易同源幻觉。现有 finalize 是写死的 `first_verified`，裁决与选择被 verifier 一把抓。
- **做法**：
  - **Code-as-Provenance**：强制 generator 提交答案时给出证据链（依赖哪些 row/col + 什么聚合）——我们 `run_python` 强契约（`PLAN.md` §2.1）已经要求输出 `evidence`，基础设施现成。
  - **Verifier 变辩护律师**：不判"38对不对"，而对证据链生成**反事实查询**（"若 Row12 为 NaN 代码会崩吗？""过滤条件加 Col4>0 答案还是 38 吗？"）。→ 对应老师1的 Differential Diagnosis（生成诊断性 query 区分候选）。
  - **Arbiter 裁"证据标准"**：第三方 arbiter/陪审团裁定证据链是否"排除合理怀疑"，而非投票选答案。
- **卖点**：Evidentiary Standard in LLM Agents——"程序正义（证据链完备）"比"结果正义（多投票）"更能解决 B 类漏检。

### 支柱 B — Provenance-Diverse Generation（多样性，退居支撑）

- 触发后才启用；用**逻辑路径变体**产真多样性：先 filter 后 groupby vs 反之、pandas vs SQL、不同 aggregation 假设。来源：老师2 TeLL 模板驱动 + 老师1 Hypothesis Space（候选=不同假设，不是不同答案）。
- **明确不做多模态**（TableDART 需截图+视觉模型，成本/infra 太高）。

---

## 3. 实验设计（杀伤力在 compute-matched）

- **主数据集**：**HiTab**（更干净、层级表有挑战）。**先跑通 HiTab 是第一优先级**（见 §5）。WTQ 作基线并报"歧义子集"。FinQA 辅、TabFact 作 verifier 专项。
- **核心对比（都在相同算力预算 compute-matched 下）**：
  - vs **Best-of-N / Self-Consistency**：证明"选择性触发"在**相同 FLOPs** 下准确率碾压全局多轨。
  - vs **全局 debate / always-verify**：证明省算力不掉分。
  - vs **传统 PRM**：证明"语义扰动/证据链"无需 step-level 标注就能达到/超越 PRM 的漏检召回。
- **消融**：Fast-path only / +扰动触发 / +证据裁决 / 完整 pipeline。
- **指标**：
  - Final Acc、PreVerify Acc、Net Gain、Damage Rate（沿用 `RESEARCH_DIRECTION.md` §6）
  - **False-Negative Recall**（漏检召回，触发核心指标）
  - **Accuracy-per-FLOP / Accuracy-per-\$**（compute-optimal 核心指标）
  - Oracle-Selector Gap
- **统计**：paired bootstrap CI + McNemar。

---

## 4. 前沿对齐（2026 顶会"黑话"）

1. **Test-Time Compute Scaling 深水区** → 我们做的是 **Compute-Optimal Agent Routing**：不增加（甚至减少）全局 FLOPs 前提下追平/超越全局多轨。
2. **Epistemic vs Aleatoric Uncertainty** → 杆A 捕捉的是**认识论不确定性**（模型不懂），要和**偶然不确定性**（gold 噪音/题目歧义，WTQ 已知问题）区分；用 semantic entropy 等 UQ 方法替代黑盒 predictor。
3. **Provenance & Grounding in Table Reasoning** → 代码生成掩盖了证据，我们主张 **Code-as-Provenance**，verifier 审数据流图而非最终输出。

---

## 5. 落地顺序（用户已定：先跑通 HiTab）

> **纪律**：不同数据集效果可能不同，先在 HiTab 上把地基跑通、拿到真实信号，再决定支柱1/2 的力度。所有头脑风暴先完整归档（见 §7），避免丢想法。

1. ✅ **多样性/可回收审计**（`scripts/candidate_diversity.py`，已完成）。
2. **P0 — HiTab adapter + pilot**（第一优先级）：
   - `DatasetAdapter` 抽象（`load_examples / load_table / evaluate`）。
   - HiTab 层级表 → flat DataFrame 转换（`kg.data` 网格，非普通 CSV，见 `HANDOFF.md` §4）。
   - 100 题 pilot，跑通现有 generator+verifier+finalize，建立 HiTab baseline。
3. **P1 — 在 HiTab 上做离线可分性分析**：B类漏检题 vs 正确题，在已有信号（答案能否在表定位、evidence 覆盖、代码复杂度、gen↔verifier 分歧）上是否可分 → 判断扰动触发方向的信号强度（跨数据集验证，不只看 WTQ）。
4. **P2 — 支柱1 原型**：语义扰动触发（先 prompt-based、零训练），在 HiTab pilot 上测 False-Negative Recall。
5. **P3 — 支柱2 原型**：证据链 + 反事实查询裁决，解耦 verdict/selection，测 Damage Rate 下降。
6. **P4 — 决定权离线重放**：现有 trace 上比较 {generator自选 / verifier / arbiter / jury / 加权投票}（近乎免费）。
7. **P5 — compute-matched 主实验**：vs Best-of-N / SC / PRM，报 Accuracy-per-FLOP。
8. **P6 — 跨数据集**：WTQ 基线 + 歧义子集、FinQA 辅、TabFact verifier 专项。

---

## 6. 风险与开放问题

- **升维 vs 可落地的平衡**：三位老师给了 30+ 想法（§7 完整归档），但一篇顶会 = 1 motivation + 1~2 机制 + 扎实实验。**警惕漂亮词汇（Belief Graph / Reasoning Ecology / 元认知模块）落不了地、难证伪**。本 proposal 已收敛到"False Consensus 动机 + 两支柱"。
- **无训练 infra 约束**：优先 prompt-based / 无训练方法（扰动、证据链、CP 都符合）；训练 PRM / MLP selector 谨慎，作为 future work。
- **Conformal Prediction 是统计外衣不是核心**：用来加覆盖率保证，别把论文押在它上面。
- **候选多样性 gate**（`RESEARCH_DIRECTION.md` 发现1/3）：离线重放必须 condition 在触发子群。
- **gold 噪音/歧义**：arbiter 修不了 gold 错；HiTab 更干净，WTQ 报歧义子集。
- **扰动触发的假设风险**：AST 突变/答案漂移是否真与"错误"相关，需 §5 P1 的可分性分析先验证——**可能某些数据集上信号弱**，这是最大不确定性。
- **成本**：扰动重跑、证据链裁决、多轨都加成本，必须用选择性触发对冲并报成本。

---

## 7. 完整头脑风暴档案（三位老师全部想法，一个不丢）

> 用户要求"先记录所有头脑风暴"。此处完整归档，标注**采纳状态**：★主线 / ☆备选 / ⚠需训练或高成本 / 💭远期。

### 7.1 老师1（`brainstorm.md`）——范式升维 + 21 方向

**核心主张**：从 Reasoning Generation 升维到 **Reasoning Control / Meta-Reasoning**；整个 agent 抽象为可学习 Policy：Information Acquisition(Trigger) → Hypothesis Exploration(Diversity) → Belief Arbitration(Selection) → Computation Allocation(Cascade) → Termination。**最后追问：这篇论文提出了什么新"世界观"？** → ★ 已采纳为 §0 世界观。

| # | 方向 | 潜力 | 采纳状态 |
|---|------|------|------|
| 1 | Reasoning Control | ★★★★★ | ★ 已作世界观 |
| 2 | Hypothesis Space Exploration（候选=假设非答案） | ★★★★★ | ★ 融入支柱B |
| 3 | Evidence Sufficiency Trigger（触发依据=证据是否充分,非一致性） | ★★★★★ | ★ 融入支柱1/2 |
| 4 | Meta Reasoning（agent 是 Reasoning Manager） | ★★★★★ | ★ 已作世界观 |
| 5 | Information Gain Termination（按信息增益停,非轮数） | ★★★★☆ | ☆ 停止策略备选 |
| 6 | Computation Allocation（算力花在哪,Pareto） | ★★★★☆ | ★ 融入 compute-optimal |
| 7 | Belief Revision（Belief Graph, 后验信念非投票） | ★★★★☆ | 💭 远期,落地难 |
| 8 | Expected Utility（Optimal Stopping,优化效用非准确率） | ★★★★☆ | ☆ 停止策略备选 |
| 9 | Program Repair（局部 patch 而非重生成） | ★★★★ | ☆ 与支柱B协同 |
| 10 | Differential Diagnosis（生成诊断query区分候选） | ★★★★ | ★ 融入支柱2反事实查询 |
| 11 | MCTS（program 是树,只展开 Filter 分支） | ★★ | 💭 远期 |
| 12 | Active Learning（Trigger≈选哪些样本值得标注） | ★★ | ☆ 触发方法论借鉴 |
| 13 | Scientific Discovery（verifier 像科学家,假设检验） | ★★ | 💭 叙事色彩 |
| 14 | Distributed Cognition（多 Judge 各看不同 evidence） | ★★ | ☆ 陪审团变体 |
| 15 | Bayesian Reasoning（Prior→Evidence→Posterior） | ★★ | 💭 远期 |
| 16 | Decision Authority（统一命名 Decision Policy；verdict/selection/termination × gen/verifier/arbiter/jury） | ★★ | ★ 融入支柱2 + P4 |
| 17 | Adaptive Exploration（简单题one-shot,难题tree-search） | ★★ | ★ 融入触发→深度 |
| 18 | Counterfactual Debate（verifier 主动构造反事实） | ★★ | ★ 融入支柱2 |
| 19 | Curriculum Debate（按难度不同 debate 策略） | ★ | ☆ 备选 |
| 20 | Reasoning Ecology（生态系统:explorer/critic/judge/...） | ★ | 💭 叙事,落地难 |

**跨领域速查表**（老师1）：Inference-time Scaling→Trigger/Cascade；MCTS→Debate展开；Active Learning→Trigger；医学→Diagnostic Query；科学发现→Debate；贝叶斯→候选融合；控制论→Meta Controller；金融→Optimal Stopping；信息论→Stop Policy；知识表示→Selection；软工→局部修复；多智能体→Jury。

### 7.2 老师2（`brainstorm2.md`）——三杆叠加的落地"大招"

**杆A 触发**：
- ★ Table-Critic 过程级 Judge（审 pandas 轨迹每步选列/过滤/聚合）→ 融入支柱2。
- ⚠ 训练 PRM 对中间结果打分 → future work（需标注）。
- ☆ 用"分歧程度"喂轻量分类器做 false-negative predictor → 被支柱1（扰动）部分替代。

**杆B 多样性**：
- ⚠ TableDART 多模态路由（文本+图像视图）→ **不做**（成本/infra）。
- ★ TeLL 模板驱动变体（先filter后聚合 / SQL vs pandas）→ 采纳为支柱B。
- ★ 对抗性多样性（verifier 主动提反例/反解）→ 融入支柱2。

**杆C 决定权**：
- ☆ CalibraEval 置信度校准 + 加权投票（缓解选择偏见）→ 决定权备选。
- ★ Conformal Prediction（候选集+覆盖率保证；辅助 arbiter / 决定是否触发）→ 采纳为统计外衣。
- ⚠ 训练轻量 MLP selector（基于 debate 特征学习选哪个）→ future work（需训练）。

**跨界**：医学二级分诊→级联；分布式共识（团队大小/发言顺序/轮数）→ debate 研究；法律陪审团（Panel-of-LLM,规模/多样性 vs 偏见）→ 决定权；A/B 测试→离线策略选择。

**老师2 优先推荐**：Conformal Prediction（杆C）+ Table-Critic 过程验证（杆A）。

### 7.3 老师3（`brainstorm3.md`）——升维的 3 个 Story + 手术刀

**诊断**：现有设计是"经典控制论/搜索框架在 LLM 上的工程映射"，缺 Scientific Insight；"多路采样+裁判投票"2023-24 已挖深。需引入跨学科认识论。

- **Story 1｜法学**：Provenance-Grounded Debate + 排除合理怀疑 → ★ 采纳为支柱2。
- **Story 2｜控制论/物理**：Semantic Perturbation Trigger（灵敏度分析,零训练） → ★★ 采纳为支柱1（主攻）。
- **Story 3｜认知科学**：Metacognitive Module（认知负荷 + 认识论不确定性 → 元认知引导算力分配） → ☆ 作为支柱1的理论包装/UQ 视角；完整"元认知模块"落地难，取其 UQ 内核。

**前沿对齐**：Test-Time Compute Scaling / Compute-Optimal Routing；Epistemic vs Aleatoric Uncertainty；Provenance & Grounding（Code-as-Provenance）。→ ★ 全部融入 §4。

**手术刀建议**（已采纳）：把发现4（漏检53-87%）作为全篇 Motivation（"False Consensus/虚假共识"）；三杆升级命名——Uncertainty-Aware Triggering / Provenance-Diverse Generation / Evidentiary Adjudication；Baseline 加 vs Best-of-N/SC（同算力）+ vs PRM。

---

## 8. 相关文件索引

| 文件 | 用途 |
|------|------|
| `docs/RESEARCH_DIRECTION.md` | 数据发现 + 三杆流水线（本 proposal 的事实基础） |
| `docs/brainstorm.md/2/3` | 三位老师原始建议（本 proposal §7 已归档） |
| `scripts/candidate_diversity.py` | 候选多样性/oracle 审计（False Consensus 数据来源） |
| `src/agent.py` (≈483–517) | 现有 finalize/选择规则（待解耦 verdict/selection） |
| `src/verifier.py` | 三层 verify（待升级为证据链裁决） |
| `src/evaluator.py` | 官方归一化匹配 |
| `PLAN.md` | 工程规划（含 run_python evidence 强契约、HiTab adapter 计划） |
| `HANDOFF.md` | 交接、既往伤害诊断、数据集结构速查 |
