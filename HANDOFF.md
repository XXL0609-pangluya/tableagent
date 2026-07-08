# TableAgent 项目交接文档（给下一对话窗口）

> 复制本段作为新会话的系统/用户提示词，让下一个 agent 快速接手。

---

## 0. 项目一句话

**TableAgent**：面向表格问答的 **Generator + Verifier Debate** 系统。Generator 用 function-calling + `run_python` 解题；独立 Verifier 在 `submit_answer` 后介入，多轮 debate 纠错；最终从 candidate pool 选答案。

**当前主目标**：不是单纯刷 WTQ 分数，而是研究 **如何控制 verify（触发、强度、选答案）**，写成可发表论文；WTQ 作基线，**HiTab 作重点新数据集**。

**工作目录**：`/Users/a1/Documents/TableAgent`

---

## 1. 当前代码状态（截至 2026-07-01）

### 分支

| 分支 | 状态 | 说明 |
|------|------|------|
| **`phase3g-v2plus-evidence`** | **当前工作分支（推荐继续用）** | commit `1fe5b7e`：checkpoint/resume + compute_recheck |
| `死马当活马医88.5%版` | 已实验、效果不佳，已回退 | Phase1 改动：last_verified、8轮debate、prompt人格强化 |
| `main` | 较旧 | Phase 3f |

### 未提交改动（本地有 diff）

- `src/data.py`：WTQ 路径改为 `datasets/WikiTableQuestions`
- `README.md`、`PLAN.md`、`docs/INTERFACES.md`：同步文档
- `datasets/`：新目录（WTQ + HiTab + FinQA + Table-Fact-Checking），**未 git add**

**smoke 已通过**：`python -m scripts.smoke` → PASS（新路径下 2831 题、200 表、评测对齐 100%）

### 核心文件

| 文件 | 职责 |
|------|------|
| `src/agent.py` | 主循环、debate、finalize candidate pool |
| `src/verifier.py` | 三层 verify：deterministic / LLM review / audit / compute_recheck |
| `src/data.py` | WTQ 加载（`DEFAULT_DATASET_ROOT`） |
| `src/schemas.py` | `Budget`: max_steps=20, max_verify_retries=6, debate_extra_steps=3 |
| `scripts/test_debate.py` | debate 测试 + **checkpoint/resume**（`*.partial.json`） |
| `prompts/AGENT.md` | Generator charter |
| `eval/disputed.json` | 10 道 disputed，评测时排除 |

### 环境变量（重要开关）

```bash
LLM_MODEL=...              # Generator
VERIFIER_MODEL=...         # Verifier（建议与 generator 不同厂/不同模型）
VERIFY_COUNT_CHECK=0       # 默认关
VERIFY_FOLLOWUP_DET=1      # 默认开：后续轮 deterministic 子集
VERIFY_COMPUTE_RECHECK=1   # 默认开：独立重算 tier
AGENT_DRIFT_GUARD=0        # 默认关
WTQ_DATASET_ROOT=...       # 可选，覆盖 WTQ 路径
```

---

## 2. 已完成工作与关键结果

### 2.1 WTQ 主跑批

- **1000 题 mine**（qwen3.6-35b-a3b + deepseek-v4-flash verifier）：**885/997 adjusted = 88.77%**
- 文件：`results/agent_qwen3.6-35b-a3b_mine_1000.json`
- 统一错题并集 **295 题**：`results/wrongset_union_mine_mine2_200_nondisputed.json`

### 2.2 50 题快筛（4 组跨厂商，固定 `wrongset_screen50_rows.json`）

| 组合 | 最终 | verify前首次submit | verify净效应 |
|------|------|-------------------|-------------|
| **glm-5.2 + claude-opus-4-8** | **19/50 (38%)** | 17/50 | **+2** |
| qwen3.7-max + claude-opus-4-8 | 16/50 (32%) | 14/50 | +2 |
| deepseek-v4-pro + gemini-2.5-pro-thinking | 13/50 (26%) | 14/50 | **-1** |
| qwen3.7-max + gemini-2.5-pro-thinking | 12/50 (24%) | 15/50 | **-3** |

**横向（50题）**：

- 四组全对：8 题
- 四组全错：**26 题**（硬骨头）
- 分析产物：`results/screen50_analysis.json`

### 2.3 纵向诊断（verify 伤害模式）

1. **首次对→最终错**（debate 改坏）：gemini verifier 组合明显，claude 几乎不伤
2. **first_candidate 回退**：后期 submit 已对，但 verify 仍 FAIL，finalize 回退到第一次提交（如 nt-11778: 36→38→38，gold=38，最终=36）
3. **verifier 口径深水区漏检**：36 错题约 58% 全程未拦；主因「可辩护就放行」
4. **compute_recheck 同坑**：verifier 与 solver 犯同样口径错误，11 题测试零提升

### 2.4 代码改动历史（phase3g）

- `followup_deterministic_issues` + `VERIFY_FOLLOWUP_DET`
- `compute_recheck` tier + `VERIFY_COMPUTE_RECHECK`
- `test_debate.py` checkpoint/resume
- `scripts/run_screen50_serial.sh` 串行跑 4 组

### 2.5 「死马当活马医88.5%版」实验（已放弃）

改动：finalize 改 `last_verified`、debate 8 轮、generator/verifier prompt 人格强化。

**用户反馈：效果不好，已回退到 phase3g。**

### 2.6 数据集目录重组（刚做）

```
datasets/
├── WikiTableQuestions/   # 从项目根移入
├── HiTab/
├── FinQA/
└── Table-Fact-Checking/  # TabFact，含原作者代码
```

`src/data.py` 默认路径已改；**尚未 commit**。

---

## 3. 核心经验（勿重犯）

### 架构层面

1. **finalize 策略是内伤点**：`passed[0]` / `first_candidate` 会在 debate 后丢掉更晚的正确答案
2. **更多 debate 轮次 ≠ 更好**：可能把对的改错；伤害案例随轮次增加
3. **verifier 净效应取决于检查器模型**：Claude 正向，Gemini 负向；换 generator 不够，要控 verify
4. **compute tier 不能解决口径歧义**：独立重算与 solver 共享偏见

### 数据层面

5. **WTQ gold 有噪音/歧义/不可达**：如 disputed、at most 字面 vs gold 反了、表内不可达答案
6. **TabFact 已饱和（SOTA ~96.6%）**：适合作 verifier 专项，不适合主创新战场
7. **HiTab 最适合做主实验**：难度够、标注相对规范、层级表有挑战；需写 hierarchical→flat 转换
8. **FinQA 适合作辅实验**：表+文本数值推理；需拼接 pre_text/table/post_text

### 论文方向（已达成共识）

- **主轴**：如何控制 verify（trigger / intensity / selector），不是单纯换模型
- **创新候选**：
  - Verify 作为 selective intervention（helpfulness predictor）
  - Debate intensity 自适应停止
  - Selector policy 离线重放（first/last/verified/vote）
  - 新指标：PreVerify-Acc、PostVerify-Acc、Net Gain、Damage Rate
- **模型配对**：Anthropic 工程报告已提，不算创新；作次要实验支撑即可
- **venue 目标**：ACL/EMNLP/NAACL；需要完整消融 + 跨数据集 + 负结果分析

---

## 4. 数据集结构速查（供 adapter 开发）

### WikiTableQuestions（已接入）

- 问题：`data/<split>.tsv`（id, utterance, context, targetValue）
- 表：`csv/.../*.csv` 或 `.tsv`
- 权威 gold：`tagged/<split>.tagged`（targetCanon）

### HiTab（优先接入）

- 样本：`data/{train,dev,test}_samples.jsonl`
- 表：`data/processed_input/tables.jsonl`（`table_id` 索引）
- 字段：question, answer (list), table_id, linked_cells, aggregation
- 表格式：`kg.data` 为 `[[{value:...},...],...]` 层级网格，**非普通 CSV**

### FinQA

- `dataset/{train,dev,test}.json`
- 字段：pre_text, post_text, table (2D list), qa{question, answer, program, steps}
- 需表+文本拼接；数值答案需容差评测

### TabFact (Table-Fact-Checking)

- `tokenized_data/full_cleaned.json`：表名 → [statements, labels(1/0), pos_tags, caption]
- 原始表：`data/all_csv/*.csv`（`#` 分隔）
- **任务不同**：entail/refute 二分类，非开放 QA

---

## 5. 下一步工作计划（优先级排序）

### P0 — 工程基础

- [ ] **commit** 数据集路径改动 + `datasets/`（或 `.gitignore` 大文件策略）
- [ ] 实现 **`src/datasets/` DatasetAdapter** 抽象：
  - `load_examples(split) -> list[Example]`
  - `load_table(example) -> TableContext`
  - `evaluate(predictions, targets) -> metrics`
- [ ] **HiTab adapter**：层级表 → flat DataFrame + 100 题 pilot

### P1 — 论文主实验（verify 控制）

- [ ] **Exp-A 辩论强度网格**：max_verify_retries × debate_extra_steps
- [ ] **Exp-B Selector 离线重放**：同一 trace 上比较 first/last/first_verified/last_verified/vote
- [ ] **Exp-C Verify 触发策略**：always / never / rule-trigger / predicted-trigger
- [ ] 报告 4 指标：Final Acc、PreVerify、Net Gain、Damage Rate
- [ ] 统计：paired bootstrap CI + McNemar

### P2 — 模型与 baseline（次要）

- [ ] Generator×Verifier 小矩阵（glm/qwen/deepseek × claude/gemini）
- [ ] 开源 baseline 用同底座重跑
- [ ] FinQA 辅实验；TabFact verifier 专项

### P3 — 全量与论文写作

- [ ] glm-5.2 + claude-opus-4-8 跑满 295 错题集（有 checkpoint）
- [ ] 26 题「四组全错」做可达性分类（可修/噪音/不可达）
- [ ] 可选：WTQ-clean 子集人工标注协议

---

## 6. 常用命令

```bash
cd /Users/a1/Documents/TableAgent
source .venv/bin/activate

# 基础检查
python -m scripts.smoke
python -m scripts.check_llm

# WTQ debate 测试（50题快筛）
LLM_MODEL="glm-5.2" VERIFIER_MODEL="claude-opus-4-8" \
  python3 -u scripts/test_debate.py \
  --from-wrong results/wrongset_screen50_rows.json \
  --max-steps 20 \
  --save results/screen50_glm52_claudeopus48.json

# 295 全量错题集（支持断点续跑）
LLM_MODEL="glm-5.2" VERIFIER_MODEL="claude-opus-4-8" \
  python3 -u scripts/test_debate.py \
  --from-wrong results/wrongset_union_rows_for_rerun.json \
  --max-steps 20 \
  --save results/debate_wrongset295_glm_claude.json
```

---

## 7. 关键结果文件索引

| 文件 | 内容 |
|------|------|
| `results/screen50_analysis.json` | 50题纵向/横向分析 |
| `results/agent_qwen3.6-35b-a3b_mine_1000.json` | 1000题主跑批 |
| `results/wrongset_union_mine_mine2_200_nondisputed.json` | 295错题并集 |
| `results/wrongset_screen50_rows.json` | 50题快筛子集 |
| `results/debate_diag52_merged_v2plus_opt1_summary.json` | 52题诊断摘要 |
| `eval/disputed.json` | 10道 disputed |

---

## 8. 给下一 agent 的明确指令

1. **继续在 `phase3g-v2plus-evidence` 分支工作**，不要默认用「死马当活马医」分支
2. **论文主轴是 verify 控制实验**，不是继续堆 verifier 规则或盲目加 debate 轮次
3. **优先接 HiTab**（DatasetAdapter + 表转换），跑 100 题 pilot 再定主战场
4. 改动前先读 `docs/INTERFACES.md` 和 `PLAN.md`
5. 任何 finalize/selector 改动都要测 **Damage Rate**，不能只看最终准确率
6. 未提交的 `datasets/` 路径改动需要用户确认后 commit

---

*交接文档生成时间：2026-07-01。上一会话主要讨论了：screen50 分析、verify 伤害机制、死马当活马医实验（失败回退）、论文方向、多数据集选型、datasets 目录迁移。*
