# 实验落地设计（具体版）：Compute-Optimal Reasoning Control for Table QA

> **本文档的目的**：把 `PAPER_PROPOSAL.md` 的两支柱叙事，写成一份**做完就能回答我们设计前所有目标和疑惑**的具体实验方案。每个阶段都写清：**用什么模型、什么数据、生成器/检查器用哪一版、什么时候裁决怎么裁决、怎么生成、要不要多轮投票、语义扰动具体怎么做**。
> **生成时间**：2026-07-08。依据见 §2 的实测锚点。

---

## 0. 我们要回答的目标和疑惑（实验的"为什么"）

做完这套实验，必须能回答以下每一个问题：

1. **"虚假共识"是不是真问题？** 检查器自信放行的题里，到底有多少是错的？这个比例在干净标尺下稳不稳？
2. **漏检是"同源盲区"吗？换不同厂商/更强的检查器能不能治？** 我们一直听说"跨厂更好"但没干净测过——而我们 65% 漏检本就是跨厂测的。同源 vs 跨厂 vs 强跨厂，漏检率到底降不降？
3. **算力到底浪费在哪？** 简单题被反复 verify/debate 了吗？硬题反而没人管？
4. **扰动能不能揪出漏检？** 对输入做点不影响意思的扰动，错的题会不会"露馅"（答案/代码漂移）？比"检查器说没问题"靠谱多少？
5. **算力花在该花的地方，能不能在同算力下打过全局多轨？**（论文主结果）
6. **裁决该不该和选择解耦？** 现在检查器又判又选（`first_verified`），换成第三方审证据链，能不能少改坏、多救回？
7. **多样性是不是裁决的前置？** 没造出别的候选，再好的裁判也没得选——这个 gate 有多硬？
8. **方法是不是只在 HiTab 上灵？** 换 WTQ/FinQA 还成立吗？

**实验纪律（贯穿全程）**：
- 所有准确率**同时报 strict 和 canonical**（`src/hitab_eval_canonical.py`）；触发/漏检类指标的分母**一律用 canonical**，否则被格式噪声地板锁死。
- 每个 run 落 `run_meta.txt`（RUN_ID/目的/输入 ids/prompt 版本/git commit）+ 完整 trace，**永不删除、永不覆盖**（`HITAB_TRACE_DIR`/`HITAB_TRACE_TAG`）。
- 每个 run 落 `compute.json`（每题 LLM 调用数 + prompt/completion tokens），作为算力口径。
- 统计：主指标给 paired bootstrap 95% CI；两系统比较用 McNemar。

---

## 1. 模型与角色配置（固定下来，别乱换）

> 端点：`LLM_BASE_URL=https://llm-center.ali.modelbest.cn/llm/v1`（OpenAI 兼容）。所有角色都是"调用次数可计、可复现"的。
>
> **重要更正（来自实测）**：我们之前在 test 上测出"false consensus 占错题 64.9%"的那两次 run，**gen=qwen3.6-35b-a3b、verifier=deepseek-v4-flash，本来就是跨厂的**。也就是说"虚假共识"是在跨厂检查器下测出来的——它**不是同源盲区**问题，换厂商的检查器治不了它。这反而强化了"需要扰动触发这种不同机制"的动机。代码里 `load_verifier_config` 默认就让 verifier 用和 solver 不同的模型（solver=qwen → verifier=deepseek），所以"跨厂"一直是我们的事实默认，不是新东西。

| 角色 | 用哪个模型 | 为什么 | 什么时候用 |
|---|---|---|---|
| **Generator（主生成器）** | `qwen3.6-35b-a3b`（`LLM_MODEL`） | 主力，写 pandas + `submit_answer` 带 `evidence` | 所有实验的"基础答案"都由它产 |
| **Verifier（检查器，默认跨厂）** | **`deepseek-v4-flash`**（`VERIFIER_MODEL`，与 gen 不同厂商） | 这是我们的**事实默认组合**，性能已很强；WikiQA 上试过 CLM5.2+Claude-opus-4.8 提升也不大 | E0 起全程默认用这个 |
| **Verifier（同源对照）** | `qwen3.6-35b-a3b`（= gen） | **我们从没做过同源 vs 跨厂对照**，跨厂更好只是传闻；E0b 专门验 | 仅 E0b 对照实验 |
| **Verifier（强跨厂对照，可选）** | Claude-opus-4.8 / Gemini 之一 | 看"换更强检查器"能不能压低漏检（预期：压不多少，因为漏检跨厂仍在） | E0b / E2 可选对照 |
| **Perturb Generator（扰动重跑器）** | 同 Generator `qwen3.6-35b-a3b`，**温度 0、固定种子** | 扰动实验要控制变量：只让输入变，不让采样抖动 | E2 |
| **Arbiter（第三方裁决）** | **跨厂强模型**（Claude / GPT-4 级） | 第三方审证据不审答案 | E4' |
| **Jury（陪审团，可选）** | 3 个便宜异构模型投票 | 对冲单一 verifier 偏见（PoLL 思路） | E4 的一个选择器变体 |

**关于"跨厂更好"的诚实立场**：我们不假设跨厂能缓解漏检——**E0b 会直接测**。如果 E0b 显示同源和跨厂的 false-consensus 率差不多（很可能，因为 65% 本就是跨厂测的），论文动机就从"同源盲区"改为"**虚假共识是跨厂商现象，换检查器治不了，需要扰动触发**"——这其实是个更干净、更可证伪的故事。

**生成方式（统一）**：函数调用 agent，工具集 `{inspect_table, search_columns, search_cells, run_python, submit_answer}`；`submit_answer` 强制带 `evidence`（依赖哪些 row/col + 聚合）——这是支柱2 "Code-as-Provenance" 的基础设施，全程保留。

**多轮投票策略（统一规则，谁用谁说明）**：
- **Self-Consistency (SC)**：同问题采样 N 次（温度 0.7，不同种子），按 canonical 语义聚类投票，最大簇为答案。
- **Best-of-N (BoN)**：同上采样 N 次，用 verifier 打分选最优。
- **我们的 pipeline 不全局投票**：只对"被触发"的题做多样性（见 E3），简单题 one-shot 直接过——这正是省算力的核心。

---

## 2. 数据集选择（关键决策：换一版没调过的）

> **用户决定**：test（1584）是我们调试 prompt 用的，**不能当最终评测集**（过拟合风险）。重新选版方案如下。

| 用途 | 数据集 | 数量 | 说明 |
|---|---|---|---|
| **开发/调参/阈值标定** | HiTab **dev** | 1671 | 没调过；用来调触发阈值 τ、δ、选择器超参 |
| **触发器训练（如需学习式触发）** | HiTab **train** | 7417 | 只在"学习式触发"备选时用；默认 prompt-based 不用 |
| **最终评测（论文报的数）** | **HiTab dev 的一个固定子集 OR 重新跑全 dev** | 1671 | **全新、未调过**，作为主评测集 |
| **外部效度** | WTQ（歧义子集）/ FinQA / TabFact | — | E6 |

**落地**：
- 把"调过的 test（1584）"降级为**只读的开发参考集**，所有 prompt 迭代以它为信号，但**论文数字一律不报它**。
- 主评测改用 **dev（1671）**：先在 dev 上用当前对齐版 prompt 跑一遍 baseline（E0），从此 dev 就是"干净评测集"。
- 若担心 dev 也被未来调过：可从 train 抽 1000 题做 holdout（但 train 可能用于触发器训练，需隔离）。**建议：dev 做主评测，test 做开发，train 留给可选的学习式触发。**

---

## 3. 实测锚点（已建立，复现脚本见 §10）

这些是方案设计的依据，不是估的：

- **干净标尺**：test 上 strict 89.4% → canonical 92.5%；格式噪声 ≈49 题（占错题 29%），真·epistemic 错 ≈119 题。
- **false consensus**：test 上未触发 debate 的错题 109/168 = **64.9%**（B 类漏检）。debate 触发率 18.7%。**⚠️ 此数字的 gen=qwen3.6-35b、verifier=deepseek-v4-flash（跨厂）——即漏检在跨厂检查器下仍占 65%，不是同源问题。**
- **决定权空间薄**：全体多候选仅 6.3%，纯选择天花板 +1.4pt；debate 子群 +7.4pt。→ **支柱B（多样性）是支柱2（裁决）的前置。**

---

## 4. 阶段总览（6 个阶段，按依赖排序）

```
阶段1 E0  基线锁定 + 干净标尺上 dev          （打地基）
       E0b 同源 vs 跨厂 vs 强跨厂 对照       （专验"漏检是不是同源问题"）
阶段2 E1  可分性分析（离线，零成本）          （验证触发有没有信号）
阶段3 E2  语义扰动触发（支柱1，主攻）         （核心方法）
阶段4 E3  多样性生成（支柱B）                 （抬天花板）
阶段5 E4  证据裁决 + 决定权解耦（支柱2）      （第二条腿）
阶段6 E5  compute-matched 主实验              （论文主结果）
        E6  跨数据集外部效度                   （可信度）
```

---

## 5. 每阶段具体方案

### 阶段 1 — E0：基线锁定（打地基）

**大白话**：用我们对齐好的 prompt，在新数据集（dev）上跑一遍现成的 agent，把"现在到底多准、漏检多严重、算力花多少"这些 baseline 数字钉死。后面所有改进都跟它比。

**这一轮的配置**：
- **数据**：HiTab **dev（1671 题）**，全新未调。
- **模型**：Generator = `qwen3.6-35b-a3b`，Verifier = `deepseek-v4-flash`（**跨厂，我们的事实默认组合**）。
- **生成器 prompt**：`prompts/skills/hitab.md` 当前版（含 THE ONE RULE + 修正符号）。
- **检查器 prompt**：`src/verifier.py` 的 `_HITAB_VERIFIER_SUFFIX` 当前版（已对齐）。
- **生成方式**：one-shot，单轨迹，温度 0。
- **要不要多轮投票**：**不要**。这就是"裸 agent"基线。
- **裁决**：现状 `first_verified`（检查器点头就选第一个），不引入第三方。
- **算力记录**：每题调用数 + tokens。

**步骤**：
1. `run_hitab_pilot.py --split dev --n 1671`，落 preds + trace + compute.json。
2. `score_hitab_canonical.py` 出 strict/canonical；`analyze_hitab_errors.py` 出错误分桶。
3. **建新脚本** `scripts/false_consensus_report.py`：从 trace 出 2×2（debate/未 debate × 对/错）+ FNR 基线 + 候选多样性 + oracle gap。

**产出**：baseline 表（Final/PreVerify/NetGain/Damage/TriggerRate/FNR/OracleGap，strict+canonical）+ compute 基线。

**判定门**：canonical baseline ≥ 90% 且 false-consensus ≥ 50% 错题 → 继续（test 上跨厂已满足 64.9%，dev 预期相近）。

---

### 阶段 1b — E0b：同源 vs 跨厂 vs 强跨厂 对照（专验"虚假共识是不是同源问题"）

**大白话**：我们一直听别人说"生成器和检查器用不同厂商的模型效果更好"，但我们**从没自己做过对照实验**——而且我们测出 65% 漏检的那次本来就用的是跨厂（qwen+deepseek）。所以这轮专门做一次干净对照：同一批题、同一版 prompt，只换检查器——分别用①同源（qwen 检查 qwen）②跨厂（deepseek，我们的默认）③强跨厂（Claude-opus-4.8，WikiQA 上最好的组合），看"漏检率"到底降不降。**预期：降不了多少**——如果真降不了，就坐实了"虚假共识是跨厂商现象，换检查器治不了，得靠扰动触发"，论文动机更干净。

**这一轮的配置**：
- **数据**：HiTab **dev 的一个固定子集（如 500 题）**，三组用**同一批题**（配对对照，McNemar 才有效）。
- **生成器**：三组都用 `qwen3.6-35b-a3b`，**同 prompt、同温度 0、同种子**——只让检查器变，控制变量。
- **检查器**三档：
  - ① 同源：`qwen3.6-35b-a3b`（设 `VERIFIER_MODEL=qwen3.6-35b-a3b`）
  - ② 跨厂默认：`deepseek-v4-flash`
  - ③ 强跨厂：`Claude-opus-4.8`（WikiQA 最佳组合之一）
- **生成器/检查器 prompt**：同 E0。
- **生成方式**：one-shot 单轨迹，不投票。
- **裁决**：三组都用现状 `first_verified`，不引入第三方（这轮只看检查器差异）。
- **算力记录**：三组分别记。

**步骤**：
1. 固定 dev 子集 ids（落 `dev_subset500_ids.json`，可复现）。
2. 三次 `run_hitab_pilot.py --ids-file dev_subset500_ids.json`，仅 `VERIFIER_MODEL` 不同。
3. `false_consensus_report.py` 出每组的 2×2 + FNR + Final Acc。

**产出**：三组的 false-consensus 率 / FNR / Final Acc 对照表（配对，McNemar）。

**判定门与分支**：
- 若三组 false-consensus 率都 ≥ 50%（预期）→ **"换检查器治不了漏检"成立**，论文动机改为"跨厂商虚假共识现象"，扰动触发是必要机制，进 E1/E2。
- 若强跨厂把漏检压到 < 30% → "换更强检查器"就能解决一大半，支柱1（扰动）的边际价值下降，需重估主攻；但注意 WikiQA 上 CLM5.2+Claude-opus 提升也不大，这个分支概率较低。
- 这轮**本身就是一个可发表的小发现**（同源 vs 跨厂对漏检的影响，目前文献基本没人干净测过）。

---

### 阶段 2 — E1：可分性分析（离线零成本，论文前置验证）

**大白话**：先不花重跑的钱。就拿 E0 的 trace 里能免费拿到的信号（代码有多复杂、证据覆盖全不全、生成器和检查器吵没吵、答案能不能在表里定位到），看看"未触发 debate 却错了"的漏检题，能不能靠这些信号从"未触发却对了"的题里挑出来。能分得开，扰动触发才值得做；分不开，得换路子。

**这一轮的配置**：
- **数据**：E0 的 dev trace（**不重跑，纯离线分析**）。
- **模型**：无新调用。
- **生成器/检查器**：不涉及。
- **裁决**：不涉及。

**信号（全部从现有 trace 提取）**：
1. **evidence 覆盖度**：`submit_answer.evidence` 引用的 row/col 数 vs 表规模。
2. **代码复杂度**：`run_python` 步数、是否有 filter/groupby/merge、AST 节点数。
3. **gen↔verifier 分歧**：verify 事件的打回轮数、concern 文本长度。
4. **答案可定位性**：答案值能否在表单元格命中（直读 vs 计算）。
5. **工具轨迹**：search 命中数、inspect 次数、总步数。

**方法**：在**未触发 debate 子群**内，以"canonical-错"为正标签，每个信号算单变量 AUC；再合逻辑回归/浅树（5 折 CV）看联合 AUC。

**产出**：单信号 AUC 表 + 联合 AUC + 固定 Trigger Precision 下的 FNR 曲线。

**判定门**：**联合 AUC ≥ 0.70** → 触发有信号，进 E2；< 0.60 → 触发信号弱，重估主攻方向（这是 proposal §6 标的最大不确定性，先验掉它）。0.60–0.70：进 E2 但降低预期，并行试"证据充分性触发"备选。

---

### 阶段 3 — E2：语义扰动触发（支柱1，主攻）

**大白话**：对一道题，把问题或表格做点**不影响答案**的小改动（比如 `2023年` 写成 `2023 年`、千分位去掉逗号、把无关的行打乱顺序、列名换个同义说法），让生成器用完全一样的逻辑重跑几遍。**对的题稳如泰山（怎么改答案都不变），错的题往往露馅（一改答案就漂移、代码逻辑也变了）**。漂移的题 = 脆弱 = 高认识论不确定 = 该深究。不漂移的 = 可信 = 跳过 debate 省算力。这是论文的主招。

**这一轮的配置**：
- **数据**：HiTab **dev（1671）**。阈值 τ/δ 在 dev 的**子集（如 500 题）**上标定，剩下做评测，避免过拟合 dev。
- **模型**：
  - 主生成器 = `qwen3.6-35b-a3b`。
  - **扰动重跑器 = 同 `qwen3.6-35b-a3b`，温度 0、固定种子**（控制变量：只让输入变，不让采样抖动）。
  - 检查器：**两个对照组**——(a) 同源 qwen；(b) 跨厂（Claude/Gemini），看去相关能否缓解漏检。
- **生成器 prompt**：同 E0（THE ONE RULE 版）。
- **检查器 prompt**：同 E0。
- **生成方式**：每题扰动 K 次（K=3 起步），每次独立 one-shot 单轨迹。
- **多轮投票**：扰动重跑**不投票**，只看"漂移与否"（二值触发信号）；漂移后才进 E3 的多样性。

**语义扰动具体怎么做（算子清单，全部"保义"）**：
1. **问句表层**：`2023年`↔`2023 年`、`what is`↔`what's`、同义词替换（"declined"→"decreased"）、语序微调（不改实体/数值/比较关系）。
2. **表格表层**：千分位 `12,467`↔`12467`、unicode 减号 `−`↔`-`、`%` 保留/去除一致化、空白规范化。
3. **无关行乱序**：打乱与问题无关的行顺序（答案不变）。
4. **无关列删除**：删掉问题没引用的列（答案不变）。
5. **同义列名替换**：`custody | number`↔`custody | count`（在 schema 里映射，不改数据）。
6. **数值探针（关键，且和 E4' 共享基础设施）**：对一个**不影响答案的**单元格 ±1 微扰，看代码是否还输出原答案；对**影响答案的**单元格微扰，看输出是否按预期变（这是反事实，归 E4' 用）。

**扰动实现工程**：建 `scripts/perturb_hitab.py`，输入 (example, K) 输出 K 个扰动副本（扰动表 JSON + 扰动问句）；agent runner 加 `--perturb K` 模式，对每题跑 1 原始 + K 扰动。

**触发判据**：
- **答案漂移**：K 次扰动答案的 canonical-不一致率 > τ（用 canonical 判，格式噪声不会漂移→正确不触发）。
- **代码 AST 突变**：`run_python` 代码的 AST 编辑距离 > δ（用 `ast.dump` + `difflib`）。
- 任一满足 → 触发深究（进 E3）。

**对照基线**：always-trigger / never / 规则触发（"含 div/percent 就触发"）/ 分歧触发（只用 verify 打回轮数）。

**产出**：FNR、Trigger Precision、Accuracy-per-Call、Drift-Error AUC；扫 τ 画 FNR-vs-算力 曲线。

**判定门**：在**相同触发预算**下，扰动触发的 FNR 显著 > 规则/分歧触发（McNemar p<0.05）→ 支柱1 成立。

**借鉴的新思路（落实点）**：semantic entropy（把 K 次扰动答案的熵当 UQ 量化，替代黑盒 predictor，老师3 Story 2）；Conformal Prediction 给 τ 一个覆盖率保证（老师2 杆C，统计外衣）。

---

### 阶段 4 — E3：多样性生成（支柱B，抬天花板）

**大白话**：上一阶段揪出了"该深究的题"。但这些题现在往往只有一个候选答案，再厉害的裁判也没得选。所以先给这些题**多造几个"想得不一样"的候选**——不是简单多采样几个一样的答案，而是真的换推理路径（先 filter 再聚合 vs 反过来、pandas vs SQL、换不同的聚合假设）。把"候选池里有对的"这个天花板抬起来，下一阶段裁决才有发挥空间。

**这一轮的配置**：
- **数据**：E2 在 dev 上**触发的那批题**（不全局做，省算力——这是 §3 实测 6.3% 多样性的教训）。
- **模型**：Generator = `qwen3.6-35b-a3b`，**多温度多路径**。
- **生成器 prompt**：同 E0 + **新增 `prompts/skills/diversity.md`**（待建）：指示"换逻辑路径"，给 TeLL 模板（先 filter 后 groupby / 反之 / SQL 写法 / 不同 aggregation 假设）。
- **检查器**：不涉及（这一步只造候选）。
- **生成方式**：每触发题产 **N=4 个候选**——2 个逻辑路径变体（温度 0，prompt 模板驱动）+ 2 个采样变体（温度 0.7，不同种子，作对照）。
- **多轮投票**：**这一步不投票**，只把 N 个候选都收进池子；投票/选择留到 E4。

**借鉴的新思路（落实点）**：Hypothesis Space Exploration（候选=不同假设，不是不同答案，老师1 方向2）；TeLL 模板驱动变体（老师2 杆B）；对抗性多样性（verifier 主动提反例）→ 反例归到 E4' 的反事实查询。

**产出**：触发子群的 oracle 天花板（E0 的 +7.4pt → 后？）、候选语义多样性（去重后候选数）、每题新增算力。

**判定门**：触发子群 oracle 天花板净增 **≥ +5pt** 且算力可接受 → 进 E4；< +3pt → 加强逻辑变体或弱化支柱2 叙事。

---

### 阶段 5 — E4：证据裁决 + 决定权解耦（支柱2）

**大白话**：现在候选池有了好几个答案，谁来拍板？现在是一段写死的代码"检查器先点头的那个"（`first_verified`），这把"判对错"和"选哪个"混在一起，容易改坏。我们试几种种拍板方式：让生成器自己选、让检查器选、让第三方裁判审、让几个便宜模型投票。重点试**第三方裁判审"证据链够不够硬"而不是"答案对不对"**——它看的是"你的推理证据排不排得掉合理怀疑"，更能抓住检查器和生成器一起看不见的盲区。

**这一轮分两小步：E4（离线重放，便宜）→ E4'（在线证据裁决，贵但关键）。**

#### E4 — 决定权离线重放（近免费）
**大白话**：E3 已经产了候选池。我们不重跑生成，只在池子上换不同的"选哪个"规则，看哪种最接近"完美裁判"。

- **数据/模型**：用 E3 的候选池，**不调用新模型**（除第三方 arbiter 变体）。
- **选择器变体**：`{generator 自选 / verifier / first_verified(现状) / verifier 置信度加权投票 / Panel-of-LLM(3 异构投票) / 各种规则}`。
- **裁决**：每个选择器独立跑，算 Final Acc / Net Gain / **Damage Rate** / Oracle Gap（触发子群）。
- **判定门**：存在某解耦选择器把 Oracle Gap 相对现状**砍半** → 决定权维度值钱。

#### E4' — 证据裁决 + 反事实查询（在线，支柱2 核心）
**大白话**：第三方裁判不投票选答案，而是当"辩护律师"：对着证据链提反事实问题——"如果那个单元格是空的，你的代码会崩吗？""过滤条件加上大于 0，答案还是 38 吗？"。这些反事实能区分"真懂"和"蒙对"。裁判最后判"证据是否排除合理怀疑"，不判"38 对不对"。

- **数据**：E3 触发子群的候选 + 证据链。
- **模型**：Arbiter = **跨厂强模型**（Claude/GPT-4 级）。
- **反事实查询**：复用 E2 的数值探针基础设施——对证据链引用的单元格做 ±1 微扰，问 arbiter "答案该不该变、变了没有"。
- **裁决标准**：arbiter 输出"证据充分/不充分"，不充分 → 打回重生成（限 1 轮，避免无限 debate）。
- **要不要多轮 debate**：**限 1 轮反事实 + 1 轮重生成**（proposal §6 教训：更多轮 ≠ 更好）。
- **产出**：B 类漏检召回、Damage Rate 下降、Final Acc（strict+canonical）。
- **判定门**：相比 E4 最优纯选择器，证据裁决在**同算力**下 Final Acc 再提升且 Damage 不升 → 支柱2 成立。

**借鉴的新思路（落实点）**：Provenance-Grounded Debate + 排除合理怀疑（老师3 Story 1，法学）；Differential Diagnosis（生成诊断性 query 区分候选，老师1 方向10）；Code-as-Provenance（代码即证据，老师3 前沿对齐）。

---

### 阶段 6 — E5：compute-matched 主实验（论文主结果）+ E6 外部效度

#### E5 — 主实验
**大白话**：把前面的好东西串成一条流水线（扰动触发 → 多样性 → 证据裁决），跟"暴力全局多轨采样"和"全局 debate"比，**条件是大家花的算力一样**。如果在同算力下我们更准，论文就成了。画一张"算力-准确率"曲线，我们的线在上面。

- **数据**：HiTab dev（1671）。
- **模型**：完整角色配置（gen qwen + verifier 跨厂 + arbiter 强模型）。
- **对照（全部 compute-matched，按调用数/tokens 配平）**：
  1. **Best-of-N / Self-Consistency**：全局多轨，N 调到和 pipeline 总调用数持平。
  2. **always-verify / 全局 debate**：证明省算力不掉分。
  3. **传统 ORM/PRM**：证明扰动+证据链无需 step 标注就追平漏检召回。
- **消融**：fast-path only / +触发 / +触发+多样性 / +全pipeline。
- **生成/投票**：pipeline 只对触发题做多轨（E3），简单题 one-shot；BoN/SC 是全局 N 轨。
- **产出**：Accuracy-per-Call、Accuracy-per-1k-tokens、Final Acc、FNR、Damage；**cost-accuracy Pareto 曲线（论文主图）**。
- **判定门**：在匹配 BoN 的算力点上，pipeline 的 Final Acc 显著更高（paired bootstrap CI 不含 0）→ 主结果成立。

#### E6 — 跨数据集外部效度
**大白话**：别让人说我们只在 HiTab 上灵。换 WTQ（但 WTQ 金标有噪音，要报"歧义子集"）、FinQA（数值推理）、TabFact（裁决专项，二分类信号更纯）。

- **WTQ**：报整体 + 歧义子集；诚实说明在"脏"数据上的退化。
- **FinQA**：数值推理辅证。
- **TabFact**：verifier/裁决专项。
- **判定门**：至少 HiTab + 1 个数据集主结论一致 → 外部效度达标。

---

## 6. 里程碑与决策节点

| 里程碑 | 实验 | 成本 | 关键交付 | 决策 |
|---|---|---|---|---|
| M1 | E0 | 中（dev 全跑一次） | dev baseline 表 | 数字稳不稳 |
| M1b | E0b | 中（同批 500 题跑 3 个检查器） | 同源/跨厂/强跨厂漏检对照 | 漏检是不是同源问题（决定动机叙事） |
| M2 | E1 | 低（纯离线） | 可分性 AUC | 触发有没有信号 |
| M3 | E2 | 高（扰动重跑） | FNR-算力曲线 | 支柱1 成不成立 |
| M4 | E3 | 中（触发子群多轨） | oracle 天花板提升 | 支柱B 有没有用 |
| M5 | E4/E4' | 中（离线+arbiter） | 选择器对比 + 证据裁决增益 | 支柱2 成不成立 |
| M6 | E5 | 高（多基线配平） | Pareto 主图 | 论文主结果 |
| M7 | E6 | 高 | 跨数据集表 | 外部效度 |

**论文 MVP**：M1+M1b+M2+M3+M6（触发 + compute-matched + 漏检跨厂商现象）已支撑主线；M4/M5 加强；M7 外部效度。

---

## 7. 风险与缓解

| 风险 | 触发条件 | 缓解 |
|---|---|---|
| 触发信号弱 | E1 AUC<0.6 | 换扰动算子；改用 semantic entropy；主攻调到"证据充分性触发" |
| 格式噪声污染 FNR | 用 strict 分母 | 已备 canonical 评估器；FNR 分母强制 canonical |
| 天花板太低 | E3 oracle 增益<+5pt | 加强逻辑路径变体；否则弱化支柱2 叙事 |
| 真错人群偏小 | dev 真错可能<100 | 扩到 train holdout 或并 dev+test（test 作开发参考） |
| 算力口径不公 | compute-matched 争议 | 统一记 calls+tokens；报 Pareto 而非单点 |
| gold 噪音 | WTQ | 主战场 HiTab；WTQ 报歧义子集 |
| 扰动重跑贵 | E2/E5 成本 | 选择性触发对冲；只对可疑题扰动；报成本 |
| dev 也被调过 | 阈值标定在 dev | dev 划子集标定/评测；或 train holdout |

---

## 8. 借鉴的新思路 → 具体落实点（一张表对清）

| 思路 | 来源 | 落在哪个实验 | 具体怎么做 |
|---|---|---|---|
| Reasoning Control / Meta-Reasoning | 老师1 | 全局世界观 | 不研究"推更好"，研究"管推理" |
| Hypothesis Space Exploration（候选=假设） | 老师1 | E3 | 逻辑路径变体，非多采样 |
| Differential Diagnosis（诊断性 query） | 老师1 | E4' | 反事实查询区分候选 |
| Decision Authority（verdict/selection 解耦） | 老师1 | E4 | 选择器矩阵实验 |
| Table-Critic 过程级 Judge | 老师2 | E4' | arbiter 审证据链/数据流 |
| TeLL 模板驱动变体 | 老师2 | E3 | 先filter后groupby / SQL vs pandas |
| Conformal Prediction | 老师2 | E2 | 给触发阈值 τ 覆盖率保证 |
| Semantic Perturbation（灵敏度分析） | 老师3 | E2 | 保义扰动测漂移 |
| Provenance-Grounded + 排除合理怀疑 | 老师3 | E4' | arbiter 审证据标准不投票 |
| Metacognitive / UQ | 老师3 | E2 | semantic entropy 量化认识论不确定 |
| PoLL（陪审团） | 老师2 | E4 | 3 异构模型投票选择器 |
| Compute-Optimal Routing | 前沿对齐 | E5 | 选择性触发，同算力超全局多轨 |

---

## 9. 立即下一步

1. **建 `scripts/false_consensus_report.py`**（E0 收尾）：固化本轮验证过的 2×2 + FNR + 候选多样性 + oracle gap 逻辑。
2. **跑 E0 在 dev 上**：用对齐版 prompt 跑 dev 1671，拿干净 baseline。
3. **建 `scripts/perturb_hitab.py` + 适配 `candidate_diversity.py` 读 HiTab trace**（E1/E2 基础设施）。
4. **跑 E1 可分性分析**（纯离线）→ 出联合 AUC → 决定支柱1 放行。

---

## 10. 相关脚本/文件索引

| 文件 | 用途 | 状态 |
|---|---|---|
| `scripts/audit_hitab_gold.py` | 金标格式审计（"逐字公式输出"证据） | ✅ |
| `src/hitab_eval_canonical.py` | 格式无关忠实评估器 | ✅ |
| `scripts/score_hitab_canonical.py` | strict vs canonical 双报 | ✅ |
| `scripts/analyze_hitab_errors.py` | 错误分桶 | ✅ |
| `scripts/run_hitab_pilot.py` | HiTab runner（trace/skill/ids） | ✅ |
| `prompts/skills/hitab.md` | 生成器 prompt（已对齐 THE ONE RULE） | ✅ |
| `src/verifier.py` suffix | 检查器 prompt（已对齐） | ✅ |
| `scripts/false_consensus_report.py` | trace → 2×2 + FNR + oracle | ⬜ 待建 |
| `scripts/perturb_hitab.py` | 语义扰动算子 + 扰动副本生成 | ⬜ 待建 |
| `prompts/skills/diversity.md` | 多样性生成 prompt（TeLL 模板） | ⬜ 待建 |
| `scripts/candidate_diversity.py` | 候选多样性审计（适配 HiTab trace） | 🔧 待适配 |
