
# TableAgent Research Brainstorm Notebook

> **目标：**突破"Verifier + Debate + Selection"的传统思路，从AI推理、认知科学、控制理论、医学、经济学等多个领域寻找新的Table QA研究方向。
>
> **原则：**不拘泥于现有Agent框架，关注"为什么这样做"而不仅是"怎么实现"。

---

# 一、Reasoning Control（⭐⭐⭐ 推荐指数：★★★★★）

## 核心思想

目前几乎所有Table QA Agent研究都在研究：

> 如何生成更好的答案（Better Reasoning）。

而可以进一步抽象为：

> 如何更好地控制推理过程（Reasoning Control）。

也就是说，研究对象从

```
Reasoning
```

提升为

```
Meta Reasoning
```

Agent不只是回答问题，而是不断决定：

```
是否继续思考？
是否值得搜索？
应该探索哪些方向？
什么时候停止？
谁拥有最终决定权？
```

因此整个系统可以统一抽象为多个可学习的 Policy。

```
Information Acquisition
↓

Hypothesis Exploration

↓

Belief Arbitration

↓

Computation Allocation
```

对应现有系统：

| 当前模块        | 抽象后的Policy              |
| ----------- | ----------------------- |
| Trigger     | Information Acquisition |
| Diversity   | Hypothesis Exploration  |
| Selection   | Belief Arbitration      |
| Cascade     | Computation Allocation  |
| Debate Stop | Termination Policy      |

论文故事可以提升为：

> We study reasoning control instead of reasoning generation.

---

# 二、Hypothesis Space Exploration（⭐⭐⭐）

目前：

Candidate = 不同答案

更合理的抽象：

Candidate = 不同假设（Hypothesis）

例如：

Question：

```
Who won most medals?
```

不同Hypothesis：

```
Gold medal

Total medal

Summer Olympics

Winter Olympics
```

而不是：

```
USA

China

Japan
```

Generator生成的不是答案。

而是：

```
Hypothesis

↓

Evidence

↓

Answer
```

Verifier验证Hypothesis是否成立。

Debate实际上变成：

```
Hypothesis Refinement
```

而不是：

```
Answer Revision
```

---

# 三、Evidence Sufficiency（⭐⭐⭐）

目前Trigger依据：

```
Generator 和 Verifier 是否一致
```

可以升级为：

```
Evidence 是否足够
```

例如：

Generator：

```
Answer = 38
```

Verifier：

```
Looks Correct.
```

但是：

```
Evidence Coverage = Low
```

仍然继续推理。

因此Trigger依据变成：

```
Evidence Sufficiency
```

而不是：

```
Agreement
```

---

# 四、Meta Reasoning（⭐⭐⭐）

真正研究对象：

不是：

```
How to reason
```

而是：

```
How to manage reasoning.
```

Agent持续回答几个问题：

```
Should I think?

Should I search?

Should I verify?

Should I debate?

Should I stop?
```

整个Agent就是：

```
Reasoning Manager
```

而不是：

```
Reasoner
```

---

# 五、Belief Revision（⭐⭐⭐）

当前：

多个Candidate互相独立。

实际上：

Candidate之间存在继承关系。

例如：

```
Candidate1

↓

修改Filter

↓

Candidate2

↓

修改Aggregation

↓

Candidate3
```

应该维护：

```
Belief Graph
```

Debate实际上就是：

```
Belief Update
```

最终：

不是Vote。

而是：

Posterior Belief。

来源：

知识表示（Belief Revision）

贝叶斯更新

Truth Maintenance System

---

# 六、Information Gain（⭐⭐⭐）

目前：

停止依据：

```
达到轮数
```

可以升级为：

```
Information Gain
```

例如：

```
Round1

Information Gain = 0.61

Round2

0.19

Round3

0.03
```

停止。

不是：

因为轮数。

而是：

没有新信息了。

---

# 七、Computation Allocation（⭐⭐⭐）

Inference-time Scaling领域核心思想：

有限算力应该花在哪。

对应：

```
Cheap Path

↓

Need More Compute?

↓

Debate

↓

Judge
```

优化目标：

```
Accuracy

vs

Cost
```

不是：

Accuracy最大。

而是：

Pareto Optimal。

---

# 八、Expected Utility（⭐⭐⭐）

来自：

经济学

Optimal Stopping

每次推理：

```
Expected Gain

>

Expected Cost ?
```

否则停止。

Agent始终优化：

```
Utility
```

而不是：

Accuracy。

---

# 九、Differential Diagnosis（⭐⭐）

来自：

医学。

医生不是猜答案。

医生不断排除可能性。

于是：

Debate不是：

```
Generate Another Answer
```

而是：

```
Generate Diagnostic Test
```

例如：

两个Program：

```
Program A

Program B
```

Verifier生成：

```
Diagnostic Query
```

专门区分两者。

---

# 十、Monte Carlo Tree Search（⭐⭐）

目前：

整个Program重新生成。

其实：

Program天然就是树。

```
Read Table

↓

Filter

↓

Sort

↓

Aggregate
```

Debate：

只展开：

```
Filter
```

而不是：

全部重来。

更加高效。

---

# 十一、Active Learning（⭐⭐）

Trigger问题：

其实和：

```
哪些样本值得标注？
```

一致。

可借鉴：

Uncertainty Sampling

BALD

Expected Error Reduction

Core-set

Expected Model Change

---

# 十二、Scientific Discovery（⭐⭐）

整个Agent过程：

```
Hypothesis

↓

Experiment

↓

Refute

↓

Survive
```

不是：

```
Generate

↓

Verify
```

因此：

Verifier：

更像：

Scientist。

---

# 十三、Distributed Cognition（⭐⭐）

当前：

一个Judge。

未来：

多个Judge。

但：

每个Judge：

只能看到不同Evidence。

例如：

```
Judge A

只看Program

Judge B

只看Table

Judge C

只看Question
```

最后：

Deliberation。

---

# 十四、Bayesian Reasoning（⭐⭐）

当前：

Vote。

可以升级：

```
Prior

↓

Evidence

↓

Posterior
```

Candidate：

不是答案。

而是：

Belief。

---

# 十五、Decision Authority（⭐⭐）

Decision Right名字略工程。

建议统一：

```
Decision Policy

Decision Authority

Control Policy
```

三个Decision：

```
Verdict

Selection

Termination
```

分别可以交给：

```
Generator

Verifier

Arbiter

Jury
```

---

# 十六、Adaptive Exploration（⭐⭐）

不同问题：

探索深度不同。

简单题：

```
One-shot
```

困难题：

```
Tree Search
```

真正决定：

```
Reasoning Budget
```

---

# 十七、Program Repair（⭐⭐）

不要重新生成Program。

而是：

```
Patch
```

Program。

Verifier指出：

```
Filter错了
```

Generator：

只修：

```
Filter
```

---

# 十八、Counterfactual Debate（⭐）

不是：

证明自己。

而是：

Verifier主动构造：

```
如果你的Aggregation改成Mean呢？
```

探索反事实。

---

# 十九、Curriculum Debate（⭐）

不同难度：

不同Debate策略。

例如：

```
Easy

No Debate

Medium

Verifier

Hard

Multi-Agent

Extreme

Tree Search
```

---

# 二十、Reasoning Ecology（⭐）

整个Agent不是：

一个Solver。

而是：

生态系统。

包含：

```
Explorer

Critic

Judge

Planner

Memory

Scheduler
```

不同角色：

动态协作。

---

# 二十一、跨领域可借鉴思想速查表

| 来源领域                   | 可迁移思想                  | 在Table QA中的对应     |
| ---------------------- | ---------------------- | ----------------- |
| Inference-time Scaling | Computation Allocation | Trigger / Cascade |
| AlphaGo / MCTS         | Selective Expansion    | Debate展开策略        |
| Active Learning        | Sample Selection       | Trigger           |
| 医学诊断                   | Differential Diagnosis | Diagnostic Query  |
| 科学发现                   | Hypothesis Testing     | Debate            |
| 贝叶斯                    | Belief Update          | Candidate融合       |
| 控制论                    | Meta Controller        | 整体Agent           |
| 金融                     | Optimal Stopping       | Debate终止          |
| 信息论                    | Information Gain       | Stop Policy       |
| 知识表示                   | Belief Revision        | Selection         |
| 软件工程                   | Program Repair         | 局部修复              |
| 多智能体                   | Distributed Cognition  | Jury              |

---

# 二十二、我目前最看好的 Top 10（按潜力排序）

| 排名 | 方向                           | 潜力    |
| -- | ---------------------------- | ----- |
| 1  | Reasoning Control            | ⭐⭐⭐⭐⭐ |
| 2  | Hypothesis Space Exploration | ⭐⭐⭐⭐⭐ |
| 3  | Evidence Sufficiency Trigger | ⭐⭐⭐⭐⭐ |
| 4  | Meta Reasoning               | ⭐⭐⭐⭐⭐ |
| 5  | Information Gain Termination | ⭐⭐⭐⭐☆ |
| 6  | Computation Allocation       | ⭐⭐⭐⭐☆ |
| 7  | Belief Revision              | ⭐⭐⭐⭐☆ |
| 8  | Expected Utility             | ⭐⭐⭐⭐☆ |
| 9  | Program Repair               | ⭐⭐⭐⭐  |
| 10 | Differential Diagnosis       | ⭐⭐⭐⭐  |

---

## 最后，我想补充一个我没有写进上面的判断，也是我这两个小时越想越坚定的观点。

**如果目标是冲 ACL / EMNLP 主会，而不是 Findings，那么真正需要的不是"一个新的 verifier"，而是一个新的"推理范式（reasoning paradigm）"。**

我认为你现在已经隐约摸到了这个范式，但还没有把它抽象出来。你真正研究的不是 **Table QA**，甚至不是 **Table Agent**，而是：

> **Agent 如何在有限测试时计算预算下，动态决定"思考什么、探索什么、验证什么、停止什么"。**

这件事天然连接了 **Inference-time Scaling、Meta-Reasoning、Reasoning Control、Agentic AI** 四条目前都非常活跃的研究主线。

如果后面我们继续一起打磨，我建议不要从"还能加什么模块"开始，而是先回答一个问题：

> **一句话，这篇论文到底提出了一个什么新的世界观？**

等这个世界观立住以后，Trigger、Debate、Selection、Judge、Cascade 都会变成支撑这个世界观的实验，而不是论文本身。这个转变，往往就是一篇"工程优化"和一篇"研究论文"之间最大的区别。