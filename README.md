# TableAgent

An agentic table question-answering system for the
[WikiTableQuestions](https://github.com/ppasupat/WikiTableQuestions) dataset.

The goal is to maximize answer accuracy by combining a tool-calling agent with
three core mechanisms — **grounding**, **verification**, and **self-consistency** —
on top of a reliable engineering harness (step budgets, structured observations,
context budgeting, full-chain tracing).

See [`PLAN.md`](PLAN.md) for the design rationale and roadmap, and
[`docs/INTERFACES.md`](docs/INTERFACES.md) for the live module/interface contract.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Dataset (not committed):
git clone https://github.com/ppasupat/WikiTableQuestions.git

# Config:
cp .env.example .env   # then fill in LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
```

## Quick checks

```bash
python -m scripts.smoke         # no-LLM foundation (data + evaluator + sandbox)
python -m scripts.check_llm     # endpoint connectivity + function-calling probe
python -m scripts.run_baseline 20   # direct-answer baseline on a quick-set sample
```

## Status

- Phase 0  — no-model foundation (schemas, data loader, Py3 evaluator, sandbox, trace). Done.
- Phase 0.5 — LLM client + direct-answer baseline. Done.
- Phase 1  — single-trajectory function-calling agent. In progress.

## Layout

```
src/        core modules (schemas, data, evaluator, llm, tools, agent, ...)
scripts/    runnable entry points (smoke, check_llm, run_baseline)
prompts/    system charter (AGENT.md) + skills
tests/      unit + golden regression tests
results/    predictions, traces, metrics (gitignored)
docs/       INTERFACES.md (live contract)
```
