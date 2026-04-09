# Replication Study: *Prompt Repetition Improves Non-Reasoning LLMs*

**Source paper:** Leviathan, Kalman & Matias (2025) — [arXiv:2512.14982](https://arxiv.org/abs/2512.14982)

> **Status: Active — Phase 1 complete, Phase 2 in progress.**
> This README will evolve into a full lab report as experiments complete. What follows documents the project's scope, architecture, and preliminary findings to date.

---

## Table of Contents

1. [What This Paper Claims](#1-what-this-paper-claims)
2. [Why It's Interesting](#2-why-its-interesting)
3. [Replication Scope](#3-replication-scope)
4. [Repository Structure](#4-repository-structure)
5. [What's Been Built](#5-whats-been-built)
6. [Preliminary Results](#6-preliminary-results)
7. [Experimental Design](#7-experimental-design)
8. [References](#10-references)

---

## 1. What This Paper Claims

The central claim of Leviathan et al. (2025) is pretty simple. If you send a model the same prompt twice in a row, such as:

```
<QUERY><QUERY>
```

, its accuracy on standard multiple-choice benchmarks improves, *without* increasing output length or inference latency.

Across 7 models and 7 benchmarks, the repeated-prompt condition outperformed the baseline in 47 out of 70 tested configurations (67%), with statistically significant wins on most benchmarks by McNemar's test. The effect is notably absent in reasoning models (o1, Gemini Thinking), which the authors interpret as evidence that the technique works by giving non-reasoning models a second pass over the question during the parallelizable *prefill* stage (a form of implicit "re-reading" that reasoning models achieve through their chain-of-thought outputs anyway).

---

## 2. Why It's Interesting

The mechanism the paper proposes connects to a fundamental property of causal (autoregressive) language models: attention is unidirectional. When generating the answer, the model attends to question tokens that were processed *left-to-right* during prefill. An early question token cannot attend to answer-choice tokens that appear later in the prompt. By repeating the prompt, every token in the second copy can attend to the full context of the first copy, effectively giving the model bidirectional access to the question before it begins generating.

This is mechanistically related to several prior lines of work:

- **RE2 (Xu et al., 2024)** — asks the model to re-read *during generation* via an instruction; prompt repetition achieves a similar effect at the *prefill* stage with no output overhead.
- **Lost in the Middle (Liu et al., 2023)** — documents that middle-of-prompt tokens are systematically underweighted; prompt repetition may counteract this for question stems buried between long answer options.
- **Speculative decoding (Leviathan et al., 2022)** — by the same first author; relevant because it motivates *why* adding prefill tokens is cheap: prefill is parallelized on modern hardware, unlike autoregressive decode.

This project aims to verify whether the effect holds under independent replication, and to develop intuition for where it does and doesn't appear.

---

## 3. Replication Scope

The original paper's scope (7 models × 7 benchmarks) is expensive to fully replicate independently. This project takes a phased approach:

| Dimension | Original Paper | This Replication |
|---|---|---|
| Models | 7 (Gemini, GPT, Claude, Deepseek) | Mistral 7B (local, Phase 1pilot run); GPT-4o-mini + Claude Haiku (Phase 2) |
| Benchmarks | 7 (incl. custom tasks) | ARC-Challenge (Phase 1); OpenBookQA + GSM8K (Phase 2) |
| Prompt variants | 5 | All 5 implemented; baseline vs. repeat is the primary comparison |
| Reasoning mode | Both | Non-reasoning only (per the paper's own framing) |
| Statistical test | McNemar | McNemar (scipy.stats.mcnemar) + odds ratios |

The decision to use a locally-run quantized model for Phase 1 was deliberate: it allows rapid iteration and debugging of the pipeline without API cost, at the expense of not being directly comparable to the paper's proprietary model results.

---

## 4. Repository Structure

```
prompt-repetition-replication/
│
├── README.md                        ← you are here
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── base_agent.py                ← abstract AgentResponse interface
│   ├── mlx_agent.py                 ← local inference via mlx-lm (Mistral 7B 4-bit)
│   ├── openai_agent.py              ← GPT wrapper (pending API key)
│   └── anthropic_agent.py           ← Claude Haiku wrapper (pending API key)
│
├── data/
│   └── loaders/
│       └── arc_loader.py            ← ARC-Challenge loader with MCQExample dataclass
│
├── experiments/
│   ├── prompt_builder.py            ← 5 prompt variants
│   ├── evaluator.py                 ← 3-tier MCQ parse cascade + GSM8K numeric path
│   ├── runner.py                    ← main loop with atomic writes + skip-if-exists cache
│   └── results/                     ← gitignored; raw JSON per run
│
├── analysis/
│   └── notebooks/
│       └── explore_results.ipynb    ← audit notebook for PoC run
│
└── tests/
    ├── test_prompt_builder.py        ✓ 11 test classes, all passing
    └── test_evaluator.py             ✓ 11 test classes, all passing
```

---

## 5. What's Been Built

### Prompt Builder (`experiments/prompt_builder.py`)

Implements all five variants described in the paper:

| Variant | Format | Purpose |
|---|---|---|
| `baseline` | `[instruction]\n\n<query>` | Control condition |
| `repeat` | `[instruction]\n\n<query><query>` | Primary experimental condition |
| `verbose` | `[instruction]\n\n<query>\n\nLet me repeat that:\n<query>` | Separator-aware repeat |
| `repeat_x3` | `[instruction]\n\n<query><query><query>` | Triple repetition |
| `padding` | `[instruction]\n\n<query> [PAD]×5 <query>` | Token-count control |

The `repeat` variant concatenates the query with *no separator*, matching the paper's specification exactly. Tests enforce this with character-level assertions (e.g., `"QQ"` not `"Q Q"`).

### ARC Loader (`data/loaders/arc_loader.py`)

Loads the ARC-Challenge test split from HuggingFace and normalizes it into `MCQExample` dataclasses. A notable implementation detail: a subset of ARC questions (from the NYSEDREGENTS source) use numeric gold answer keys (`"1"`, `"2"`, `"3"`, `"4"`) rather than letter keys. The loader handles this via a `_KEY_MAP` normalization step discovered and patched during Phase 1 debugging.

### MLX Agent (`agents/mlx_agent.py`)

Runs `mlx-community/Mistral-7B-Instruct-v0.3-4bit` (~4.5 GB) locally via `mlx-lm`. The agent uses the `sampler=make_sampler(temp)` pattern introduced in mlx-lm v0.22, which was a breaking change from the previous `temp=` keyword argument. Worth noting if you're running this on a different mlx-lm version.

### Evaluator (`experiments/evaluator.py`)

Parses model responses using a three-tier cascade for MCQ:

1. **Exact format** — looks for `"The answer is X"` where X is A–D
2. **Delimited fallback** — catches formats like `"(B)"`, `"B)"`, `"B."` with regex named groups
3. **Keyword/isolated fallback** — finds a standalone letter, excluding `"I"` (to avoid false positives on first-person responses while preserving `"I)"` for MMLU-Pro style formats)

A separate numeric path handles GSM8K's free-form math answers. The evaluator also tracks McNemar-compatible pairing contracts — each question gets a `(baseline_correct, repeat_correct)` pair, not just individual accuracy.

### Runner (`experiments/runner.py`)

The main experiment loop writes results atomically (write to `.tmp`, then rename) and skips question IDs whose results already exist on disk. This means interrupted runs resume cleanly without duplicating API calls. Provider ordering is round-robin to avoid systematic latency bias between models.

---

## 6. Preliminary Results

### Proof-of-Concept Run (Phase 1)

- **Model:** `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- **Benchmark:** ARC-Challenge test split
- **Sample size:** 100 questions (paired)
- **Date:** March 2026

| Condition | Accuracy |
|---|---|
| Baseline | 69% |
| Repeat | 66% |
| Δ | −3% |
| McNemar p | ≈ 0.45 |

The repeat condition did not outperform baseline in this run. The difference is not statistically significant (p = 0.45 is well above any conventional threshold).

**Two artifacts were identified and resolved during auditing that affected the quality of this PoC run:**

1. **`max_tokens=128` cap** — the initial run used a hard cap of 128 output tokens, which caused truncation on verbose answers. The paper does not impose this constraint. This has been patched; re-running with a higher cap is pending.

2. **Numeric gold key bug** — NYSEDREGENTS questions with numeric keys (`"1"`, `"2"`) were being scored as always-wrong because the evaluator expected letter keys. The `_KEY_MAP` normalization fix was subsequently validated with a dedicated test class.

A secondary observation: wrong answers used approximately 2.5× more output tokens than correct ones in this run. This "verbosity on uncertainty" pattern is worth tracking as a secondary signal in Phase 2.

**Interpretation:** The PoC run is best understood as a pipeline validation, not an experimental result. The model (Mistral 7B quantized) differs from the paper's test set (GPT-4o, Gemini 1.5 Flash, etc.), the sample is small, and two known artifacts were present. Phase 2 with API models is needed before drawing any conclusions.

---

## 7. Experimental Design

### Pairing

All runs maintain paired results: the same question ID is answered under both `baseline` and `repeat` conditions. This is required for McNemar's test, which operates on a 2×2 contingency table of per-question win/loss outcomes, not aggregate accuracy.

### McNemar's Test

For each (model, benchmark, variant pair) combination, the analysis builds:

```
                   Repeat: Correct    Repeat: Wrong
Baseline: Correct       a                  b
Baseline: Wrong         c                  d
```

McNemar's statistic tests whether `b ≠ c` (i.e., whether the off-diagonal cells are symmetric). A significant result with `c > b` means the repeated prompt produced more *recoveries* (wrong→right) than *regressions* (right→wrong). The paper reports 47/70 tested configurations meeting this criterion.

### Extended Thinking

The Anthropic agent has extended thinking explicitly disabled. Enabling it on Haiku would make it a reasoning model, invalidating the non-reasoning comparison that is the paper's primary framing.

---

## 8. References

Leviathan, Y., Kalman, M., & Matias, Y. (2025). *Prompt repetition improves non-reasoning LLMs*. arXiv:2512.14982.

Leviathan, Y., Kalman, M., & Matias, Y. (2022). *Fast inference from transformers via speculative decoding*. arXiv:2211.17192.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). *Lost in the middle: How language models use long contexts*. arXiv:2307.03172.

Xu, X., Tao, C., Shen, T., Xu, C., Xu, H., Long, G., & Lou, J. Y. (2024). *Re-reading improves reasoning in large language models*. arXiv:2309.06275.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). *Chain-of-thought prompting elicits reasoning in large language models*. arXiv:2201.11903.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention is all you need*. NeurIPS 2017.

---

*Last updated: April 2026. This README will be revised to a full lab report upon completion of Phase 2 experimental runs.*
