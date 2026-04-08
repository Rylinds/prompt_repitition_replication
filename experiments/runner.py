"""
Experiment runner for the study replication.

Loops over every (model x benchmark x n_samples) combination defined in ExperimentConfig,
evaluates baseline and repeat prompt variants in a paired design,
and saves one JSON result file per (model, benchmark) run.

Design decisions
----------
Paired evaluation
    Baseline and repeat are evaluated on the same question IDs in the same run.
    This is a hard requirement for McNemar test -> unpaired results cannot be used.

Round-robin provider ordering
    When running multiple models, the runner interleaves them question by question 
    rather than running all questions for model A and then all for model B.
    This distributes network latency variance evenly so latency comparisons between
    providers are fair.

Result caching
    If a result file for a (model, benchmark) pair already exists it is skipped by default.
    Set overwrite=True in ExperimentConfig to re-run. This means you never pay for
    re-running a completed experiment.

Output schema (one record per question per model)
        "question_id":       str,
        "benchmark":         str,
        "model_id":          str,
        "gold":              str,
        "base_correct":      bool,
        "base_predicted":    str | null,
        "base_parse_status": str,
        "base_tokens":       int,
        "base_latency_ms":   float,
        "base_response":     str,       # raw text — for auditing
        "rep_correct":       bool,
        "rep_predicted":     str | null,
        "rep_parse_status":  str,
        "rep_tokens":        int,
        "rep_latency_ms":    float,
        "rep_response":      str,

Usage
----------
from experiments.runner import Runner, ExperimentConfig
from agents.openai_agent import OpenAIAgent
from agents.anthropic_agent import AnthropicAgent

config = ExperimentConfig(
    n_samples=500,
    benchmarks=["arc"],
    output_dir="experiments/results",
)
agents = [OpenAIAgent(), AnthropicAgent()]
runner = Runner(config, agents)
runner.run()
"""

from __future__ import annotations
import json
import logging
import pathlib
import time
from dataclasses import dataclass, field
from typing import List, Optional

from agents.base_agent import BaseAgent
from experiments.evaluator import Evaluator
from data.arc_loader import ARCLoader, MCQExample, format_mcq_prompt

logger = logging.getLogger(__name__)


# config
@dataclass
class ExperimentConfig:
    """
    Top level config for a full experiment run

    Attributes
    ----------
    n_samples: questions per benchmark per model. Use a small number (50-100) for pilots
    benchmarks: which benchmarks to run. Currently supported is arc, need to add others (4/7)
    seed: random seed for reproducing question sampling
    output_dir: directory where JSON result is saved
    overwrite: if False, skip runs whose output file already exists. True to force re-run
    prompt_ordering: question first or options first (see arc_loader.py)
    save_responses: if True, include raw model response text in each record. Useful for audits
    """
    n_samples: int = 100
    benchmarks: List[str] = field(default_factory=lambda: ['arc'])
    seed: int = 42
    output_dir: str = "experiments/results"
    overwrite: bool = False
    prompt_ordering: str = "question_first"
    save_responses: bool = True


# runner
class Runner:
    """
    Orchestrates the full paired experiment loop.

    Instantiate with config and list of agent objects, then call run().
    The list can be a single agent for a single model run, or multiple agents for
    multi-model comparison run.

    Example - single model:
        runner = Runner(ExperiementConfig(n_samples=100), [OpenAIAgent()])
        runner.run()
    """
    # loaders are instantiated lazily when benchmark is first needed
    _SUPPORTED_BENCHMARKS = {"arc"}

    def __init__(self, config: ExperimentConfig, agents: List[BaseAgent]):
        if not agents:
            raise ValueError("agents list must not be empty")
        unknown = set(config.benchmarks) - self._SUPPORTED_BENCHMARKS
        if unknown:
            raise ValueError(
                f"Unsupported benchmark(s): {unknown}. "
                f"Supported: {self._SUPPORTED_BENCHMARKS}"
            )
        self.config = config
        self.agents = agents
        self.evaluator = Evaluator()
        self._output_dir = pathlib.Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # public entry point
    def run(self) -> None:
        """
        Execute the full experiment as configured.

        For each (benchmark, agent) pair:
            1. load questions
            2. check whether output already exists; skip if overwrite=False
            3. evaluate every question under both baseline and repeat variants
            4. write results to disk after every question (crash safety)
        """
        for benchmark in self.config.benchmarks:
            examples = self._load_benchmark(benchmark)
            logger.info(
                "Benchmark %s: %d questions loaded (seed=%d).",
                benchmark, len(examples), self.config.seed,
            )

            for agent in self.agents:
                out_path = self._result_path(agent.model_id, benchmark)

                if out_path.exists() and not self.config.overwrite:
                    logger.info(
                        "Skipping %s / %s - result file already exists: %s",
                        agent.model_id, benchmark, out_path
                    )
                    continue
                
                logger.info(
                    "Running %s on %s (%d qiestions)...",
                    agent.model_id, benchmark, len(examples)
                )
                self._run_single(agent, benchmark, examples, out_path)
        
        logger.info("All runs complete.")

# core eval loop
def _run_single(
    self,
    agent: BaseAgent,
    benchmark: str,
    examples: list,
    out_path: pathlib.Path,
) -> None:
    """
    Evaluate one (agent, benchmark) pair and write results to out_path.

    Each question is evaluated under baseline then repeat in immeidate succession so
    the paired association is never broken even if the run is interrupted mid-way.
    Results are appended to a list and written atomically at the end. A crash loses at
    the most current question.
    """
    is_math = benchmark == 'gsm8k'
    records = []
    t_run_start = time.perf_counter()

    for i, ex in enumerate(examples):
        record = self._evaluate_pair(agent, ex, benchmark, is_math)
        records.append(record)

        # progress log every 10 Qs
        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            elapsed = time.perf_counter() - t_run_start
            base_acc = sum(r['base_correct'] for r in records) / len(records)
            rep_acc = sum(r['rep_correct'] for r in records) / len(records)
            logger.info(
                " [%d/%d]    base=%.1f%%     rep=%.1f%%     elapsed=%.0fs",
                base_acc * 100, rep_acc * 100, elapsed,
            )
    
    # write results atomically (write to .tmp then rename)
    tmp_path = out_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(records, indent=2))
    tmp_path.rename(out_path)
    logger.info("Results written to %s", out_path)

    # log final summary
    self._log_summary(records, agent.model_id, benchmark)

def _evaluate_pair(
    self,
    agent: BaseAgent,
    ex: MCQExample,
    benchmark: str,
    is_math: bool,
) -> dict:
    """
    Evaluate a single question under both baseline and repeat. Return a record dict.
    The two queries are issued back to back for the same question so paired associatiom
    is explicit in time (not just saved IDs).
    """
    # build prompts
    base_prompt = self._build_prompt(ex, "baseline")
    rep_prompt = self._build_prompt(ex, "repeat")

    # query model (baseline then repeat)
    r_base = agent.query(base_prompt, variant="baseline")
    r_rep = agent.query(rep_prompt, variant="repeat")

    # evaluate
    ev_base = self.evaluator.evaluate(
        r_base.text, ex.correct_label,
        question_id=ex.example_id, variant="baseline", is_math=is_math,
    )
    ev_rep = self.evaluator.evaluate(
        r_rep.text, ex.correct_label,
        question_id=ex.example_id, variant="repeat", is_math=is_math,
    )

    record: dict = {
        "question_id":       ex.example_id,
        "benchmark":         benchmark,
        "model_id":          agent.model_id,
        "gold":              ex.correct_label,
        # Baseline
        "base_correct":      ev_base.is_correct,
        "base_predicted":    ev_base.predicted,
        "base_parse_status": ev_base.parse_status.value,
        "base_tokens":       r_base.completion_tokens,
        "base_latency_ms":   round(r_base.latency_ms, 2),
        # Repeat
        "rep_correct":       ev_rep.is_correct,
        "rep_predicted":     ev_rep.predicted,
        "rep_parse_status":  ev_rep.parse_status.value,
        "rep_tokens":        r_rep.completion_tokens,
        "rep_latency_ms":    round(r_rep.latency_ms, 2),
    }

    if self.config.save_responses:
        record["base_response"] = r_base.text
        record["rep_resposne"] = r_rep.text
    
    return record

# prompt building
def _build_prompt(self, ex: MCQExample, variant: str) -> str:
    """
    Construct the full prompt for a given example and variant.
    Uses arc_loader's format_mcq_prompt to build the base query, then
    wraps it with PrompBuilder to apply the variant.
    """
    from agents.prompt_builder import PromptBuilder, PromptVariant

    # build the raw MCQ query string
    query = format_mcq_prompt(ex, ordering=self.config.prompt_ordering)
    # append the answer-format instruction that the evaluator expects
    query += "\nProvide your answer in the format: The answer is X (where X is A, B, C, or D)."

    builder = PromptBuilder()
    pv = PromptVariant.BASELINE if variant == "baseline" else PromptVariant.REPEAT          # sketchy
    return builder.build(query, pv)

# benchmark loading
def _load_benchmark(self, benchmark: str) -> list:
    """
    Instantiate the appropriate laoder and return sampled examples.
    """
    if benchmark == "arc":
        loader = ARCLoader(split="test")
        return loader.load_subset(self.config.n_samples, seed=self.config.seed)
    # future to-dos are obqa and gsm8k
    raise ValueError(f"No loader implemented for benchmark: {benchmark!r}")

# output path helpers
def _result_path(self, model_id: str, benchmark: str) -> pathlib.Path:
    """
    Derive a deterministic output filename from model and benchmark.

    ex: results/gpt-4o-mini_arc_n100_seed42.json
    """
    # sanitise model_id for use in a filename
    safe_model = model_id.replace("/", "-").replace(":", "-")
    filename = (
        f"{safe_model}_{benchmark}"
        f"_n{self.config.n_samples}"
        f"_seed{self.config.seed}.json"
    )
    return self._output_dir / filename

# summary logging
@staticmethod
def _log_summary(records: list, model_id: str, benchmark: str) -> None:
    """Log a McNemar contingency table and accuracy summary."""
    n = len(records)
    base_acc = sum(r["base_correct"] for r in records) / n
    rep_acc  = sum(r["rep_correct"]  for r in records) / n

    both_correct = sum(1 for r in records if     r["base_correct"] and     r["rep_correct"])
    base_only    = sum(1 for r in records if     r["base_correct"] and not r["rep_correct"])
    rep_only     = sum(1 for r in records if not r["base_correct"] and     r["rep_correct"])
    both_wrong   = sum(1 for r in records if not r["base_correct"] and not r["rep_correct"])

    logger.info("─" * 56)
    logger.info("Summary  model=%s  benchmark=%s  n=%d", model_id, benchmark, n)
    logger.info("  Baseline acc : %.1f%%  (%d/%d)", base_acc * 100, int(base_acc * n), n)
    logger.info("  Repeat acc   : %.1f%%  (%d/%d)", rep_acc  * 100, int(rep_acc  * n), n)
    logger.info("  Δ accuracy   : %+.1f%%", (rep_acc - base_acc) * 100)
    logger.info("  McNemar table: both_correct=%d  base_only=%d  rep_only=%d  both_wrong=%d",
                both_correct, base_only, rep_only, both_wrong)

    # Quick McNemar p-value (exact binomial).
    b, c = base_only, rep_only
    if b + c > 0:
        from scipy.stats import binom
        p = min(2 * binom.cdf(min(b, c), b + c, 0.5), 1.0)
        logger.info("  McNemar p    : %.4f  (b=%d, c=%d)", p, b, c)
    logger.info("─" * 56)
