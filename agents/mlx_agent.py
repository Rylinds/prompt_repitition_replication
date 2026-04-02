"""
This entire experiment is being run on a 2022 Macbook Air M2 (yikes...), hence this file
and MLX choice for the proof-of-concept stage of the experiment.

MLX-based local inference agent for prompt repitition experiements.

Uses mlx-lm to run Mistral-7B-Instruct (4-bit quantized) locally on Apple Silicon M2.
This agent is designed for PoC / pilot before it runs before switching to cloud APIs (which cost money and I'm frugal).

Usage:
    agent = MLXAgent()               <- load the model once and reuse across questions
    result = agent.query(prompt)
    agent.unload()                   <- free up my memory before switching to API agents :')
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MLXAgentConfig:
    """Configuration for the MLX local agent."""
    model_id: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    max_tokens: int = 128               # MCQ answers are short; keep low for speed
    temperature: float = 0.0            # greedy decoding (I love reproducibility)
    verbose: bool = False               # True to see token-by-token generation


@dataclass
class AgentResponse:
    """Structured response from the agent, mirroring the API agent contract."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model_id: str
    variant: str = "unknown"            # set by runner.py after construction
    metadata: dict = field(default_factory=dict)

class MLXAgent:
    """
    Local inference agent using mlx-lm on Apple Silicon.

    Loads the model once on init and holds it in memory.  Call unload() when
    done to release unified memory before the API-based agents are initialised
    (important on 16 GB M2 where you cannot hold two large models at once).

    The public interface intentionally matches the cloud-API agents so that
    runner.py can swap them with zero changes:

        agent.query(prompt: str) -> AgentResponse
        agent.unload() -> None
        agent.model_id -> str
    """

    def __init__(self, config: Optional[MLXAgentConfig] = None):
        self.config = config or MLXAgentConfig()
        self._model = None
        self._tokenizer = None
        self._load_model()

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def query(self, prompt: str, variant: str = "unknown") -> AgentResponse:
        """
        Run inference for a single prompt and return a structured response.

        Args:
            prompt:  The full formatted prompt string (already built by PromptBuilder).
            variant: Label for the experiment variant (e.g. "baseline", "repeat").

        Returns:
            AgentResponse with the generated text and timing/token metadata.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model is not loaded. Call _load_model() first.")

        # local imports so the file is importable on non-Apple machines (people who escaped the cult)
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        # apply the chat template so mistral's instruction format is respected
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # count prompt tokens (approx via tokenizer encode).
        prompt_token_ids = self._tokenizer.encode(formatted_prompt)
        prompt_tokens = len(prompt_token_ids)

        # recent mlx-lm removed the temp kwarg from generate() in favor of an explicit sampler callable.
        # make_sampler(temp, top_p, min_p, min_tokens_to_keep)
        # FAAAAAAAAAHHHHHH
        sampler = make_sampler(self.config.temperature)

        # time the generation
        t0 = time.perf_counter()
        output_text = generate(
            self._model,
            self._tokenizer,
            prompt=formatted_prompt,
            max_tokens=self.config.max_tokens,
            sampler=sampler,
            verbose=self.config.verbose,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        # completion tokens (approx)
        completion_tokens = len(self._tokenizer.encode(output_text))

        logger.debug(
            "MLXAgent query | variant=%s | prompt_tokens=%d | "
            "completion_tokens=%d | latency_ms=%.1f",
            variant, prompt_tokens, completion_tokens, latency_ms,
        )

        return AgentResponse(
            text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            model_id=self.config.model_id,
            variant=variant,
        )

    def unload(self) -> None:
        """
        Release the model and tokenizer from unified memory.

        Call this before initialising an API-based agent or before the process
        exits to avoid holding 4-5GB of weights.
        """
        self._model = None
        self._tokenizer = None
        # mlx doesn't expose an explicit free; dropping references lets the
        # metal allocator reclaim the memory on the next GC cycle.
        import gc
        gc.collect()
        logger.info("MLXAgent: model unloaded.")

    # private helper
    def _load_model(self) -> None:
        """Download (first run) and load the quantized model into unified memory."""
        try:
            from mlx_lm import load             # guarded import
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is not installed.  Run: pip install mlx-lm"
            ) from exc

        logger.info("Loading MLX model: %s …", self.config.model_id)
        t0 = time.perf_counter()
        self._model, self._tokenizer = load(self.config.model_id)
        elapsed = time.perf_counter() - t0
        logger.info("Model loaded in %.1f s.", elapsed)


# smoke test (python mlx_agent.py)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    agent = MLXAgent()

    test_prompt = (
        "Answer the following question. Choose from the options provided.\n\n"
        "What is the powerhouse of the cell?\n\n"
        "A) Nucleus\nB) Mitochondria\nC) Ribosome\nD) Golgi apparatus\n\n"
        "Provide your answer in the format: The answer is X (where X is A, B, C, or D)."
    )

    print("\n=== Baseline ===")
    r = agent.query(test_prompt, variant="baseline")
    print(f"Response : {r.text}")
    print(f"Latency  : {r.latency_ms:.0f} ms")
    print(f"Tokens   : {r.prompt_tokens} prompt / {r.completion_tokens} completion")

    agent.unload()
    print("\nModel unloaded.")
