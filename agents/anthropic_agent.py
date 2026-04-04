"""
Anthropic agent for the prompt repetition experiment.

Wraps the anthropic Python SDK to expose the same interface as MLXAgent and
OpenAIAgent so runner.py requires no changes when switching providers.

Supported models
----------
    claude-haiku-4-5-20251001   (recommended — cheapest Anthropic option)
    claude-sonnet-4-6           (stronger, if Haiku results look noisy)

Setup
----------
    pip install anthropic
    export ANTHROPIC_API_KEY=...
    # or set it in a .env file and load with python-dotenv

Cost estimate for my low-budget folks
----------
    ~3 000 questions * 2 variants * ~300 input tokens  * $0.80/1M ≈ $0.29
    ~3 000 questions * 2 variants * ~30 output tokens  * $4.00/1M ≈ $0.07
    Total: < $0.50 for a full ARC + OpenBookQA + GSM8K run with Haiku

    
Note on non-reasoning models
----------
The paper specifically tests non-reasoning models (no CoT in the prompt).
Claude Haiku is a non-reasoning model by default —> do NOT pass
thinking={"type": "enabled"} in the API call, as that would activate extended
thinking and invalidate the comparison.
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

from agents.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class AnthropicAgentConfig:
    """Configuration for the Anthropic agent."""
    model_id: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 256
    temperature: float = 0.0        # greedy; consistent with other agents
    timeout: float = 60.0           # seconds


class AnthropicAgent(BaseAgent):
    """
    Anthropic messages-API agent.

    The client is init once on construction and reused across queries.

    The API key is read from the ANTHROPIC_API_KEY environment variable:

        from dotenv import load_dotenv
        load_dotenv()
        agent = AnthropicAgent()
    """

    def __init__(self, config: Optional[AnthropicAgentConfig] = None):
        self.config = config or AnthropicAgentConfig()
        self._client = self._build_client()

    # BaseAgent interface
    @property
    def model_id(self) -> str:
        return self.config.model_id

    def query(self, prompt: str, variant: str = "unknown") -> AgentResponse:
        """
        Send a single prompt via the Anthropic messages endpoint.

        The prompt is passed as a user message with no system prompt,
        matching the setup used for OpenAIAgent and MLXAgent.
        """
        t0 = time.perf_counter()

        response = self._client.messages.create(
            model=self.config.model_id,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        # extract text from the first content block.
        output_text = ""
        for block in response.content:
            if block.type == "text":
                output_text = block.text
                break

        usage = response.usage

        logger.debug(
            "AnthropicAgent query | model=%s | variant=%s | "
            "input_tokens=%d | output_tokens=%d | latency_ms=%.1f",
            self.config.model_id, variant,
            usage.input_tokens, usage.output_tokens, latency_ms,
        )

        return AgentResponse(
            text=output_text,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            latency_ms=latency_ms,
            model_id=self.config.model_id,
            variant=variant,
            metadata={
                "stop_reason": response.stop_reason,
                "response_id": response.id,
            },
        )

    # unload() inherits the no-op from BaseAgent

    # private helpers
    def _build_client(self):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic is not installed. Run: pip install anthropic"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it or add it to your .env file."
            )

        return anthropic.Anthropic(
            api_key=api_key,
            timeout=self.config.timeout,
        )


# smoke test
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    agent = AnthropicAgent()

    test_prompt = (
        "Answer the following question. Choose from the options provided.\n\n"
        "What is the powerhouse of the cell?\n\n"
        "A) Nucleus\nB) Mitochondria\nC) Ribosome\nD) Golgi apparatus\n\n"
        "Provide your answer in the format: The answer is X (where X is A, B, C, or D)."
    )

    r = agent.query(test_prompt, variant="baseline")
    print(f"Response : {r.text}")
    print(f"Latency  : {r.latency_ms:.0f} ms")
    print(f"Tokens   : {r.prompt_tokens} prompt / {r.completion_tokens} completion")
    print(f"Stop     : {r.metadata['stop_reason']}")
