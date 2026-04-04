"""
OpenAI agent for the experiment.

Wraps the openai Python SDK to expose the same interface as MLXAgent so
runner.py requires no changes when switching between local and cloud inference.

Just for using gpt-4o-mini

Setup
----------
    pip install openai
    export OPENAI_API_KEY=...
    # or set in .env with python-dotenv if you prefer idk

Cost estimate for my low-budget folks
----------
    ~3,000 questions * 2 variants * ~300 input tokens * $0.15/1M = $0.27
    ~3,000 questions * 2 variants * ~30 input tokens * $0.60/1M = $0.11
    Total: < $0.50 for a full ARC + OpenBookQA + GSM8K run with mini
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional
from agents.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class OpenAIAgentConfig:
    """Config for the OpenAI agent"""
    model_id: str = "gpt-4o-mini"
    max_tokens: int = 256
    temperature: float = 0.0            # greedy; same as MLX agent to reproducibility
    timeout: float = 60.0               # seconds before raising TimeoutError
    max_retries: int = 3                # automatic retries on transient errors (built in to SDK)

class OpenAIAgent(BaseAgent):
    """
    OpenAI chat-completions agent.

    The client is init once on construction and reused for every query() call,
    which avoids repeated SSL handshakes across a long run.

    The API key is read from the OPENAI_API_KEY environment var.
    Load is from .env before constructing the class:

        from dotenv import load_dotenv
        load_dotenv()
        agent = OpenAIAgent()
    """
    def __init__(self, config: Optional[OpenAIAgentConfig] = None):
        self.config = config or OpenAIAgentConfig()
        self._client = self._build_client()
    
    # BaseAgent interface
    @property
    def model_id(self) -> str:
        return self.config.model_id
    
    def query(self, prompt: str, variant: str = "unknown") -> AgentResponse:
        """
        Send a single prompt via chat-completions endpoint.

        The prompt is passed as the only user message. No sys prompt
        is added so that the repitition effect is measured cleanly
        without additional instruction contexts.
        """
        t0 = time.perf_counter()

        response = self._client.chat.completions.create(
            model=self.config.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        choice = response.choices[0]
        output_text = choice.message.content or ""
        usage = response.usage

        logger.debug(
            "OpenAIAgent query | model=%s | variant=%s | "
            "prompt_tokens=%d | completion_tokens=%d | latency_ms=%.1f",
            self.config.model_id, variant,
            usage.prompt_tokens, usage.completions_tokens, latency_ms
        )

        return AgentResponse(
            text=output_text,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_ms=latency_ms,
            model_id=self.config.model_id,
            variant=variant,
            metadata={
                "finish_reason": choice.finish_reason,
                "response_id": response.id,
            },
        )

        # unload() inherits no-op from BaseAgent
    
    # private helpers

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
            "openai is not installed. Run: pip install openai"
        ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Export it or addit to .env file."
            )
        
        return OpenAI(
            api_key=api_key,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

# smoke test
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
 
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
 
    agent = OpenAIAgent()
 
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
    print(f"Finish   : {r.metadata['finish_reason']}")
