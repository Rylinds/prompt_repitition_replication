"""
OpenBookQA loader for the prompt repetition experiment.

Dataset: allenai/openbookqa (HuggingFace)
Config: "main"
Split: "test" (500 questions)

HuggingFace schema
----------
    id: str   e.g. "7-980"
    question_stem: str; the question text  (note: different key from ARC's "question")
    choices: {text: [str, ...], label: [str, ...]} always A/B/C/D
    answerKey: str; "A" | "B" | "C" | "D"

The loader converts every raw example to the shared MCQExample dataclass so
that prompt_builder, evaluator, and runner work without modification.
"""

from datasets import load_dataset
from typing import List
import random

from data.arc_loader import MCQExample, format_mcq_prompt    # reuse shared types


class OBQALoader:
    """Load OpenBookQA dataset from HuggingFace"""

    def __init__(self, split: str = "test", cache_dir: str = "data/cache"):
        """
        Args:
            split: "train", "validation", or "test"
            cache_dir: local cache directory for downloaded data
        """
        self.split = split
        self.cache_dir = cache_dir
        self._dataset = None

    @property
    def dataset(self):
        """Lazy-load on first access."""
        if self._dataset is None:
            self._dataset = load_dataset(
                "allenai/openbookqa",
                "main",
                split=self.split,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        return self._dataset

    def load_all(self) -> List[MCQExample]:
        """Load all examples from the split."""
        return [self._parse_example(raw) for raw in self.dataset]

    def load_subset(self, n: int, seed: int = 42) -> List[MCQExample]:
        """
        Load a random subset of n examples.

        Args:
            n: number of examples to sample
            seed: random seed for reproducibility
        """
        random.seed(seed)
        indices = random.sample(range(len(self.dataset)), min(n, len(self.dataset)))
        return [self._parse_example(self.dataset[idx]) for idx in sorted(indices)]

    def _parse_example(self, raw: dict) -> MCQExample:
        """
        Convert a raw HuggingFace OpenBookQA record to MCQExample.

        The only structural difference from ARC is the question field name:
        OpenBookQA uses "question_stem" where ARC uses "question".
        Answer keys are always A/B/C/D so no numeric normalisation is needed.
        """
        return MCQExample(
            benchmark="OBQA",
            example_id=raw["id"],
            question=raw["question_stem"],          # OBQA-specific field name
            options=raw["choices"]["text"],
            option_labels=raw["choices"]["label"],
            correct_label=raw["answerKey"],
        )

    def __len__(self) -> int:
        return len(self.dataset)


if __name__ == "__main__":
    loader = OBQALoader(split="test")
    examples = loader.load_subset(5, seed=42)

    for ex in examples:
        print(f"\n{'='*80}")
        print(f"ID      : {ex.example_id}")
        print(f"Question: {ex.question}")
        print(f"Options : {ex.options}")
        print(f"Correct : {ex.correct_label} ({ex.correct_text})")
        print(format_mcq_prompt(ex, "question_first"))
