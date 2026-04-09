"""
GSM8K loader for the prompt repetition experiment.

Dataset: openai/gsm8k (HuggingFace)
Config: "main"
Split: "test"  (1,319 questions)

HuggingFace schema
----------
    question: str; the math word problem
    answer: str; full worked solution ending with "#### <integer>"
                    ex: "She has 3 cats.\n#### 3"

This benchmark is different from ARC and OBQA:
    - No multiple-choice options
    - Gold answer is a numeric value, not a letter
    - The evaluator must be called with is_math=True

MathExample is a separate dataclass from MCQExample because the data shape
is different enough that forcing it into MCQExample would require awkward
empty fields. The runner detects GSM8K by benchmark name and sets is_math=True.
"""

from dataclasses import dataclass
from datasets import load_dataset
from typing import List
import random
import re

@dataclass
class MathExample:
    """
    A single GSM8K math word problem.

    Attr
    ----------
    benchmark: always GSM8K
    example_id: positional index as a string (GSM8K has no IDs)
    question: word problem text
    full_answer: complete worked solution from the dataset
    gold_answer: extracted final numeric answer (ex: '42') derived from '#### <n>'
                 in full answer
    """
    benchmark: str
    example_id: str
    question: str
    full_answer: str

    _ANSWER_RE = re.compile(r"####\s*([\d,]+(?:\.\d+)?)")

    @property
    def gold_answer(self):
        """
        Extract and normalise the numeric answer from the full solution.

        Strips commas and converts float-valued intsj so that the evaluator's
        _normalize_number() will always find a match.
        """
        match = self._ANSWER_RE.search(self.full_answer)
        if not match:
            raise ValueError(
                f"No '#### <answer>' marjer found in GSM8K example {self.example_id}."
                f"Full answer: {self.full_answer!r}"
            )
        raw = match.group(1).replace(",", "")
        try:
            return str(int(float(raw)))
        except ValueError:
            return raw

def format_math_prompt(example: MathExample) -> str:
    """
    Format MathExample as a prompt str for the model

    GSM8K has no option ordering variant. The prompt instructs the model to solve
    the answer and end with the answer in the canonical '#### <n>' format that the
    evaluator expects.

    NOTE:
        debating saying: 'solve the following math problem step by step'
                         'show your reasoning, then write the final answer on a new line'
        but this could be dangerous -> noisy thinking when this is trying to be
        explicitly non-reasoning.
    """
    return (
        f"Solve the following math problem.\n\n"
        f"{example.question}\n\n"
        f"Write your final answer on a new line "
        f"in the format: #### <number>"
    )

class GSM8KLoader:
    """Load GSM8K dataset from HuggingFace"""

    def __init__(self, split="test", cache_dir: str = "data/cache"):
        """
        Args:
            split: train or test
            cache_dir: local cache directory for downloaded data
        """
        self.split = split
        self.cache_dir = cache_dir
        self._dataset = None
    
    @property
    def dataset(self):
        """Lazy load on first access"""
        if self._dataset is None:
            self._dataset = load_dataset(
                "openai/gsm8k",
                "main",
                split=self.split,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        return self._dataset

    def load_all(self) -> List[MathExample]:
        """Load all examples from the split"""
        return [self._parse_example(raw, idx) for idx, raw in enumerate(self.dataset)]

    def load_subset(self, n: int, seed: int = 42) -> List[MathExample]:
        """
        Load a random subset of n examples

        Args:
            n: number of examples to sample
            seed: random seed for reproducibility
        """
        random.seed(seed)
        indices = random.sample(range(len(self.dataset)), min(n, len(self.dataset)))
        return [self._parse_example(self.dataset[idx], idx) for idx in sorted(indices)]

    def _parse_example(self, raw: dict, idx: int) -> MathExample:
        """
        Covert raw HuggingFace GSM8K record to MathExample

        GSM8K has no named IDs so use the dataset index as example_id zero-padded to
        4 digits for stable sorting in result filenames.
        """
        return MathExample(
            benchmark="GSM8K",
            example_id=f"gsm8k_{idx:04d}",
            question=raw["question"],
            full_answer=raw["answer"],
        )

    def __len__(self) -> int:
        return len(self.dataset)

# quick test
if __name__ == "__main__":
    loader = GSM8KLoader(split="test")
    examples = loader.load_subset(3, seed=42)

    for ex in examples:
        print(f"\n{'='*80}")
        print(f"ID          : {ex.example_id}")
        print(f"Question    : {ex.question[:120]}...")
        print(f"Gold answer : {ex.gold_answer}")
        print(f"\nPrompt:\n{format_math_prompt(ex)}")
