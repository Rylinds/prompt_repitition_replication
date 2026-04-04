"""
The paper suggests that the model's attention may weight earlier tokesn more heavily.
By reversing the order (question-first vs option-first), I can test whether the benefit
of prompt repition holds regardless of prompt structure.
"""
from datasets import load_dataset
from dataclasses import dataclass
from typing import List
import random

@dataclass
class MCQExample:
    """Standardized multiple-choice question format"""
    benchmark: str                      # ARC
    example_id: str                     # Mercury_SC_####
    question: str                       # A student observes a green plant...
    options: list[str]                  # [choice A text, choice B text, ...]
    option_labels: list[str]            # [A, B, ...]
    correct_label: str                  # B

    @property
    def correct_text(self) -> str:
        """Return the text of the correct option"""
        idx = self.option_labels.index(self.correct_label)
        return self.options[idx]

class ARCLoader:
    """Load ARC-Challenge dataset from HuggingFace"""

    def __init__(self, split: str="test", cache_dir: str="data/cache"):
        """
        Args:
            split: 'train', 'validation', or 'test'
            cache_dir: where to cache downloaded data
        """
        self.split = split
        self.cache_dir = cache_dir
        self._dataset = None
    
    @property
    def dataset(self):
        """Lazy-load the dataset on first access"""
        if self._dataset is None:
            self._dataset = load_dataset(
                "allenai/ai2_arc",
                "ARC-Challenge",
                split=self.split,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
        return self._dataset
    
    def load_all(self) -> List[MCQExample]:
        """Load all examples from the split"""
        examples = []
        for raw_example in self.dataset:
            example = self._parse_example(raw_example)
            examples.append(example)
        return examples
    
    def load_subset(self, n: int, seed: int = 42) -> List[MCQExample]:
        """
        Load a random subset of n examples

        Args:
            n: number of examples to use
            seed: random seed for reproducibility
        """
        random.seed(seed)
        indices = random.sample(range(len(self.dataset)), min(n, len(self.dataset)))
        examples = []
        for idx in sorted(indices):
            raw_example = self.dataset[idx]
            example = self._parse_example(raw_example)
            examples.append(example)
        return examples
    
    # a small subset of ARC questions (NYSEDREGENTS) store their
    # answer keys as digits 1/2/3/4 -> option_labels and answerKey can be
    # affected, so normalise both so the rest of the pipeline always
    # see the letter keys A/B/C/D
    _KEY_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}

    @classmethod
    def _normalize_key(cls, key: str) -> str:
        """Map numeric answers to letters; pass letters through unchanged"""
        return cls._KEY_MAP.get(key, key)
    
    def _parse_example(self, raw_example: dict) -> MCQExample:
        """Convert raw HuggingFace example to MCQExample"""
        raw_labels = raw_example["choices"]["label"]
        
        return MCQExample(
            benchmark="ARC",
            example_id=raw_example["id"],
            question=raw_example["question"],
            options=raw_example["choices"]["text"],
            option_labels=[self._normalize_key(l) for l in raw_labels],
            correct_label=self._normalize_key(raw_example["answerKey"]),
        )
    
    def __len__(self) -> int:
        """Total number of examples in a split"""
        return len(self.dataset)

def format_mcq_prompt(example: MCQExample, ordering: str = "question_first") -> str:
    """
    Format an MCQExample as a prompt string.

    Args:
        example: MCQExample instance
        ordering: question_first or options_first
    
    Returns:
        Formatted prompt string (without the 'The answer is' suffix)
    """
    options_str = "\n".join(
        f"{label}) {text}"
        for label, text in zip(example.option_labels, example.options)
    )

    if ordering == "question_first":
        return f"Question: {example.question}\n\n{options_str}\n"
    elif ordering == "options_first":
        return f"{options_str}\n\nQuestion: {example.question}\n"
    else:
        raise ValueError(f"Unknown ordering: {ordering}")

if __name__ == "__main__":
    # example usage
    loader = ARCLoader(split="test")

    # load 5 examples
    examples = loader.load_subset(5, seed=42)

    for example in examples:
        print(f"\n{'='*80}")
        print(f"ID: {example.example_id}")
        print(f"Question: {example.question}")
        print(f"Options: {example.options}")
        print(f"Correct: {example.correct_label} ({example.correct_text})")
    
        # show prompt formatting
        prompt_qf = format_mcq_prompt(example, "question_first")
        print(f"\nQuestion-First Ordering:\n{prompt_qf}")

        prompt_of = format_mcq_prompt(example, "options_first")
        print(f"Options-First Ordering:\n{prompt_of}")
