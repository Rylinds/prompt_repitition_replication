from enum import Enum
from dataclasses import dataclass

class PromptVariant(Enum):
    """Enumeration of prompt variants"""
    BASELINE = "baseline"
    REPEAT = "repeat"
    VERBOSE = "verbose"
    REPEAT_X3 = "repeat_x3"
    PADDING = "padding"

@dataclass
class PromptConfig:
    """Configuration for prompt variant generation"""
    include_instructions: bool = True
    instruction_text: str = "Answer the following question. Choose from the options provided"
    answer_format: str = "The answer is" # should respond 'The answer is X.'

class PromptBuilder:
    """
    Builds prompt variants for the reptition experiment.
    Ensures exact formatting as per the paper to avoid whitespace bugs.
    """

    def __init__(self, config: PromptConfig = None):
        self.config = config or PromptConfig()
    
    def build(
        self,
        query: str,
        variant: PromptVariant = PromptVariant.BASELINE,
        padding_token: str = "[PAD]"
    ) -> str:
        """
        Build a prompt variant

        Args:
            query: the question/prompt to format
            variant: which variant to generate
            padding_token: token to use for padding variant
        """
        if variant == PromptVariant.BASELINE:
            return self._build_baseline(query)
        elif variant == PromptVariant.REPEAT:
            return self._build_repeat(query)
        elif variant == PromptVariant.VERBOSE:
            return self._build_verbose(query)
        elif variant == PromptVariant.REPEAT_X3:
            return self._build_repeat_x3(query)
        elif variant == PromptVariant.PADDING:
            return self._build_padding(query, padding_token)
        else:
            raise ValueError(f"Unknown variant: {variant}")
    
    def _build_baseline(self, query: str) -> str:
        """
        Baseline: single query with instructions

        Format:
            [instruction]

            <query>
        """
        if self.config.include_instructions:
            return f"{self.config.instruction_text}\n\n{query}"
        return query
    
    def _build_repeat(self, query: str) -> str:
        """
        Repeat variant: exact duplication without separator

        Format:
            [instruction]

            <query><query>
        """
        repeated = f"{query}{query}"
        if self.config.include_instructions:
            return f"{self.config.instruction_text}\n\n{repeated}"
        return repeated
    
    def _build_verbose(self, query: str) -> str:
        """
        Verbose variant: repitition with separator

        Format:
            [instruction]

            <query>

            Let me repeat that:
            <query>
        """
        verbose = f"{query}\n\nLet me repeat that:\n{query}"
        if self.config.include_instructions:
            return f"{self.config.instruction_text}\n\n{verbose}"
        return verbose
    
    def _build_repeat_x3(self, query: str) -> str:
        """
        Repeat x3 variant: triple repitition without separators

        Format:
            [instruction]

            <query><query><query>
        """
        repeated = f"{query}{query}{query}"
        if self.config.include_instructions:
            return f"{self.config.instruction_text}\n\n{repeated}"
        return repeated
    
    def _build_padding(self, query: str, padding_token: str = "[PAD]") -> str:
        """
        Padding variant: repitition with padding tokens between

        Format:
            [instruction]

            <query> [PAD] [PAD] ... [PAD] <query>
        
        Thought: add tokens between repititions to maintain token count
        similar to other variants while testing if pure token count (not
        actual content) drives the effect
        """
        # 5 for middle ground because I'm poor
        padding = " ".join([padding_token] * 5)
        padded = f"{query} {padding} {query}"
        if self.config.include_instructions:
            return f"{self.config.instruction_text}\n\n{padded}"
        return padded
    
    def build_all_variants(self, query: str) -> dict:
        """
        Build all 5 variants at once for a given query

        Returns:
            Dict mapping variant names to formatted prompts
        """
        return {
            variant.value: self.build(query, variant)
            for variant in PromptVariant
        }

def format_mcq_prompt(
    question: str,
    options: list,
    variant: PromptVariant = PromptVariant.BASELINE,
    option_ordering: str = "alphabetic",
) -> str:
    """
    Made for pure convenience for formatting multiple-choice questions

    Args:
        question: question text
        options: list of option strings (ex: ['A) Option 1', 'B) Option 2'])
        variant: which prompt variant to use
        option_ordering: 'alphabetic' (default) or 'original' (no reordering)
    
    Returns:
        Formatted prompt ready to input into a model
    """
    # build full query including options
    options_text = "\n".join(options)
    full_query = f"{question}\n\n{options_text}\n\nProvide your answer in the format: The answer is X (where X is A, B, C, or D)."

    builder = PromptBuilder()
    return builder.build(full_query, variant=variant)

if __name__ == "__main__":
    # example usage
    test_query = "What is 2+2?"
    builder = PromptBuilder()

    print("=== BASELINE ===")
    print(builder.build(test_query, PromptVariant.BASELINE))
    print("\n=== REPEAT ===")
    print(builder.build(test_query, PromptVariant.REPEAT))
    print("\n=== VERBOSE ===")
    print(builder.build(test_query, PromptVariant.VERBOSE))
    print("\n=== REPEAT X3 ===")
    print(builder.build(test_query, PromptVariant.REPEAT_X3))
    print("\n=== PADDING ===")
    print(builder.build(test_query, PromptVariant.PADDING))

    # show lengths (I should calc budget with this)
    print("\n=== TOKEN COUNTS (approx) ===")
    all_variants = builder.build_all_variants(test_query)
    for name, prompt in all_variants.items():
        word_count = len(prompt.split())
        print(f"{name:12} : {word_count:3} words")
