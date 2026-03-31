"""
Test zone for agents/prompt_builder.py

Validates:
1. Exact formatting matches specification
2. Repeat is exactly 2x baseline (no whitespace)
3. Verbose maintains content integrity
4. Repeat X3 is exactly x3 baseline
5. Padding maintains option structure
6. No encoding/unicode bugs
"""

import pytest
from agents.prompt_builder import(
    PromptBuilder,
    PromptVariant,
    PromptConfig,
    format_mcq_prompt
)


class TestPromptBuilderBaseline:
    """Test baseline variant generation"""

    def test_baseline_includes_instruction(self):
        builder = PromptBuilder()
        query = "What is the capital of France?"
        result = builder.build(query, PromptVariant.BASELINE)

        assert "Answer the following question" in result
        assert query in result
        assert result.count(query) == 1
    
    def test_baseline_no_instruction_when_disabled(self):
        builder = PromptBuilder(PromptConfig(include_instructions=False))
        query = "What is the capital of France?"
        result = builder.build(query, PromptVariant.BASELINE)

        assert result == query
        assert "Answer the following question" not in result
    
    def test_baseline_preserves_exact_query(self):
        builder = PromptBuilder()
        query = "Complex query:\n Option A) First\n Option B) Second"
        result = builder.build(query, PromptVariant.BASELINE)

        assert query in result

class TestPromptBuilderRepeat:
    """Test repeat (x2) variant"""

    def test_repeat_exact_duplication(self):
        builder = PromptBuilder()
        query = "What is 2+2?"
        result = builder.build(query, PromptVariant.REPEAT)

        assert "Answer the following question" in result
        assert result.count(query) == 2
    
    def test_repeat_no_sep(self):
        builder = PromptBuilder(PromptConfig(include_instructions=False))
        query = "TEST"
        result = builder.build(query, PromptVariant.REPEAT)

        assert result == "TESTTEST"
    
    def test_repeat_versus_baseline_length(self):
        builder = PromptBuilder()
        query = "What is the capital of France?"

        baseline = builder.build(query, PromptVariant.BASELINE)
        repeat = builder.build(query, PromptVariant.REPEAT)

        baseline_query_only = baseline.split("\n\n", 1)[1]
        repeat_query_only = repeat.split("\n\n", 1)[1]

        assert len(repeat_query_only) == 2 * len(baseline_query_only)
    
    def test_repeat_no_trailing_whitespace(self):
        builder = PromptBuilder(PromptConfig(include_instructions=False))
        query = "Q"
        result = builder.build(query, PromptVariant.REPEAT)

        assert result == "QQ", f"Excpected 'QQ' got '{result}'"
        assert not result.endswith(" ")
        assert not result.endswith("\n")

class TestPromptBuilderVerbose:
    """Test verbose variant with explicit separator"""

    def test_verbose_has_sep(self):
        builder = PromptBuilder()
        query = "What is 2+2?"
        result = builder.build(query, PromptVariant.VERBOSE)

        assert "Let me repeat that:" in result
        assert result.count(query) == 2
    
    def test_verbose_maintains_query(self):
        builder = PromptBuilder(PromptConfig(include_instructions=False))
        query = "Option A) First\nOption B) Second"
        result = builder.build(query, PromptVariant.VERBOSE)

        assert "Let me repeat that:" in result
        assert result.count(query) == 2
        # check newlines
        assert "Option A) First" in result
        assert "Option B) Second" in result

class TestPromptBuilderRepeatX3:
    """Test x3 repetition variant"""

    def test_repeat_x3_triples(self):
        builder = PromptBuilder(PromptConfig(include_instructions=False))
        query = "TEST"
        result = builder.build(query, PromptVariant.REPEAT_X3)

        assert result == "TESTTEST" + "TEST"
        assert result.count(query) == 3
    
    def test_repeat_x3_no_sep(self):
        builder = PromptBuilder(PromptConfig(include_instructions=False))
        query = "Q"
        result = builder.build(query, PromptVariant.REPEAT_X3)

        assert result == "QQQ"

    def test_repeat_x3_with_instruction(self):
        builder = PromptBuilder()
        query = "What is the capital of France?"
        result = builder.build(query, PromptVariant.REPEAT_X3)

        assert "Answer the following question" in result
        assert result.count(query) == 3

class TestPromptBuilderPadding:
    """Test padding variant"""

    def test_padding_has_pad_tokens(self):
        builder = PromptBuilder(PromptConfig(include_instructions=False))
        query = "TEST"
        result = builder.build(query, PromptVariant.PADDING, padding_token="[PAD]")

        assert "[PAD]" in result
        assert result.count("TEST") == 2
    
    def test_padding_token_customizable(self):
        builder = PromptBuilder(PromptConfig(include_instructions=False))
        query = "TEST"
        result = builder.build(query, PromptVariant.PADDING, padding_token="[X]")

        assert "[X]" in result
        assert "[PAD]" not in result
    
    def test_padding_maintains_structure(self):
        builder = PromptBuilder()
        query = "What is 2+2?"
        result = builder.build(query , PromptVariant.PADDING)

        assert "Answer the following question" in result
        assert "[PAD]" in result
        assert query in result

class TestBuildAllVariants:
    """Test batch building all the variants"""

    def test_build_all_variants_returns_five(self):
        builder = PromptBuilder()
        query = "Test Query"
        variants = builder.build_all_variants(query)

        assert len(variants) == 5
        assert "baseline" in variants
        assert "repeat" in variants
        assert "verbose" in variants
        assert "repeat_x3" in variants
        assert "padding" in variants
    
    def test_build_all_variants_consistent_individuals(self):
        builder = PromptBuilder()
        query = "Test Query"
        variants = builder.build_all_variants(query)

        for variant_enum in PromptVariant:
            individual = builder.build(query, variant_enum)
            batch = variants[variant_enum.value]
            assert individual == batch
    
class TestFormatMCQPrompt:
    """Test convenience function for multiple choice questions"""

    def test_format_mcq_baseline(self):
        question = "What is 2+2?"
        options = ["A) 3", "B) 4", "C) 5", "D) 6"]
        result = format_mcq_prompt(question, options, PromptVariant.VERBOSE)

        assert question in result
        assert all(opt in result for opt in options)
        assert "The answer is X" in result
    
    def test_format_mcq_repeat(self):
        question = "What is 2 + 2?"
        options = ["A) 3", "B) 4", "C) 5", "D) 6"]
        result = format_mcq_prompt(question, options, PromptVariant.REPEAT)
        
        full_query = f"{question}\n\nA) 3\nB) 4\nC) 5\nD) 6\n\nProvide your answer in the format: The answer is X (where X is A, B, C, or D)."
        # should appear twice
        assert result.count(full_query) == 2
    
    def test_format_mcq_verbose(self):
        question = "What is 2 + 2?"
        options = ["A) 3", "B) 4", "C) 5", "D) 6"]
        result = format_mcq_prompt(question, options, PromptVariant.VERBOSE)
        
        assert "Let me repeat that:" in result
        assert question in result

class TestEdgeCases:
    """Test edge cases and potential bugs"""

    def test_empty_query(self):
        """Empty query should still produce valid variants."""
        builder = PromptBuilder()
        result = builder.build("", PromptVariant.BASELINE)
        assert "Answer the following question" in result
    
    def test_query_with_special_characters(self):
        """Query with special chars should not break formatting."""
        builder = PromptBuilder(PromptConfig(include_instructions=False))
        query = "What is $100 × 2?"
        result = builder.build(query, PromptVariant.REPEAT)
        
        assert result == query + query
    
    def test_query_with_newlines(self):
        """Query containing newlines should be preserved exactly."""
        builder = PromptBuilder(PromptConfig(include_instructions=False))
        query = "Line 1\nLine 2\nLine 3\n"
        result = builder.build(query, PromptVariant.REPEAT)
        
        assert result == query + query
        assert result.count("\n") == 6  # 3 newlines * 2 repetitions
    
    def test_unicode_handling(self):
        """Unicode characters should be handled correctly."""
        builder = PromptBuilder(PromptConfig(include_instructions=False))
        query = "What is the capital of France? (Français: Paris) 🇫🇷"
        result = builder.build(query, PromptVariant.REPEAT)
        
        assert result == query + query
        assert "Paris" in result
        assert "🇫🇷" in result
    
    def test_very_long_query(self):
        """Long queries should not cause formatting issues."""
        builder = PromptBuilder(PromptConfig(include_instructions=False))
        query = "Question: " + "option text " * 500
        result = builder.build(query, PromptVariant.REPEAT)
        
        assert result == query + query
        assert len(result) == 2 * len(query)
    
    def test_invalid_variant_raises_error(self):
        """Passing an invalid variant should raise ValueError."""
        builder = PromptBuilder()
        with pytest.raises(ValueError):
            builder.build("test", "invalid_variant")

class TestQuantitativeProperties:
    """Test quantitative properties of variants."""
    
    def test_repeat_is_approximately_double_baseline(self):
        """Character count of repeat should be ~2x baseline (ignoring instruction)."""
        builder = PromptBuilder(PromptConfig(include_instructions=False))
        query = "This is a medium-length query testing character counts."
        
        baseline = builder.build(query, PromptVariant.BASELINE)
        repeat = builder.build(query, PromptVariant.REPEAT)
        
        assert len(repeat) == 2 * len(baseline)
    
    def test_repeat_x3_is_triple_baseline(self):
        """Character count of x3 should be exactly 3x baseline."""
        builder = PromptBuilder(PromptConfig(include_instructions=False))
        query = "Test query for length comparison."
        
        baseline = builder.build(query, PromptVariant.BASELINE)
        repeat_x3 = builder.build(query, PromptVariant.REPEAT_X3)
        
        assert len(repeat_x3) == 3 * len(baseline)
    
    def test_verbose_slightly_longer_than_repeat(self):
        """Verbose should be longer than repeat due to separator text."""
        builder = PromptBuilder(PromptConfig(include_instructions=False))
        query = "Test query"
        
        repeat = builder.build(query, PromptVariant.REPEAT)
        verbose = builder.build(query, PromptVariant.VERBOSE)
        
        assert len(verbose) > len(repeat)
    
    def test_padding_reasonable_length(self):
        """Padding should add moderate number of tokens between repetitions."""
        builder = PromptBuilder(PromptConfig(include_instructions=False))
        query = "Test"
        
        repeat = builder.build(query, PromptVariant.REPEAT, padding_token="[PAD]")
        padding = builder.build(query, PromptVariant.PADDING, padding_token="[PAD]")
        
        # padding should be longer than repeat but not too much ig
        assert len(padding) > len(repeat)

# python -m pytest tests/test_prompt_builder.py -vs
