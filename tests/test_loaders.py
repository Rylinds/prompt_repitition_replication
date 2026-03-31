import pytest
from data.arc_loader import ARCLoader, MCQExample, format_mcq_prompt

def test_arc_loader_structure():
    """Verify dataset loads and has expected schema"""
    loader = ARCLoader(split="test")
    examples = loader.load_subset(5, seed=42)

    assert len(examples) == 5
    for ex in examples:
        assert ex.benchmark == "ARC"
        assert len(ex.options) == 4
        assert len(ex.option_labels) == 4
        assert ex.correct_label in ex.option_labels
        assert ex.correct_text in ex.options

def test_arc_loader_reproducibility():
    """Verify that the random seed yields the same results"""
    loader = ARCLoader(split="test")
    set_a = loader.load_subset(3, seed=123)
    set_b = loader.load_subset(3, seed=123)
    
    for a, b in zip(set_a, set_b):
        assert a.example_id == b.example_id

def test_prompt_formatting():
    """Verify both ordering types are formatted correctly"""
    # mock example
    example = MCQExample(
        benchmark="ARC",
        example_id="test_123",
        question="What is 2+2?",
        options=["3", "4", "5", "6"],
        option_labels=["A", "B", "C", "D"],
        correct_label="B"
    )

    qf = format_mcq_prompt(example, "question_first")
    assert "Question: What is 2+2?" in qf
    assert "A) 3" in qf

    of = format_mcq_prompt(example, "options_first")
    assert "Question: What is 2+2?" in of
    assert of.find("A) 3") < of.find("Question: What is 2+2?") # option should come first
    assert of.strip().endswith("What is 2+2?")

# python -m pytest tests/test_loaders.py -vs
