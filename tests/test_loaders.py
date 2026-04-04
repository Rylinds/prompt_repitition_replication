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

def test_numeric_answer_key_normalisation():
    """
    NYSEDREGENTS questions use numeric keys 1/2/3/4 instead of A/B/C/D.
    The loader must map them to letters so the evaluator never sees a digit
    as a gold answer.
    """
    from data.arc_loader import ARCLoader
 
    # simulate exactly what the raw HuggingFace dataset returns for a
    # NYSEDREGENTS question: numeric labels and a numeric answerKey
    raw = {
        "id": "NYSEDREGENTS_mock",
        "question": "What is the primary source of energy for Earth's surface processes?",
        "choices": {
            "text":  ["The Sun", "The Moon", "Earth's core", "Ocean currents"],
            "label": ["1", "2", "3", "4"],
        },
        "answerKey": "1",
    }
 
    loader = ARCLoader.__new__(ARCLoader)
    example = loader._parse_example(raw)
 
    # option_labels must all be letters
    assert example.option_labels == ["A", "B", "C", "D"], (
        f"option_labels not normalised: {example.option_labels}"
    )
    # correct_label must be a letter
    assert example.correct_label == "A", (
        f"correct_label not normalised: {example.correct_label!r}"
    )
    # correct_label must still be in option_labels (internal consistency)
    assert example.correct_label in example.option_labels
    # correct_text must still resolve correctly
    assert example.correct_text == "The Sun"
 
    # check each digit -> letter mapping
    for digit, letter in [("1","A"), ("2","B"), ("3","C"), ("4","D")]:
        raw_variant = {**raw, "choices": {**raw["choices"], "label": ["1","2","3","4"]},
                       "answerKey": digit}
        ex = loader._parse_example(raw_variant)
        assert ex.correct_label == letter, f"Expected {digit} → {letter}, got {ex.correct_label}"
 
    # letter keys must pass through unchanged (regression guard)
    raw_letter = {**raw, "choices": {**raw["choices"], "label": ["A","B","C","D"]},
                  "answerKey": "C"}
    ex_letter = loader._parse_example(raw_letter)
    assert ex_letter.correct_label == "C"
    assert ex_letter.option_labels == ["A", "B", "C", "D"]

# python -m pytest tests/test_loaders.py -vs
