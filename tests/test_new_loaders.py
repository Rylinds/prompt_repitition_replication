"""
Unit tests for obqa and gsm8k loaders
"""

import pytest
from data.arc_loader import MCQExample, format_mcq_prompt

def make_obqa_raw(
        id_="7-9890",
        question_stem="Which of the following is a source of light?",
        texts=("The Sun", "A rock", "Water", "Soil"),
        labels=("A", "B", "C", "D"),
        answer_key = "A",
):
    return {
        "id":id_,
        "question_stem": question_stem,
        "choices": {"text": list(texts), "label": list(labels)},
        "answerKey": answer_key
    }

def make_gsm8k_raw(
    question="Janet has 3 cats. She buys 2 more. How many cats does she have?",
    answer="She started with 3 cats and bought 2 more.\n3 + 2 = 5\n#### 5",
):
    return {"question": question, "answer": answer}


# OBQALoader unit tests (no download needed)
class TestOBQALoaderParse:

    @pytest.fixture
    def loader(self):
        from data.obqa_loader import OBQALoader
        return OBQALoader.__new__(OBQALoader)
    
    def test_returns_mcq_example(self, loader):
        ex = loader._parse_example(make_obqa_raw())
        assert isinstance(ex, MCQExample)
    
    def test_benchmark_label(self, loader):
        ex = loader._parse_example(make_obqa_raw())
        assert ex.benchmark == "OBQA"
    
    def test_question_stem_mapped_to_question(self, loader):
        """OBQA uses 'question_stem'; must land in MCQExample.question"""
        ex = loader._parse_example(make_obqa_raw(question_stem="What is 2+2?"))
        assert ex.question == "What is 2+2?"

    def test_id_preserved(self, loader):
        ex = loader._parse_example(make_obqa_raw(id_="test-001"))
        assert ex.example_id == "test-001"

    def test_options_and_labels(self, loader):
        ex = loader._parse_example(make_obqa_raw())
        assert ex.options == ["The Sun", "A rock", "Water", "Soil"]
        assert ex.option_labels == ["A", "B", "C", "D"]

    def test_correct_label(self, loader):
        ex = loader._parse_example(make_obqa_raw(answer_key="C"))
        assert ex.correct_label == "C"

    def test_correct_label_in_option_labels(self, loader):
        for letter in "ABCD":
            ex = loader._parse_example(make_obqa_raw(answer_key=letter))
            assert ex.correct_label in ex.option_labels

    def test_correct_text_resolves(self, loader):
        ex = loader._parse_example(make_obqa_raw(answer_key="B"))
        assert ex.correct_text == "A rock"

    def test_all_four_answer_keys(self, loader):
        texts = ("Opt1", "Opt2", "Opt3", "Opt4")
        for letter, expected_text in zip("ABCD", texts):
            ex = loader._parse_example(make_obqa_raw(texts=texts, answer_key=letter))
            assert ex.correct_label == letter
            assert ex.correct_text == expected_text

    def test_prompt_question_first(self, loader):
        ex = loader._parse_example(make_obqa_raw(question_stem="What glows?"))
        prompt = format_mcq_prompt(ex, "question_first")
        assert "What glows?" in prompt
        assert prompt.index("What glows?") < prompt.index("A)")

    def test_prompt_options_first(self, loader):
        ex = loader._parse_example(make_obqa_raw(question_stem="What glows?"))
        prompt = format_mcq_prompt(ex, "options_first")
        assert "What glows?" in prompt
        assert prompt.index("A)") < prompt.index("What glows?")

    def test_invalid_ordering_raises(self, loader):
        ex = loader._parse_example(make_obqa_raw())
        with pytest.raises(ValueError):
            format_mcq_prompt(ex, "random_order")


class TestOBQALoaderReproducibility:
    """seed based reproducibility -> use __new__ trick"""

    def test_same_seed_indices(self):
        from data.obqa_loader import OBQALoader
        import random

        fake_dataset = [make_obqa_raw(id_=str(i)) for i in range(50)]

        random.seed(7)
        indices_a = random.sample(range(50), 5)
        random.seed(7)
        indices_b = random.sample(range(50), 5)

        assert indices_a == indices_b
    
    def test_different_seeds_different_indices(self):
        import random
        random.seed(1)
        a = random.sample(range(100), 10)
        random.seed(2)
        b = random.sample(range(100), 10)
        assert a != b

#@pytest.mark.integration
class TestOBQALoaderIntegration:
    """Requires caches HuggingFace dataset"""

    def test_load_subset_returns_correct_count(self):
        from data.obqa_loader import OBQALoader
        loader = OBQALoader(split="test")
        examples = loader.load_subset(5, seed=42)
        assert len(examples) == 5

    def test_all_examples_valid(self):
        from data.obqa_loader import OBQALoader
        loader = OBQALoader(split="test")
        for ex in loader.load_subset(10, seed=42):
            assert ex.benchmark == "OBQA"
            assert len(ex.options) == 4
            assert ex.correct_label in ex.option_labels
            assert ex.correct_text in ex.options

    def test_reproducibility(self):
        from data.obqa_loader import OBQALoader
        loader = OBQALoader(split="test")
        a = loader.load_subset(5, seed=99)
        b = loader.load_subset(5, seed=99)
        assert [e.example_id for e in a] == [e.example_id for e in b]


# GSM8KLoader unit tests (no download needed)
class TestGSM8KLoaderParse:

    @pytest.fixture
    def loader(self):
        from data.gsm8k_loader import GSM8KLoader
        return GSM8KLoader.__new__(GSM8KLoader)

    def test_returns_math_example(self, loader):
        from data.gsm8k_loader import MathExample
        ex = loader._parse_example(make_gsm8k_raw(), idx=0)
        assert isinstance(ex, MathExample)
    
    def test_benchmark_label(self, loader):
        ex = loader._parse_example(make_gsm8k_raw(), idx=0)
        assert ex.benchmark == "GSM8K"
    
    def test_example_id_format(self, loader):
        """ID should be zero-padded 4-digit index"""
        for idx, expected in [(0, "gsm8k_0000"), (7, "gsm8k_0007"), (123, "gsm8k_0123")]:
            ex = loader._parse_example(make_gsm8k_raw(), idx=idx)
            assert ex.example_id == expected

    def test_question_preserved(self, loader):
        q = "How many apples does Sara have?"
        ex = loader._parse_example(make_gsm8k_raw(question=q), idx=0)
        assert ex.question == q

    def test_full_answer_preserved(self, loader):
        ans = "She had 5.\n#### 5"
        ex = loader._parse_example(make_gsm8k_raw(answer=ans), idx=0)
        assert ex.full_answer == ans


class TestMathExampleGoldAnswer:

    @pytest.fixture
    def loader(self):
        from data.gsm8k_loader import GSM8KLoader
        return GSM8KLoader.__new__(GSM8KLoader)
    
    def _ex(self, loader, answer_str):
        return loader._parse_example(make_gsm8k_raw(answer=answer_str), idx=0)
    
    def test_simple_integer(self, loader):
        ex = self._ex(loader, "3 + 2 = 5\n##### 5")
        assert ex.gold_answer == "5"
    
    def test_large_number_with_comma(self, loader):
        """Comma-formatted numbers must be normalised"""
        ex = self._ex(loader, "#### 1,024")
        assert ex.gold_answer == "1024"

    def test_float_answer_normalised_to_int(self, loader):
        """GSM8K answers are always integers; 42.0 -> '42'"""
        ex = self._ex(loader, "#### 42.0")
        assert ex.gold_answer == "42"

    def test_answer_after_multiline_reasoning(self, loader):
        answer = (
            "She started with 10 apples.\n"
            "She gave away 3.\n"
            "10 - 3 = 7\n"
            "#### 7"
        )
        ex = self._ex(loader, answer)
        assert ex.gold_answer == "7"

    def test_missing_marker_raises(self, loader):
        ex = self._ex(loader, "The answer is somewhere in here but no marker.")
        with pytest.raises(ValueError, match="####"):
            _ = ex.gold_answer

    def test_zero_answer(self, loader):
        ex = self._ex(loader, "Nothing remains.\n#### 0")
        assert ex.gold_answer == "0"


#@pytest.mark.integration
class TestGSM8KLoaderIntegration:
    """Required cached HuggingFace dataset"""

    def test_load_subset_count(self):
        from data.gsm8k_loader import GSM8KLoader
        loader = GSM8KLoader(split="test")
        examples = loader.load_subset(5, seed=42)
        assert len(examples) == 5

    def test_all_examples_valid(self):
        from data.gsm8k_loader import GSM8KLoader
        loader = GSM8KLoader(split="test")
        for ex in loader.load_subset(10, seed=42):
            assert ex.benchmark == "GSM8K"
            assert ex.question
            assert ex.gold_answer.lstrip("-").isdigit()

    def test_reproducibility(self):
        from data.gsm8k_loader import GSM8KLoader
        loader = GSM8KLoader(split="test")
        a = loader.load_subset(5, seed=42)
        b = loader.load_subset(5, seed=42)
        assert [e.example_id for e in a] == [e.example_id for e in b]

    def test_different_seeds_differ(self):
        from data.gsm8k_loader import GSM8KLoader
        loader = GSM8KLoader(split="test")
        a = loader.load_subset(10, seed=1)
        b = loader.load_subset(10, seed=2)
        assert [e.example_id for e in a] != [e.example_id for e in b]


#if __name__ == "__main__":
    #pytest.main([__file__, "-v", "-m", "not integration"])
  
# python -m pytest tests/test_loaders.py -vs