"""
Test zone for experiments/evaluator.py

Each class maps to one logical concern so failures pinpoint exactly which
parsing branch broken. Tests are ordered to mirror cascade in _evaluate_mcq:
1. Extract format (_EXACT_PATTERN)
2. Fallback - delimited search          (B) of _LETTER_PATTERN
3. Fallback - keyword branch            (K) of _LETTER_PATTERN    
4. Fallback - isolated branch           (Is) of _LETTER_PATTERN
5. Ambiguous (multiple unique letters found)
6. Unparseable (no letter found)
7. GSM8K math eval
8. Gold answer normalization
9. Metadata propagation (question_id, variant)
10. evaluate_paur() contract
11. _normalize_number() unit tests
"""

import pytest
from experiments.evaluator import Evaluator, EvalResult, ParseStatus, evaluate_pair


# shared fixture
@pytest.fixture
def ev():
    return Evaluator()

# 1. Exact format - 'The answer is X.'
class TestExactFormat:

    def test_standard_correct(self, ev):
        r = ev.evaluate("The answer is B.", "B")

        assert r.parse_status == ParseStatus.EXACT_FORMAT
        assert r.is_correct is True
        assert r.predicted == "B"

    def test_standard_wrong(self, ev):
        r = ev.evaluate("The answer is C.", "B")

        assert r.parse_status == ParseStatus.EXACT_FORMAT
        assert r.is_correct is False
        assert r.predicted == "C"
    
    def test_lowercase_response(self, ev):
        """Case insensitive: 'the answer is b.' must still match"""
        r = ev.evaluate("the answer is b.", "B")
        
        assert r.parse_status == ParseStatus.EXACT_FORMAT
        assert r.predicted == "B"
    
    def test_no_trailing_period(self, ev):
        """Period is not required since the pattern doesn't enforce it"""
        r = ev.evaluate("The answer is B", "B")

        assert r.parse_status == ParseStatus.EXACT_FORMAT
        assert r.predicted == "B"
    
    def test_answer_buried(self, ev):
        """Exact pattern should match even if there's extra text after"""
        response = "Let me think step by step...\n\nThe answer is D. This is because..."
        r = ev.evaluate(response, "D")

        assert r.parse_status == ParseStatus.EXACT_FORMAT
        assert r.predicted == "D"
    
    def test_letters_a_to_j(self, ev):
        """MMLU-Pro uses up to option J so all need to be recognized"""
        for letter in "ABCDEFGHIJ":
            r = ev.evaluate(f"The answer is {letter}.", letter)

            assert r.parse_status == ParseStatus.EXACT_FORMAT, f"Failed for letter {letter}"
            assert r.predicted == letter
    
    def test_letter_beyond_j_not_matched(self, ev):
        """K is outside the valid range so go to fallback"""
        r = ev.evaluate("The answer is K.", "K")

        # K is not in A-J so _EXACT_PATTERN won't catch it
        # result will NOT be _EXACT_FORMAT
        assert r.parse_status != ParseStatus.EXACT_FORMAT
    
    def test_exact_over_fallback(self, ev):
        """
        When exact format is present, fallback must not fire even if
        other lettter appear elsewhere in the response
        """
        response = "I considered option A but the answer is B."
        r = ev.evaluate(response, "B")

        assert r.parse_status == ParseStatus.EXACT_FORMAT
        assert r.predicted == "B"


# 2. Fallback - delimited branch (letter immediately before ')')
class TestFallbackDelimited:

    def test_letter_before_paren_correct(self, ev):
        r = ev.evaluate("I think it's B) Mitochondria.", "B")

        assert r.parse_status == ParseStatus.FALLBACK_REGEX
        assert r.predicted == "B"
        assert r.is_correct is True
    
    def test_letter_before_paren_wrong(self, ev):
        r = ev.evaluate("I think it's C) Nucleus.", "B")

        assert r.parse_status == ParseStatus.FALLBACK_REGEX
        assert r.predicted == "C"
        assert r.is_correct is False
    
    def test_lowercase_letter(self, ev):
        r = ev.evaluate("Probably b) is correct", "B")

        assert r.parse_status == ParseStatus.FALLBACK_REGEX
        assert r.predicted == "B"
    
    def test_i_before_paren(self, ev):
        """
        'I' is excluded from the ISOLATED branch but is valid when delimited
        ex: MMLU-PRO option I) should be catchable via the delimited branch
        """
        r = ev.evaluate("The best option is I) None of the above.", "I")

        assert r.parse_status == ParseStatus.FALLBACK_REGEX
        assert r.predicted == "I"


# 3. Fallback - keyword branch (answer/option/choice/correct + letter)
class TestFallbackKeyword:

    def test_option_keyword(self, ev):
        r = ev.evaluate("option B is the most accurate.", "B")

        assert r.parse_status == ParseStatus.FALLBACK_REGEX
        assert r.predicted == "B"
    
    def test_choice_keyword_colon(self, ev):
        r = ev.evaluate("choice: C", "C")

        assert r.parse_status == ParseStatus.FALLBACK_REGEX
        assert r.predicted == "C"
    
    def test_correct_keyword(self, ev):
        r = ev.evaluate("The correct answer is D.", "D")

        # Note: _EXACT_PATTERN requires exactly 'The answer is X.'
        # 'correct answer' doesn't match it so I expect the keyword branch
        assert r.parse_status == ParseStatus.FALLBACK_REGEX
        assert r.predicted == "D"
    
    def test_answer_keyword_equals(self, ev):
        r = ev.evaluate("answer = A", "A")

        assert r.parse_status == ParseStatus.FALLBACK_REGEX
        assert r.predicted == "A"
    
    def test_keyword_lowercase(self, ev):
        r = ev.evaluate("option d seems right", "D")

        assert r.parse_status == ParseStatus.FALLBACK_REGEX
        assert r.predicted == "D"


# 4. Fallback - isolated branch (bare uppercase letter, no I)
class TestFallbackIsolated:

    def test_isolated_letter_mid_sentence(self, ev):
        """Single unambiguous isolated letter should resolve as fallback"""
        r = ev.evaluate("Select B fro the options", "B")

        assert r.parse_status == ParseStatus.FALLBACK_REGEX
        assert r.predicted == "B"
    
    def test_pronoun_i_never_captured(self, ev):
        """'I' must be excluded since it's an english pronoun, not a signal"""
        r = ev.evaluate("I think the question is ambiguous.", "A")

        assert r.predicted != "I"
        # with no other signals, should be unparseable
        assert r.parse_status == ParseStatus.UNPARSEABLE

    def test_i_in_option_context(self, ev):
        """Even 'I' surrounded by spaces must not be captured by the isolated branch"""
        r = ev.evaluate("Neither A nor I make sense here.", "A")

        # A is isolated and valid; I should be excluded
        # so only A found -> fallback, not ambiguous
        assert r.parse_status == ParseStatus.FALLBACK_REGEX
        assert r.predicted == "A"
    
    def test_letter_inside_word_not_captured(self, ev):
        """Letters inside words (ex: Because) must not be captured"""
        r = ev.evaluate("Because the nucleus controls the cell", "A")

        assert r.parse_status == ParseStatus.UNPARSEABLE
    
    def test_all_isolated_letters_a_to_j(self, ev):
        """Every valid isolated letter except I should be captured"""
        for letter in "ABCDEFGHJ":
            response = f"The best choice here seems to be {letter} overall."
            r = ev.evaluate(response, letter)

            assert r.parse_status == ParseStatus.FALLBACK_REGEX, f"Failed for {letter}"
            assert r.predicted == letter


# 5. Ambiguous - multiple distinct letters found
class TestAmbiguous:

    def test_two_isolated_letters(self, ev):
        r = ev.evaluate("Both A and C seem right.", "A")

        assert r.parse_status == ParseStatus.AMBIGUOUS
        assert r.is_correct is False
        assert r.predicted is None
    
    def test_two_deliminted_letters(self, ev):
        r = ev.evaluate("Either B) or D) would workd.", "B")

        assert r.parse_status == ParseStatus.AMBIGUOUS
        assert r.predicted is None
    
    def test_same_letter_repeated_not_ambig(self, ev):
        """Dedup logic: seeing 'B' twice should resolve as fallback, not ambiguous"""
        r = ev.evaluate("B) is correct. I'll go with B).", "B")

        assert r.parse_status == ParseStatus.FALLBACK_REGEX
        assert r.predicted == "B"
    
    def test_ambig_never_marked_correct(self, ev):
        """Even if the gold answer is among candidates, ambig -> is_correct = False"""
        r = ev.evaluate("Could be A or B", "A")

        assert r.parse_status == ParseStatus.AMBIGUOUS
        assert r.is_correct is False
    
    def test_three_letters(self, ev):
        r = ev.evaluate("A, B, or C could all be valid.", "A")

        assert r.parse_status == ParseStatus.AMBIGUOUS


# 6. Unparseable - no valid letter found at all
class TestUnparseable:

    def test_purpose(self, ev):
        r = ev.evaluate("None of the above makes sense.", "A")

        assert r.parse_status == ParseStatus.UNPARSEABLE
        assert r.is_correct is False
    
    def test_pronoun_i(self, ev):
        r = ev.evaluate("I really cannot determine the answer.", "A")

        assert r.parse_status == ParseStatus.UNPARSEABLE

    def test_numbers(self, ev):
        r = ev.evaluate("The answer is 42.", "A")

        # is_math = False so numeric response can't be parsed as MCQ
        assert r.parse_status == ParseStatus.UNPARSEABLE
    
    def test_empty_response(self, ev):
        r = ev.evaluate("", "A")

        assert r.parse_status == ParseStatus.UNPARSEABLE
        assert r.predicted is None
    
    def test_letter_beyond_valid_range(self, ev):
        """Letter K is outside A-J so should not be captured and response is unparseable"""
        r = ev.evaluate("The answer is K.", "K")

        assert r.parse_status != ParseStatus.EXACT_FORMAT
        # out of range so unparseable
        assert r.parse_status == ParseStatus.UNPARSEABLE
    
    def test_unparseable_never_correct(self, ev):
        r = ev.evaluate("I have no idea", "A")

        assert r.is_correct is False


# 7. GSM8K math eval
class TestGSM8K:

    def test_gsm8k_canonical_format(self, ev):
        r = ev.evaluate("So the total is 42.\n#### 42", "42", is_math=True)

        assert r.parse_status == ParseStatus.NUMERIC_MATCH
        assert r.is_correct is True
        assert r.predicted == "42"
 
    def test_gsm8k_with_comma_in_number(self, ev):
        r = ev.evaluate("#### 1,024", "1024", is_math=True)

        assert r.is_correct is True
        assert r.predicted == "1024"
 
    def test_answer_is_format(self, ev):
        r = ev.evaluate("The answer is 100.", "100", is_math=True)

        assert r.parse_status == ParseStatus.NUMERIC_MATCH
        assert r.is_correct is True
 
    def test_equals_format(self, ev):
        r = ev.evaluate("Total = 256", "256", is_math=True)

        assert r.parse_status == ParseStatus.NUMERIC_MATCH
        assert r.predicted == "256"
 
    def test_wrong_numeric_answer(self, ev):
        r = ev.evaluate("#### 99", "100", is_math=True)

        assert r.parse_status == ParseStatus.NUMERIC_MATCH
        assert r.is_correct is False
        assert r.predicted == "99"
 
    def test_float_normalised_to_int(self, ev):
        """GSM8K answers are always ints; 42.0 should normalise to '42'"""
        r = ev.evaluate("#### 42.0", "42", is_math=True)

        assert r.is_correct is True
        assert r.predicted == "42"
 
    def test_gold_with_comma_normalised(self, ev):
        """Gold answer supplied with comma should still match"""
        r = ev.evaluate("#### 1000", "1,000", is_math=True)

        assert r.is_correct is True
 
    def test_no_number_in_response(self, ev):
        r = ev.evaluate("I'm not sure how to solve this.", "42", is_math=True)

        assert r.parse_status == ParseStatus.UNPARSEABLE
        assert r.is_correct is False
        assert r.predicted is None
 
    def test_is_math_false_ignores_numeric(self, ev):
        """When is_math = False, numeric-only responses must NOT be parsed as MCQ answers"""
        r = ev.evaluate("#### 42", "42", is_math=False)

        assert r.parse_status == ParseStatus.UNPARSEABLE


# 8. Gold answer normalisation
class TestGoldNormalisation:

    def test_lowercase_gold_normalised(self, ev):
        r = ev.evaluate("The answer is B.", "b")

        assert r.gold == "B"
        assert r.is_correct is True
    
    def test_gold_whitespace(self, ev):
        r = ev.evaluate("The answer is B.", " B ")

        assert r.gold == "B"
        assert r.is_correct is True
    
    def test_math_gold_comma(self, ev):
        r = ev.evaluate("#### 1000", "1,000", is_math=True)

        assert r.gold == "1000"
        assert r.is_correct is True


# 9. Metadata propagation
class TestMetadataPropagation:
    """
    question_id and variant must survive the nuclear winter (aka every parse branch).
    They are critical for the survival of humanity (aka McNemar test downstream)
    """

    def test_question_id_preserved_exact(self, ev):
        r = ev.evaluate("The answer is B.", "B", question_id="arc_q001")

        assert r.question_id == "arc_q001"
 
    def test_question_id_preserved_fallback(self, ev):
        r = ev.evaluate("I think it's B) Mitochondria.", "B", question_id="arc_q002")
        
        assert r.question_id == "arc_q002"
 
    def test_question_id_preserved_ambiguous(self, ev):
        r = ev.evaluate("Both A and C seem right.", "A", question_id="arc_q003")
        
        assert r.question_id == "arc_q003"
 
    def test_question_id_preserved_unparseable(self, ev):
        r = ev.evaluate("I have no idea.", "A", question_id="arc_q004")
        
        assert r.question_id == "arc_q004"
 
    def test_variant_preserved(self, ev):
        r = ev.evaluate("The answer is B.", "B", variant="repeat")
        
        assert r.variant == "repeat"
 
    def test_raw_response_preserved(self, ev):
        response = "The answer is B."
        r = ev.evaluate(response, "B")
        
        assert r.raw_response == response
 
    def test_default_metadata_empty_strings(self, ev):
        r = ev.evaluate("The answer is B.", "B")
        
        assert r.question_id == ""
        assert r.variant == ""


# 10. evaluate_pair()
class TestEvaluatePair:

    def test_returns_two_results(self):
        b, r = evaluate_pair("The answer is A.", "The answer is B.", "A", "q001")
        
        assert isinstance(b, EvalResult)
        assert isinstance(r, EvalResult)
 
    def test_baseline_variant_label(self):
        b, _ = evaluate_pair("The answer is A.", "The answer is A.", "A", "q001")
        
        assert b.variant == "baseline"
 
    def test_repeat_variant_label(self):
        _, r = evaluate_pair("The answer is A.", "The answer is A.", "A", "q001")
        
        assert r.variant == "repeat"
 
    def test_shared_question_id(self):
        """Both results must carry the same question_id for McNemar pairing"""
        b, r = evaluate_pair("The answer is A.", "The answer is B.", "A", "mcnemar_q42")
        
        assert b.question_id == "mcnemar_q42"
        assert r.question_id == "mcnemar_q42"
 
    def test_correct_correctness_values(self):
        """Baseline correct, repeat wrong"""
        b, r = evaluate_pair("The answer is A.", "The answer is B.", "A", "q001")
        
        assert b.is_correct is True
        assert r.is_correct is False
 
    def test_both_correct(self):
        b, r = evaluate_pair("The answer is C.", "The answer is C.", "C", "q001")
        
        assert b.is_correct is True
        assert r.is_correct is True
 
    def test_both_wrong(self):
        b, r = evaluate_pair("The answer is D.", "The answer is D.", "A", "q001")
        
        assert b.is_correct is False
        assert r.is_correct is False
 
    def test_math_pair(self):
        b, r = evaluate_pair("#### 42", "#### 43", "42", "gsm_q01", is_math=True)
        
        assert b.is_correct is True
        assert r.is_correct is False
        assert b.variant == "baseline"
        assert r.variant == "repeat"


# 11. _normalize_number() (tested via public math path)
class TestNormalizeNumber:                                  # ok but is it normalized or normalised wtf is happening
    """
    Rather than testing the private static method directly, exercise it through
    evaluate() with is_math = true to stay in the public interface
    """

    def test_integer_string(self, ev):
        r = ev.evaluate("#### 100", "100", is_math=True)

        assert r.predicted == "100"
 
    def test_float_becomes_int(self, ev):
        r = ev.evaluate("#### 7.0", "7", is_math=True)

        assert r.predicted == "7"
        assert r.is_correct is True
 
    def test_comma_stripped(self, ev):
        r = ev.evaluate("#### 10,000", "10000", is_math=True)

        assert r.predicted == "10000"
        assert r.is_correct is True
 
    def test_leading_trailing_whitespace(self, ev):
        r = ev.evaluate("####  99 ", "99", is_math=True)

        assert r.predicted == "99"
        assert r.is_correct is True

# python -m pytest tests/test_evaluator.py -vs
