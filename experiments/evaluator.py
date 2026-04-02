"""
Response evaluator for prompt repitition experiments

Parses raw model output into structured correctness judgements.
Handles both multiple-choice (ARC, OpenBookQA, MMLU-Pro) and free-form
math (GSM8K) answer formats.

Design notes
------------
* primary parsing always tries the paper's prescribed format first:
    'The answer is X.'
* a cascade of fallback strats handles imperfect responses
* ambiguous / unparsable responses are flagged rather than just silently wrong
    * helps audit them seperately during analysis
* all logic is deterministic and free of side-effects so the test suite
  can validate it without real model calls
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# RESULT TYPES
class ParseStatus(Enum):
    """How confidently the answer was extracted"""
    EXACT_FORMAT = "exact_format"               # 'The answer is X.'
    FALLBACK_REGEX = "fallback_regex"           # found a lone letter / number via regex
    NUMERIC_MATCH = "numeric_match"             # GSM8K: extracted final numeric value
    AMBIGUOUS = "ambiguous"                     # multiple valid candidates found
    UNPARSEABLE = "unparseable"                 # no answer found -> flagged for review

@dataclass
class EvalResult:
    """
    Result of evaluating a single model response

    Attr:
        is_correct: whether the model's answer matches the gold label
        predicted: the extracted answer str (letter or num), or None
        gold: the truth answer str
        parse_status: how the answer was extracted
        raw_response: the original model output preserved for auditing
        question_id: identifier from the dataset for paired McNemar analysis
        variant: prompt variant that produced the response
    """
    is_correct: bool
    predicted: Optional[str]
    gold: str
    parse_status: ParseStatus
    raw_response: str
    question_id: str = ""
    variant: str = ""

# EVALUATOR
class Evaluator:
    """
    Parses and evaluates model responses for MCQ and GSM8K benchmarks

    Instantiate once and call evaluate() for each response. The class is stateless
    between calls so it's safe to reuse across benchmark types
    """

    # regex for 'The answer is X.' <- the paper's prescribed output format
    _EXACT_PATTERN = re.compile(
        r"[Tt]he\s+answer\s+is\s+([A-Ja-j])\b",            # letter up to J (MMLU-Pro has 10 options)
        re.IGNORECASE,
    )

    # callback pattern using three named groups. First non-None group is the answer
    #
    # The three contexts that reliably signal an answer letter are fundamentally different and cannot share a single prefix rule:
    #
    #   delimited  — letter immediately before ")"  e.g. "it's B) Mitochondria"
    #                Strongest signal; almost never a coincidence
    #
    #   keyword    — letter after answer/option/choice/correct
    #                e.g. "option B", "choice: C"
    #
    #   isolated   — bare uppercase letter surrounded by non-letter chars
    #                ex: "Both A and C seem right" -> catches BOTH for AMBIGUOUS
    #                Excludes "I" because it is an extremely common English pronoun
    #                and would fire on every response that starts with "I think…".
    #                "A" is kept because answer option A is far more common than the
    #                article "a" appearing as an isolated uppercase letter
    _LETTER_PATTERN = re.compile(
        r"""
        (?:
            # letter directly before ) ex: "B)" or "b)"
            \b(?P<delimited>[A-Ja-j])(?=\s*\))
            |
            # after an answer-indicating keyword ex: "answer: B", "option B"
            (?:answer|option|choice|correct)\s*[:\-=]?\s*(?P<keyword>[A-Ja-j])\b
            |
            # isolated uppercase letter (A–J, but NOT I which is a pronoun); requires a non-letter on both sides to avoid matching inside words
            (?<![A-Za-z])(?P<isolated>[A-HJ])(?![A-Za-z])
        )
        """,
        re.VERBOSE | re.IGNORECASE | re.MULTILINE,
    )

    _NUMERIC_PATTERN = re.compile(
        r"####\s*([\d,]+(?:\.\d+)?)"                            # GSM8K canonical '#### <answer>'
        r"|"
        r"(?:answer\s+is|=)\s*([\d,]+(?:\.\d+)?)\s*\.?\s*$",    # 'answer is 42'
        re.IGNORECASE | re.MULTILINE,
    )

    def evaluate(
        self,
        raw_response: str,
        gold_answer: str,
        question_id: str = "",
        variant: str = "",
        is_math: bool = False,
    ) -> EvalResult:
        """
        Parse a model respomse and compare to the gold answer

        Args:
            raw_response: the raw text returned by the model
            gold_answer: ground-truth answer (letter like 'B' or num like '42')
            question_id: dataset question ID for pairing in McNemar analysis
            variant: prompt variant label
            is_math: if True, use numeric extraction logic (GSM8K)
        
        Returns:
            EvalResult with correctness verdict and diagnostic metadata
        """
        if is_math:
            return self._evaluate_math(raw_response, gold_answer, question_id, variant)
        return self._evaluate_mcq(raw_response, gold_answer, question_id, variant)
    
    # MCQ EVALUATION
    def _evaluate_mcq(
        self, response: str, gold: str, qid: str, variant: str
    ) -> EvalResult:
        gold_norm = gold.strip().upper()

        # try prescribed format first
        exact_match = self._EXACT_PATTERN.search(response)
        if exact_match:
            predicted = exact_match.group(1).upper()
            return EvalResult(
                is_correct=predicted == gold_norm,
                predicted=predicted,
                gold=gold_norm,
                parse_status=ParseStatus.EXACT_FORMAT,
                raw_response=response,
                question_id=qid,
                variant=variant,
            )

        # fallback: scan for any standalone letter
        # _LETTER_PATTERN has three named groups; exactly one will be non-None per match.
        all_letters = [
            next(v for v in m.groupdict().values() if v is not None).upper()
            for m in self._LETTER_PATTERN.finditer(response)
        ]
        # de-duplicate while preserving order
        seen = set()
        unique_letters = []
        for letter in all_letters:
            if letter not in seen:
                seen.add(letter)
                unique_letters.append(letter)
        
        # questionable
        if len(unique_letters) == 1:
            predicted = unique_letters[0]
            logger.debug("qid=%s | fallback letter march: %s", qid, predicted)
            return EvalResult(
                is_correct=predicted == gold_norm,
                predicted=predicted,
                gold=gold_norm,
                parse_status=ParseStatus.FALLBACK_REGEX,
                raw_response=response,
                question_id=qid,
                variant=variant,
            )
        
        if len(unique_letters) > 1:
            # multiple candidate letters yikes -> ambiguous
            logger.warning("qid=%s | ambiguous MCQ response: %s", qid, unique_letters)
            return EvalResult(
                is_correct=False,
                predicted=None,
                gold=gold_norm,
                parse_status=ParseStatus.AMBIGUOUS,
                raw_response=response,
                question_id=qid,
                variant=variant,
            )
        
        # no letter found at all wtf
        logger.warning("qid=%s | unparseable MCQ response (first 120 chars): %.120s", qid, response)
        return EvalResult(
            is_correct=False,
            predicted=None,
            gold=gold_norm,
            parse_status=ParseStatus.UNPARSEABLE,
            raw_response=response,
            question_id=qid,
            variant=variant
        )
    
    # GSM8K EVALUATION
    def _evaluate_math(
        self, response: str, gold: str, qid: str, variant: str
    ) -> EvalResult:
        gold_norm = self._normalize_number(gold)

        match = self._NUMERIC_PATTERN.search(response)
        if match:
            raw_num = match.group(1) or match.group(2)
            predicted_norm = self._normalize_number(raw_num)
            return EvalResult(
                is_correct=predicted_norm == gold_norm,
                predicted=predicted_norm,
                gold=gold_norm,
                parse_status=ParseStatus.NUMERIC_MATCH,
                raw_response=response,
                question_id=qid,
                variant=variant,
            )

        logger.warning("qid=%s | unparseable math response (first 120 chars): %.120s", qid, response)
        return EvalResult(
            is_correct=False,
            predicted=None,
            gold=gold_norm,
            parse_status=ParseStatus.UNPARSEABLE,
            raw_response=response,
            question_id=qid,
            variant=variant,
        )
    
    @staticmethod
    def _normalize_number(s: str) -> str:
        """Strip commas, trailing zeros, and whitespace for numeric comparison"""
        s = s.strip().replace(",", "")
        try:
            return str(int(float(s)))
        except ValueError:
            return s.strip()


# convenience wrapper for pair experiment eval
def evaluate_pair(
        baseline_response: str,
        repeat_response: str,
        gold_answer: str,
        question_id: str,
        is_math: bool = False,
) -> tuple[EvalResult, EvalResult]:
    """
    Evaluate a matched baseline / repeat pair for a single question

    Returns (baseline_result, repear_result). The shared question_id is
    essential for the paired McNemar test downstream
    """
    ev = Evaluator()
    b = ev.evaluate(baseline_response, gold_answer, question_id, variant="baseline", is_math=is_math)
    r = ev.evaluate(repeat_response, gold_answer, question_id, variant="repeat", is_math=is_math)
    return b, r


# smoke test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ev = Evaluator()

    cases = [
        # (response,                              gold, is_math, description)
        ("The answer is B.",                      "B",  False,  "exact format — correct"),
        ("the answer is C.",                      "B",  False,  "exact format — wrong"),
        ("I think it's B) Mitochondria.",         "B",  False,  "fallback letter"),
        ("Both A and C seem right.",              "A",  False,  "ambiguous"),
        ("None of the above makes sense.",        "A",  False,  "unparseable"),
        ("So the total is 42.\n#### 42",          "42", True,   "GSM8K exact"),
        ("The answer is 1,024.",                  "1024", True, "GSM8K with comma"),
    ]

    for response, gold, is_math, desc in cases:
        result = ev.evaluate(response, gold, question_id="test", is_math=is_math)
        status = "good" if result.is_correct else "bad"
        print(f"{status}  [{result.parse_status.value:16}]  {desc}")
        print(f"   predicted = {result.predicted!r}  gold = {result.gold!r}\n")
    