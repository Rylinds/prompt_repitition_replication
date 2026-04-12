"""
McNemar's test for the prompt repetition experiment.

McNemar's test evaluates whether two classifiers (baseline vs. repeat prompt)
differ significantly on the *same* set of questions. It operates only on the
discordant pairs (questions where one variant was correct and the other was not).

Contingency table
----------
                      Repeat correct    Repeat wrong
  Baseline correct  |      a           |     b        |
  Baseline wrong    |      c           |     d        |

    a = both correct
    b = baseline only correct  (repeat hurt)
    c = repeat only correct    (repeat helped)
    d = both wrong

The test statistic uses only b and c. a and d provide no information about
whether one variant is better than the other.

Statistical method
----------
Use the exact two-sided binomial test rather than the chi-squared
approximation. The approximation is unreliable when b + c < 25, which is
common at 100-question pilot scale. The exact test is always valid.

    H0: P(baseline correct, repeat wrong) = P(baseline wrong, repeat correct)
        --> neither variant is better than the other
    H1: the two variants differ (two-sided)

    p = 2 * P(X <= min(b, c))  where X ~ Binomial(b+c, 0.5)
    p is capped at 1.0.

References
----------
    McNemar, Q. (1947). Note on the sampling error of the difference between
    correlated proportions or percentages. Psychometrika, 12(2), 153–157.

Usage
----------
    from analysis.mcnemar_test import McNemar, McNemanResult

    # From a list of runner result records:
    result = McNemar.from_records(records)
    print(result)

    # From two boolean lists:
    result = McNemar.from_bool_lists(baseline_correct, repeat_correct)
    print(result.p_value, result.significant)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
from scipy.stats import binom


# result dataclass
@dataclass
class McNemarResult:
    """
    Output of McNemar's test

    Attr
    ----------
    n: total number of paired questions
    a: both correct
    b: baseline only correct (repeat hurt)
    c: repeat only correct (repeat helped)
    d: both wrong
    p_value: exact two-sided p-value
    signficant: true if p-value < alpha
    alpha: significance threshold (0.05)
    direction: 'repeat_better' | 'baseline_better' | 'no_difference'
                base on whether c > b, b > c, or b == c
    """
    n : int
    a: int
    b: int
    c: int
    d: int
    p_value: float
    significant: bool
    alpha: float
    direction: str

    def __str__(self) -> str:
        lines = [
             f"McNemar's Test  (n={self.n}, α={self.alpha})",
            f"  Contingency  a={self.a}  b={self.b}  c={self.c}  d={self.d}",
            f"  Discordant   b+c={self.b + self.c}  (b=repeat hurt, c=repeat helped)",
            f"  p-value      {self.p_value:.4f}  {'*SIGNIFICANT*' if self.significant else '(not significant)'}",
            f"  Direction    {self.direction}",
        ]
        return "\n".join(lines)


# McNemar class
class McNemar:
    """
    Computes McNemar's two-sided test for paired binary outcomes.

    Instantiate with a/b/c/d counts directly, or use on of the convenience class methods
    to construct from result records or bool lists
    """

    def __init__(self, a: int, b: int, c: int, d:int, alpha: float = 0.05):
        if any(v < 0 for v in (a, b, c, d)):
            raise ValueError("Contingency table counts must be non-negative")
        if a + b + c + d == 0:
            raise ValueError("Contingency table is empty (all counts are zero)")
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.alpha = alpha
    
    # constructor class methods
    @classmethod
    def from_records(
        cls,
        records: List[dict],
        alpha: float = 0.05,
        base_key: str = "base_correct",
        rep_key: str = "rep_correct",
    ) -> "McNemar":
        """
        Construct from a list of runner result records.

        Each record must have boolean fields named base_key and rep_key
        matching the runner.py output schema by default.

        Args:
            records: list of dicts from runner.py JSON output
            alpha: significance threshold
            base_key: dict key for baseline correctness
            rep_key: dict key for repeat correctness
        """
        a = sum(1 for r in records if r[base_key] and r[rep_key])
        b = sum(1 for r in records if r[base_key] and not r[rep_key])
        c = sum(1 for r in records if not r[base_key] and r[rep_key])
        d = sum(1 for r in records if not r[base_key] and not r[rep_key])
        return cls(a, b, c, d, alpha=alpha)
    
    @classmethod
    def from_bool_lists(
        cls,
        baseline_correct: List[bool],
        repeat_correct: List[bool],
        alpha: float = 0.05,
    ) -> "McNemar":
        """
        Construct two parallel boolean lists.

        Args:
            baseline_correct: per-question correctness under baseline prompt
            repeat_correct: per-question correctness under repeat prompt
            alpha: significance threshold
        """
        if len(baseline_correct) != len(repeat_correct):
            raise ValueError(
                f"Lists must be the same length. "
                f"Got {len(baseline_correct)} and {len(repeat_correct)}."
            )
        if not baseline_correct:
            raise ValueError("Lists must not be empty")
        
        pairs = list(zip(baseline_correct, repeat_correct))
        a = sum(1 for bc, rc in pairs if bc and rc)
        b = sum(1 for bc, rc in pairs if bc and not rc)
        c = sum(1 for bc, rc in pairs if not bc and rc)
        d = sum(1 for bc, rc in pairs if not bc and not rc)
        return cls(a, b, c, d, alpha=alpha)


    # test computation
    def test(self) -> McNemarResult:
        p_value = self._exact_p(self.b, self.c)
        significant = p_value < self.alpha

        if self.c > self.b:
            direction = "repeat_better"
        elif self.b > self.c:
            direction = "baseline_better"
        else:
            direction = "no_difference"

        return McNemarResult(
            n=self.a + self.b + self.c + self.d,
            a=self.a,
            b=self.b,
            c=self.c,
            d=self.d,
            p_value=p_value,
            significant=significant,
            alpha=self.alpha,
            direction=direction,
        )

    # private helper
    @staticmethod
    def _exact_p(b: int, c: int) -> float:
        """
        Two-sided binomial p-val.

        Under H0, b and c are realizztions of a Binomial(b+c, 0.5).
        p = 2 * P(X <= min(b, c)) capped at 1.0.

        When b + c == 0 there are no discordant pairs and H0 cannot be
        rejected regardless of a and d so we return 1.0
        """
        n = b + c
        if n == 0:
            return 1.0
        p = 2.0 * float(binom.cdf(min(b, c), n, 0.5))
        return min(p, 1.0)


# convenience wrapper
def run_mcnemar(
        records: List[dict],
        alpha: float = 0.05,
        base_key: str = "base_correct",
        rep_key: str = "rep_correct",
) -> McNemarResult:
    """
    One call wrapper: load records, run test, return result

    Example:
        import json
        records = json.loads(Path("results/gpt-4o-mini_arc_n500_seed42.json").read_text())
        result = run_mcnemar(records)
        print(result)
    """
    return McNemar.from_records(records, alpha=alpha, base_key=base_key, rep_key=rep_key).test()

# smoke test
if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) == 2:
        # python -m analysis.mcnemar_test results/some_run.json
        records = json.loads(open(sys.argv[1]).read())
        result = run_mcnemar(records)
        print(result)
    else:
        # Built-in demo with known values from the pilot run.
        print("Demo — pilot run values (b=5, c=2):")
        mc = McNemar(a=64, b=5, c=2, d=29)
        print(mc.test())
        print()
        print("Demo — clearly significant (b=0, c=15):")
        mc2 = McNemar(a=50, b=0, c=15, d=35)
        print(mc2.test())
