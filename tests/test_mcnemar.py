"""
Units tests for analysis/mcnemar_test.py

Test classes
--------
TestMcNemarExactPValues: p-value correctness against known values
TestMcNemarContingencyTable: a/b/c/d counts built correctly from inputs
TestMcNemarDirection: direction label (repeat_better / baseline_better)
TestMcNemarSignificance: significant flag respects alpha
TestMcNemarFromRecords: from_records() constructor
TestMcNemarFromBoolLists: from_bool_lists() constructor
TestMcNemarEdgeCases: zero discordants, all-same, mismatched inputs
TestRunMcNemar: convenience wrapper smoke test
"""

import pytest
from scipy.stats import binom as scipy_binom
from analysis.mcnemar_test import McNemar, McNemarResult, run_mcnemar


# ref p-value helper
def ref_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    p = 2.0 * float(scipy_binom.cdf(min(b, c), n, 0.5))
    return min(p, 1.0)


# 1. exact p-values
class TestMcNemarExtactPValues:

    def _p(self, b, c):
        return McNemar(a=10, b=b, c=c, d=10).test().p_value
    
    def test_no_discordant_returns(self):
        assert self._p(0, 0) == 1.0
    
    def test_pilot_values_b5_c2(self):
        """Values from the 100Q mistral pilot (see ipynbs)"""
        result = McNemar(a=64, b=5, c=2, d=29).test()
        assert abs(result.p_value - ref_p(5, 2)) < 1e-9
    
    def test_pilot_values_b5_c1(self):
        """Values from the 100-question Mistral pilot (v1)."""
        result = McNemar(a=62, b=5, c=1, d=32).test()
        assert abs(result.p_value - ref_p(5, 1)) < 1e-9

    def test_symmetric_b_equals_c(self):
        """When b == c the test is maximally non-significant for those counts."""
        result = McNemar(a=40, b=5, c=5, d=50).test()
        assert abs(result.p_value - ref_p(5, 5)) < 1e-9
        assert result.p_value == 1.0   # 2 * P(X <= 5 | Bin(10,0.5)) = 2*1 = capped at 1

    def test_strongly_significant(self):
        """b=0, c=15 — repeat is clearly better."""
        result = McNemar(a=50, b=0, c=15, d=35).test()
        assert abs(result.p_value - ref_p(0, 15)) < 1e-9
        assert result.p_value < 0.001

    def test_b_and_c_equal_one(self):
        result = McNemar(a=10, b=1, c=1, d=10).test()
        assert abs(result.p_value - ref_p(1, 1)) < 1e-9

    def test_large_discordant_count(self):
        result = McNemar(a=400, b=30, c=70, d=500).test()
        assert abs(result.p_value - ref_p(30, 70)) < 1e-9

    def test_p_value_never_exceeds_one(self):
        for b, c in [(0, 0), (3, 3), (1, 0), (10, 10)]:
            assert McNemar(a=10, b=b, c=c, d=10).test().p_value <= 1.0

    def test_p_value_never_negative(self):
        for b, c in [(0, 0), (5, 1), (1, 5), (10, 10)]:
            assert McNemar(a=10, b=b, c=c, d=10).test().p_value >= 0.0


# 2. contingency table counts
class TestMcNemarContingencyTable:

    def test_counts_sum_to_n(self):
        mc = McNemar(a=64, b=5, c=2, d=29)
        r = mc.test()
        assert r.a + r.b + r.c + r.d == r.n
        assert r.n == 100

    def test_abcd_stored_correctly(self):
        mc = McNemar(a=1, b=2, c=3, d=4)
        assert mc.a == 1
        assert mc.b == 2
        assert mc.c == 3
        assert mc.d == 4

    def test_result_carries_abcd(self):
        r = McNemar(a=10, b=3, c=7, d=80).test()
        assert r.a == 10
        assert r.b == 3
        assert r.c == 7
        assert r.d == 80

    def test_n_is_total_questions(self):
        r = McNemar(a=60, b=5, c=3, d=32).test()
        assert r.n == 100


# 3. direction label
class TestMcNemarDirection:

    def test_repeat_better_when_c_greater(self):
        r = McNemar(a=10, b=2, c=8, d=10).test()
        assert r.direction == "repeat_better"

    def test_baseline_better_when_b_greater(self):
        r = McNemar(a=10, b=8, c=2, d=10).test()
        assert r.direction == "baseline_better"

    def test_no_difference_when_equal(self):
        r = McNemar(a=10, b=5, c=5, d=10).test()
        assert r.direction == "no_difference"

    def test_no_difference_when_both_zero(self):
        r = McNemar(a=50, b=0, c=0, d=50).test()
        assert r.direction == "no_difference"

    def test_pilot_direction_baseline_better(self):
        """Mistral pilot v2: b=5 > c=2 → baseline better."""
        r = McNemar(a=64, b=5, c=2, d=29).test()
        assert r.direction == "baseline_better"


# 4. significance flag
class TestMcNemarSignificance:

    def test_not_significant_at_default_alpha(self):
        """Pilot values — p ≈ 0.45, not significant at α=0.05."""
        r = McNemar(a=64, b=5, c=2, d=29).test()
        assert r.significant is False

    def test_significant_when_p_below_alpha(self):
        r = McNemar(a=50, b=0, c=15, d=35).test()
        assert r.significant is True

    def test_custom_alpha_respected(self):
        """With α=0.5, almost everything should be significant."""
        r = McNemar(a=10, b=3, c=1, d=10, alpha=0.5).test()
        assert r.alpha == 0.5
        # ref_p(3,1) ≈ 0.625 — not significant even at 0.5
        # use a more extreme case
        r2 = McNemar(a=10, b=0, c=5, d=10, alpha=0.5).test()
        assert r2.significant is True   # ref_p(0,5) = 2*(1/32) ≈ 0.063

    def test_alpha_stored_in_result(self):
        r = McNemar(a=10, b=2, c=3, d=10, alpha=0.01).test()
        assert r.alpha == 0.01

    def test_boundary_p_equals_alpha_not_significant(self):
        """p == alpha is not significant (strict less-than)."""
        mc = McNemar(a=10, b=3, c=3, d=10, alpha=ref_p(3, 3))
        r = mc.test()
        assert r.significant is False


# 5. from_records constructor()
class TestMcNemarFromRecords:

    def _make_record(self, base_correct: bool, rep_correct: bool) -> dict:
        return {"base_correct": base_correct, "rep_correct": rep_correct}

    def test_counts_correct(self):
        records = [
            self._make_record(True,  True),   # a
            self._make_record(True,  True),   # a
            self._make_record(True,  False),  # b
            self._make_record(True,  False),  # b
            self._make_record(True,  False),  # b
            self._make_record(False, True),   # c
            self._make_record(False, False),  # d
        ]
        mc = McNemar.from_records(records)
        assert mc.a == 2
        assert mc.b == 3
        assert mc.c == 1
        assert mc.d == 1

    def test_returns_mcnemar_instance(self):
        records = [self._make_record(True, True)]
        mc = McNemar.from_records(records)
        assert isinstance(mc, McNemar)

    def test_custom_keys(self):
        records = [{"bc": True, "rc": False}, {"bc": False, "rc": True}]
        mc = McNemar.from_records(records, base_key="bc", rep_key="rc")
        assert mc.b == 1
        assert mc.c == 1

    def test_pilot_v2_values(self):
        """Reconstruct pilot v2 table from individual records."""
        records = (
            [{"base_correct": True,  "rep_correct": True}]  * 64 +
            [{"base_correct": True,  "rep_correct": False}] * 5  +
            [{"base_correct": False, "rep_correct": True}]  * 2  +
            [{"base_correct": False, "rep_correct": False}] * 29
        )
        mc = McNemar.from_records(records)
        assert mc.a == 64
        assert mc.b == 5
        assert mc.c == 2
        assert mc.d == 29

    def test_p_value_matches_direct_construction(self):
        records = (
            [{"base_correct": True,  "rep_correct": True}]  * 10 +
            [{"base_correct": True,  "rep_correct": False}] * 4  +
            [{"base_correct": False, "rep_correct": True}]  * 1  +
            [{"base_correct": False, "rep_correct": False}] * 10
        )
        r_from_records = McNemar.from_records(records).test()
        r_direct       = McNemar(a=10, b=4, c=1, d=10).test()
        assert abs(r_from_records.p_value - r_direct.p_value) < 1e-12


# 6. from_bool_lists() constructor
class TestMcNemarFromBoolLists:

    def test_counts_correct(self):
        base = [True,  True,  True,  False, False]
        rep  = [True,  False, False, True,  False]
        mc = McNemar.from_bool_lists(base, rep)
        assert mc.a == 1   # both True  at index 0
        assert mc.b == 2   # base True, rep False at indices 1,2
        assert mc.c == 1   # base False, rep True at index 3
        assert mc.d == 1   # both False at index 4

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            McNemar.from_bool_lists([True, False], [True])

    def test_empty_lists_raise(self):
        with pytest.raises(ValueError):
            McNemar.from_bool_lists([], [])

    def test_all_correct_both(self):
        base = [True] * 10
        rep  = [True] * 10
        mc = McNemar.from_bool_lists(base, rep)
        assert mc.a == 10
        assert mc.b == mc.c == mc.d == 0

    def test_p_value_matches_from_records(self):
        base = [True, True, True, False, False, False]
        rep  = [True, False, False, True, True, False]
        mc_lists   = McNemar.from_bool_lists(base, rep)
        records    = [{"bc": b, "rc": r} for b, r in zip(base, rep)]
        mc_records = McNemar.from_records(records, base_key="bc", rep_key="rc")
        assert mc_lists.b == mc_records.b
        assert mc_lists.c == mc_records.c
        assert abs(mc_lists.test().p_value - mc_records.test().p_value) < 1e-12


# 7. edge cases and input validation
class TestMcNemarEdgeCases:

    def test_negative_count_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            McNemar(a=-1, b=1, c=1, d=1)

    def test_all_zero_raises(self):
        with pytest.raises(ValueError, match="empty"):
            McNemar(a=0, b=0, c=0, d=0)

    def test_only_a_and_d_nonzero(self):
        """No discordant pairs — p must be 1.0."""
        r = McNemar(a=50, b=0, c=0, d=50).test()
        assert r.p_value == 1.0
        assert r.significant is False

    def test_single_discordant_pair(self):
        """Minimum meaningful discordance: b=1, c=0 or b=0, c=1."""
        r = McNemar(a=10, b=1, c=0, d=10).test()
        assert abs(r.p_value - ref_p(1, 0)) < 1e-9

    def test_result_is_mcnemar_result_instance(self):
        r = McNemar(a=5, b=3, c=1, d=5).test()
        assert isinstance(r, McNemarResult)

    def test_str_representation_contains_key_info(self):
        r = McNemar(a=64, b=5, c=2, d=29).test()
        s = str(r)
        assert "p-value" in s.lower() or "p_value" in s.lower() or "p=" in s.lower() or "p-" in s
        assert "5" in s   # b value
        assert "2" in s   # c value


# 8. run_mcnemar() conveniance wrapper
class TestRunMcNemar:

    def test_returns_mcnemar_result(self):
        records = [
            {"base_correct": True,  "rep_correct": True},
            {"base_correct": True,  "rep_correct": False},
            {"base_correct": False, "rep_correct": True},
        ]
        r = run_mcnemar(records)
        assert isinstance(r, McNemarResult)

    def test_matches_direct_construction(self):
        records = (
            [{"base_correct": True,  "rep_correct": True}]  * 64 +
            [{"base_correct": True,  "rep_correct": False}] * 5  +
            [{"base_correct": False, "rep_correct": True}]  * 2  +
            [{"base_correct": False, "rep_correct": False}] * 29
        )
        r_wrapper = run_mcnemar(records)
        r_direct  = McNemar(a=64, b=5, c=2, d=29).test()
        assert abs(r_wrapper.p_value - r_direct.p_value) < 1e-12

    def test_custom_alpha_passed_through(self):
        records = [{"base_correct": True, "rep_correct": False}] * 20
        r = run_mcnemar(records, alpha=0.01)
        assert r.alpha == 0.01


#if __name__ == "__main__":
    #pytest.main([__file__, "-v"])

# python -m pytest tests/test_mcnemar.py -vs