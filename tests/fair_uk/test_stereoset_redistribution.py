import numpy as np
import pytest

from lm_eval.fair_uk.stereoset_redistribution import (
    FamilyBootstrap,
    harm_reduction,
    margin_thresholds,
    redistribution,
)


def test_development_repeat_floor_and_linear_quantile():
    floor, tau = margin_thresholds([0, 1, 2], [0.5, 1, 2], [0.1, 0.2, 0.3])
    assert floor == tau[0.1] == tau[0.2] == 0.5
    assert tau[0.3] == pytest.approx(0.6)


def test_nnr_boundaries_family_weighting_and_empty_cohort():
    bootstrap = FamilyBootstrap(["a", "b", "c"], 20, 42)
    r = bootstrap.neutralization([1, 2, 2, 2], [0, 1, -1, -2], ["c", "a", "a", "b"], 1)
    assert (r["eligible"], r["families"], r["cured"], r["reversed"]) == (3, 2, 2, 1)
    assert r["value"] == 0 and r["raw_item_nnr"] == pytest.approx(1 / 3)
    one = bootstrap.neutralization([2], [-1], ["a"], 1)
    assert one["value"] == 1 and one["ci_low"] is None
    assert bootstrap.neutralization([1], [0], ["a"], 1)["value"] is None


def test_linked_component_keeps_native_target_weights_and_paired_scale():
    boot = FamilyBootstrap(["a", "b", "c"], 100, 42)
    w = boot.target_macro_weights(["X", "X", "Y"], ["a", "b", "a"])
    assert np.allclose(w.sum(axis=1), 100)
    assert np.allclose(w[:, 2], 50)
    assert w[0] @ [1, 0, 1] == 75
    assert np.array_equal(harm_reduction(w @ [1, 0, 1], w @ [1, 0, 1]), np.zeros(101))


def test_ri_weights_gates_and_defined_zero():
    d = np.full(101, 4.0)
    r = redistribution(d, [-d, d, d * 0], [1 / 3] * 3, 1)
    assert r["value"] == pytest.approx(1 / 3)
    assert redistribution(d, [d], [1], 1)["value"] == 0
    assert (
        redistribution(d, [d], [1], 4)["status"] == "below_or_equal_practical_minimum"
    )
    d[1:20] = -2
    assert (
        redistribution(d, [d], [1], 1)["status"] == "denominator_interval_reaches_zero"
    )
    with pytest.raises(ValueError):
        redistribution(d, [d], [0.5], 1)
