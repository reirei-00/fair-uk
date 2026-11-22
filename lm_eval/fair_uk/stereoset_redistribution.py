"""Paired native StereoSet effects with shared semantic-family uncertainty draws."""

import numpy as np

from lm_eval.fair_uk.warbias.stats import percentile_interval


def paired_arrays(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a) or not np.isfinite([a, b]).all():
        raise ValueError("Expected two matched finite vectors")
    return a, b


def margin_thresholds(first, repeat, quantiles):
    first, repeat = paired_arrays(first, repeat)
    floor = float(np.max(np.abs(first - repeat)))
    return floor, {
        q: max(floor, float(np.quantile(np.abs(first), q, method="linear")))
        for q in quantiles
    }


class FamilyBootstrap:
    def __init__(self, families, draws, seed):
        self.families = sorted(set(families))
        if len(self.families) < 2 or draws < 2:
            raise ValueError("Need multiple semantic families and draws")
        self.index = {f: i for i, f in enumerate(self.families)}
        samples = np.random.default_rng(seed).exponential(
            size=(draws, len(self.families))
        )
        self.weights = np.vstack([np.ones(len(self.families)), samples])

    def target_macro_weights(self, groups, families):
        if len(groups) != len(families) or not len(groups):
            raise ValueError("Each item needs one group and one family")
        w = self.weights[:, [self.index[f] for f in families]].copy()
        groups = np.asarray(groups)
        names = np.unique(groups)
        for name in names:
            mask = groups == name
            part = w[:, mask]
            w[:, mask] = part / part.sum(axis=1, keepdims=True)
        return w * (100.0 / len(names))

    def neutralization(self, first, second, families, tau):
        before, after = paired_arrays(first, second)
        if len(families) != len(before) or not np.isfinite(tau) or tau < 0:
            raise ValueError("Invalid margin calibration or family mapping")
        eligible = before > tau
        groups, codes = np.unique(np.asarray(families)[eligible], return_inverse=True)
        cured = np.abs(after[eligible]) <= tau
        reversed_ = after[eligible] < -tau
        n = int(eligible.sum())
        row = {
            "eligible": n,
            "families": len(groups),
            "cured": int(cured.sum()),
            "reversed": int(reversed_.sum()),
            "persistent": n - int(cured.sum()) - int(reversed_.sum()),
        }
        if not n:
            return dict(
                row,
                value=None,
                ci_low=None,
                ci_high=None,
                cured_rate=None,
                reversed_rate=None,
                raw_item_nnr=None,
                status="no_baseline_eligible_items",
            )
        counts = np.bincount(codes)
        cured_by_family = (
            np.bincount(codes, weights=cured, minlength=len(groups)) / counts
        )
        reversed_by_family = (
            np.bincount(codes, weights=reversed_, minlength=len(groups)) / counts
        )
        weights = self.weights[:, [self.index[f] for f in groups]]
        values = (weights @ (cured_by_family - reversed_by_family)) / weights.sum(
            axis=1
        )
        interval = percentile_interval(values[1:]) if len(groups) > 1 else (None, None)
        return dict(
            row,
            value=float(values[0]),
            ci_low=interval[0],
            ci_high=interval[1],
            cured_rate=float(cured_by_family.mean()),
            reversed_rate=float(reversed_by_family.mean()),
            raw_item_nnr=(row["cured"] - row["reversed"]) / n,
            status="available",
        )


def harm_reduction(before_ss, after_ss):
    before, after = paired_arrays(before_ss, after_ss)
    values = abs(before - 50) - abs(after - 50)
    values[np.abs(values) < 1e-12] = 0
    return values


def redistribution(target_reduction, other_reductions, weights, practical_minimum):
    target = np.asarray(target_reduction, dtype=float)
    others = np.asarray(other_reductions, dtype=float)
    w = np.asarray(weights, dtype=float)
    if (
        target.ndim != 1
        or len(target) < 3
        or not np.isfinite(target).all()
        or others.shape != (len(w), len(target))
        or not np.isfinite(others).all()
        or not np.isfinite(w).all()
        or (w < 0).any()
        or not np.isclose(w.sum(), 1)
        or not np.isfinite(practical_minimum)
        or practical_minimum <= 0
    ):
        raise ValueError(
            "Expected paired draws, fixed normalized weights and a positive practical minimum"
        )
    numerator = w @ np.maximum(-others, 0)
    low, high = percentile_interval(target[1:])
    lower, upper = percentile_interval(numerator[1:])
    positive = target[1:] > 0
    row = {
        "target_reduction": float(target[0]),
        "denominator_low": low,
        "denominator_high": high,
        "numerator": float(numerator[0]),
        "numerator_low": lower,
        "numerator_high": upper,
        "practical_minimum": practical_minimum,
        "positive_draws": int(positive.sum()),
        "draws": len(positive),
    }
    if target[0] <= practical_minimum:
        status = "below_or_equal_practical_minimum"
    elif low <= 0:
        status = "denominator_interval_reaches_zero"
    else:
        lo, hi = percentile_interval(numerator[1:][positive] / target[1:][positive])
        return dict(
            row,
            value=float(numerator[0] / target[0]),
            ci_low=lo,
            ci_high=hi,
            status="available",
        )
    return dict(row, value=None, ci_low=None, ci_high=None, status=status)
