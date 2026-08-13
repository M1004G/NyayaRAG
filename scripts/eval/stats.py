"""
Statistical rigor utilities, proposal Section 6.2: "paired bootstrap
resampling (10k resamples) for significance between configurations;
Holm-Bonferroni correction across the ablation grid; confidence intervals
reported for all headline numbers."

Pure computation, CPU-only, no external dependency beyond the standard
library (random) -- matches the proposal's own framing of this section as
compute, not GPU/API-bound.
"""
import random

DEFAULT_RESAMPLES = 10_000


def bootstrap_ci(scores: list, n_resamples: int = DEFAULT_RESAMPLES, ci: float = 0.95, seed: int = 42) -> dict:
    """95% CI (default) via the percentile bootstrap over a list of
    per-item scores for ONE configuration. `seed` is fixed for
    reproducibility across repeated runs of the same eval."""
    if not scores:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}

    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_resamples):
        resample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    alpha = (1 - ci) / 2
    low = means[int(alpha * n_resamples)]
    high = means[int((1 - alpha) * n_resamples) - 1]
    return {"mean": sum(scores) / n, "low": low, "high": high, "n": n}


def paired_bootstrap_test(scores_a: list, scores_b: list, n_resamples: int = DEFAULT_RESAMPLES, seed: int = 42) -> dict:
    """Paired significance test between two configurations run on the
    SAME benchmark items (scores_a[i] and scores_b[i] must be the same
    item i). Returns a two-sided p-value: the fraction of bootstrap
    resamples where the sign of (mean_b - mean_a) flips relative to the
    observed difference -- the standard paired-bootstrap significance
    procedure."""
    if len(scores_a) != len(scores_b):
        raise ValueError(f"paired scores must be the same length, got {len(scores_a)} vs {len(scores_b)}")
    if not scores_a:
        return {"observed_diff": float("nan"), "p_value": float("nan"), "n": 0}

    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    observed_diff = sum(diffs) / len(diffs)

    rng = random.Random(seed)
    n = len(diffs)
    count_extreme = 0
    for _ in range(n_resamples):
        resample = [diffs[rng.randrange(n)] for _ in range(n)]
        resample_mean = sum(resample) / n
        # two-sided: does the resampled mean, RE-CENTERED to test the null
        # of no difference, exceed the observed difference in magnitude
        centered = resample_mean - observed_diff
        if abs(centered) >= abs(observed_diff):
            count_extreme += 1

    p_value = count_extreme / n_resamples
    return {"observed_diff": observed_diff, "p_value": p_value, "n": n}


def holm_bonferroni(p_values: dict, alpha: float = 0.05) -> dict:
    """p_values: {comparison_label: p_value}. Returns
    {comparison_label: {"p": ..., "significant": bool, "adjusted_alpha": ...}}
    using the Holm step-down procedure (uniformly more powerful than plain
    Bonferroni while controlling the same family-wise error rate -- the
    standard choice for an ablation grid's multiple comparisons, per
    Section 6.2)."""
    m = len(p_values)
    ranked = sorted(p_values.items(), key=lambda kv: kv[1])

    results = {}
    still_significant = True
    for i, (label, p) in enumerate(ranked):
        adjusted_alpha = alpha / (m - i)
        if still_significant and p <= adjusted_alpha:
            results[label] = {"p": p, "significant": True, "adjusted_alpha": adjusted_alpha}
        else:
            still_significant = False  # Holm step-down: once one comparison fails, all subsequent (larger-p) ones fail too
            results[label] = {"p": p, "significant": False, "adjusted_alpha": adjusted_alpha}
    return results
