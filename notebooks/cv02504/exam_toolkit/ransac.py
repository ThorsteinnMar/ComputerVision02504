"""RANSAC formula helpers for Priority A templates."""

from __future__ import annotations

import math

import numpy as np


def ransac_iterations(s: int, m: int, p: float, n: int) -> tuple[float, int]:
    """
    Compute required RANSAC iterations.

    Args:
        s: best inlier count
        m: total points/matches
        p: desired confidence (e.g. 0.90, 0.95, 0.99)
        n: sample size per model

    Returns:
        raw_float_N, ceil_int_N
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1).")
    if not (0 < s <= m):
        raise ValueError("Require 0 < s <= m.")
    if n <= 0:
        raise ValueError("n must be positive.")

    inlier_ratio = s / m
    denom = math.log(1.0 - inlier_ratio**n)
    if np.isclose(denom, 0.0):
        return 1.0, 1

    raw = math.log(1.0 - p) / denom
    return float(raw), int(math.ceil(raw))


def ransac_threshold_from_sigma(
    sigma: float,
    confidence: float = 0.95,
    dof: int = 1,
    squared: bool = True,
) -> float:
    """
    Compute threshold from Gaussian noise sigma using chi-square quantile.

    Typical use:
        - dof=1 for one-dimensional squared residual
        - dof=2 for 2D squared Euclidean residual
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0,1).")
    if dof <= 0:
        raise ValueError("dof must be positive integer.")

    try:
        from scipy.stats import chi2

        chi_value = float(chi2.ppf(confidence, dof))
    except Exception:
        # Minimal fallback for common settings.
        fallback = {
            (1, 0.95): 3.84,
            (2, 0.95): 5.99,
            (1, 0.99): 6.63,
            (2, 0.99): 9.21,
        }
        key = (dof, round(confidence, 2))
        if key not in fallback:
            raise ValueError("scipy unavailable and no fallback chi-square value for these settings.")
        chi_value = fallback[key]

    tau2 = chi_value * (sigma**2)
    if squared:
        return float(tau2)
    return float(np.sqrt(tau2))
