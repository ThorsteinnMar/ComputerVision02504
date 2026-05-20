"""Miscellaneous computation helpers used in exam questions."""

from __future__ import annotations

import numpy as np


def ransac_iterations_required(
    best_inliers: int,
    total_points: int,
    sample_size: int,
    confidence: float,
) -> float:
    """Compute minimum RANSAC iterations N.

    N = log(1-p) / log(1-w^n), where w = inlier ratio.
    """
    if total_points <= 0:
        raise ValueError("total_points must be positive")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0,1)")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    w = best_inliers / total_points
    if w <= 0.0:
        return float("inf")
    denom = np.log(1.0 - w**sample_size)
    if np.isclose(denom, 0.0):
        return 1.0
    return float(np.log(1.0 - confidence) / denom)


def squared_reprojection_threshold(
    sigma: float,
    chi_square_value: float = 3.84,
) -> float:
    """Squared reprojection threshold tau^2 = chi2 * sigma^2."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float(chi_square_value * sigma**2)


def fft_first_harmonic_phase(signal: np.ndarray) -> float:
    """Phase of first non-zero FFT harmonic for phase-shift signal."""
    s = np.asarray(signal, dtype=float)
    fft = np.fft.rfft(s, axis=0)
    return float(np.angle(fft[1]))


def structured_light_unwrap_phase(
    primary: np.ndarray,
    secondary: np.ndarray,
    n1: int,
) -> float:
    """Unwrap phase for two-frequency structured light pattern."""
    theta_primary = fft_first_harmonic_phase(primary)
    theta_secondary = fft_first_harmonic_phase(secondary)
    theta_c = np.mod(theta_secondary - theta_primary, 2.0 * np.pi)
    o_primary = np.rint((n1 * theta_c - theta_primary) / (2.0 * np.pi))
    theta_est = np.mod((2.0 * np.pi * o_primary + theta_primary) / n1, 2.0 * np.pi)
    return float(theta_est)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean square error between arrays."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Inputs must have same shape")
    return float(np.sqrt(np.mean((x - y) ** 2)))
