"""Structured-light phase helpers for Priority A templates."""

from __future__ import annotations

import numpy as np


def phase_from_shift_sequence(intensities: np.ndarray) -> float:
    """Extract phase from first non-DC FFT component."""
    seq = np.asarray(intensities, dtype=float).reshape(-1)
    if seq.size < 2:
        raise ValueError("Need at least two samples in sequence.")
    fft_vals = np.fft.rfft(seq)
    if fft_vals.size < 2:
        raise ValueError("Sequence too short for first harmonic.")
    return float(np.angle(fft_vals[1]))


def structured_light_phase_unwrap_pixel(
    primary: np.ndarray,
    secondary: np.ndarray,
    n1: int,
    n2: int,
) -> dict:
    """
    Unwrap per-pixel phase from primary/secondary phase-shift sequences.

    Args:
        primary: primary-period intensity sequence.
        secondary: secondary-period intensity sequence.
        n1: number of periods in primary pattern.
        n2: number of periods in secondary pattern (kept for clarity).

    Returns dict with intermediate and final phases.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 and n2 must be positive.")

    theta_primary = phase_from_shift_sequence(primary)
    theta_secondary = phase_from_shift_sequence(secondary)

    theta_c = float(np.mod(theta_secondary - theta_primary, 2.0 * np.pi))
    o_primary = float(np.rint((n1 * theta_c - theta_primary) / (2.0 * np.pi)))
    theta_est = float(np.mod((2.0 * np.pi * o_primary + theta_primary) / n1, 2.0 * np.pi))

    return {
        "theta_primary": theta_primary,
        "theta_secondary": theta_secondary,
        "theta_c": theta_c,
        "o_primary": o_primary,
        "theta_est": theta_est,
    }
