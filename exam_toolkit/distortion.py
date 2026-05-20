"""Radial distortion helpers for exam computations."""

from __future__ import annotations

import numpy as np

from .core import ensure_points, pi, pi_inv


def _radial_scale(r2: np.ndarray, dist_coeffs: list[float] | tuple[float, ...]) -> np.ndarray:
    """Compute radial scale: 1 + k3 r^2 + k5 r^4 + k7 r^6 + ..."""
    scale = np.ones_like(r2, dtype=float)
    for i, k in enumerate(dist_coeffs):
        scale = scale + float(k) * (r2 ** (i + 1))
    return scale


def distort_normalized_points(
    q: np.ndarray,
    dist_coeffs: list[float] | tuple[float, ...],
) -> np.ndarray:
    """Apply radial distortion to normalized points.

    Input q shape: (2,N)
    """
    qn = ensure_points(q, 2)
    r2 = qn[0] ** 2 + qn[1] ** 2
    scale = _radial_scale(r2, dist_coeffs)
    return qn * scale.reshape(1, -1)


def pixel_to_normalized_points(p: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert image pixels to normalized camera points."""
    pts = ensure_points(p, 2)
    k = np.asarray(K, dtype=float)
    if k.shape != (3, 3):
        raise ValueError("K must be 3x3")
    return pi(np.linalg.inv(k) @ pi_inv(pts))


def normalized_to_pixel_points(q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert normalized camera points to image pixels."""
    qn = ensure_points(q, 2)
    k = np.asarray(K, dtype=float)
    if k.shape != (3, 3):
        raise ValueError("K must be 3x3")
    return pi(k @ pi_inv(qn))


def distort_pixel_points(
    p: np.ndarray,
    K: np.ndarray,
    dist_coeffs: list[float] | tuple[float, ...],
) -> np.ndarray:
    """Apply radial distortion to pixel points using camera matrix K."""
    q = pixel_to_normalized_points(p, K)
    qd = distort_normalized_points(q, dist_coeffs)
    return normalized_to_pixel_points(qd, K)


def undistort_normalized_points_iterative(
    qd: np.ndarray,
    dist_coeffs: list[float] | tuple[float, ...],
    max_iters: int = 10,
) -> np.ndarray:
    """Approximate undistortion by fixed-point iterations in normalized frame."""
    qd_arr = ensure_points(qd, 2)
    q = qd_arr.copy()
    for _ in range(max_iters):
        r2 = q[0] ** 2 + q[1] ** 2
        scale = _radial_scale(r2, dist_coeffs).reshape(1, -1)
        q = qd_arr / scale
    return q


def undistort_pixel_points_iterative(
    pd: np.ndarray,
    K: np.ndarray,
    dist_coeffs: list[float] | tuple[float, ...],
    max_iters: int = 10,
) -> np.ndarray:
    """Approximate undistorted pixel coordinates from distorted pixel points."""
    qd = pixel_to_normalized_points(pd, K)
    q = undistort_normalized_points_iterative(qd, dist_coeffs, max_iters=max_iters)
    return normalized_to_pixel_points(q, K)
