"""Radial distortion helpers for Priority A templates."""

from __future__ import annotations

import numpy as np

from .core import ensure_column, from_homogeneous, to_homogeneous


def distort_normalized_points(q: np.ndarray, dist_coeffs: list[float] | tuple[float, ...]) -> np.ndarray:
    """Apply radial distortion to normalized image points q (2xn)."""
    pts = np.asarray(q, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(2, 1)
    if pts.shape[0] != 2:
        raise ValueError(f"q must be 2xn. Got {pts.shape}.")

    r2 = pts[0] ** 2 + pts[1] ** 2
    scale = np.ones_like(r2, dtype=float)
    for i, k in enumerate(dist_coeffs, start=1):
        scale += float(k) * (r2**i)

    return pts * scale


def map_undistorted_to_distorted_pixel(
    undistorted_pixel: np.ndarray,
    K: np.ndarray,
    dist_coeffs: list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map one undistorted pixel to distorted pixel.

    Returns:
        distorted_pixel (2x1), normalized_undistorted (2x1), normalized_distorted (2x1)
    """
    K = np.asarray(K, dtype=float)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3. Got {K.shape}.")

    p_u = ensure_column(undistorted_pixel, name="undistorted_pixel")
    if p_u.shape != (2, 1):
        raise ValueError(f"undistorted_pixel must be 2x1. Got {p_u.shape}.")

    q_u_h = np.linalg.inv(K) @ to_homogeneous(p_u)
    q_u = from_homogeneous(q_u_h)
    q_d = distort_normalized_points(q_u, dist_coeffs)
    p_d_h = K @ to_homogeneous(q_d)
    p_d = from_homogeneous(p_d_h)
    return p_d, q_u, q_d
