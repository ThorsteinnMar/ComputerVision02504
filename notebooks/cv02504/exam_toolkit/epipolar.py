"""Epipolar geometry helpers for Priority A templates."""

from __future__ import annotations

import numpy as np

from .core import ensure_column, skew, to_homogeneous
from .geometry import point_line_distance


def essential_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute E = [t]_x R."""
    R = np.asarray(R, dtype=float)
    t = ensure_column(t, name="t")
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3. Got {R.shape}.")
    if t.shape != (3, 1):
        raise ValueError(f"t must be 3x1. Got {t.shape}.")
    return skew(t) @ R


def fundamental_matrix_from_rt(
    K1: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    K2: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
) -> np.ndarray:
    """
    Compute fundamental matrix from camera intrinsics and poses.

    Convention: x_cam = R x_world + t.
    """
    K1 = np.asarray(K1, dtype=float)
    K2 = np.asarray(K2, dtype=float)
    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)
    t1 = ensure_column(t1, name="t1")
    t2 = ensure_column(t2, name="t2")

    if K1.shape != (3, 3) or K2.shape != (3, 3):
        raise ValueError("K1 and K2 must be 3x3.")
    if R1.shape != (3, 3) or R2.shape != (3, 3):
        raise ValueError("R1 and R2 must be 3x3.")

    R_tilde = R2 @ R1.T
    t_tilde = t2 - R_tilde @ t1
    E = essential_matrix(R_tilde, t_tilde)
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F


def point_to_epipolar_line_distance(
    F: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    distance_in_image: int = 1,
) -> tuple[float, np.ndarray]:
    """
    Distance from point to induced epipolar line.

    Args:
        F: maps q1 to line in image2 (l2 = F q1).
        p1: point in image1 (2x1).
        p2: point in image2 (2x1).
        distance_in_image:
            - 1: distance from p1 to line in image1 induced by p2 (l1 = p2^T F)
            - 2: distance from p2 to line in image2 induced by p1 (l2 = F p1)

    Returns:
        distance, epipolar_line_used (3x1)
    """
    F = np.asarray(F, dtype=float)
    if F.shape != (3, 3):
        raise ValueError(f"F must be 3x3. Got {F.shape}.")

    p1h = to_homogeneous(ensure_column(p1, name="p1"))
    p2h = to_homogeneous(ensure_column(p2, name="p2"))

    if distance_in_image == 1:
        l1 = (p2h.T @ F).reshape(3, 1)
        d = point_line_distance(l1, p1h)
        return d, l1
    if distance_in_image == 2:
        l2 = F @ p1h
        d = point_line_distance(l2, p2h)
        return d, l2

    raise ValueError("distance_in_image must be 1 or 2.")
