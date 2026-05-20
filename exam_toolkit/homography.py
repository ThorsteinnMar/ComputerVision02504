"""Homography estimation and evaluation helpers."""

from __future__ import annotations

import numpy as np

from .core import ensure_points, pi, pi_inv, svd_last_vector


def normalize_points_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization of 2D points.

    Returns:
    - points_norm: (2,N)
    - T: (3,3), so q_norm = pi(T @ pi_inv(q))
    """
    q = ensure_points(points, 2)
    mean = np.mean(q, axis=1, keepdims=True)
    q_centered = q - mean
    d = np.sqrt(np.sum(q_centered**2, axis=0))
    mean_dist = float(np.mean(d))
    if np.isclose(mean_dist, 0.0):
        raise ValueError("Cannot normalize identical points")
    s = np.sqrt(2.0) / mean_dist
    T = np.array(
        [
            [s, 0.0, -s * mean[0, 0]],
            [0.0, s, -s * mean[1, 0]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    qn = pi(T @ pi_inv(q))
    return qn, T


def homography_dlt(
    q1: np.ndarray,
    q2: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Estimate homography H from q1 ~ H q2.

    Inputs q1, q2 shape: (2,N), N>=4
    Returns H shape: (3,3)
    """
    p1 = ensure_points(q1, 2)
    p2 = ensure_points(q2, 2)
    if p1.shape[1] != p2.shape[1]:
        raise ValueError("q1 and q2 must contain same number of points")
    if p1.shape[1] < 4:
        raise ValueError("Need at least 4 correspondences")

    T1 = np.eye(3)
    T2 = np.eye(3)
    if normalize:
        p1, T1 = normalize_points_2d(p1)
        p2, T2 = normalize_points_2d(p2)

    n = p1.shape[1]
    A = np.zeros((2 * n, 9), dtype=float)
    for i in range(n):
        u, v = p1[:, i]
        x, y = p2[:, i]
        A[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]

    h = svd_last_vector(A, normalize=False)
    Hn = h.reshape(3, 3)

    H = np.linalg.inv(T1) @ Hn @ T2
    if not np.isclose(H[2, 2], 0.0):
        H = H / H[2, 2]
    return H


def apply_homography(H: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Apply homography to 2D points: q_out = pi(H @ pi_inv(q))."""
    h = np.asarray(H, dtype=float)
    if h.shape != (3, 3):
        raise ValueError("H must be 3x3")
    pts = ensure_points(q, 2)
    return pi(h @ pi_inv(pts))


def transfer_error(H: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Forward transfer error ||q1 - H q2||_2 per correspondence."""
    p1 = ensure_points(q1, 2)
    p2 = ensure_points(q2, 2)
    q1_hat = apply_homography(H, p2)
    return np.linalg.norm(p1 - q1_hat, axis=0)


def symmetric_transfer_error(H: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Symmetric transfer error per correspondence."""
    h = np.asarray(H, dtype=float)
    if h.shape != (3, 3):
        raise ValueError("H must be 3x3")
    e12 = transfer_error(h, q1, q2)
    e21 = transfer_error(np.linalg.inv(h), q2, q1)
    return e12 + e21


def homography_inlier_mask(
    H: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    threshold: float,
    symmetric: bool = True,
) -> np.ndarray:
    """Return boolean inlier mask under transfer error threshold."""
    if symmetric:
        err = symmetric_transfer_error(H, q1, q2)
    else:
        err = transfer_error(H, q1, q2)
    return err < threshold
