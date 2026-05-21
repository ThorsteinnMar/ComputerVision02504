"""Homography + minimal Zhang intrinsics helpers for Priority A templates."""

from __future__ import annotations

import numpy as np

from .core import from_homogeneous, to_homogeneous


def normalize_points_2d(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize 2D points to zero mean and unit std per axis."""
    pts = np.asarray(q, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(2, 1)
    if pts.shape[0] != 2:
        raise ValueError(f"q must be 2xn. Got {pts.shape}.")

    mu = np.mean(pts, axis=1, keepdims=True)
    sd = np.std(pts, axis=1, keepdims=True)
    if np.any(np.isclose(sd, 0.0)):
        raise ValueError("Cannot normalize when std is zero on an axis.")

    T_inv = np.array(
        [
            [sd[0, 0], 0.0, mu[0, 0]],
            [0.0, sd[1, 0], mu[1, 0]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    T = np.linalg.inv(T_inv)
    qn = from_homogeneous(T @ to_homogeneous(pts))
    return qn, T


def estimate_homography_dlt(p_target: np.ndarray, p_source: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Estimate homography H such that p_target ~ H p_source.

    Args:
        p_target: 2xn target points.
        p_source: 2xn source points.
    """
    q1 = np.asarray(p_target, dtype=float)
    q2 = np.asarray(p_source, dtype=float)
    if q1.ndim == 1:
        q1 = q1.reshape(2, 1)
    if q2.ndim == 1:
        q2 = q2.reshape(2, 1)

    if q1.shape[0] != 2 or q2.shape[0] != 2:
        raise ValueError("p_target and p_source must be 2xn.")
    if q1.shape[1] != q2.shape[1]:
        raise ValueError("p_target and p_source must have same number of points.")
    if q1.shape[1] < 4:
        raise ValueError("At least 4 correspondences are required.")

    T1 = np.eye(3)
    T2 = np.eye(3)
    if normalize:
        q1, T1 = normalize_points_2d(q1)
        q2, T2 = normalize_points_2d(q2)

    n = q1.shape[1]
    A = np.zeros((2 * n, 9), dtype=float)
    for i in range(n):
        x, y = q1[:, i]
        u, v = q2[:, i]
        A[2 * i] = [u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v, -x]
        A[2 * i + 1] = [0.0, 0.0, 0.0, u, v, 1.0, -y * u, -y * v, -y]

    _, _, vt = np.linalg.svd(A)
    Hn = vt[-1].reshape(3, 3)

    H = Hn
    if normalize:
        H = np.linalg.inv(T1) @ Hn @ T2

    if not np.isclose(H[-1, -1], 0.0):
        H = H / H[-1, -1]
    return H


def apply_homography(H: np.ndarray, p_source: np.ndarray) -> np.ndarray:
    """Apply H to source 2D point(s)."""
    pts = np.asarray(p_source, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(2, 1)
    return from_homogeneous(np.asarray(H, dtype=float) @ to_homogeneous(pts))


def checkerboard_world_points(rows: int, cols: int, square_size: float = 1.0) -> np.ndarray:
    """Generate 3D checkerboard points (3 x (rows*cols)) on Z=0 plane."""
    pts = np.array(
        [
            (i - (rows - 1) / 2.0, j - (cols - 1) / 2.0, 0.0)
            for i in range(rows)
            for j in range(cols)
        ],
        dtype=float,
    ).T
    return pts * float(square_size)


def reorder_checkerboard_world_points_for_opencv(Q_world: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    Reorder row-major world points to match cv2.findChessboardCorners order used in course notebooks.
    """
    Q = np.asarray(Q_world, dtype=float)
    if Q.shape != (3, rows * cols):
        raise ValueError(f"Q_world must be (3, {rows*cols}). Got {Q.shape}.")

    out = np.zeros_like(Q)
    for i in range(rows):
        for j in range(cols):
            old_idx = i * cols + j
            new_idx = j * rows + (rows - 1 - i)
            out[:, new_idx] = Q[:, old_idx]
    return out


def _v_ij(H: np.ndarray, i: int, j: int) -> np.ndarray:
    """Zhang v_ij row for homography H with 0-based i,j."""
    h = H
    return np.array(
        [
            h[0, i] * h[0, j],
            h[0, i] * h[1, j] + h[1, i] * h[0, j],
            h[1, i] * h[1, j],
            h[2, i] * h[0, j] + h[0, i] * h[2, j],
            h[2, i] * h[1, j] + h[1, i] * h[2, j],
            h[2, i] * h[2, j],
        ],
        dtype=float,
    )


def estimate_homographies_from_checkerboards(
    qs: list[np.ndarray],
    Q_world: np.ndarray,
    normalize: bool = True,
) -> list[np.ndarray]:
    """Estimate one homography per checkerboard view using 2D-2D correspondences."""
    Q_plane = np.asarray(Q_world, dtype=float)[:2]
    Hs: list[np.ndarray] = []
    for q in qs:
        H = estimate_homography_dlt(np.asarray(q, dtype=float), Q_plane, normalize=normalize)
        Hs.append(H)
    return Hs


def estimate_intrinsics_zhang_from_homographies(Hs: list[np.ndarray]) -> np.ndarray:
    """Estimate K from homographies via Zhang's linear constraints."""
    if len(Hs) < 2:
        raise ValueError("At least two checkerboard views are recommended for stable Zhang intrinsics.")

    V_rows = []
    for H in Hs:
        v12 = _v_ij(H, 0, 1)
        v11 = _v_ij(H, 0, 0)
        v22 = _v_ij(H, 1, 1)
        V_rows.append(v12)
        V_rows.append(v11 - v22)

    V = np.vstack(V_rows)
    _, _, vt = np.linalg.svd(V)
    b = vt[-1]

    B11, B12, B22, B13, B23, B33 = b
    denom = (B11 * B22 - B12**2)
    if abs(float(denom)) < 1e-14:
        raise ValueError("Degenerate homography set for Zhang intrinsics (denominator near zero).")

    v0 = (B12 * B13 - B11 * B23) / denom
    lam = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lam / B11)
    beta = np.sqrt(lam * B11 / denom)
    gamma = -B12 * alpha**2 * beta / lam
    u0 = lam * v0 / beta - B13 * alpha**2 / lam

    K = np.array(
        [
            [alpha, gamma, u0],
            [0.0, beta, v0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return K


def zhang_focal_from_checkerboards(qs: list[np.ndarray], Q_world: np.ndarray) -> tuple[float, float, np.ndarray, list[np.ndarray]]:
    """Compute focal lengths fx, fy from checkerboard correspondences using Zhang linear stage."""
    Hs = estimate_homographies_from_checkerboards(qs, Q_world, normalize=True)
    K = estimate_intrinsics_zhang_from_homographies(Hs)
    return float(K[0, 0]), float(K[1, 1]), K, Hs
