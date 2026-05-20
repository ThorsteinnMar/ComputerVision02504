"""Essential/fundamental matrix and epipolar geometry helpers."""

from __future__ import annotations

import numpy as np

from .core import as_col, cross_op, ensure_points, pi_inv
from .geometry import point_line_distance


def relative_pose(
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Relative pose from camera 1 frame to camera 2 frame.

    Assumes world-to-camera extrinsics: p_i = R_i X + t_i.
    Returns R_21, t_21 so p2 = R_21 p1 + t_21.
    """
    r1 = np.asarray(R1, dtype=float)
    r2 = np.asarray(R2, dtype=float)
    tt1 = as_col(t1, 3)
    tt2 = as_col(t2, 3)
    if r1.shape != (3, 3) or r2.shape != (3, 3):
        raise ValueError("R1 and R2 must be 3x3")

    r21 = r2 @ r1.T
    t21 = tt2 - r21 @ tt1
    return r21, t21


def essential_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute essential matrix E = [t]_x R."""
    r = np.asarray(R, dtype=float)
    tt = as_col(t, 3)
    if r.shape != (3, 3):
        raise ValueError("R must be 3x3")
    return cross_op(tt) @ r


def fundamental_from_essential(E: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """Compute fundamental matrix F = K2^-T E K1^-1."""
    e = np.asarray(E, dtype=float)
    k1 = np.asarray(K1, dtype=float)
    k2 = np.asarray(K2, dtype=float)
    if e.shape != (3, 3):
        raise ValueError("E must be 3x3")
    if k1.shape != (3, 3) or k2.shape != (3, 3):
        raise ValueError("K1 and K2 must be 3x3")
    return np.linalg.inv(k2).T @ e @ np.linalg.inv(k1)


def essential_from_fundamental(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """Compute essential matrix E = K2^T F K1."""
    f = np.asarray(F, dtype=float)
    k1 = np.asarray(K1, dtype=float)
    k2 = np.asarray(K2, dtype=float)
    if f.shape != (3, 3):
        raise ValueError("F must be 3x3")
    if k1.shape != (3, 3) or k2.shape != (3, 3):
        raise ValueError("K1 and K2 must be 3x3")
    return k2.T @ f @ k1


def fundamental_matrix_from_extrinsics(
    K1: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    K2: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
) -> np.ndarray:
    """Compute F from two camera extrinsics/intrinsics."""
    r21, t21 = relative_pose(R1, t1, R2, t2)
    E = essential_matrix(r21, t21)
    return fundamental_from_essential(E, K1, K2)


def enforce_rank2(F: np.ndarray) -> np.ndarray:
    """Project matrix to closest rank-2 matrix (SVD)."""
    f = np.asarray(F, dtype=float)
    if f.shape != (3, 3):
        raise ValueError("F must be 3x3")
    U, S, Vt = np.linalg.svd(f)
    S[-1] = 0.0
    return U @ np.diag(S) @ Vt


def epipolar_line(F: np.ndarray, point: np.ndarray, in_image: int = 2) -> np.ndarray:
    """Compute epipolar line induced by a point.

    in_image=2: line in image 2 from point in image 1 => l2 = F q1
    in_image=1: line in image 1 from point in image 2 => l1 = F^T q2
    """
    f = np.asarray(F, dtype=float)
    if f.shape != (3, 3):
        raise ValueError("F must be 3x3")
    p = ensure_points(point, 2)
    qh = pi_inv(p)
    if in_image == 2:
        return f @ qh
    if in_image == 1:
        return f.T @ qh
    raise ValueError("in_image must be 1 or 2")


def point_to_epipolar_distance(
    F: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    line_in_image: int = 2,
) -> float:
    """Distance from point to induced epipolar line.

    line_in_image=2: distance of p2 to l2 induced by p1
    line_in_image=1: distance of p1 to l1 induced by p2
    """
    p1c = ensure_points(p1, 2)
    p2c = ensure_points(p2, 2)
    if line_in_image == 2:
        l2 = epipolar_line(F, p1c, in_image=2)
        return point_line_distance(l2, pi_inv(p2c))
    if line_in_image == 1:
        l1 = epipolar_line(F, p2c, in_image=1)
        return point_line_distance(l1, pi_inv(p1c))
    raise ValueError("line_in_image must be 1 or 2")


def sampson_distance(F: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Sampson distance for point correspondences.

    Inputs p1, p2 shape: (2,N)
    Output shape: (N,)
    """
    f = np.asarray(F, dtype=float)
    q1 = pi_inv(ensure_points(p1, 2))
    q2 = pi_inv(ensure_points(p2, 2))

    l2 = f @ q1
    l1 = f.T @ q2
    num = np.sum(q2 * l2, axis=0) ** 2
    den = l2[0] ** 2 + l2[1] ** 2 + l1[0] ** 2 + l1[1] ** 2
    den = np.where(np.isclose(den, 0.0), np.finfo(float).eps, den)
    return num / den
