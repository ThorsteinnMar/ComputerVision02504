"""Triangulation and reprojection error helpers."""

from __future__ import annotations

import numpy as np

from .core import ensure_points, pi, pi_inv


def triangulate_linear(q_list: list[np.ndarray], P_list: list[np.ndarray]) -> np.ndarray:
    """Linear triangulation of one 3D point from multiple views.

    q_list: list of (2,1) points
    P_list: list of (3,4) projection matrices
    Returns: Q (3,1)
    """
    if len(q_list) != len(P_list):
        raise ValueError("q_list and P_list must have same length")
    if len(q_list) < 2:
        raise ValueError("Need at least 2 views")

    A_rows: list[np.ndarray] = []
    for q, P in zip(q_list, P_list):
        qi = ensure_points(q, 2)
        p = np.asarray(P, dtype=float)
        if p.shape != (3, 4):
            raise ValueError("Each projection matrix must be 3x4")
        x, y = qi[:, 0]
        A_rows.append(x * p[2] - p[0])
        A_rows.append(y * p[2] - p[1])

    A = np.vstack(A_rows)
    _, _, vt = np.linalg.svd(A)
    Xh = vt[-1]
    if np.isclose(Xh[-1], 0.0):
        raise ValueError("Degenerate triangulation result (homogeneous scale zero)")
    X = (Xh[:3] / Xh[3]).reshape(3, 1)
    return X


def reprojection_errors(
    Q: np.ndarray,
    q_list: list[np.ndarray],
    P_list: list[np.ndarray],
) -> np.ndarray:
    """Per-view reprojection errors for one 3D point.

    Returns shape (n_views,)
    """
    q3 = ensure_points(Q, 3)
    Qh = pi_inv(q3)
    errs = []
    for q, P in zip(q_list, P_list):
        qi = ensure_points(q, 2)
        p = np.asarray(P, dtype=float)
        qhat = pi(p @ Qh)
        errs.append(float(np.linalg.norm(qi - qhat)))
    return np.array(errs, dtype=float)


def reprojection_rmse(
    Q: np.ndarray,
    q_list: list[np.ndarray],
    P_list: list[np.ndarray],
) -> float:
    """RMSE of reprojection residuals for one 3D point across views."""
    errs = reprojection_errors(Q, q_list, P_list)
    return float(np.sqrt(np.mean(errs**2)))


def triangulate_nonlinear(
    q_list: list[np.ndarray],
    P_list: list[np.ndarray],
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Nonlinear triangulation minimizing reprojection residuals.

    Returns Q (3,1).
    """
    try:
        from scipy.optimize import least_squares
    except Exception as exc:  # pragma: no cover
        raise ImportError("scipy is required for nonlinear triangulation") from exc

    if x0 is None:
        x0 = triangulate_linear(q_list, P_list)
    x0 = ensure_points(x0, 3).reshape(-1)

    def residuals(x: np.ndarray) -> np.ndarray:
        Q = x.reshape(3, 1)
        Qh = pi_inv(Q)
        res = []
        for q, P in zip(q_list, P_list):
            qi = ensure_points(q, 2)
            p = np.asarray(P, dtype=float)
            qhat = pi(p @ Qh)
            res.extend((qi - qhat).reshape(-1))
        return np.asarray(res, dtype=float)

    out = least_squares(residuals, x0)
    return out.x.reshape(3, 1)
