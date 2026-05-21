"""Triangulation helpers for Priority A templates."""

from __future__ import annotations

import numpy as np

from .core import ensure_column, from_homogeneous, to_homogeneous


def triangulate_linear(q_list: list[np.ndarray], P_list: list[np.ndarray]) -> np.ndarray:
    """Linear triangulation of one 3D point from multiple views."""
    if len(q_list) != len(P_list):
        raise ValueError("q_list and P_list must have equal length.")

    A_rows = []
    for q, P in zip(q_list, P_list):
        qi = ensure_column(q, name="q")
        if qi.shape != (2, 1):
            raise ValueError(f"Each q must be 2x1. Got {qi.shape}.")
        Pi = np.asarray(P, dtype=float)
        if Pi.shape != (3, 4):
            raise ValueError(f"Each P must be 3x4. Got {Pi.shape}.")

        A_rows.append(Pi[2] * qi[0, 0] - Pi[0])
        A_rows.append(Pi[2] * qi[1, 0] - Pi[1])

    A = np.array(A_rows, dtype=float)
    _, _, vt = np.linalg.svd(A)
    Qh = vt[-1].reshape(4, 1)
    Q = from_homogeneous(Qh)
    return Q


def reprojection_residual_vector(Q: np.ndarray, q_list: list[np.ndarray], P_list: list[np.ndarray]) -> np.ndarray:
    """Return stacked 2n residual vector for nonlinear triangulation."""
    Qc = ensure_column(Q, name="Q")
    if Qc.shape != (3, 1):
        raise ValueError(f"Q must be 3x1. Got {Qc.shape}.")

    Qh = to_homogeneous(Qc)
    residuals = np.zeros(2 * len(q_list), dtype=float)
    for i, (q, P) in enumerate(zip(q_list, P_list)):
        q_obs = ensure_column(q, name="q_obs")
        q_est = from_homogeneous(np.asarray(P, dtype=float) @ Qh)
        res = (q_obs - q_est).reshape(-1)
        residuals[2 * i : 2 * (i + 1)] = res
    return residuals


def triangulate_nonlinear(
    q_list: list[np.ndarray],
    P_list: list[np.ndarray],
    max_iters: int = 50,
    tol_step: float = 1e-10,
    fd_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Nonlinear triangulation by minimizing reprojection residuals (pure NumPy Gauss-Newton).

    Returns:
        Q_nonlinear, Q_linear_init, info
    """
    Q0 = triangulate_linear(q_list, P_list)
    x = Q0.reshape(3,).copy()

    def residual_fn(x_vec: np.ndarray) -> np.ndarray:
        return reprojection_residual_vector(x_vec.reshape(3, 1), q_list, P_list)

    nfev = 0
    damping = 1e-9
    success = False
    message = "max iterations reached"

    for _ in range(max_iters):
        r = residual_fn(x)
        nfev += 1
        cost = 0.5 * float(r @ r)

        # Finite-difference Jacobian J: (2n x 3)
        J = np.zeros((r.size, 3), dtype=float)
        for j in range(3):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += fd_eps
            x_minus[j] -= fd_eps
            r_plus = residual_fn(x_plus)
            r_minus = residual_fn(x_minus)
            nfev += 2
            J[:, j] = (r_plus - r_minus) / (2.0 * fd_eps)

        JTJ = J.T @ J + damping * np.eye(3, dtype=float)
        g = J.T @ r
        try:
            step = -np.linalg.solve(JTJ, g)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(JTJ, g, rcond=None)[0]

        if float(np.linalg.norm(step)) < tol_step:
            success = True
            message = "step below tolerance"
            break

        x_candidate = x + step
        r_candidate = residual_fn(x_candidate)
        nfev += 1
        candidate_cost = 0.5 * float(r_candidate @ r_candidate)

        if candidate_cost < cost:
            x = x_candidate
            damping = max(damping * 0.5, 1e-12)
        else:
            damping = min(damping * 10.0, 1e6)

    Qn = x.reshape(3, 1)
    info = {
        "cost": 0.5 * float(residual_fn(x) @ residual_fn(x)),
        "nfev": int(nfev),
        "success": bool(success),
        "message": str(message),
    }
    return Qn, Q0, info
