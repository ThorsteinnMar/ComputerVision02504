"""Camera calibration helpers (DLT + Zhang-style planar calibration)."""

from __future__ import annotations

import numpy as np

from .core import cross_op, ensure_points, pi, pi_inv, svd_last_vector
from .homography import homography_dlt


def checkerboard_points(nx: int, ny: int, square_size: float = 1.0) -> np.ndarray:
    """Return planar checkerboard corner coordinates as (3, N)."""
    pts = []
    for i in range(nx):
        for j in range(ny):
            pts.append([i * square_size, j * square_size, 0.0])
    return np.asarray(pts, dtype=float).T


def estimate_projection_dlt(
    Q_world: np.ndarray,
    q_image: np.ndarray,
    normalize_image_points: bool = True,
) -> np.ndarray:
    """Estimate projection matrix P from 3D-2D correspondences using DLT.

    Q_world: (3,N)
    q_image: (2,N)
    Returns P: (3,4)
    """
    Q = ensure_points(Q_world, 3)
    q = ensure_points(q_image, 2)
    if Q.shape[1] != q.shape[1]:
        raise ValueError("Q_world and q_image must have same number of points")

    if normalize_image_points:
        # Use Hartley-style normalization for 2D points
        from .homography import normalize_points_2d

        qn, T = normalize_points_2d(q)
    else:
        qn, T = q, np.eye(3)

    qh = pi_inv(qn)
    Qh = pi_inv(Q)

    rows = []
    for i in range(Qh.shape[1]):
        rows.append(np.kron(Qh[:, i], cross_op(qh[:, i])))
    A = np.vstack(rows)
    p = svd_last_vector(A, normalize=False)
    Pn = p.reshape(4, 3).T
    P = np.linalg.inv(T) @ Pn
    return P


def reprojection_rmse(P: np.ndarray, Q_world: np.ndarray, q_image: np.ndarray) -> float:
    """RMSE of 2D reprojection error for projection matrix P."""
    p = np.asarray(P, dtype=float)
    if p.shape != (3, 4):
        raise ValueError("P must be 3x4")
    Q = ensure_points(Q_world, 3)
    q = ensure_points(q_image, 2)
    q_hat = pi(p @ pi_inv(Q))
    err = q_hat - q
    return float(np.sqrt(np.mean(err**2)))


def estimate_planar_homographies(
    Q_planar: np.ndarray,
    q_list: list[np.ndarray],
    normalize: bool = True,
) -> list[np.ndarray]:
    """Estimate one homography per image from planar world points.

    Q_planar: (3,N) or (2,N) planar points. If (3,N), z row is ignored.
    q_list: list of image points each shape (2,N)
    """
    Q = np.asarray(Q_planar, dtype=float)
    if Q.ndim != 2:
        raise ValueError("Q_planar must be 2D")
    if Q.shape[0] == 3:
        Q2 = Q[:2]
    elif Q.shape[0] == 2:
        Q2 = Q
    else:
        raise ValueError("Q_planar must have 2 or 3 rows")

    Hs = []
    for q in q_list:
        qi = ensure_points(q, 2)
        H = homography_dlt(qi, Q2, normalize=normalize)
        Hs.append(H)
    return Hs


def form_vi(H: np.ndarray, a: int, b: int) -> np.ndarray:
    """Construct Zhang's v_ij row from homography H using 1-based indices a,b."""
    h = np.asarray(H, dtype=float)
    if h.shape != (3, 3):
        raise ValueError("H must be 3x3")
    i = a - 1
    j = b - 1
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


def estimate_b_from_homographies(Hs: list[np.ndarray]) -> np.ndarray:
    """Estimate Zhang b vector (6,1) from homographies."""
    rows = []
    for H in Hs:
        v12 = form_vi(H, 1, 2)
        v11 = form_vi(H, 1, 1)
        v22 = form_vi(H, 2, 2)
        rows.append(v12)
        rows.append(v11 - v22)
    V = np.vstack(rows)
    b = svd_last_vector(V, normalize=False).reshape(6, 1)
    return b


def estimate_intrinsics_zhang(Hs: list[np.ndarray]) -> np.ndarray:
    """Estimate intrinsic matrix K from planar homographies (Zhang)."""
    b = estimate_b_from_homographies(Hs)
    B11, B12, B22, B13, B23, B33 = [float(x) for x in b.ravel()]

    denom = B11 * B22 - B12**2
    if np.isclose(denom, 0.0):
        raise ValueError("Degenerate homography set for intrinsic estimation")

    v0 = (B12 * B13 - B11 * B23) / denom
    lam = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11

    alpha = np.sqrt(lam / B11)
    beta = np.sqrt(lam * B11 / denom)
    gamma = -B12 * alpha**2 * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha**2 / lam

    return np.array(
        [
            [alpha, gamma, u0],
            [0.0, beta, v0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def estimate_extrinsics_zhang(K: np.ndarray, Hs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Estimate extrinsic R,t for each homography using known K."""
    k = np.asarray(K, dtype=float)
    if k.shape != (3, 3):
        raise ValueError("K must be 3x3")
    Kinv = np.linalg.inv(k)

    R_list = []
    t_list = []
    for H in Hs:
        h = np.asarray(H, dtype=float)
        h1, h2, h3 = h[:, 0], h[:, 1], h[:, 2]

        lam = 1.0 / np.linalg.norm(Kinv @ h1)
        r1 = lam * (Kinv @ h1)
        r2 = lam * (Kinv @ h2)
        r3 = np.cross(r1, r2)
        R_approx = np.column_stack([r1, r2, r3])

        # Orthonormalize R with SVD
        U, _, Vt = np.linalg.svd(R_approx)
        R = U @ Vt
        t = (lam * (Kinv @ h3)).reshape(3, 1)

        R_list.append(R)
        t_list.append(t)

    return np.asarray(R_list), np.asarray(t_list)


def calibrate_zhang(
    q_list: list[np.ndarray],
    Q_planar: np.ndarray,
    normalize_homography_points: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run planar Zhang calibration pipeline.

    Returns:
    - K: (3,3)
    - Rs: (M,3,3)
    - ts: (M,3,1)
    """
    Hs = estimate_planar_homographies(
        Q_planar,
        q_list,
        normalize=normalize_homography_points,
    )
    K = estimate_intrinsics_zhang(Hs)
    Rs, ts = estimate_extrinsics_zhang(K, Hs)
    return K, Rs, ts
