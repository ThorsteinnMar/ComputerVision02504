"""Camera and projection helpers for exam questions."""

from __future__ import annotations

import numpy as np

from .core import as_col, ensure_points, pi, pi_inv
from .distortion import distort_normalized_points


def camera_intrinsic(
    f: float,
    c: tuple[float, float],
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """Build intrinsic matrix K from course notation."""
    return np.array(
        [
            [f, beta * f, c[0]],
            [0.0, alpha * f, c[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def resize_intrinsics(K: np.ndarray, sx: float, sy: float | None = None) -> np.ndarray:
    """Scale intrinsic matrix for image resize.

    sx: x-scale factor
    sy: y-scale factor (defaults to sx)
    """
    k = np.asarray(K, dtype=float)
    if k.shape != (3, 3):
        raise ValueError("K must be 3x3")
    if sy is None:
        sy = sx
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return S @ k


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute projection matrix P = K [R|t]."""
    k = np.asarray(K, dtype=float)
    r = np.asarray(R, dtype=float)
    tt = as_col(t, 3)
    if k.shape != (3, 3):
        raise ValueError("K must be 3x3")
    if r.shape != (3, 3):
        raise ValueError("R must be 3x3")
    return k @ np.hstack([r, tt])


def project_points(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    Q: np.ndarray,
    dist_coeffs: list[float] | tuple[float, ...] | None = None,
) -> np.ndarray:
    """Project 3D world points to image pixels.

    Inputs:
    - K: (3,3)
    - R: (3,3)
    - t: (3,1)
    - Q: (3,N)
    Returns:
    - p: (2,N)
    """
    q_world = ensure_points(Q, 3)
    r = np.asarray(R, dtype=float)
    tt = as_col(t, 3)
    k = np.asarray(K, dtype=float)
    if r.shape != (3, 3):
        raise ValueError("R must be 3x3")
    if k.shape != (3, 3):
        raise ValueError("K must be 3x3")

    q_cam = r @ q_world + tt
    q_norm = pi(q_cam)
    if dist_coeffs is not None and len(dist_coeffs) > 0:
        q_norm = distort_normalized_points(q_norm, dist_coeffs)

    p = pi(k @ pi_inv(q_norm))
    return p


def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues rotation vector to 3x3 matrix using OpenCV."""
    import cv2

    rv = as_col(rvec, 3)
    R, _ = cv2.Rodrigues(rv)
    return R


def matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Rodrigues vector using OpenCV."""
    import cv2

    r = np.asarray(R, dtype=float)
    if r.shape != (3, 3):
        raise ValueError("R must be 3x3")
    rvec, _ = cv2.Rodrigues(r)
    return rvec.reshape(3, 1)


def camera_center_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Camera center in world coordinates: C = -R^T t."""
    r = np.asarray(R, dtype=float)
    tt = as_col(t, 3)
    if r.shape != (3, 3):
        raise ValueError("R must be 3x3")
    return -r.T @ tt


def world_to_camera(R: np.ndarray, t: np.ndarray, Q_world: np.ndarray) -> np.ndarray:
    """Transform world points (3,N) to camera frame (3,N)."""
    r = np.asarray(R, dtype=float)
    tt = as_col(t, 3)
    q = ensure_points(Q_world, 3)
    if r.shape != (3, 3):
        raise ValueError("R must be 3x3")
    return r @ q + tt


def camera_to_world(R: np.ndarray, t: np.ndarray, Q_cam: np.ndarray) -> np.ndarray:
    """Transform camera-frame points (3,N) to world frame (3,N)."""
    r = np.asarray(R, dtype=float)
    tt = as_col(t, 3)
    q = ensure_points(Q_cam, 3)
    if r.shape != (3, 3):
        raise ValueError("R must be 3x3")
    return r.T @ (q - tt)


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform from world to camera."""
    r = np.asarray(R, dtype=float)
    tt = as_col(t, 3)
    if r.shape != (3, 3):
        raise ValueError("R must be 3x3")
    T = np.eye(4, dtype=float)
    T[:3, :3] = r
    T[:3, 3:] = tt
    return T


def transform_points(T: np.ndarray, points3: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to 3D points (3,N), return (3,N)."""
    t = np.asarray(T, dtype=float)
    if t.shape != (4, 4):
        raise ValueError("T must be 4x4")
    p = ensure_points(points3, 3)
    return pi(t @ pi_inv(p))


def camera_to_camera(
    R_from: np.ndarray,
    t_from: np.ndarray,
    R_to: np.ndarray,
    t_to: np.ndarray,
    points_in_from: np.ndarray,
) -> np.ndarray:
    """Convert 3D points from camera A frame to camera B frame."""
    q_world = camera_to_world(R_from, t_from, points_in_from)
    return world_to_camera(R_to, t_to, q_world)
