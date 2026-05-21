"""Projection and camera-frame helpers for Priority A templates."""

from __future__ import annotations

import numpy as np

from .core import ensure_column, from_homogeneous, to_homogeneous


def camera_intrinsic(f: float, c: tuple[float, float], alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """Build intrinsic matrix K from focal length and principal point."""
    cx, cy = c
    return np.array(
        [
            [f, beta * f, cx],
            [0.0, alpha * f, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return P = K [R|t]."""
    K = np.asarray(K, dtype=float)
    R = np.asarray(R, dtype=float)
    t = ensure_column(t, name="t")
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3. Got {K.shape}.")
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3. Got {R.shape}.")
    if t.shape != (3, 1):
        raise ValueError(f"t must be 3x1. Got {t.shape}.")
    return K @ np.hstack((R, t))


def project_points(K: np.ndarray, R: np.ndarray, t: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Project 3D world point(s) Q (3xn) to image coordinates (2xn)."""
    Q = np.asarray(Q, dtype=float)
    if Q.ndim == 1:
        Q = Q.reshape(3, 1)
    if Q.shape[0] != 3:
        raise ValueError(f"Q must be 3xn. Got {Q.shape}.")

    P = projection_matrix(K, R, t)
    q_h = P @ to_homogeneous(Q)
    return from_homogeneous(q_h)


def camera_center_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return camera center C in world coordinates from x_cam = R x_world + t."""
    R = np.asarray(R, dtype=float)
    t = ensure_column(t, name="t")
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3. Got {R.shape}.")
    if t.shape != (3, 1):
        raise ValueError(f"t must be 3x1. Got {t.shape}.")
    return -R.T @ t


def rt_to_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 transform T_world_to_cam from R,t."""
    R = np.asarray(R, dtype=float)
    t = ensure_column(t, name="t")
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3. Got {R.shape}.")
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


def transform_point_between_cameras(
    point_in_src_cam: np.ndarray,
    R_src: np.ndarray,
    t_src: np.ndarray,
    R_dst: np.ndarray,
    t_dst: np.ndarray,
) -> np.ndarray:
    """
    Transform 3D point from source camera frame to destination camera frame.

    Convention: x_cam = R x_world + t.
    """
    p_src = ensure_column(point_in_src_cam, name="point_in_src_cam")
    if p_src.shape != (3, 1):
        raise ValueError(f"point_in_src_cam must be 3x1. Got {p_src.shape}.")

    R_src = np.asarray(R_src, dtype=float)
    R_dst = np.asarray(R_dst, dtype=float)
    t_src = ensure_column(t_src, name="t_src")
    t_dst = ensure_column(t_dst, name="t_dst")

    x_world = R_src.T @ (p_src - t_src)
    x_dst = R_dst @ x_world + t_dst
    return x_dst
