"""Core utilities for Priority A exam templates."""

from __future__ import annotations

import numpy as np


def ensure_column(x, name: str = "x") -> np.ndarray:
    """Return x as a float column vector of shape (n, 1)."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] != 1:
        raise ValueError(f"{name} must be a vector convertible to shape (n,1). Got {arr.shape}.")
    return arr


def to_homogeneous(p: np.ndarray) -> np.ndarray:
    """Convert inhomogeneous points (d x n) to homogeneous ((d+1) x n)."""
    arr = np.asarray(p, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.vstack((arr, np.ones((1, arr.shape[1]), dtype=float)))


def from_homogeneous(ph: np.ndarray) -> np.ndarray:
    """Convert homogeneous points ((d+1) x n) to inhomogeneous (d x n)."""
    arr = np.asarray(ph, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    w = arr[-1:]
    if np.any(np.isclose(w, 0.0)):
        raise ValueError("Homogeneous scale contains zeros; cannot dehomogenize.")
    return arr[:-1] / w


def skew(v: np.ndarray) -> np.ndarray:
    """Return 3x3 skew-symmetric matrix [v]_x for a 3-vector."""
    vc = ensure_column(v, name="v")
    if vc.shape[0] != 3:
        raise ValueError(f"v must have 3 elements. Got shape {vc.shape}.")
    x, y, z = vc.reshape(3,)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues rotation vector (3,) or (3,1) to 3x3 rotation matrix."""
    rv = ensure_column(rvec, name="rvec")
    if rv.shape[0] != 3:
        raise ValueError(f"rvec must have 3 elements. Got shape {rv.shape}.")

    r = rv.reshape(3,)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3, dtype=float)

    k = r / theta
    K = skew(k)
    I = np.eye(3, dtype=float)
    return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
