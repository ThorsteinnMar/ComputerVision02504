"""Core numerical helpers for DTU 02504 exam computations."""

from __future__ import annotations

import numpy as np


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


def as_float_array(x: ArrayLike) -> np.ndarray:
    """Return input as a float NumPy array."""
    return np.asarray(x, dtype=float)


def as_col(x: ArrayLike, size: int | None = None) -> np.ndarray:
    """Return input as a column vector with optional size check."""
    a = as_float_array(x).reshape(-1, 1)
    if size is not None and a.shape != (size, 1):
        raise ValueError(f"Expected shape {(size, 1)}, got {a.shape}")
    return a


def ensure_points(p: ArrayLike, dim: int) -> np.ndarray:
    """Ensure a point matrix is shaped (dim, N)."""
    a = as_float_array(p)
    if a.ndim == 1:
        if a.size != dim:
            raise ValueError(f"Expected {dim} values, got {a.size}")
        return a.reshape(dim, 1)
    if a.ndim != 2:
        raise ValueError("Points must be 1D or 2D array")
    if a.shape[0] == dim:
        return a
    if a.shape[1] == dim:
        return a.T
    raise ValueError(f"Expected shape ({dim}, N) or (N, {dim}), got {a.shape}")


def pi(ph: ArrayLike) -> np.ndarray:
    """Homogeneous to inhomogeneous conversion.

    Input shape: (d+1, N)
    Output shape: (d, N)
    """
    ph_arr = as_float_array(ph)
    if ph_arr.ndim == 1:
        ph_arr = ph_arr.reshape(-1, 1)
    if ph_arr.ndim != 2 or ph_arr.shape[0] < 2:
        raise ValueError("Input must have shape (d+1, N)")
    denom = ph_arr[-1:]
    if np.any(np.isclose(denom, 0.0)):
        raise ValueError("Last homogeneous coordinate contains zeros")
    out = ph_arr[:-1] / denom
    return out


def pi_inv(p: ArrayLike) -> np.ndarray:
    """Inhomogeneous to homogeneous conversion.

    Input shape: (d, N)
    Output shape: (d+1, N)
    """
    p_arr = as_float_array(p)
    if p_arr.ndim == 1:
        p_arr = p_arr.reshape(-1, 1)
    if p_arr.ndim != 2:
        raise ValueError("Input must be 1D or 2D")
    ones = np.ones((1, p_arr.shape[1]), dtype=float)
    return np.vstack([p_arr, ones])


def normalize_homogeneous(ph: ArrayLike) -> np.ndarray:
    """Scale homogeneous vectors so last coordinate equals 1."""
    ph_arr = as_float_array(ph)
    if ph_arr.ndim == 1:
        ph_arr = ph_arr.reshape(-1, 1)
    if ph_arr.ndim != 2:
        raise ValueError("Input must be 1D or 2D")
    scale = ph_arr[-1:]
    if np.any(np.isclose(scale, 0.0)):
        raise ValueError("Cannot normalize homogeneous vector with last=0")
    return ph_arr / scale


def cross_op(r: ArrayLike) -> np.ndarray:
    """Return 3x3 cross-product matrix [r]_x for a 3-vector."""
    v = as_col(r, 3).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def svd_last_vector(a: ArrayLike, normalize: bool = True) -> np.ndarray:
    """Solve homogeneous system A x = 0 using SVD and return last singular vector."""
    A = as_float_array(a)
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    _, _, vt = np.linalg.svd(A)
    x = vt[-1]
    if normalize and not np.isclose(x[-1], 0.0):
        x = x / x[-1]
    return x


def least_squares(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Least-squares solution to A x ≈ b using NumPy lstsq."""
    A = as_float_array(a)
    B = as_float_array(b)
    x, *_ = np.linalg.lstsq(A, B, rcond=None)
    return x
