"""Feature-matching helpers for Priority A templates."""

from __future__ import annotations

import numpy as np


def rootsift(des: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert SIFT descriptors to RootSIFT descriptors."""
    D = np.asarray(des, dtype=float)
    if D.ndim != 2:
        raise ValueError(f"des must be 2D array (N,D). Got shape {D.shape}.")

    l1 = np.sum(np.abs(D), axis=1, keepdims=True)
    l1 = np.maximum(l1, eps)
    return np.sqrt(D / l1)


def knn_match_l2(des1: np.ndarray, des2: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Brute-force KNN match by squared L2 distance using NumPy.

    Returns:
        indices (N1 x k), distances (N1 x k)
    """
    A = np.asarray(des1, dtype=float)
    B = np.asarray(des2, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("des1 and des2 must be 2D arrays.")
    if A.shape[1] != B.shape[1]:
        raise ValueError("Descriptor dimensions must match.")
    if B.shape[0] < k:
        raise ValueError(f"des2 has too few descriptors ({B.shape[0]}) for k={k}.")

    dists = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
    idx_part = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]

    row_idx = np.arange(A.shape[0])[:, None]
    d_part = dists[row_idx, idx_part]
    order = np.argsort(d_part, axis=1)

    idx_sorted = np.take_along_axis(idx_part, order, axis=1)
    d_sorted = np.take_along_axis(d_part, order, axis=1)
    return idx_sorted, np.sqrt(d_sorted)


def rootsift_ratio_match_count(
    des1: np.ndarray,
    des2: np.ndarray,
    ratio: float = 0.8,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Count RootSIFT matches passing Lowe ratio test.

    Returns:
        count, passed_mask, knn_indices, knn_distances
    """
    rs1 = rootsift(des1)
    rs2 = rootsift(des2)

    idx, dist = knn_match_l2(rs1, rs2, k=2)
    passed = dist[:, 0] < ratio * dist[:, 1]
    return int(np.sum(passed)), passed, idx, dist
