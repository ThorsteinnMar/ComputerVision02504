"""2D/3D geometry helpers for Priority A templates."""

from __future__ import annotations

import numpy as np

from .core import ensure_column, to_homogeneous


def line_from_points_2d(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Return homogeneous line l (3x1) through two 2D points."""
    p1c = ensure_column(p1, name="p1")
    p2c = ensure_column(p2, name="p2")
    if p1c.shape[0] != 2 or p2c.shape[0] != 2:
        raise ValueError("p1 and p2 must be 2x1 points.")

    p1h = to_homogeneous(p1c)
    p2h = to_homogeneous(p2c)
    l = np.cross(p1h.reshape(3,), p2h.reshape(3,)).reshape(3, 1)
    return l


def point_line_distance(line: np.ndarray, point_h: np.ndarray) -> float:
    """Shortest distance from homogeneous point to homogeneous line."""
    l = ensure_column(line, name="line")
    p = ensure_column(point_h, name="point_h")
    if l.shape != (3, 1):
        raise ValueError(f"line must be 3x1. Got {l.shape}.")
    if p.shape != (3, 1):
        raise ValueError(f"point_h must be 3x1. Got {p.shape}.")

    num = float(np.abs((l.T @ p).item()))
    den = float(np.abs(p[2, 0]) * np.sqrt(l[0, 0] ** 2 + l[1, 0] ** 2))
    if den == 0.0:
        raise ValueError("Degenerate line/point configuration gives zero denominator.")
    return num / den


def line_ransac_inlier_count(
    x1: np.ndarray,
    x2: np.ndarray,
    points: np.ndarray,
    tau: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Count inliers to line through x1,x2 under distance threshold tau.

    Args:
        x1, x2: 2-vectors defining the line.
        points: 2xn points.
        tau: inlier threshold.

    Returns:
        count, inlier_mask, line
    """
    l = line_from_points_2d(x1, x2)
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] != 2:
        raise ValueError(f"points must be 2xn. Got {pts.shape}.")

    inlier_mask = np.zeros(pts.shape[1], dtype=bool)
    for i in range(pts.shape[1]):
        p = pts[:, i].reshape(2, 1)
        p_h = to_homogeneous(p)
        d = point_line_distance(l, p_h)
        inlier_mask[i] = d < tau

    return int(inlier_mask.sum()), inlier_mask, l


def harris_response(gxx: np.ndarray, gyy: np.ndarray, gxy: np.ndarray, k: float = 0.06) -> np.ndarray:
    """Compute Harris response from structure tensor terms."""
    a = np.asarray(gxx, dtype=float)
    b = np.asarray(gyy, dtype=float)
    c = np.asarray(gxy, dtype=float)
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError("gxx, gyy, gxy must share shape.")
    return a * b - c**2 - k * (a + b) ** 2


def harris_nms(response: np.ndarray, tau: float, neighborhood: int = 8) -> list[tuple[int, int]]:
    """
    Extract corners with threshold + local non-maximum suppression.

    Returns corners as (row, col).
    """
    r = np.asarray(response, dtype=float)
    if neighborhood not in (4, 8):
        raise ValueError("neighborhood must be 4 or 8.")

    corners: list[tuple[int, int]] = []
    rows, cols = r.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = r[i, j]
            if center <= tau:
                continue

            if neighborhood == 4:
                neighbors = np.array([r[i - 1, j], r[i + 1, j], r[i, j - 1], r[i, j + 1]])
            else:
                neighbors = r[i - 1 : i + 2, j - 1 : j + 2].copy().reshape(-1)
                neighbors = neighbors[np.arange(9) != 4]

            if center > neighbors.max():
                corners.append((i, j))

    return corners


def harris_from_tensor_with_nms(
    gxx: np.ndarray,
    gyy: np.ndarray,
    gxy: np.ndarray,
    k: float,
    tau: float,
    neighborhood: int = 8,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Convenience wrapper used by the Priority A Harris template."""
    r = harris_response(gxx, gyy, gxy, k=k)
    corners = harris_nms(r, tau=tau, neighborhood=neighborhood)
    return r, corners
