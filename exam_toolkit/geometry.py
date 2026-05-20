"""2D geometric primitives used in exam computations."""

from __future__ import annotations

import numpy as np

from .core import as_col, ensure_points, pi, pi_inv


def line_from_points(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Return homogeneous line l through two 2D points.

    Inputs: p1, p2 each shape (2,1) or (2,)
    Output: l shape (3,1)
    """
    a = ensure_points(p1, 2)[:, 0]
    b = ensure_points(p2, 2)[:, 0]
    l = np.cross(pi_inv(a)[:, 0], pi_inv(b)[:, 0]).reshape(3, 1)
    return l


def line_intersection(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """Return inhomogeneous intersection point of two homogeneous lines.

    Inputs: l1, l2 shape (3,1) or (3,)
    Output: p shape (2,1)
    """
    a = as_col(l1, 3).reshape(3)
    b = as_col(l2, 3).reshape(3)
    ph = np.cross(a, b).reshape(3, 1)
    return pi(ph)


def point_line_distance(line: np.ndarray, point_h: np.ndarray) -> float:
    """Distance between homogeneous 2D point and homogeneous 2D line.

    Inputs: line shape (3,1), point_h shape (3,1)
    """
    l = as_col(line, 3)
    p = as_col(point_h, 3)
    denom = abs(p[2, 0]) * np.sqrt(l[0, 0] ** 2 + l[1, 0] ** 2)
    if np.isclose(denom, 0.0):
        raise ValueError("Invalid line/point for distance computation")
    return float(abs((l.T @ p)[0, 0]) / denom)


def point_to_line_distance_xy(point_xy: np.ndarray, line: np.ndarray) -> float:
    """Distance from inhomogeneous 2D point to homogeneous 2D line."""
    p = ensure_points(point_xy, 2)
    return point_line_distance(line, pi_inv(p))


def batch_point_line_distance(points_xy: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Distance from multiple 2D points to one line.

    Input points shape: (2,N)
    Output: shape (N,)
    """
    pts = ensure_points(points_xy, 2)
    l = as_col(line, 3)
    ph = pi_inv(pts)
    num = np.abs((l.T @ ph).reshape(-1))
    den = np.abs(ph[2]) * np.sqrt(l[0, 0] ** 2 + l[1, 0] ** 2)
    if np.any(np.isclose(den, 0.0)):
        raise ValueError("Found invalid point for distance computation")
    return num / den
