"""Predicted computation templates based on exercises and exam patterns."""

from __future__ import annotations

import numpy as np

from exam_toolkit import (
    apply_homography,
    camera_center_from_rt,
    essential_from_fundamental,
    fundamental_from_essential,
    homography_dlt,
    line_from_points,
    line_intersection,
    pixel_to_normalized_points,
    point_line_distance,
    point_to_epipolar_distance,
    project_points,
    resize_intrinsics,
    rodrigues_to_matrix,
    triangulate_linear,
)


def template_project_given_KRtQ(K: np.ndarray, R: np.ndarray, t: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Given K, R, t, Q -> project 3D point(s) to image."""
    return project_points(K, R, t, Q)


def template_project_given_rodrigues(K: np.ndarray, rvec: np.ndarray, t: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Same as projection template but with Rodrigues vector input."""
    R = rodrigues_to_matrix(rvec)
    return project_points(K, R, t, Q)


def template_resize_intrinsics(K: np.ndarray, sx: float, sy: float | None = None) -> np.ndarray:
    """Given image resize factors -> compute new intrinsic matrix."""
    return resize_intrinsics(K, sx=sx, sy=sy)


def template_distortion_pixel_to_normalized(pixel: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Given pixel and K -> normalized camera coordinates."""
    return pixel_to_normalized_points(pixel, K)


def template_line_from_two_points_and_distance(p1: np.ndarray, p2: np.ndarray, p_query: np.ndarray) -> tuple[np.ndarray, float]:
    """Given two points define line, compute query-point distance."""
    l = line_from_points(p1, p2)
    d = point_line_distance(l, np.vstack([np.asarray(p_query).reshape(2, 1), [[1.0]]]))
    return l, d


def template_line_intersection(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """Given two homogeneous lines -> inhomogeneous intersection point."""
    return line_intersection(l1, l2)


def template_homography_from_correspondences(p1: np.ndarray, p2: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Given matched points -> estimate homography q1 ~ H q2."""
    return homography_dlt(p1, p2, normalize=normalize)


def template_apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply known homography to point(s)."""
    return apply_homography(H, points)


def template_epipolar_distance(F: np.ndarray, p1: np.ndarray, p2: np.ndarray, line_in_image: int = 2) -> float:
    """Given F and two points -> point-to-epiline distance."""
    return point_to_epipolar_distance(F, p1, p2, line_in_image=line_in_image)


def template_triangulation(P_list: list[np.ndarray], p_list: list[np.ndarray]) -> np.ndarray:
    """Given camera matrices and observations -> linear triangulated 3D point."""
    return triangulate_linear(p_list, P_list)


def template_camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Given R,t -> camera center in world coordinates."""
    return camera_center_from_rt(R, t)


def template_convert_E_and_F(E_or_F: np.ndarray, K1: np.ndarray, K2: np.ndarray, input_type: str = "E") -> np.ndarray:
    """Convert between essential and fundamental matrix when K1,K2 are known."""
    if input_type.upper() == "E":
        return fundamental_from_essential(E_or_F, K1, K2)
    if input_type.upper() == "F":
        return essential_from_fundamental(E_or_F, K1, K2)
    raise ValueError("input_type must be 'E' or 'F'")


PREDICTED_EXAMPLE = {
    "line_intersection": template_line_intersection(
        np.array([[1.0], [2.0], [-3.0]]),
        np.array([[-1.0], [1.0], [-3.0]]),
    ),
}
