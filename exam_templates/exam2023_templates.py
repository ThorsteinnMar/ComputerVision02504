"""Ready-to-run computational templates for Exam 2023 notebook questions."""

from __future__ import annotations

import numpy as np

from exam_toolkit import (
    camera_intrinsic,
    difference_of_gaussians,
    fundamental_matrix_from_extrinsics,
    harris_response_from_tensor,
    line_from_points,
    non_max_suppression_2d,
    pi_inv,
    point_line_distance,
    point_to_epipolar_distance,
    project_points,
    ransac_iterations_required,
    resize_intrinsics,
    triangulate_linear,
)


def q1_intrinsic_matrix(f: float, cx: float, cy: float, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """Q1 template: build K from focal length and principal point."""
    return camera_intrinsic(f, (cx, cy), alpha=alpha, beta=beta)


def q2_resize_intrinsics_or_pixel(
    K: np.ndarray,
    sx: float,
    sy: float,
    point_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Q2 template: resize intrinsics and optionally transform pixel coordinates."""
    K_new = resize_intrinsics(K, sx=sx, sy=sy)
    if point_xy is None:
        return K_new, None
    p = np.asarray(point_xy, dtype=float).reshape(2, 1)
    p_new = np.array([[sx, 0.0], [0.0, sy]], dtype=float) @ p
    return K_new, p_new


def q3_project_single_point(K: np.ndarray, rvec: np.ndarray, t: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Q3 template: project one 3D point with K, Rodrigues vector, and t."""
    from exam_toolkit import rodrigues_to_matrix

    R = rodrigues_to_matrix(rvec)
    return project_points(K, R, t, Q)


def q5_epipolar_distance(
    K: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> float:
    """Q5 template: distance from p2 to epipolar line induced by p1."""
    F = fundamental_matrix_from_extrinsics(K, R1, t1, K, R2, t2)
    return point_to_epipolar_distance(F, p1, p2, line_in_image=2)


def q6_linear_triangulation(p_list: list[np.ndarray], P_list: list[np.ndarray]) -> np.ndarray:
    """Q6 template: triangulate one point from multiple views."""
    return triangulate_linear(p_list, P_list)


def q11_point_line_distance(line_h: np.ndarray, point_h: np.ndarray) -> float:
    """Q11 template: point-line distance in homogeneous coordinates."""
    return point_line_distance(line_h, point_h)


def q12_harris_from_tensor(gxx: np.ndarray, gyy: np.ndarray, gxy: np.ndarray, k: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Q12 template: Harris response and 4-neighborhood NMS corners."""
    r = harris_response_from_tensor(gxx, gyy, gxy, k=k)
    corners = non_max_suppression_2d(r, threshold=tau, connectivity=4)
    return r, corners


def q13_line_ransac_inlier_count(x1: np.ndarray, x2: np.ndarray, points: np.ndarray, tau: float) -> int:
    """Q13 template: count inliers to the line through two points."""
    l = line_from_points(x1, x2)
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] != 2:
        pts = pts.T
    inliers = 0
    for i in range(pts.shape[1]):
        d = point_line_distance(l, pi_inv(pts[:, i].reshape(2, 1)))
        if d < tau:
            inliers += 1
    return inliers


def q14_ransac_iterations(best_inliers: int, total_points: int, confidence: float, sample_size: int) -> float:
    """Q14 template: compute required RANSAC iterations."""
    return ransac_iterations_required(best_inliers, total_points, sample_size, confidence)


def q15_difference_of_gaussians(im: np.ndarray, sigma0: float, num_scales: int) -> tuple[list[np.ndarray], list[float]]:
    """Q15 template: compute DoG pyramid."""
    return difference_of_gaussians(im, sigma0=sigma0, num_scales=num_scales)


# Small examples with 2023 notebook values
EXAMPLE_2023 = {
    "q1": q1_intrinsic_matrix(1200, 400, 350),
    "q2_pixel_scaling": q2_resize_intrinsics_or_pixel(
        np.array([[1000.0, 0.0, 400.0], [0.0, 1000.0, 350.0], [0.0, 0.0, 1.0]]),
        sx=0.5,
        sy=0.5,
        point_xy=np.array([[0.4], [0.0]]),
    )[1],
    "q14": q14_ransac_iterations(103, 404, 0.95, 4),
}
