"""Ready-to-run computational templates for Exam 2024 questions."""

from __future__ import annotations

import numpy as np

from exam_toolkit import (
    calibrate_zhang,
    camera_center_from_rt,
    camera_intrinsic,
    camera_to_camera,
    distort_normalized_points,
    fundamental_matrix_from_extrinsics,
    harris_response_from_tensor,
    homography_dlt,
    non_max_suppression_2d,
    pixel_to_normalized_points,
    point_to_epipolar_distance,
    project_points,
    ransac_iterations_required,
    squared_reprojection_threshold,
    structured_light_unwrap_phase,
    triangulate_nonlinear,
)


def q1_projection(K: np.ndarray, rvec: np.ndarray, t: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Q1 template: project 3D world point to image."""
    from exam_toolkit import rodrigues_to_matrix

    R = rodrigues_to_matrix(rvec)
    return project_points(K, R, t, Q)


def q2_homography_from_4_points(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Q2 template: estimate H where q1 ~ H q2."""
    return homography_dlt(p1, p2, normalize=False)


def q4_distortion_mapping(K: np.ndarray, dist_coeffs: list[float], pixel_xy: np.ndarray) -> np.ndarray:
    """Q4 template: map undistorted pixel -> distorted normalized point."""
    q = pixel_to_normalized_points(pixel_xy, K)
    qd = distort_normalized_points(q, dist_coeffs)
    return qd


def q6_harris_from_tensor(gxx: np.ndarray, gyy: np.ndarray, gxy: np.ndarray, k: float, tau: float, connectivity: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Q6 template: Harris response and NMS corners."""
    r = harris_response_from_tensor(gxx, gyy, gxy, k=k)
    corners = non_max_suppression_2d(r, threshold=tau, connectivity=connectivity)
    return r, corners


def q9_rootsift_ratio_count(des1: np.ndarray, des2: np.ndarray, ratio: float = 0.8) -> int:
    """Q9 template: RootSIFT + ratio test match count."""
    from exam_toolkit import match_descriptors_ratio

    matches = match_descriptors_ratio(des1, des2, ratio=ratio, use_root_sift=True)
    return len(matches)


def q10_calibrate_from_corners(q_list: list[np.ndarray], board_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q10 template: Zhang calibration from detected corners."""
    return calibrate_zhang(q_list, board_points)


def q10_detect_corners_from_images(images: list[np.ndarray], pattern_size: tuple[int, int]) -> list[np.ndarray]:
    """Q10 helper: detect chessboard corners and return list of (2,N) points."""
    import cv2

    qs: list[np.ndarray] = []
    for im in images:
        found, corners = cv2.findChessboardCorners(im, pattern_size)
        if not found:
            continue
        qs.append(corners.squeeze().T)
    return qs


def q12_homography_from_matches(points_im1: np.ndarray, points_im2: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Q12 template: estimate homography from matched points (pure rotation case)."""
    return homography_dlt(points_im1, points_im2, normalize=normalize)


def q15_epipolar_distance(
    K: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> float:
    """Q15 template: distance of p1 to line in image1 induced by p2."""
    F = fundamental_matrix_from_extrinsics(K, R1, t1, K, R2, t2)
    return point_to_epipolar_distance(F, p1, p2, line_in_image=1)


def q16_triangulate_nonlinear(p_list: list[np.ndarray], P_list: list[np.ndarray]) -> np.ndarray:
    """Q16 template: nonlinear triangulation from multi-view observations."""
    return triangulate_nonlinear(p_list, P_list)


def q17_camera_position(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Q17 template: compute camera center from R,t."""
    return camera_center_from_rt(R, t)


def q18_cam3_to_cam2(
    R2: np.ndarray,
    t2: np.ndarray,
    R3: np.ndarray,
    t3: np.ndarray,
    point_cam3: np.ndarray,
) -> np.ndarray:
    """Q18 template: convert 3D point from camera 3 frame to camera 2 frame."""
    return camera_to_camera(R3, t3, R2, t2, point_cam3)


def q19_ransac_iterations(best_inliers: int, total_points: int, confidence: float = 0.90, sample_size: int = 4) -> float:
    """Q19 template: compute minimum RANSAC iterations."""
    return ransac_iterations_required(best_inliers, total_points, sample_size, confidence)


def q20_squared_threshold(sigma: float, chi2: float = 3.84) -> float:
    """Q20 template: compute squared inlier threshold."""
    return squared_reprojection_threshold(sigma=sigma, chi_square_value=chi2)


def q21_structured_light(primary: np.ndarray, secondary: np.ndarray, n1: int) -> float:
    """Q21 template: unwrap phase from two phase-shift sequences."""
    return structured_light_unwrap_phase(primary, secondary, n1=n1)


# Small examples with 2024 exam values
EXAMPLE_2024 = {
    "q1_K": camera_intrinsic(1400, (750, 520)),
    "q19": q19_ransac_iterations(465, 1177, confidence=0.90, sample_size=4),
    "q20": q20_squared_threshold(1.4),
    "q21": q21_structured_light(
        np.array([12, 9, 10, 13, 18, 25, 33, 40, 46, 49, 48, 45, 39, 31, 23, 17]),
        np.array([15, 29, 43, 49, 43, 29, 15, 10]),
        n1=40,
    ),
}
