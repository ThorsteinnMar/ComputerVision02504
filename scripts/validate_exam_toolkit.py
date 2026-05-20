"""Validation checks for the exam_toolkit against existing notebook patterns."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from cv02504 import util_functions as old  # noqa: E402
from exam_templates import exam2024_templates as t24  # noqa: E402
from exam_toolkit import (  # noqa: E402
    camera_center_from_rt,
    camera_to_camera,
    distort_normalized_points,
    fundamental_matrix_from_extrinsics,
    harris_response_from_tensor,
    homography_dlt,
    match_descriptors_ratio,
    non_max_suppression_2d,
    pi,
    pi_inv,
    pixel_to_normalized_points,
    point_to_epipolar_distance,
    project_points,
    projection_matrix,
    ransac_iterations_required,
    rootsift_descriptors,
    squared_reprojection_threshold,
    structured_light_unwrap_phase,
    triangulate_nonlinear,
)


def assert_close(name: str, a: np.ndarray | float, b: np.ndarray | float, tol: float = 1e-6) -> None:
    if not np.allclose(a, b, atol=tol, rtol=tol):
        raise AssertionError(f"{name} failed.\nA={a}\nB={b}")


def same_homography(H1: np.ndarray, H2: np.ndarray, tol: float = 1e-6) -> bool:
    A = H1 / np.linalg.norm(H1)
    B = H2 / np.linalg.norm(H2)
    return np.allclose(A, B, atol=tol, rtol=tol) or np.allclose(A, -B, atol=tol, rtol=tol)


def manual_harris_corners(r: np.ndarray, tau: float) -> list[tuple[int, int]]:
    corners = []
    for i in range(1, r.shape[0] - 1):
        for j in range(1, r.shape[1] - 1):
            if (
                r[i, j] > r[i + 1, j]
                and r[i, j] >= r[i - 1, j]
                and r[i, j] > r[i, j + 1]
                and r[i, j] >= r[i, j - 1]
                and r[i, j] > tau
            ):
                corners.append((i, j))
    return corners


def validate() -> None:
    passed = 0
    skipped = 0

    # 1) Core invariant
    p = np.array([[1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])
    assert_close("pi(pi_inv)", pi(pi_inv(p)), p)
    passed += 1

    # 2) 2023 Q3 projection parity
    K = old.camera_intrinsic(1720, (680, 610), 1, 0)
    R = cv2.Rodrigues(np.array([-0.1, 0.1, -0.2]))[0]
    t = np.array([[0.09], [0.05], [0.05]])
    Q = np.array([[-0.03, 0.01, 0.59]]).T
    new_p = project_points(K, R, t, Q)
    old_p = old.project_points(K, R, t, Q)
    assert_close("2023 Q3 projection", new_p, old_p, tol=1e-9)
    passed += 1

    # 3) 2024 Q2 homography parity (up to scale)
    p1 = np.array(
        [
            [1.45349587e2, -1.12915131e-1, 1.91640565e0, -6.08129962e-1],
            [1.05603820e2, 5.62792554e-2, 1.79040110e0, -2.32182177e-1],
        ],
    )
    p2 = np.array(
        [
            [1.3753556, -1.77072961, 2.94511795, 0.04032374],
            [0.30936653, 0.37172814, 1.44007577, -0.03173825],
        ],
    )
    H_new = homography_dlt(p1, p2, normalize=False)
    H_old = old.hest(p1, p2, False)
    if not same_homography(H_new, H_old, tol=1e-5):
        raise AssertionError("2024 Q2 homography mismatch up to scale")
    passed += 1

    # 4) 2024 Q4 distortion parity in normalized coordinates
    K = np.array([[300, 0, 840], [0, 300, 620], [0, 0, 1]], float)
    dist = [-0.2, 0.01, -0.03]
    pixel = np.array([[400], [500]])
    q_new = pixel_to_normalized_points(pixel, K)
    qd_new = distort_normalized_points(q_new, dist)
    q_manual = (np.linalg.inv(K) @ old.PiInv(pixel))[:2]
    r = np.sqrt(q_manual[0] ** 2 + q_manual[1] ** 2)
    corr = 1 + dist[0] * r**2 + dist[1] * r**4 + dist[2] * r**6
    qd_manual = q_manual * corr
    assert_close("2024 Q4 distortion", qd_new, qd_manual)
    passed += 1

    # 5) 2024 Q15 epipolar distance parity
    K = np.array([[300, 0, 840], [0, 300, 620.0], [0, 0, 1]], float)
    R1 = cv2.Rodrigues(np.array([-2.3, -0.7, 1.0]))[0]
    t1 = np.array([0.0, -1.0, 4.0], float).reshape(3, 1)
    R2 = cv2.Rodrigues(np.array([-0.6, 0.5, -0.9]))[0]
    t2 = np.array([0.0, 0.0, 9.0], float).reshape(3, 1)
    p1 = np.array([853.0, 656.0]).reshape(2, 1)
    p2 = np.array([814.0, 655.0]).reshape(2, 1)

    F_new = fundamental_matrix_from_extrinsics(K, R1, t1, K, R2, t2)
    d_new = point_to_epipolar_distance(F_new, p1, p2, line_in_image=1)

    F_old = old.fundamental_matrix(K, R1, t1, K, R2, t2)
    l1_old = old.PiInv(p2).T @ F_old
    d_old = old.point_line_distance(l1_old.T, old.PiInv(p1))
    assert_close("2024 Q15 epipolar distance", d_new, d_old)
    passed += 1

    # 6) 2024 Q16 nonlinear triangulation parity
    R3 = cv2.Rodrigues(np.array([-0.1, 0.9, -1.2]))[0]
    t3 = np.array([-1.0, -6.0, 28.0], float).reshape(3, 1)
    p3 = np.array([798.0, 535.0]).reshape(2, 1)

    P1 = projection_matrix(K, R1, t1)
    P2 = projection_matrix(K, R2, t2)
    P3 = projection_matrix(K, R3, t3)

    try:
        Q_new = triangulate_nonlinear([p1, p2, p3], [P1, P2, P3])
        Q_old = old.triangulate_nonlin([p1, p2, p3], [P1, P2, P3]).reshape(3, 1)
        assert_close("2024 Q16 nonlinear triangulation", Q_new, Q_old, tol=1e-4)
        passed += 1
    except ImportError:
        print("Skipping Q16 nonlinear triangulation check (scipy not installed).")
        skipped += 1

    # 7) 2024 Q17 camera center
    R = cv2.Rodrigues(np.array([-1.9, 0.1, -0.2]))[0]
    t = np.array([-1.7, 1.3, 1.5], float)
    c_new = camera_center_from_rt(R, t)
    c_manual = (-R.T @ t.reshape(3, 1))
    assert_close("2024 Q17 camera center", c_new, c_manual)
    passed += 1

    # 8) 2024 Q18 camera-to-camera transform parity
    p_cam3 = np.array([[-0.38], [0.1], [1.32]])
    p2_new = camera_to_camera(R3, t3, R2, t2, p_cam3)
    T02 = np.vstack((np.hstack((R2, t2)), np.array([0, 0, 0, 1])))
    T03 = np.vstack((np.hstack((R3, t3)), np.array([0, 0, 0, 1])))
    p2_manual = pi(T02 @ np.linalg.inv(T03) @ pi_inv(p_cam3))
    assert_close("2024 Q18 frame transform", p2_new, p2_manual)
    passed += 1

    # 9) 2024 Q6 Harris + NMS parity (4-neighborhood)
    h = np.load(ROOT / "notebooks" / "materials_2024" / "harris.npy", allow_pickle=True).item()
    gxx = h["g*(I_x^2)"]
    gyy = h["g*(I_y^2)"]
    gxy = h["g*(I_x I_y)"]
    tau = 5

    r_new = harris_response_from_tensor(gxx, gyy, gxy, k=0.06)
    corners_new = non_max_suppression_2d(r_new, threshold=tau, connectivity=4)
    corners_old = manual_harris_corners(r_new, tau)

    if set(map(tuple, corners_new.tolist())) != set(corners_old):
        raise AssertionError("2024 Q6 Harris corners mismatch")
    passed += 1

    # 10) Q19 RANSAC iteration formula
    N_new = ransac_iterations_required(465, 1177, 4, 0.90)
    N_manual = np.log(1 - 0.90) / np.log(1 - (465 / 1177) ** 4)
    assert_close("2024 Q19 RANSAC iterations", N_new, N_manual)
    passed += 1

    # 11) Q20 threshold formula
    t_new = squared_reprojection_threshold(1.4, 3.84)
    t_manual = 3.84 * 1.4**2
    assert_close("2024 Q20 threshold", t_new, t_manual)
    passed += 1

    # 12) Q21 structured-light unwrap parity
    primary = np.array([12, 9, 10, 13, 18, 25, 33, 40, 46, 49, 48, 45, 39, 31, 23, 17])
    secondary = np.array([15, 29, 43, 49, 43, 29, 15, 10])
    theta_new = structured_light_unwrap_phase(primary, secondary, n1=40)
    theta_old = t24.q21_structured_light(primary, secondary, n1=40)
    assert_close("2024 Q21 unwrap phase", theta_new, theta_old)
    passed += 1

    # 13) Q9 RootSIFT + ratio test parity with manual implementation
    sift_data = np.load(ROOT / "notebooks" / "materials_2024" / "sift_data.npy", allow_pickle=True).item()
    d1 = sift_data["des1"]
    d2 = sift_data["des2"]

    matches_new = match_descriptors_ratio(d1, d2, ratio=0.8, use_root_sift=True)

    bf = cv2.BFMatcher()
    d1_root = rootsift_descriptors(d1).astype(np.float32)
    d2_root = rootsift_descriptors(d2).astype(np.float32)
    knn = bf.knnMatch(d1_root, d2_root, k=2)
    manual = [m for m, n in knn if m.distance < 0.8 * n.distance]

    if len(matches_new) != len(manual):
        raise AssertionError("Q9 RootSIFT ratio count mismatch")
    passed += 1

    print(f"Validation checks passed: {passed}/13, skipped: {skipped}")


if __name__ == "__main__":
    validate()
