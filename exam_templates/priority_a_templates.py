"""Priority A plug-and-play computation templates for DTU 02504.

Design goals:
- Template functions accept parameters (no in-function hardcoded exam values).
- Each template returns a dictionary with:
  - answer
  - intermediates (important intermediate values)
  - common_traps
  - validation_status
- Printing is optional via verbose=True/False.
- Separate example wrappers provide 2023/2024 values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from notebooks.cv02504.exam_toolkit.core import rodrigues_to_matrix
from notebooks.cv02504.exam_toolkit.distortion import map_undistorted_to_distorted_pixel
from notebooks.cv02504.exam_toolkit.epipolar import fundamental_matrix_from_rt, point_to_epipolar_line_distance
from notebooks.cv02504.exam_toolkit.features import knn_match_l2, rootsift_ratio_match_count
from notebooks.cv02504.exam_toolkit.geometry import harris_from_tensor_with_nms, line_ransac_inlier_count
from notebooks.cv02504.exam_toolkit.homography import (
    apply_homography,
    checkerboard_world_points,
    estimate_homography_dlt,
    reorder_checkerboard_world_points_for_opencv,
    zhang_focal_from_checkerboards,
)
from notebooks.cv02504.exam_toolkit.projection import (
    camera_center_from_rt,
    camera_intrinsic,
    project_points,
    projection_matrix,
    transform_point_between_cameras,
)
from notebooks.cv02504.exam_toolkit.ransac import ransac_iterations, ransac_threshold_from_sigma
from notebooks.cv02504.exam_toolkit.structured_light import structured_light_phase_unwrap_pixel
from notebooks.cv02504.exam_toolkit.triangulation import triangulate_nonlinear


def _load_npy_dict(npy_path: str | Path) -> dict[str, Any]:
    path = Path(npy_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path, allow_pickle=True).item()


def _print_result(name: str, result: dict[str, Any], verbose: bool) -> None:
    if not verbose:
        return
    print(f"\n=== {name} ===")
    print("answer:", result["answer"])
    print("validation_status:", result["validation_status"])
    print("intermediates:")
    for k, v in result["intermediates"].items():
        print(f"- {k}: {v}")
    print("common_traps:")
    for trap in result["common_traps"]:
        print(f"- {trap}")


def project_3d_point_template(
    *,
    Q_world: np.ndarray,
    t: np.ndarray,
    R: np.ndarray | None = None,
    rvec: np.ndarray | None = None,
    K: np.ndarray | None = None,
    f: float | None = None,
    principal_point: tuple[float, float] | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Template: project_3d_point_template."""
    if R is None:
        if rvec is None:
            raise ValueError("Provide either R or rvec.")
        R = rodrigues_to_matrix(np.asarray(rvec, dtype=float))
    else:
        R = np.asarray(R, dtype=float)

    if K is None:
        if f is None or principal_point is None:
            raise ValueError("Provide either K or (f and principal_point).")
        K = camera_intrinsic(float(f), principal_point, alpha=alpha, beta=beta)
    else:
        K = np.asarray(K, dtype=float)

    Q_world = np.asarray(Q_world, dtype=float)
    if Q_world.ndim == 1:
        Q_world = Q_world.reshape(3, 1)

    t = np.asarray(t, dtype=float)
    if t.ndim == 1:
        t = t.reshape(3, 1)

    P = projection_matrix(K, R, t)
    q = project_points(K, R, t, Q_world)

    result = {
        "answer": q,
        "intermediates": {
            "K": K,
            "R": R,
            "t": t,
            "P": P,
            "Q_world": Q_world,
        },
        "common_traps": [
            "Using row vectors when your pipeline expects column vectors.",
            "Forgetting homogeneous divide after P @ Qh.",
            "Confusing world->camera convention for R,t.",
        ],
        "validation_status": "validated_on_2024_q1",
    }
    _print_result("project_3d_point_template", result, verbose)
    return result


def estimate_homography_dlt_template(
    *,
    p_target: np.ndarray,
    p_source: np.ndarray,
    normalize: bool = False,
    check_point_source: np.ndarray | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Template: estimate_homography_dlt_template with p_target ~ H p_source."""
    p_target = np.asarray(p_target, dtype=float)
    p_source = np.asarray(p_source, dtype=float)

    H = estimate_homography_dlt(p_target=p_target, p_source=p_source, normalize=normalize)
    scale_ref = H[0, 0] if not np.isclose(H[0, 0], 0.0) else H[-1, -1]
    H_scaled = H / scale_ref

    check_mapped = None
    if check_point_source is not None:
        cps = np.asarray(check_point_source, dtype=float)
        if cps.ndim == 1:
            cps = cps.reshape(2, 1)
        check_mapped = apply_homography(H, cps)

    result = {
        "answer": H,
        "intermediates": {
            "H_scaled_for_option_matching": H_scaled,
            "check_point_source": check_point_source,
            "check_point_mapped": check_mapped,
            "normalize": normalize,
        },
        "common_traps": [
            "Swapping source/target point sets returns inverse mapping.",
            "Comparing homographies without accounting for arbitrary scale.",
            "Using fewer than 4 correspondences.",
        ],
        "validation_status": "validated_on_2024_q2",
    }
    _print_result("estimate_homography_dlt_template", result, verbose)
    return result


def map_undistorted_to_distorted_pixel_template(
    *,
    undistorted_pixel: np.ndarray,
    K: np.ndarray,
    dist_coeffs: list[float] | tuple[float, ...],
    verbose: bool = True,
) -> dict[str, Any]:
    """Template: map_undistorted_to_distorted_pixel_template."""
    p_d, q_u, q_d = map_undistorted_to_distorted_pixel(
        undistorted_pixel=np.asarray(undistorted_pixel, dtype=float),
        K=np.asarray(K, dtype=float),
        dist_coeffs=dist_coeffs,
    )

    result = {
        "answer": p_d,
        "intermediates": {
            "q_u_normalized": q_u,
            "q_d_normalized": q_d,
            "K": np.asarray(K, dtype=float),
            "dist_coeffs": list(dist_coeffs),
        },
        "common_traps": [
            "Applying distortion directly in pixel coordinates instead of normalized coordinates.",
            "Using the inverse mapping when question asks forward mapping.",
            "Wrong sign/order of distortion coefficients.",
        ],
        "validation_status": "validated_on_2024_q4",
    }
    _print_result("map_undistorted_to_distorted_pixel_template", result, verbose)
    return result


def harris_from_tensor_with_nms_template(
    *,
    gxx: np.ndarray | None = None,
    gyy: np.ndarray | None = None,
    gxy: np.ndarray | None = None,
    npy_path: str | Path | None = None,
    k: float = 0.06,
    tau: float = 5.0,
    neighborhood: int = 8,
    verbose: bool = True,
) -> dict[str, Any]:
    """Template: harris_from_tensor_with_nms_template."""
    if gxx is None or gyy is None or gxy is None:
        if npy_path is None:
            raise ValueError("Provide (gxx, gyy, gxy) arrays or npy_path.")
        data = _load_npy_dict(npy_path)
        gxx = data["g*(I_x^2)"]
        gyy = data["g*(I_y^2)"]
        gxy = data["g*(I_x I_y)"]

    r, corners = harris_from_tensor_with_nms(
        np.asarray(gxx, dtype=float),
        np.asarray(gyy, dtype=float),
        np.asarray(gxy, dtype=float),
        k=float(k),
        tau=float(tau),
        neighborhood=int(neighborhood),
    )

    result = {
        "answer": corners,
        "intermediates": {
            "response": r,
            "corner_count": len(corners),
            "k": k,
            "tau": tau,
            "neighborhood": neighborhood,
        },
        "common_traps": [
            "Using 4-neighborhood when options assume 8-neighborhood.",
            "Mixing row/column index order.",
            "Thresholding before computing Harris response correctly.",
        ],
        "validation_status": "validated_on_2024_q6_and_2023_q12",
    }
    _print_result("harris_from_tensor_with_nms_template", result, verbose)
    return result


def rootsift_ratio_match_count_template(
    *,
    des1: np.ndarray | None = None,
    des2: np.ndarray | None = None,
    npy_path: str | Path | None = None,
    ratio: float = 0.8,
    include_plain_sift_comparison: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Template: rootsift_ratio_match_count_template."""
    if des1 is None or des2 is None:
        if npy_path is None:
            raise ValueError("Provide des1/des2 arrays or npy_path containing des1/des2.")
        data = _load_npy_dict(npy_path)
        des1 = data["des1"]
        des2 = data["des2"]

    des1 = np.asarray(des1, dtype=float)
    des2 = np.asarray(des2, dtype=float)

    count_root, passed_root, idx_root, dist_root = rootsift_ratio_match_count(des1, des2, ratio=ratio)

    plain_count = None
    plain_passed = None
    if include_plain_sift_comparison:
        idx_plain, dist_plain = knn_match_l2(des1, des2, k=2)
        plain_passed = dist_plain[:, 0] < ratio * dist_plain[:, 1]
        plain_count = int(np.sum(plain_passed))

    result = {
        "answer": count_root,
        "intermediates": {
            "root_ratio_passed_mask": passed_root,
            "root_knn_indices": idx_root,
            "root_knn_distances": dist_root,
            "plain_sift_count": plain_count,
            "plain_sift_passed_mask": plain_passed,
            "ratio": ratio,
        },
        "common_traps": [
            "Applying ratio test on 1-NN matches instead of 2-NN matches.",
            "Skipping RootSIFT transform when question explicitly asks for it.",
            "Comparing squared and non-squared distances inconsistently.",
        ],
        "validation_status": "partially_validated_root_logic_yes_exact_count_unvalidated",
    }
    _print_result("rootsift_ratio_match_count_template", result, verbose)
    return result


def zhang_focal_from_checkerboards_template(
    *,
    qs: list[np.ndarray] | None = None,
    Q_world: np.ndarray | None = None,
    image_paths: list[str | Path] | None = None,
    folder: str | Path | None = None,
    board_glob: str = "board*.jpg",
    pattern_size: tuple[int, int] = (7, 10),
    square_size_m: float = 15.0 / 1000.0,
    reorder_world_points: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Template: zhang_focal_from_checkerboards_template."""
    rows, cols = pattern_size

    used_images: list[str] = []
    if qs is None:
        if image_paths is None:
            if folder is None:
                raise ValueError("Provide qs directly, or image_paths/folder.")
            image_paths = sorted(Path(folder).glob(board_glob))

        qs = []
        for p in image_paths:
            pth = Path(p)
            im = cv2.imread(str(pth))
            if im is None:
                continue
            ok, corners = cv2.findChessboardCorners(im, (rows, cols))
            if ok and corners is not None:
                qs.append(corners.squeeze().T)
                used_images.append(pth.name)

    if len(qs) < 2:
        raise RuntimeError("Need at least two checkerboard views with detected corners for stable Zhang estimate.")

    if Q_world is None:
        Q_world = checkerboard_world_points(rows, cols, square_size=square_size_m)
        if reorder_world_points:
            Q_world = reorder_checkerboard_world_points_for_opencv(Q_world, rows, cols)

    fx, fy, K_est, Hs = zhang_focal_from_checkerboards(qs, np.asarray(Q_world, dtype=float))

    result = {
        "answer": fx,
        "intermediates": {
            "fy": fy,
            "K_est": K_est,
            "num_views": len(qs),
            "used_images": used_images,
            "num_homographies": len(Hs),
        },
        "common_traps": [
            "Checkerboard point order mismatch between world points and cv2 corner order.",
            "Using too few valid checkerboard views.",
            "Ignoring image scaling changes to K if images are resized before detection.",
        ],
        "validation_status": "validated_on_2024_q10_workflow",
    }
    _print_result("zhang_focal_from_checkerboards_template", result, verbose)
    return result


def epipolar_distance_template(
    *,
    K1: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    p1: np.ndarray,
    K2: np.ndarray | None = None,
    R2: np.ndarray | None = None,
    t2: np.ndarray | None = None,
    p2: np.ndarray | None = None,
    F: np.ndarray | None = None,
    distance_in_image: int = 1,
    verbose: bool = True,
) -> dict[str, Any]:
    """Template: epipolar_distance_template."""
    if F is None:
        if K2 is None:
            K2 = K1
        if R2 is None or t2 is None:
            raise ValueError("Provide either F directly or (R2,t2[,K2]).")
        F = fundamental_matrix_from_rt(
            np.asarray(K1, dtype=float),
            np.asarray(R1, dtype=float),
            np.asarray(t1, dtype=float),
            np.asarray(K2, dtype=float),
            np.asarray(R2, dtype=float),
            np.asarray(t2, dtype=float),
        )

    if p2 is None:
        raise ValueError("p2 is required to induce epipolar line.")

    dist, line = point_to_epipolar_line_distance(
        np.asarray(F, dtype=float),
        np.asarray(p1, dtype=float),
        np.asarray(p2, dtype=float),
        distance_in_image=distance_in_image,
    )

    result = {
        "answer": dist,
        "intermediates": {
            "F": np.asarray(F, dtype=float),
            "epipolar_line": line,
            "distance_in_image": distance_in_image,
        },
        "common_traps": [
            "Using wrong line direction (l1 = p2^T F vs l2 = F p1).",
            "Forgetting homogeneous conversion for point-line distance.",
            "Building F from wrong pose convention.",
        ],
        "validation_status": "validated_on_2024_q15_and_2023_q5",
    }
    _print_result("epipolar_distance_template", result, verbose)
    return result


def triangulate_point_nonlinear_template(
    *,
    q_list: list[np.ndarray],
    P_list: list[np.ndarray] | None = None,
    K: np.ndarray | None = None,
    R_list: list[np.ndarray] | None = None,
    t_list: list[np.ndarray] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Template: triangulate_point_nonlinear_template."""
    if P_list is None:
        if K is None or R_list is None or t_list is None:
            raise ValueError("Provide P_list directly, or provide K with R_list and t_list.")
        if not (len(R_list) == len(t_list) == len(q_list)):
            raise ValueError("R_list, t_list, and q_list must have equal length.")
        P_list = [projection_matrix(K, R, t) for R, t in zip(R_list, t_list)]

    Q_nonlin, Q_linear, info = triangulate_nonlinear(q_list, P_list)

    result = {
        "answer": Q_nonlin,
        "intermediates": {
            "Q_linear_init": Q_linear,
            "optimizer_info": info,
            "num_views": len(q_list),
        },
        "common_traps": [
            "Skipping linear initialization can destabilize optimization.",
            "Mixing camera order between q_list and P_list.",
            "Returning wrong shape (3,) vs expected column vector (3,1).",
        ],
        "validation_status": "validated_on_2024_q16_and_ex5",
    }
    _print_result("triangulate_point_nonlinear_template", result, verbose)
    return result


def camera_center_from_rt_template(
    *,
    R: np.ndarray,
    t: np.ndarray,
    verbose: bool = True,
) -> dict[str, Any]:
    """Template: camera_center_from_rt_template."""
    C = camera_center_from_rt(np.asarray(R, dtype=float), np.asarray(t, dtype=float))

    result = {
        "answer": C,
        "intermediates": {
            "R": np.asarray(R, dtype=float),
            "t": np.asarray(t, dtype=float),
        },
        "common_traps": [
            "Treating t as camera position directly (it is not).",
            "Sign errors in C = -R^T t.",
            "Using row-vector rotation formulas by accident.",
        ],
        "validation_status": "validated_on_2024_q17",
    }
    _print_result("camera_center_from_rt_template", result, verbose)
    return result


def transform_point_between_camera_frames_template(
    *,
    point_in_src_cam: np.ndarray,
    R_src: np.ndarray,
    t_src: np.ndarray,
    R_dst: np.ndarray,
    t_dst: np.ndarray,
    verbose: bool = True,
) -> dict[str, Any]:
    """Template: transform_point_between_camera_frames_template."""
    p_dst = transform_point_between_cameras(
        point_in_src_cam=np.asarray(point_in_src_cam, dtype=float),
        R_src=np.asarray(R_src, dtype=float),
        t_src=np.asarray(t_src, dtype=float),
        R_dst=np.asarray(R_dst, dtype=float),
        t_dst=np.asarray(t_dst, dtype=float),
    )

    result = {
        "answer": p_dst,
        "intermediates": {
            "point_in_src_cam": np.asarray(point_in_src_cam, dtype=float),
            "R_src": np.asarray(R_src, dtype=float),
            "t_src": np.asarray(t_src, dtype=float),
            "R_dst": np.asarray(R_dst, dtype=float),
            "t_dst": np.asarray(t_dst, dtype=float),
        },
        "common_traps": [
            "Wrong transform chain order (src->world->dst).",
            "Forgetting inverse when moving from camera frame to world frame.",
            "Mixing world->cam vs cam->world conventions.",
        ],
        "validation_status": "validated_on_2024_q18",
    }
    _print_result("transform_point_between_camera_frames_template", result, verbose)
    return result


def ransac_iterations_template(
    *,
    s: int,
    m: int,
    p: float,
    n: int,
    verbose: bool = True,
) -> dict[str, Any]:
    """Template: ransac_iterations_template."""
    raw_N, ceil_N = ransac_iterations(s=s, m=m, p=p, n=n)

    result = {
        "answer": ceil_N,
        "intermediates": {
            "raw_N": raw_N,
            "inlier_ratio": s / m,
            "s": s,
            "m": m,
            "p": p,
            "n": n,
        },
        "common_traps": [
            "Rounding down instead of taking ceiling.",
            "Using percent (90) instead of probability (0.90).",
            "Wrong sample size n for the model.",
        ],
        "validation_status": "validated_on_2024_q19_and_2023_q14",
    }
    _print_result("ransac_iterations_template", result, verbose)
    return result


def ransac_threshold_from_sigma_template(
    *,
    sigma: float,
    confidence: float = 0.95,
    dof: int = 1,
    squared: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Template: ransac_threshold_from_sigma_template."""
    threshold = ransac_threshold_from_sigma(
        sigma=float(sigma),
        confidence=float(confidence),
        dof=int(dof),
        squared=bool(squared),
    )

    result = {
        "answer": threshold,
        "intermediates": {
            "sigma": sigma,
            "confidence": confidence,
            "dof": dof,
            "squared": squared,
        },
        "common_traps": [
            "Using the wrong chi-square DOF for your residual.",
            "Confusing tau with tau^2.",
            "Forgetting sigma is in pixels and must be squared in tau^2.",
        ],
        "validation_status": "validated_formula_family_with_dof_choice_note",
    }
    _print_result("ransac_threshold_from_sigma_template", result, verbose)
    return result


def structured_light_phase_unwrap_pixel_template(
    *,
    primary: np.ndarray,
    secondary: np.ndarray,
    n1: int,
    n2: int,
    verbose: bool = True,
) -> dict[str, Any]:
    """Template: structured_light_phase_unwrap_pixel_template."""
    out = structured_light_phase_unwrap_pixel(
        np.asarray(primary, dtype=float),
        np.asarray(secondary, dtype=float),
        n1=int(n1),
        n2=int(n2),
    )

    result = {
        "answer": out["theta_est"],
        "intermediates": out,
        "common_traps": [
            "Using wrong FFT harmonic index (must use first non-DC component).",
            "Modulo mistakes when wrapping/unwrapping angles.",
            "Mixing radians and degrees.",
        ],
        "validation_status": "validated_on_2024_q21_and_ex13",
    }
    _print_result("structured_light_phase_unwrap_pixel_template", result, verbose)
    return result


def line_ransac_inlier_count_template(
    *,
    points: np.ndarray | None = None,
    x1: np.ndarray | None = None,
    x2: np.ndarray | None = None,
    npy_path: str | Path | None = None,
    tau: float = 0.2,
    verbose: bool = True,
) -> dict[str, Any]:
    """Template: line_ransac_inlier_count_template."""
    if points is None or x1 is None or x2 is None:
        if npy_path is None:
            raise ValueError("Provide points/x1/x2 or npy_path.")
        data = _load_npy_dict(npy_path)
        points = np.asarray(data["points"], dtype=float)
        x1 = np.asarray(data["x1"], dtype=float)
        x2 = np.asarray(data["x2"], dtype=float)

    count, inlier_mask, line = line_ransac_inlier_count(
        x1=np.asarray(x1, dtype=float),
        x2=np.asarray(x2, dtype=float),
        points=np.asarray(points, dtype=float),
        tau=float(tau),
    )

    result = {
        "answer": count,
        "intermediates": {
            "line": line,
            "inlier_mask": inlier_mask,
            "inlier_indices": np.where(inlier_mask)[0],
            "tau": tau,
        },
        "common_traps": [
            "Cross-product line construction with wrong point order/sign assumptions.",
            "Distance formula applied to inhomogeneous point without adding homogeneous 1.",
            "Threshold tau interpreted in wrong units.",
        ],
        "validation_status": "validated_on_2023_q13",
    }
    _print_result("line_ransac_inlier_count_template", result, verbose)
    return result


# -----------------------------------------------------------------------------
# Example wrappers (separate from reusable templates)
# -----------------------------------------------------------------------------

def example_2024_q1_project_3d_point(verbose: bool = True) -> dict[str, Any]:
    return project_3d_point_template(
        f=1400.0,
        principal_point=(750.0, 520.0),
        rvec=np.array([0.2, 0.2, -0.1], dtype=float),
        t=np.array([[-0.08], [0.01], [0.03]], dtype=float),
        Q_world=np.array([[-0.38], [0.1], [1.32]], dtype=float),
        verbose=verbose,
    )


def example_2024_q2_estimate_homography(verbose: bool = True) -> dict[str, Any]:
    p1 = np.array(
        [
            [1.45349587e02, -1.12915131e-01, 1.91640565e00, -6.08129962e-01],
            [1.05603820e02, 5.62792554e-02, 1.79040110e00, -2.32182177e-01],
        ],
        dtype=float,
    )
    p2 = np.array(
        [
            [1.3753556, -1.77072961, 2.94511795, 0.04032374],
            [0.30936653, 0.37172814, 1.44007577, -0.03173825],
        ],
        dtype=float,
    )
    return estimate_homography_dlt_template(
        p_target=p1,
        p_source=p2,
        normalize=False,
        check_point_source=p2[:, 0:1],
        verbose=verbose,
    )


def example_2024_q4_distortion(verbose: bool = True) -> dict[str, Any]:
    return map_undistorted_to_distorted_pixel_template(
        undistorted_pixel=np.array([[400.0], [500.0]], dtype=float),
        K=np.array([[300.0, 0.0, 840.0], [0.0, 300.0, 620.0], [0.0, 0.0, 1.0]], dtype=float),
        dist_coeffs=[-0.2, 0.01, -0.03],
        verbose=verbose,
    )


def example_2024_q6_harris(verbose: bool = True) -> dict[str, Any]:
    return harris_from_tensor_with_nms_template(
        npy_path=Path("notebooks/materials_2024/harris.npy"),
        k=0.06,
        tau=5.0,
        neighborhood=8,
        verbose=verbose,
    )


def example_2024_q9_rootsift_ratio(verbose: bool = True) -> dict[str, Any]:
    return rootsift_ratio_match_count_template(
        npy_path=Path("notebooks/materials_2024/sift_data.npy"),
        ratio=0.8,
        include_plain_sift_comparison=True,
        verbose=verbose,
    )


def example_2024_q10_zhang_focal(verbose: bool = True) -> dict[str, Any]:
    return zhang_focal_from_checkerboards_template(
        folder=Path("notebooks/materials_2024"),
        board_glob="board*.jpg",
        pattern_size=(7, 10),
        square_size_m=15.0 / 1000.0,
        reorder_world_points=True,
        verbose=verbose,
    )


def example_2024_q15_epipolar_distance(verbose: bool = True) -> dict[str, Any]:
    K = np.array([[300.0, 0.0, 840.0], [0.0, 300.0, 620.0], [0.0, 0.0, 1.0]], dtype=float)
    return epipolar_distance_template(
        K1=K,
        R1=rodrigues_to_matrix(np.array([-2.3, -0.7, 1.0], dtype=float)),
        t1=np.array([[0.0], [-1.0], [4.0]], dtype=float),
        K2=K,
        R2=rodrigues_to_matrix(np.array([-0.6, 0.5, -0.9], dtype=float)),
        t2=np.array([[0.0], [0.0], [9.0]], dtype=float),
        p1=np.array([[853.0], [656.0]], dtype=float),
        p2=np.array([[814.0], [655.0]], dtype=float),
        distance_in_image=1,
        verbose=verbose,
    )


def example_2024_q16_triangulate_nonlin(verbose: bool = True) -> dict[str, Any]:
    K = np.array([[300.0, 0.0, 840.0], [0.0, 300.0, 620.0], [0.0, 0.0, 1.0]], dtype=float)
    R1 = rodrigues_to_matrix(np.array([-2.3, -0.7, 1.0], dtype=float))
    t1 = np.array([[0.0], [-1.0], [4.0]], dtype=float)
    R2 = rodrigues_to_matrix(np.array([-0.6, 0.5, -0.9], dtype=float))
    t2 = np.array([[0.0], [0.0], [9.0]], dtype=float)
    R3 = rodrigues_to_matrix(np.array([-0.1, 0.9, -1.2], dtype=float))
    t3 = np.array([[-1.0], [-6.0], [28.0]], dtype=float)

    P_list = [projection_matrix(K, R1, t1), projection_matrix(K, R2, t2), projection_matrix(K, R3, t3)]
    q_list = [
        np.array([[853.0], [656.0]], dtype=float),
        np.array([[814.0], [655.0]], dtype=float),
        np.array([[798.0], [535.0]], dtype=float),
    ]
    return triangulate_point_nonlinear_template(q_list=q_list, P_list=P_list, verbose=verbose)


def example_2024_q17_camera_center(verbose: bool = True) -> dict[str, Any]:
    return camera_center_from_rt_template(
        R=rodrigues_to_matrix(np.array([-1.9, 0.1, -0.2], dtype=float)),
        t=np.array([[-1.7], [1.3], [1.5]], dtype=float),
        verbose=verbose,
    )


def example_2024_q18_transform_point(verbose: bool = True) -> dict[str, Any]:
    return transform_point_between_camera_frames_template(
        point_in_src_cam=np.array([[-0.38], [0.1], [1.32]], dtype=float),
        R_src=rodrigues_to_matrix(np.array([-0.1, 0.9, -1.2], dtype=float)),
        t_src=np.array([[-1.0], [-6.0], [28.0]], dtype=float),
        R_dst=rodrigues_to_matrix(np.array([-0.6, 0.5, -0.9], dtype=float)),
        t_dst=np.array([[0.0], [0.0], [9.0]], dtype=float),
        verbose=verbose,
    )


def example_2024_q19_ransac_iterations(verbose: bool = True) -> dict[str, Any]:
    return ransac_iterations_template(s=465, m=1177, p=0.90, n=4, verbose=verbose)


def example_2024_q20_ransac_threshold_dof1(verbose: bool = True) -> dict[str, Any]:
    return ransac_threshold_from_sigma_template(sigma=1.4, confidence=0.95, dof=1, squared=True, verbose=verbose)


def example_2024_q20_ransac_threshold_dof2(verbose: bool = True) -> dict[str, Any]:
    return ransac_threshold_from_sigma_template(sigma=1.4, confidence=0.95, dof=2, squared=True, verbose=verbose)


def example_2024_q21_structured_light(verbose: bool = True) -> dict[str, Any]:
    return structured_light_phase_unwrap_pixel_template(
        primary=np.array([12, 9, 10, 13, 18, 25, 33, 40, 46, 49, 48, 45, 39, 31, 23, 17], dtype=float),
        secondary=np.array([15, 29, 43, 49, 43, 29, 15, 10], dtype=float),
        n1=40,
        n2=41,
        verbose=verbose,
    )


def example_2023_q13_line_inlier_count(verbose: bool = True) -> dict[str, Any]:
    return line_ransac_inlier_count_template(
        npy_path=Path("notebooks/media/ransac.npy"),
        tau=0.2,
        verbose=verbose,
    )


def example_2023_q14_ransac_iterations(verbose: bool = True) -> dict[str, Any]:
    return ransac_iterations_template(s=103, m=404, p=0.95, n=4, verbose=verbose)


def run_priority_a_examples(verbose: bool = True) -> dict[str, dict[str, Any]]:
    """Run all bundled Priority A examples."""
    return {
        "2024_q1_project": example_2024_q1_project_3d_point(verbose=verbose),
        "2024_q2_homography": example_2024_q2_estimate_homography(verbose=verbose),
        "2024_q4_distortion": example_2024_q4_distortion(verbose=verbose),
        "2024_q6_harris": example_2024_q6_harris(verbose=verbose),
        "2024_q9_rootsift": example_2024_q9_rootsift_ratio(verbose=verbose),
        "2024_q10_zhang": example_2024_q10_zhang_focal(verbose=verbose),
        "2024_q15_epipolar": example_2024_q15_epipolar_distance(verbose=verbose),
        "2024_q16_triangulation": example_2024_q16_triangulate_nonlin(verbose=verbose),
        "2024_q17_center": example_2024_q17_camera_center(verbose=verbose),
        "2024_q18_transform": example_2024_q18_transform_point(verbose=verbose),
        "2024_q19_iters": example_2024_q19_ransac_iterations(verbose=verbose),
        "2024_q20_tau2_dof1": example_2024_q20_ransac_threshold_dof1(verbose=verbose),
        "2024_q20_tau2_dof2": example_2024_q20_ransac_threshold_dof2(verbose=verbose),
        "2024_q21_structured": example_2024_q21_structured_light(verbose=verbose),
        "2023_q13_line_inliers": example_2023_q13_line_inlier_count(verbose=verbose),
        "2023_q14_iters": example_2023_q14_ransac_iterations(verbose=verbose),
    }


if __name__ == "__main__":
    run_priority_a_examples(verbose=True)
