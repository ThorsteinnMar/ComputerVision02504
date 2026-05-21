"""Validation script for rewritten Priority A templates.

Runs bundled 2023/2024 examples and verifies each template returns:
- answer
- intermediates
- common_traps
- validation_status
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exam_templates.priority_a_templates import (
    example_2023_q13_line_inlier_count,
    example_2023_q14_ransac_iterations,
    example_2024_q10_zhang_focal,
    example_2024_q15_epipolar_distance,
    example_2024_q16_triangulate_nonlin,
    example_2024_q17_camera_center,
    example_2024_q18_transform_point,
    example_2024_q19_ransac_iterations,
    example_2024_q1_project_3d_point,
    example_2024_q20_ransac_threshold_dof1,
    example_2024_q20_ransac_threshold_dof2,
    example_2024_q21_structured_light,
    example_2024_q2_estimate_homography,
    example_2024_q4_distortion,
    example_2024_q6_harris,
    example_2024_q9_rootsift_ratio,
    harris_from_tensor_with_nms_template,
    line_ransac_inlier_count_template,
    rootsift_ratio_match_count_template,
)

REQUIRED_KEYS = {"answer", "intermediates", "common_traps", "validation_status"}


def _validate_result(name: str, result: dict[str, Any]) -> None:
    missing = REQUIRED_KEYS.difference(result.keys())
    if missing:
        raise AssertionError(f"{name}: missing keys {sorted(missing)}")
    if not isinstance(result["common_traps"], list) or len(result["common_traps"]) == 0:
        raise AssertionError(f"{name}: common_traps must be a non-empty list")


def main() -> None:
    checks: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("2024_q1_project_3d_point", lambda: example_2024_q1_project_3d_point(verbose=False)),
        ("2024_q2_estimate_homography", lambda: example_2024_q2_estimate_homography(verbose=False)),
        ("2024_q4_distortion", lambda: example_2024_q4_distortion(verbose=False)),
        ("2024_q6_harris", lambda: example_2024_q6_harris(verbose=False)),
        ("2024_q9_rootsift_ratio", lambda: example_2024_q9_rootsift_ratio(verbose=False)),
        ("2024_q10_zhang_focal", lambda: example_2024_q10_zhang_focal(verbose=False)),
        ("2024_q15_epipolar_distance", lambda: example_2024_q15_epipolar_distance(verbose=False)),
        ("2024_q16_triangulate_nonlin", lambda: example_2024_q16_triangulate_nonlin(verbose=False)),
        ("2024_q17_camera_center", lambda: example_2024_q17_camera_center(verbose=False)),
        ("2024_q18_transform_point", lambda: example_2024_q18_transform_point(verbose=False)),
        ("2024_q19_ransac_iterations", lambda: example_2024_q19_ransac_iterations(verbose=False)),
        ("2024_q20_ransac_threshold_dof1", lambda: example_2024_q20_ransac_threshold_dof1(verbose=False)),
        ("2024_q20_ransac_threshold_dof2", lambda: example_2024_q20_ransac_threshold_dof2(verbose=False)),
        ("2024_q21_structured_light", lambda: example_2024_q21_structured_light(verbose=False)),
        ("2023_q13_line_inliers", lambda: example_2023_q13_line_inlier_count(verbose=False)),
        ("2023_q14_ransac_iterations", lambda: example_2023_q14_ransac_iterations(verbose=False)),
    ]

    for name, fn in checks:
        out = fn()
        _validate_result(name, out)
        print(f"[PASS] {name}")

    # Extra checks for file-path driven templates (explicit requirement).
    out_harris = harris_from_tensor_with_nms_template(
        npy_path="notebooks/materials_2024/harris.npy", verbose=False
    )
    _validate_result("harris_from_tensor_with_nms_template(npy_path)", out_harris)
    print("[PASS] harris_from_tensor_with_nms_template(npy_path)")

    out_root = rootsift_ratio_match_count_template(
        npy_path="notebooks/materials_2024/sift_data.npy", ratio=0.8, verbose=False
    )
    _validate_result("rootsift_ratio_match_count_template(npy_path)", out_root)
    print("[PASS] rootsift_ratio_match_count_template(npy_path)")

    out_line = line_ransac_inlier_count_template(
        npy_path="notebooks/media/ransac.npy", tau=0.2, verbose=False
    )
    _validate_result("line_ransac_inlier_count_template(npy_path)", out_line)
    print("[PASS] line_ransac_inlier_count_template(npy_path)")

    print("\nAll Priority A template validations passed.")


if __name__ == "__main__":
    main()
