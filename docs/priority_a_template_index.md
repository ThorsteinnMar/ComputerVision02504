# Priority A Template Index

This index maps each Priority A template to the minimal helper functions in `exam_toolkit` and the validation source.

| Template | Main module/helper functions | Validation status | Source evidence |
|---|---|---|---|
| `project_3d_point_template` | `projection.camera_intrinsic`, `projection.project_points`, `core.rodrigues_to_matrix` | validated | `notebooks/exam2024.ipynb` Q1 |
| `estimate_homography_dlt_template` | `homography.estimate_homography_dlt`, `homography.apply_homography` | validated | `notebooks/exam2024.ipynb` Q2 |
| `map_undistorted_to_distorted_pixel_template` | `distortion.map_undistorted_to_distorted_pixel` | validated | `notebooks/exam2024.ipynb` Q4 |
| `harris_from_tensor_with_nms_template` | `geometry.harris_from_tensor_with_nms` | validated | `notebooks/exam2024.ipynb` Q6, `notebooks/exam2023.ipynb` Q12 |
| `rootsift_ratio_match_count_template` | `features.rootsift_ratio_match_count`, `features.knn_match_l2` | partially validated (`RootSIFT` exact count unvalidated) | ratio-test flow in `notebooks/exam2024.ipynb` Q9 |
| `zhang_focal_from_checkerboards_template` | `homography.checkerboard_world_points`, `homography.reorder_checkerboard_world_points_for_opencv`, `homography.zhang_focal_from_checkerboards` | validated | `notebooks/exam2024.ipynb` Q10, `notebooks/ex5.ipynb` |
| `epipolar_distance_template` | `epipolar.fundamental_matrix_from_rt`, `epipolar.point_to_epipolar_line_distance` | validated | `notebooks/exam2024.ipynb` Q15 |
| `triangulate_point_nonlinear_template` | `triangulation.triangulate_nonlinear`, `projection.projection_matrix` | validated | `notebooks/exam2024.ipynb` Q16, `notebooks/ex5.ipynb` |
| `camera_center_from_rt_template` | `projection.camera_center_from_rt` | validated | `notebooks/exam2024.ipynb` Q17 |
| `transform_point_between_camera_frames_template` | `projection.transform_point_between_cameras` | validated | `notebooks/exam2024.ipynb` Q18 |
| `ransac_iterations_template` | `ransac.ransac_iterations` | validated | `notebooks/exam2024.ipynb` Q19, `notebooks/exam2023.ipynb` Q14 |
| `ransac_threshold_from_sigma_template` | `ransac.ransac_threshold_from_sigma` | validated with DOF note | `notebooks/exam2024.ipynb` Q20 |
| `structured_light_phase_unwrap_pixel_template` | `structured_light.structured_light_phase_unwrap_pixel` | validated | `notebooks/exam2024.ipynb` Q21, `notebooks/ex13.ipynb` |
| `line_ransac_inlier_count_template` | `geometry.line_ransac_inlier_count` | validated | `notebooks/exam2023.ipynb` Q13 |

## Notes
- Scope is **Priority A only** from `docs/computation_question_universe.md`.
- No Priority B/C templates are implemented in this phase.
- `rootsift_ratio_match_count_template` is marked partially validated because the notebook solution path used plain SIFT ratio test, while the exam text asks RootSIFT.
- Reusable template functions live in `exam_templates/priority_a_templates.py` and accept parameters (no source edits required during exam use).
- The exam-facing workflow is in `notebooks/exam_ready_priority_a.ipynb`.
- End-to-end validation runner: `python3 scripts/validate_priority_a_templates.py`.
