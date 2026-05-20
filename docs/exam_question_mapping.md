# Exam Question Mapping

This mapping is computation-focused only.

## Exam 2023 (from `notebooks/exam2023.ipynb`)

| Question | Topic | Toolkit function(s) | Template |
|---|---|---|---|
| Q1 | Camera intrinsic matrix | `camera_intrinsic` | `exam_templates.exam2023_templates.q1_intrinsic_matrix` |
| Q2 | Resize / pixel scaling | `resize_intrinsics` | `exam_templates.exam2023_templates.q2_resize_intrinsics_or_pixel` |
| Q3 | 3D projection | `rodrigues_to_matrix`, `project_points` | `exam_templates.exam2023_templates.q3_project_single_point` |
| Q5 | Epipolar line distance | `fundamental_matrix_from_extrinsics`, `point_to_epipolar_distance` | `exam_templates.exam2023_templates.q5_epipolar_distance` |
| Q6 | Linear triangulation | `projection_matrix`, `triangulate_linear` | `exam_templates.exam2023_templates.q6_linear_triangulation` |
| Q11 | Point-line distance | `point_line_distance` | `exam_templates.exam2023_templates.q11_point_line_distance` |
| Q12 | Harris response + NMS | `harris_response_from_tensor`, `non_max_suppression_2d` | `exam_templates.exam2023_templates.q12_harris_from_tensor` |
| Q13 | RANSAC line inlier counting | `line_from_points`, `point_line_distance` | `exam_templates.exam2023_templates.q13_line_ransac_inlier_count` |
| Q14 | RANSAC iteration formula | `ransac_iterations_required` | `exam_templates.exam2023_templates.q14_ransac_iterations` |
| Q15 | DoG computation | `difference_of_gaussians` | `exam_templates.exam2023_templates.q15_difference_of_gaussians` |

## Exam 2024 (from `stoney/Exam24.pdf` + `notebooks/exam2024.ipynb`)

| Question | Topic | Toolkit function(s) | Template |
|---|---|---|---|
| Q1 | 3D projection | `camera_intrinsic`, `rodrigues_to_matrix`, `project_points` | `exam_templates.exam2024_templates.q1_projection` |
| Q2 | Homography from 4 correspondences | `homography_dlt` | `exam_templates.exam2024_templates.q2_homography_from_4_points` |
| Q3 | Minimum correspondences for F | Not computation toolkit (theory/memory) | N/A |
| Q4 | Distortion mapping | `pixel_to_normalized_points`, `distort_normalized_points` | `exam_templates.exam2024_templates.q4_distortion_mapping` |
| Q5 | NMS topics | Not computation toolkit (theory) | N/A |
| Q6 | Harris from tensor | `harris_response_from_tensor`, `non_max_suppression_2d` | `exam_templates.exam2024_templates.q6_harris_from_tensor` |
| Q7 | Pose recoverability from sequence | Not computation toolkit (theory) | N/A |
| Q8 | SIFT pipeline step | Not computation toolkit (theory) | N/A |
| Q9 | RootSIFT + ratio test count | `rootsift_descriptors`, `match_descriptors_ratio` | `exam_templates.exam2024_templates.q9_rootsift_ratio_count` |
| Q10 | Zhang calibration from board images | `checkerboard_points`, `calibrate_zhang` | `exam_templates.exam2024_templates.q10_calibrate_from_corners` |
| Q11 | Zhang statements | Not computation toolkit (theory) | N/A |
| Q12 | Homography between pure-rotation images | `homography_dlt` | `exam_templates.exam2024_templates.q12_homography_from_matches` |
| Q13 | F vs E conceptual difference | Not computation toolkit (theory) | N/A |
| Q14 | Rectified pair identification | Not computation toolkit (theory) | N/A |
| Q15 | Epipolar distance in image 1 | `fundamental_matrix_from_extrinsics`, `point_to_epipolar_distance` | `exam_templates.exam2024_templates.q15_epipolar_distance` |
| Q16 | Nonlinear triangulation | `triangulate_nonlinear` | `exam_templates.exam2024_templates.q16_triangulate_nonlinear` |
| Q17 | Camera position from `R,t` | `camera_center_from_rt` | `exam_templates.exam2024_templates.q17_camera_position` |
| Q18 | Camera-frame transformation | `camera_to_camera` | `exam_templates.exam2024_templates.q18_cam3_to_cam2` |
| Q19 | RANSAC iterations | `ransac_iterations_required` | `exam_templates.exam2024_templates.q19_ransac_iterations` |
| Q20 | Squared threshold from `sigma` | `squared_reprojection_threshold` | `exam_templates.exam2024_templates.q20_squared_threshold` |
| Q21 | Structured-light unwrap | `structured_light_unwrap_phase` | `exam_templates.exam2024_templates.q21_structured_light` |
| Q22 | Pose from essential chains | Not computation toolkit (theory) | N/A |

## Predicted Variations

Use `exam_templates.predicted_question_templates.*` for likely rewordings:
- projection with Rodrigues or `R`
- resize `K`
- point-line and line-line tasks
- homography estimate/apply
- epipolar distance
- triangulation
- camera center
- `E <-> F` conversion
