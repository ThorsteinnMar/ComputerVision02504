"""Priority A exam toolkit package."""

from .core import ensure_column, from_homogeneous, rodrigues_to_matrix, skew, to_homogeneous
from .distortion import distort_normalized_points, map_undistorted_to_distorted_pixel
from .epipolar import fundamental_matrix_from_rt, point_to_epipolar_line_distance
from .features import rootsift, rootsift_ratio_match_count
from .geometry import (
    harris_from_tensor_with_nms,
    harris_nms,
    harris_response,
    line_from_points_2d,
    line_ransac_inlier_count,
    point_line_distance,
)
from .homography import (
    apply_homography,
    checkerboard_world_points,
    estimate_homography_dlt,
    estimate_intrinsics_zhang_from_homographies,
    normalize_points_2d,
    reorder_checkerboard_world_points_for_opencv,
    zhang_focal_from_checkerboards,
)
from .projection import (
    camera_center_from_rt,
    camera_intrinsic,
    project_points,
    projection_matrix,
    rt_to_transform,
    transform_point_between_cameras,
)
from .ransac import ransac_iterations, ransac_threshold_from_sigma
from .structured_light import structured_light_phase_unwrap_pixel
from .triangulation import reprojection_residual_vector, triangulate_linear, triangulate_nonlinear

__all__ = [
    "ensure_column",
    "from_homogeneous",
    "rodrigues_to_matrix",
    "skew",
    "to_homogeneous",
    "camera_intrinsic",
    "projection_matrix",
    "project_points",
    "camera_center_from_rt",
    "rt_to_transform",
    "transform_point_between_cameras",
    "line_from_points_2d",
    "point_line_distance",
    "line_ransac_inlier_count",
    "harris_response",
    "harris_nms",
    "harris_from_tensor_with_nms",
    "distort_normalized_points",
    "map_undistorted_to_distorted_pixel",
    "normalize_points_2d",
    "estimate_homography_dlt",
    "apply_homography",
    "checkerboard_world_points",
    "reorder_checkerboard_world_points_for_opencv",
    "estimate_intrinsics_zhang_from_homographies",
    "zhang_focal_from_checkerboards",
    "fundamental_matrix_from_rt",
    "point_to_epipolar_line_distance",
    "triangulate_linear",
    "reprojection_residual_vector",
    "triangulate_nonlinear",
    "rootsift",
    "rootsift_ratio_match_count",
    "ransac_iterations",
    "ransac_threshold_from_sigma",
    "structured_light_phase_unwrap_pixel",
]
