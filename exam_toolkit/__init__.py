"""Exam-ready computation toolkit for DTU 02504 Computer Vision."""

from .core import (
    as_col,
    as_float_array,
    cross_op,
    ensure_points,
    least_squares,
    normalize_homogeneous,
    pi,
    pi_inv,
    svd_last_vector,
)
from .geometry import (
    batch_point_line_distance,
    line_from_points,
    line_intersection,
    point_line_distance,
    point_to_line_distance_xy,
)
from .projection import (
    camera_center_from_rt,
    camera_intrinsic,
    camera_to_camera,
    camera_to_world,
    make_transform,
    matrix_to_rodrigues,
    project_points,
    projection_matrix,
    resize_intrinsics,
    rodrigues_to_matrix,
    transform_points,
    world_to_camera,
)
from .distortion import (
    distort_normalized_points,
    distort_pixel_points,
    normalized_to_pixel_points,
    pixel_to_normalized_points,
    undistort_normalized_points_iterative,
    undistort_pixel_points_iterative,
)
from .homography import (
    apply_homography,
    homography_dlt,
    homography_inlier_mask,
    normalize_points_2d,
    symmetric_transfer_error,
    transfer_error,
)
from .epipolar import (
    enforce_rank2,
    epipolar_line,
    essential_from_fundamental,
    essential_matrix,
    fundamental_from_essential,
    fundamental_matrix_from_extrinsics,
    point_to_epipolar_distance,
    relative_pose,
    sampson_distance,
)
from .triangulation import (
    reprojection_errors,
    reprojection_rmse,
    triangulate_linear,
    triangulate_nonlinear,
)
from .calibration import (
    calibrate_zhang,
    checkerboard_points,
    estimate_b_from_homographies,
    estimate_extrinsics_zhang,
    estimate_intrinsics_zhang,
    estimate_planar_homographies,
    estimate_projection_dlt,
    form_vi,
    reprojection_rmse as calibration_reprojection_rmse,
)
from .features import (
    detect_blobs_dog,
    difference_of_gaussians,
    gaussian_1d_kernel,
    gaussian_scale_space,
    gaussian_smoothing,
    harris_corners,
    harris_response,
    harris_response_from_tensor,
    match_descriptors_ratio,
    non_max_suppression_2d,
    rootsift_descriptors,
    structure_tensor,
)
from .misc import (
    fft_first_harmonic_phase,
    ransac_iterations_required,
    rmse,
    squared_reprojection_threshold,
    structured_light_unwrap_phase,
)

__all__ = [name for name in globals() if not name.startswith("_")]
