# Function Index

All functions below use NumPy arrays and column-vector conventions (`2xN`, `3xN`, `4xN`).

## Core (`exam_toolkit.core`)

| Function | Purpose | Input shapes | Minimal usage |
|---|---|---|---|
| `as_float_array` | Convert to float array | any array-like | `a = as_float_array([1,2,3])` |
| `as_col` | Force column vector | `(n,)` or `(n,1)` | `t = as_col([0,0,1], 3)` |
| `ensure_points` | Normalize point matrix orientation | `(d,N)` or `(N,d)` | `q = ensure_points(q, 2)` |
| `pi` | Homogeneous -> inhomogeneous | `(d+1,N)` | `q = pi(qh)` |
| `pi_inv` | Inhomogeneous -> homogeneous | `(d,N)` | `qh = pi_inv(q)` |
| `normalize_homogeneous` | Scale last coordinate to 1 | `(d+1,N)` | `qh = normalize_homogeneous(qh)` |
| `cross_op` | 3D skew-symmetric cross matrix | `(3,)`/`(3,1)` | `tx = cross_op(t)` |
| `svd_last_vector` | Solve homogeneous system by SVD | `A (m,n)` | `x = svd_last_vector(A)` |
| `least_squares` | Solve `Ax≈b` | `A,b` | `x = least_squares(A,b)` |

## Geometry (`exam_toolkit.geometry`)

| Function | Purpose | Input shapes | Minimal usage |
|---|---|---|---|
| `line_from_points` | Line through 2 points | `p1,p2 (2,1)` | `l = line_from_points(p1,p2)` |
| `line_intersection` | Intersection of 2 lines | `l1,l2 (3,1)` | `p = line_intersection(l1,l2)` |
| `point_line_distance` | Distance point-to-line | `l (3,1), p_h (3,1)` | `d = point_line_distance(l,p_h)` |
| `point_to_line_distance_xy` | Distance from Euclidean point to line | `p (2,1), l (3,1)` | `d = point_to_line_distance_xy(p,l)` |
| `batch_point_line_distance` | Vectorized point-to-line distances | `points (2,N), l (3,1)` | `d = batch_point_line_distance(P,l)` |

## Projection (`exam_toolkit.projection`)

| Function | Purpose | Input shapes | Minimal usage |
|---|---|---|---|
| `camera_intrinsic` | Build intrinsic matrix `K` | `f,(cx,cy),alpha,beta` | `K = camera_intrinsic(1400,(750,520))` |
| `resize_intrinsics` | Scale intrinsics for resized image | `K (3,3), sx, sy` | `K2 = resize_intrinsics(K,0.5,0.5)` |
| `projection_matrix` | Build `P = K[R|t]` | `K,R,t` | `P = projection_matrix(K,R,t)` |
| `project_points` | Project world 3D points | `K,R,t,Q (3,N)` | `p = project_points(K,R,t,Q)` |
| `rodrigues_to_matrix` | Rodrigues vector -> rotation matrix | `rvec (3,)` | `R = rodrigues_to_matrix(rvec)` |
| `matrix_to_rodrigues` | Rotation matrix -> Rodrigues vector | `R (3,3)` | `rvec = matrix_to_rodrigues(R)` |
| `camera_center_from_rt` | Compute camera center in world | `R,t` | `C = camera_center_from_rt(R,t)` |
| `world_to_camera` | World -> camera coordinates | `R,t,Qw (3,N)` | `Qc = world_to_camera(R,t,Qw)` |
| `camera_to_world` | Camera -> world coordinates | `R,t,Qc (3,N)` | `Qw = camera_to_world(R,t,Qc)` |
| `make_transform` | Build 4x4 homogeneous transform | `R,t` | `T = make_transform(R,t)` |
| `transform_points` | Apply 4x4 transform to 3D points | `T (4,4), Q (3,N)` | `Q2 = transform_points(T,Q)` |
| `camera_to_camera` | Camera-A frame -> camera-B frame | `R_a,t_a,R_b,t_b,Qa` | `Qb = camera_to_camera(...)` |

## Distortion (`exam_toolkit.distortion`)

| Function | Purpose | Input shapes | Minimal usage |
|---|---|---|---|
| `distort_normalized_points` | Radial distortion in normalized frame | `q (2,N), coeffs` | `qd = distort_normalized_points(q,[k3,k5])` |
| `pixel_to_normalized_points` | Pixel -> normalized point | `p (2,N), K` | `q = pixel_to_normalized_points(p,K)` |
| `normalized_to_pixel_points` | Normalized -> pixel point | `q (2,N), K` | `p = normalized_to_pixel_points(q,K)` |
| `distort_pixel_points` | Distort pixel points via `K` | `p,K,coeffs` | `pd = distort_pixel_points(p,K,coeffs)` |
| `undistort_normalized_points_iterative` | Approximate inverse radial model | `qd, coeffs` | `q = undistort_normalized_points_iterative(qd,c)` |
| `undistort_pixel_points_iterative` | Approximate undistorted pixels | `pd,K,coeffs` | `p = undistort_pixel_points_iterative(pd,K,c)` |

## Homography (`exam_toolkit.homography`)

| Function | Purpose | Input shapes | Minimal usage |
|---|---|---|---|
| `normalize_points_2d` | Hartley normalization | `q (2,N)` | `qn,T = normalize_points_2d(q)` |
| `homography_dlt` | Estimate `H` from correspondences (`q1~Hq2`) | `q1,q2 (2,N)` | `H = homography_dlt(q1,q2)` |
| `apply_homography` | Transfer points with `H` | `H (3,3), q (2,N)` | `q1 = apply_homography(H,q2)` |
| `transfer_error` | Forward transfer error | `H,q1,q2` | `e = transfer_error(H,q1,q2)` |
| `symmetric_transfer_error` | Symmetric transfer error | `H,q1,q2` | `e = symmetric_transfer_error(H,q1,q2)` |
| `homography_inlier_mask` | Inlier mask from transfer error | `H,q1,q2,tau` | `m = homography_inlier_mask(H,q1,q2,5)` |

## Epipolar (`exam_toolkit.epipolar`)

| Function | Purpose | Input shapes | Minimal usage |
|---|---|---|---|
| `relative_pose` | Pose cam1->cam2 from extrinsics | `R1,t1,R2,t2` | `R21,t21 = relative_pose(...)` |
| `essential_matrix` | Compute `E=[t]_xR` | `R,t` | `E = essential_matrix(R,t)` |
| `fundamental_from_essential` | Convert `E` to `F` | `E,K1,K2` | `F = fundamental_from_essential(E,K1,K2)` |
| `essential_from_fundamental` | Convert `F` to `E` | `F,K1,K2` | `E = essential_from_fundamental(F,K1,K2)` |
| `fundamental_matrix_from_extrinsics` | Compute `F` from camera parameters | `K1,R1,t1,K2,R2,t2` | `F = fundamental_matrix_from_extrinsics(...)` |
| `enforce_rank2` | Force rank-2 matrix | `F (3,3)` | `F2 = enforce_rank2(F)` |
| `epipolar_line` | Epipolar line from point and `F` | `F,p,in_image` | `l2 = epipolar_line(F,p1,2)` |
| `point_to_epipolar_distance` | Distance to induced epiline | `F,p1,p2,line_in_image` | `d = point_to_epipolar_distance(F,p1,p2,1)` |
| `sampson_distance` | Sampson distances for correspondences | `F,p1,p2` | `d = sampson_distance(F,p1,p2)` |

## Triangulation (`exam_toolkit.triangulation`)

| Function | Purpose | Input shapes | Minimal usage |
|---|---|---|---|
| `triangulate_linear` | Linear multi-view triangulation | `q_list, P_list` | `Q = triangulate_linear([p1,p2],[P1,P2])` |
| `reprojection_errors` | Per-view reprojection residual norm | `Q,q_list,P_list` | `e = reprojection_errors(Q,qs,Ps)` |
| `reprojection_rmse` | RMSE reprojection error | `Q,q_list,P_list` | `r = reprojection_rmse(Q,qs,Ps)` |
| `triangulate_nonlinear` | Nonlinear triangulation | `q_list,P_list` | `Q = triangulate_nonlinear(qs,Ps)` |

## Calibration (`exam_toolkit.calibration`)

| Function | Purpose | Input shapes | Minimal usage |
|---|---|---|---|
| `checkerboard_points` | Build planar board coordinates | `nx,ny,square_size` | `Q = checkerboard_points(7,10,0.015)` |
| `estimate_projection_dlt` | Estimate projection matrix | `Q (3,N), q (2,N)` | `P = estimate_projection_dlt(Q,q)` |
| `reprojection_rmse` | RMSE for `P` calibration fit | `P,Q,q` | `r = reprojection_rmse(P,Q,q)` |
| `estimate_planar_homographies` | Per-view homography list | `Q_planar, q_list` | `Hs = estimate_planar_homographies(Q,qs)` |
| `form_vi` | Zhang `v_ij` row | `H,a,b` | `v12 = form_vi(H,1,2)` |
| `estimate_b_from_homographies` | Estimate Zhang `b` vector | `Hs` | `b = estimate_b_from_homographies(Hs)` |
| `estimate_intrinsics_zhang` | Intrinsics from homographies | `Hs` | `K = estimate_intrinsics_zhang(Hs)` |
| `estimate_extrinsics_zhang` | Per-view `R,t` from `K,Hs` | `K,Hs` | `Rs,ts = estimate_extrinsics_zhang(K,Hs)` |
| `calibrate_zhang` | Full Zhang pipeline | `q_list,Q_planar` | `K,Rs,ts = calibrate_zhang(qs,Q)` |

## Features (`exam_toolkit.features`)

| Function | Purpose | Input shapes | Minimal usage |
|---|---|---|---|
| `gaussian_1d_kernel` | Gaussian + derivative kernels | `sigma` | `g,gd = gaussian_1d_kernel(1)` |
| `gaussian_smoothing` | Smoothed image + derivatives | `image,sigma` | `I,Ix,Iy = gaussian_smoothing(im,1)` |
| `structure_tensor` | Compute `g*(Ix^2), g*(Iy^2), g*(IxIy)` | `image,sigma,epsilon` | `gxx,gyy,gxy = structure_tensor(im,1,2)` |
| `harris_response_from_tensor` | Harris from tensor entries | `gxx,gyy,gxy,k` | `r = harris_response_from_tensor(...)` |
| `harris_response` | Harris directly from image | `image,sigma,epsilon,k` | `r = harris_response(im,1,2,0.06)` |
| `non_max_suppression_2d` | Threshold + NMS corners | `r,tau,neighborhood,connectivity` | `c = non_max_suppression_2d(r,5,1,4)` |
| `harris_corners` | Combined Harris + NMS | `image,sigma,epsilon,k,tau` | `r,c = harris_corners(im,1,2,0.06,5)` |
| `gaussian_scale_space` | Build Gaussian pyramid (no resize) | `image,sigma0,num_scales` | `Gs,s = gaussian_scale_space(im,1,4)` |
| `difference_of_gaussians` | Build DoG pyramid | `image,sigma0,num_scales` | `DoG,s = difference_of_gaussians(im,1,4)` |
| `detect_blobs_dog` | Detect DoG blob maxima | `image,sigma0,num_scales,tau` | `blobs = detect_blobs_dog(im,1,4,3)` |
| `rootsift_descriptors` | Convert SIFT -> RootSIFT | `des (N,128)` | `d = rootsift_descriptors(des)` |
| `match_descriptors_ratio` | KNN ratio-test matching | `des1,des2,ratio` | `m = match_descriptors_ratio(des1,des2,0.8)` |

## Misc (`exam_toolkit.misc`)

| Function | Purpose | Input shapes | Minimal usage |
|---|---|---|---|
| `ransac_iterations_required` | Compute required iterations `N` | `inliers,total,sample,confidence` | `N = ransac_iterations_required(465,1177,4,0.9)` |
| `squared_reprojection_threshold` | Compute `tau^2=chi2*sigma^2` | `sigma,chi2` | `tau2 = squared_reprojection_threshold(1.4)` |
| `fft_first_harmonic_phase` | Get phase from 1D phase-shift sequence | `signal` | `th = fft_first_harmonic_phase(primary)` |
| `structured_light_unwrap_phase` | Two-frequency phase unwrapping | `primary,secondary,n1` | `theta = structured_light_unwrap_phase(p,s,40)` |
| `rmse` | Generic RMSE | `a,b` same shape | `r = rmse(x,y)` |
