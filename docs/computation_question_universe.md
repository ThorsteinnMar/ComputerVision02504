# Computation Question Universe (DTU 02504)

## Source Coverage (Computational)
- Primary 2024 sources: `stoney/Exam24.pdf` (official wording) and `notebooks/exam2024.ipynb` (worked computational solutions for Q1, Q2, Q4, Q6, Q9, Q10, Q12, Q15-Q21).
- Primary 2023 source: `notebooks/exam2023.ipynb` (requested primary source; image PDF is backup only).
- 2023 backup visual source: `stoney/2023 exam_full.pdf` (image-based; text extraction is empty).
- Exercises: `notebooks/ex1.ipynb`-`notebooks/ex13.ipynb` and `stoney/Exercises_Combined.pdf`.
- Lectures: `stoney/Lectures_Combinede.pdf` (formulas, derivations, and algorithm choices).

## 1) Main Computational Topic Areas (Short Overview)
- Homogeneous geometry: point/line representation, conversion, line intersection, point-line distance.
- Camera geometry: projection with `K, R, t`, projection matrix `P = K[R|t]`, camera position vs translation.
- Distortion/undistortion: radial models, point mapping, and image remapping.
- Planar geometry: homography estimation (DLT + SVD), normalization, and robust fitting with RANSAC.
- Multi-view geometry: essential/fundamental matrices, epipolar lines, point-to-epipolar-line distance.
- 3D recovery: linear triangulation, nonlinear triangulation, and reprojection error minimization.
- Feature pipelines: SIFT/RootSIFT matching + ratio test.
- Detector pipelines: Harris response + non-maximum suppression; DoG/blob detection.
- Robust estimation math: RANSAC iteration counts, adaptive stopping, chi-square thresholding logic.
- Calibration and reconstruction: Zhang calibration and structured-light phase unwrapping + triangulation.

## 2) Exact Computational Exam Question Types from 2024

| Question # | Problem type | Source file | What is given | What must be computed | Required computation pipeline | Minimal helper functions needed later | Common traps | Plug-and-play template later? |
|---|---|---|---|---|---|---|---|---|
| Q1 | 3D point projection with known camera | `stoney/Exam24.pdf` (Q1), `notebooks/exam2024.ipynb` (Q1 cell) | `f`, principal point, Rodrigues rotation vector, `t`, one 3D world point | 2D image projection | Build `K` -> Rodrigues to `R` -> `q = Pi(K[R|t]PiInv(Q))` | `camera_intrinsic`, `projection_matrix`/`project_points`, `Pi`, `PiInv` | Wrong `t` shape; mixing world->cam direction; forgetting homogeneous divide | yes |
| Q2 | Homography from 4 correspondences (no normalization) | `stoney/Exam24.pdf` (Q2), `notebooks/exam2024.ipynb` (Q2 cells) | `p1`, `p2` arrays (4 matches), relation `q1 = H q2` | Correct `H` among options | DLT setup -> SVD -> reshape `H` -> scale-normalize for comparison | `hest`, `Pi`, `PiInv` | Using inverse direction (`H21` vs `H12`); comparing without scale normalization | yes |
| Q4 | Radial distortion point mapping | `stoney/Exam24.pdf` (Q4), `notebooks/exam2024.ipynb` (Q4 cells) | `K`, `k3,k5,k7`, pixel in undistorted image | Corresponding pixel in distorted image | Pixel->normalized (`K^-1`) -> apply radial model -> map back (if needed) | `distort`, `Pi`, `PiInv` | Using distortion in wrong direction; skipping normalization; sign mistakes in coefficients | yes |
| Q6 | Harris corner from structure tensor values | `stoney/Exam24.pdf` (Q6), `notebooks/exam2024.ipynb` (Q6 cell) | `g*(Ix^2)`, `g*(Iy^2)`, `g*(IxIy)`, `k`, threshold `tau` | Whether/where corners are detected | Compute `r = det(C) - k*trace(C)^2` -> threshold -> local NMS | `harris_measure`, NMS helper (`corner_detector` pattern) | 4-neighborhood vs 8-neighborhood ambiguity; row/column index confusion | yes |
| Q9 | SIFT/RootSIFT matching count with ratio test | `stoney/Exam24.pdf` (Q9), `notebooks/exam2024.ipynb` (Q9 cell) | Stored `kp1,des1,kp2,des2`, ratio `0.8` | Number of accepted matches | (RootSIFT transform if required) -> KNN matching (`k=2`) -> Lowe ratio filter -> count | `root_sift_transform` (to add later), matcher wrapper | Notebook uses BF+ratio directly; RootSIFT preprocessing may be implicit/omitted (`uncertain`) | yes |
| Q10 | Camera focal length from checkerboard images | `stoney/Exam24.pdf` (Q10), `notebooks/exam2024.ipynb` (Q10 cells) | 5 board images, no lens distortion assumption | Best focal length | Detect corners -> map checkerboard 3D points with correct ordering -> calibrate (`K`) -> read `fx` | `checkerboard_points`, `calibrate_camera` (Zhang), RMSE helper | Corner ordering mismatch; not filtering failed boards; image scaling changes `K` scale | yes |
| Q12 | Homography between two real images (pure rotation context) | `stoney/Exam24.pdf` (Q12), `notebooks/exam2024.ipynb` (Q12 cells) | `im1.jpg`, `im2.jpg`, matrix `A` such that `A q2` localizes in `im1` | Best homography matrix | SIFT features -> robust matches -> RANSAC homography -> scale-normalized comparison | `find_features`, `ransac_homography`, `hest` | Wrong mapping direction (`im2 -> im1` required); weak threshold selection | yes |
| Q15 | Point-to-epipolar-line distance | `stoney/Exam24.pdf` (Q15), `notebooks/exam2024.ipynb` (Q15 cell) | Shared `K`, three camera extrinsics, noisy `p1,p2,p3` | Distance from `p1` to epipolar line in cam1 induced by `p2` | Build `F12` -> line in cam1 from `p2` -> point-line distance | `fundamental_matrix`, `point_line_distance`, `PiInv` | Wrong side of `F` (`l1 = p2^T F12` vs `F p1` form); forgetting homogeneous conversion | yes |
| Q16 | Nonlinear triangulation (reprojection minimization) | `stoney/Exam24.pdf` (Q16), `notebooks/exam2024.ipynb` (Q16 cell) | Same setup as Q15, 3 observations | 3D point estimate | Build `P1..P3` -> linear init -> nonlinear least-squares reprojection minimization | `projection_matrix`, `triangulate`, `triangulate_nonlin`, reprojection residual helper | Bad initialization; returning wrong shape `(3,)` vs `(3,1)`; not using all views | yes |
| Q17 | Camera position from extrinsics | `stoney/Exam24.pdf` (Q17), `notebooks/exam2024.ipynb` (Q17 cell) | `R`, `t` | Camera position in world frame | `C = -R^T t` | `camera_center_from_rt` helper | Confusing translation vector with camera position | yes |
| Q18 | Frame-to-frame 3D point transform | `stoney/Exam24.pdf` (Q18), `notebooks/exam2024.ipynb` (Q18 cell) | Cameras from Q15 and point in camera-3 frame | Point in camera-2 frame | Build homogeneous transforms -> cam3->world (`T03^-1`) -> world->cam2 (`T02`) | `rt_to_T`, `transform_points_h`, `Pi`, `PiInv` | Using inverse on wrong transform; wrong multiplication order | yes |
| Q19 | RANSAC iteration count for homography | `stoney/Exam24.pdf` (Q19), `notebooks/exam2024.ipynb` (Q19 cell) | Inliers `s`, matches `m`, confidence `p`, sample size `n=4` | Minimum iterations `N` | `N = log(1-p)/log(1-(s/m)^n)` | `ransac_iterations` helper | Log sign mistakes; rounding policy (ceil vs nearest) | yes |
| Q20 | RANSAC threshold from noise model | `stoney/Exam24.pdf` (Q20), `notebooks/exam2024.ipynb` (Q20 cell) | `sigma_x=sigma_y=1.4`, squared reprojection distance, 95% inliers | Squared threshold `tau^2` | Chi-square quantile * variance scale | `ransac_threshold_chi2` helper | DOF ambiguity: notebook uses `3.84*sigma^2`; for 2D squared Euclidean many workflows use `5.99*sigma^2` (`uncertain`, keep both variants) | yes |
| Q21 | Structured-light phase unwrapping (heterodyne) | `stoney/Exam24.pdf` (Q21), `notebooks/exam2024.ipynb` (Q21 cell) | `n1=40`, `n2=41`, primary and secondary intensity sequences | Unwrapped phase `theta` at pixel | FFT first harmonic for both patterns -> wrapped phases -> phase cue -> order -> unwrapped phase | `unwrap_phase_pixel` (FFT+heterodyne logic) | Using wrong harmonic index; modulo handling mistakes; radians vs degrees | yes |

Notes:
- Computationally lighter/theory-heavy 2024 questions (e.g., Q3, Q5, Q7, Q8, Q11, Q13, Q14, Q22) are excluded from this bank by design (`theory later`).

## 3) Exact Computational Exam Question Types from 2023 (Primary: notebook)

| Question # | Problem type | Source notebook/cell reference | What is given | What must be computed | Required computation pipeline | Minimal helper functions needed later | Common traps | Plug-and-play template later? |
|---|---|---|---|---|---|---|---|---|
| Q1 | Build intrinsic matrix | `notebooks/exam2023.ipynb` (Q1 cell) | `f=1200`, principal point `(400,350)`, `alpha,beta` | Camera matrix `K` | Fill intrinsic matrix from parameters | `camera_intrinsic` | Mixing `alpha/beta` placement; center order `(cx,cy)` | yes |
| Q2 | Image resize / pixel mapping | `notebooks/exam2023.ipynb` (Q2 cell) | Normalized coordinate (`Px`), two focal/center scales | Pixel coordinates before/after resizing context | Use `u = f*x + cx` at each scale | `scale_intrinsics` helper, direct projection scalar helper | Confusing normalized coordinate with pixel coordinate | yes |
| Q3 | 3D projection | `notebooks/exam2023.ipynb` (Q3 cells) | `K,R,t,Q` | 2D projection | `q = Pi(K[R|t]PiInv(Q))` | `project_points`, `Pi`, `PiInv` | Wrong transpose/shape; Rodrigues vector conversion | yes |
| Q5 | Epipolar line + distance | `notebooks/exam2023.ipynb` (Q5 cell) | Three camera extrinsics, `K`, observed `p1,p2,p3` | Distance from observed point to induced epipolar line | `F` from cam pair -> `l = F p` or `p^T F` form -> distance | `fundamental_matrix`, `point_line_distance`, `PiInv` | Wrong camera pair/orientation for `F`; homogeneous mismatch | yes |
| Q6 | Linear triangulation (no normalization) | `notebooks/exam2023.ipynb` (Q6 cell) | Multiple `P` matrices and image observations | 3D point | Build linear system from each view -> SVD -> dehomogenize | `triangulate`, `projection_matrix` | Using only two of three views; forgetting dehomogenization | yes |
| Q11 | Point-to-line distance in homogeneous form | `notebooks/exam2023.ipynb` (Q11 cell) | Homogeneous point `q` and line `l` | Shortest distance | Closed-form homogeneous line-point distance | `point_line_distance` | Not normalizing line term `sqrt(a^2+b^2)` | yes |
| Q12 | Harris response + NMS | `notebooks/exam2023.ipynb` (Q12 cell) | Tensor terms, `k=0.06`, threshold `tau=516` | Corner coordinates/count | Compute Harris metric -> threshold -> NMS | `harris_measure`, NMS helper | Neighborhood definition sensitivity; threshold interpretation | yes |
| Q13 | RANSAC line inlier count | `notebooks/exam2023.ipynb` (Q13 cell) | Two seed points, candidate points, threshold | Number of inliers | Line from cross product -> evaluate distance for all points | `fit_line_from_two_points`, `point_line_distance`, `count_inliers` | Treating row/column incorrectly for point arrays | yes |
| Q14 | RANSAC iteration count | `notebooks/exam2023.ipynb` (Q14 cell) | `s,m,p,n` | Required iterations | Standard RANSAC iteration formula | `ransac_iterations` | Using integer division or wrong exponent | yes |
| Q15 (`uncertain`) | Difference-of-Gaussians / blob-related | `notebooks/exam2023.ipynb` (Q15 placeholder only) | Not implemented in notebook (comment only) | Likely DoG-related quantity | Use exercise/slide DoG pipeline as fallback | `difference_of_gaussians`, `detect_blobs` | Source is incomplete; treat as likely but uncertain | yes |

## 4) Topic-Based Variation Bank

## Topic: Homogeneous coordinates

### Base question type
- Source(s): `notebooks/ex1.ipynb` (Ex 1.1, 1.2, 1.12), `stoney/Exercises_Combined.pdf` (Exercise 1), `stoney/Lectures_Combinede.pdf` (homogeneous-coordinate recap).
- Given: 2D/3D homogeneous points with arbitrary scales.
- Compute: Inhomogeneous coordinates and/or re-homogenized vectors.
- Pipeline: divide by last coordinate -> convert back by appending ones.
- Needed helper functions: `Pi`, `PiInv`.
- Common traps: forgetting scale equivalence; shape mistakes (`(n,)` vs `(n,1)`).

### Likely variations
1. Variation:
   - Given: mixed valid/invalid homogeneous points where `w` may be negative/small.
   - Compute: consistent inhomogeneous points and detect equivalent points up to scale.
   - Pipeline: robust `Pi` conversion + scale normalization for comparisons.
   - Why likely: frequent warm-up pattern in exercises + lectures.
   - Template priority: High

2. Variation:
   - Given: Euclidean transform matrix in homogeneous form and input point(s).
   - Compute: transformed inhomogeneous coordinates.
   - Pipeline: homogeneous multiply -> dehomogenize.
   - Why likely: directly linked to early-week exercise format.
   - Template priority: Medium

## Topic: 2D lines and point-line distance; line intersection

### Base question type
- Source(s): `notebooks/ex1.ipynb` (Ex 1.3-1.8), `notebooks/exam2023.ipynb` (Q11, Q13), `stoney/Exercises_Combined.pdf` (Exercise 1).
- Given: homogeneous line(s), homogeneous point(s), or two points defining a line.
- Compute: line equation, intersection point, distance, or inlier counts.
- Pipeline: line from cross product -> point-line formula -> optional thresholding.
- Needed helper functions: `fit_line_from_two_points`, `point_line_distance`, `line_intersection`.
- Common traps: wrong cross-product order; non-normalized line coefficients.

### Likely variations
1. Variation:
   - Given: two lines in homogeneous form.
   - Compute: their intersection in both homogeneous and inhomogeneous coordinates.
   - Pipeline: `q = l1 x l2` -> `Pi(q)`.
   - Why likely: canonical exercise+exam building block.
   - Template priority: High

2. Variation:
   - Given: candidate points and a line with threshold `tau`.
   - Compute: inlier set and count.
   - Pipeline: iterate distances -> mask by `dist < tau`.
   - Why likely: direct precursor to RANSAC exam tasks.
   - Template priority: High

## Topic: camera projection with K, R, t; projection matrix P = K[R|t]

### Base question type
- Source(s): `notebooks/ex1.ipynb` (Ex 1.13-1.15), `notebooks/exam2023.ipynb` (Q3), `notebooks/exam2024.ipynb` (Q1), `stoney/Exam24.pdf` (Q1).
- Given: intrinsic/extrinsic parameters and one or more 3D points.
- Compute: image coordinates.
- Pipeline: form `P = K[R|t]` -> multiply homogeneous 3D points -> dehomogenize.
- Needed helper functions: `camera_intrinsic`, `projection_matrix`, `project_points`, `Pi`, `PiInv`.
- Common traps: wrong frame convention; forgetting `t` as `(3,1)`.

### Likely variations
1. Variation:
   - Given: same `Q`, two different poses.
   - Compute: both projections and pixel displacement.
   - Pipeline: evaluate projection under each `(R,t)` and subtract.
   - Why likely: common multiple-choice style distractor setup.
   - Template priority: High

2. Variation:
   - Given: `K`, `R`, `t`, and already-assembled `P` options.
   - Compute: which `P` is correct.
   - Pipeline: construct `K[R|t]`, compare up to scale.
   - Why likely: exam likes matrix-option questions.
   - Template priority: High

## Topic: camera center from R,t; camera coordinate transforms

### Base question type
- Source(s): `notebooks/exam2024.ipynb` (Q17, Q18), `stoney/Exam24.pdf` (Q17, Q18), `stoney/Lectures_Combinede.pdf` (pose vs position slides).
- Given: extrinsics and/or a point in one camera frame.
- Compute: camera world position or point in another camera frame.
- Pipeline: `C = -R^T t`; or transform chaining via `T_world->cam` and inverse.
- Needed helper functions: `camera_center_from_rt`, `rt_to_T`, `transform_point_between_frames`, `Pi`, `PiInv`.
- Common traps: translation is not camera position; wrong transform order.

### Likely variations
1. Variation:
   - Given: two cameras `(R1,t1),(R2,t2)` and point in cam1.
   - Compute: same point in cam2.
   - Pipeline: cam1->world (`T1^-1`) then world->cam2 (`T2`).
   - Why likely: exact pattern of 2024 Q18.
   - Template priority: High

2. Variation:
   - Given: pose matrix `T` directly.
   - Compute: camera center and orientation axes in world frame.
   - Pipeline: invert `T`; extract translation and rotation.
   - Why likely: appears repeatedly in lectures/ex11 commentary.
   - Template priority: Medium

## Topic: image resizing and camera matrix scaling

### Base question type
- Source(s): `notebooks/exam2023.ipynb` (Q2), `notebooks/ex5.ipynb` (resize calibration workflow comments), `stoney/Lectures_Combinede.pdf` (distortion notes under resizing).
- Given: scale factors and camera intrinsics.
- Compute: resized pixel mapping and updated intrinsics.
- Pipeline: scale `fx, fy, cx, cy` by resize factors.
- Needed helper functions: `scale_intrinsics`, `project_scalar`.
- Common traps: scaling focal but forgetting principal point; anisotropic resize handling.

### Likely variations
1. Variation:
   - Given: original `K` and image resized by `sx, sy`.
   - Compute: new `K'`.
   - Pipeline: multiply first row by `sx`, second row by `sy` (except `[2,2]=1`).
   - Why likely: practical calibration post-processing pattern.
   - Template priority: High

2. Variation:
   - Given: normalized coordinate and two camera parameter sets.
   - Compute: corresponding pixel coordinates in each image.
   - Pipeline: apply `u = fx*x + cx`, `v = fy*y + cy` twice.
   - Why likely: directly mirrors 2023 Q2 style.
   - Template priority: High

## Topic: radial distortion / undistortion point mapping

### Base question type
- Source(s): `notebooks/ex2.ipynb` (Ex 2.2-2.4), `notebooks/exam2024.ipynb` (Q4), `stoney/Exam24.pdf` (Q4), lecture distortion slides.
- Given: `K`, distortion coefficients, pixel(s).
- Compute: distorted or undistorted location.
- Pipeline: pixel->normalized -> radial polynomial -> map back; for image-level, remap grid.
- Needed helper functions: `distort`, `undistort_image`, `Pi`, `PiInv`.
- Common traps: using inverse direction incorrectly; mixing pixel and normalized coordinates.

### Likely variations
1. Variation:
   - Given: one undistorted pixel and `k3,k5,k7`.
   - Compute: distorted pixel.
   - Pipeline: identical to 2024 Q4.
   - Why likely: exact exam pattern already appeared.
   - Template priority: High

2. Variation:
   - Given: distorted image and calibration parameters.
   - Compute: undistorted image via coordinate remap.
   - Pipeline: build dense pixel grid -> forward map into distorted domain -> bilinear remap.
   - Why likely: core exercise and lecture pipeline.
   - Template priority: Medium

## Topic: homography estimation and application

### Base question type
- Source(s): `notebooks/ex2.ipynb` (Ex 2.5-2.11), `notebooks/ex10.ipynb`, `notebooks/exam2024.ipynb` (Q2, Q12), `stoney/Exam24.pdf` (Q2, Q12).
- Given: 4+ correspondences or SIFT matches across images.
- Compute: `H` and/or mapped points.
- Pipeline: DLT estimate `H` -> optional RANSAC -> map points/image warp.
- Needed helper functions: `hest`, `Hest_dist`, `ransac_homography`, `warpImage`, `Pi`, `PiInv`.
- Common traps: swapped source/target points giving inverse homography.

### Likely variations
1. Variation:
   - Given: exactly 4 point pairs.
   - Compute: homography among matrix options.
   - Pipeline: no normalization DLT + scale-normalized comparison.
   - Why likely: exact 2024 Q2 pattern.
   - Template priority: High

2. Variation:
   - Given: two images and feature matches with outliers.
   - Compute: robust homography and inlier count.
   - Pipeline: feature matching -> RANSAC -> refit on inliers.
   - Why likely: exact course and exam Q12 style.
   - Template priority: High

## Topic: point normalization; SVD-based solving

### Base question type
- Source(s): `notebooks/ex2.ipynb` (Ex 2.7-2.9), `notebooks/ex4.ipynb` (DLT), lecture normalization + SVD slides.
- Given: noisy correspondences for DLT-type estimation.
- Compute: normalized points and stable linear solution.
- Pipeline: center+scale normalization -> build linear system -> SVD smallest singular vector -> denormalize.
- Needed helper functions: `normalize2d`, generic `solve_homogeneous_svd`.
- Common traps: forgetting denormalization transform.

### Likely variations
1. Variation:
   - Given: homography correspondences with large coordinate magnitudes.
   - Compute: `H` with/without normalization; compare reprojection quality.
   - Pipeline: run both DLT variants and evaluate error.
   - Why likely: explicit in exercises and lecture motivation.
   - Template priority: High

2. Variation:
   - Given: 3D-2D correspondences.
   - Compute: projection matrix `P` via DLT + SVD.
   - Pipeline: `pest` style kron/cross linearization -> SVD solve.
   - Why likely: appears in exercise 4 and supports calibration tasks.
   - Template priority: High

## Topic: fundamental matrix; essential matrix

### Base question type
- Source(s): `notebooks/ex3.ipynb` (Ex 3.3, 3.8), `notebooks/ex9.ipynb`, `notebooks/ex11.ipynb`, lecture epipolar slides.
- Given: camera poses or correspondences.
- Compute: `E` and/or `F`.
- Pipeline: relative pose (`R_tilde,t_tilde`) -> `E=[t]_xR` -> `F=K^-T E K^-1` or 8-point + SVD.
- Needed helper functions: `CrossOp`, `essential_matrix`, `fundamental_matrix`, `Fest_8point`.
- Common traps: pose direction and sign conventions.

### Likely variations
1. Variation:
   - Given: `R1,t1,R2,t2,K`.
   - Compute: `F12`.
   - Pipeline: convert to relative pose then to `E` and `F`.
   - Why likely: routine step in multiple exam computations.
   - Template priority: High

2. Variation:
   - Given: 8+ noisy point matches.
   - Compute: `F` via normalized 8-point, optional rank-2 enforcement.
   - Pipeline: normalize points -> SVD solve -> enforce rank 2 -> denormalize.
   - Why likely: core week-9 exercise and lecture objective.
   - Template priority: Medium

## Topic: epipolar lines; point-to-epipolar-line distance

### Base question type
- Source(s): `notebooks/ex3.ipynb` (Ex 3.4-3.5, 3.9-3.10), `notebooks/exam2023.ipynb` (Q5), `notebooks/exam2024.ipynb` (Q15), `stoney/Exam24.pdf` (Q15).
- Given: `F` and observed points.
- Compute: epipolar line equation and distance.
- Pipeline: line from `l2=F q1` or `l1=q2^T F` -> normalized line-point distance.
- Needed helper functions: `compute_epiline`, `point_line_distance`, `DrawLine`, `PiInv`.
- Common traps: left/right multiplication mismatch.

### Likely variations
1. Variation:
   - Given: point in image 1 and `F12`.
   - Compute: epipolar line in image 2 and verify if given point in image 2 is consistent.
   - Pipeline: `l2=F12 q1`; check `q2^T l2 ≈ 0`.
   - Why likely: direct week-3 and exam style.
   - Template priority: High

2. Variation:
   - Given: `p1,p2` with noise.
   - Compute: geometric distance to epipolar line for scoring candidates.
   - Pipeline: compute both line and distance; optionally symmetric distance.
   - Why likely: used in robust fitting and exam Q15.
   - Template priority: High

## Topic: triangulation; reprojection error; nonlinear optimization / least squares

### Base question type
- Source(s): `notebooks/ex3.ipynb` (Ex 3.11), `notebooks/ex5.ipynb` (Ex 5.3-5.4), `notebooks/exam2023.ipynb` (Q6), `notebooks/exam2024.ipynb` (Q16), lectures on linear vs nonlinear optimization.
- Given: multiple 2D observations and projection matrices.
- Compute: 3D point; optionally refined point and reprojection error.
- Pipeline: linear triangulation (SVD) -> reprojection residual -> nonlinear least squares refinement.
- Needed helper functions: `triangulate`, `triangulate_nonlin`, `compute_rmse`, reprojection residual helper.
- Common traps: not using homogeneous formulation correctly; reporting wrong frame.

### Likely variations
1. Variation:
   - Given: two or three cameras with noisy point observations.
   - Compute: linear and nonlinear triangulated points + compare residuals.
   - Pipeline: `triangulate` -> residual computation -> `least_squares` refinement.
   - Why likely: exact exercise-5 and 2024 Q16 pattern.
   - Template priority: High

2. Variation:
   - Given: triangulated point and camera set.
   - Compute: per-camera reprojection errors and RMSE.
   - Pipeline: project estimated `Q` into each camera and compare to observations.
   - Why likely: frequent grading metric in calibration/reconstruction tasks.
   - Template priority: High

## Topic: Harris response; non-maximum suppression

### Base question type
- Source(s): `notebooks/ex6.ipynb` (Ex 6.4-6.5), `notebooks/exam2023.ipynb` (Q12), `notebooks/exam2024.ipynb` (Q6), `stoney/Exam24.pdf` (Q6).
- Given: structure tensor terms, `k`, threshold.
- Compute: Harris metric and detected corners.
- Pipeline: `r = ab - c^2 - k(a+b)^2` -> threshold -> NMS.
- Needed helper functions: `harris_measure`, `corner_detector`.
- Common traps: neighborhood definition (4-neighbor vs 8-neighbor), index ordering.

### Likely variations
1. Variation:
   - Given: tiny tensor grids in table form.
   - Compute: whether any corner exists and at which index.
   - Pipeline: explicit metric table + NMS.
   - Why likely: exact exam format.
   - Template priority: High

2. Variation:
   - Given: full image and parameter ranges for `sigma,epsilon,k,tau`.
   - Compute: corner count and overlay plot.
   - Pipeline: gradients -> tensor smoothing -> Harris metric -> threshold+NMS.
   - Why likely: direct exercise progression.
   - Template priority: Medium

## Topic: blob detection / scale space

### Base question type
- Source(s): `notebooks/ex8.ipynb` (Ex 8.2-8.3), `notebooks/exam2023.ipynb` (Q15 placeholder), `stoney/Exercises_Combined.pdf` (Exercise 8), lecture blobs/DoG slides.
- Given: image, `sigma`, number of scales `n`, threshold `tau`.
- Compute: DoG pyramid and blob coordinates (with scale).
- Pipeline: Gaussian scale space -> adjacent differences -> 2D/3D NMS + threshold.
- Needed helper functions: `scale_spaced`, `difference_of_gaussians`, `detect_blobs`.
- Common traps: NMS only in-space but not across scale.

### Likely variations
1. Variation:
   - Given: `sigma,n,tau` with one image.
   - Compute: total number of blobs + sample coordinates.
   - Pipeline: DoG creation + threshold + cross-scale maxima filtering.
   - Why likely: explicit exercise deliverable and 2023 DoG hint.
   - Template priority: Medium

2. Variation:
   - Given: same image and two parameter sets.
   - Compute: sensitivity comparison (blob counts/locations).
   - Pipeline: run detector twice and compare statistics.
   - Why likely: common “parameter reasoning” exam variation.
   - Template priority: Low

## Topic: RANSAC iteration formulas (and thresholding where relevant)

### Base question type
- Source(s): `notebooks/ex7.ipynb` (Ex 7.13), `notebooks/ex9.ipynb`, `notebooks/exam2023.ipynb` (Q14), `notebooks/exam2024.ipynb` (Q19, Q20), `stoney/Exam24.pdf` (Q19, Q20), lecture RANSAC slides.
- Given: inlier ratios/confidence/sample size and/or measurement noise `sigma`.
- Compute: required iterations and/or inlier threshold.
- Pipeline: use closed-form `N` formula; use chi-square quantiles for threshold.
- Needed helper functions: `ransac_iterations`, `ransac_iterations_adaptive`, `ransac_threshold_chi2`.
- Common traps: wrong residual DOF and rounding.

### Likely variations
1. Variation:
   - Given: current best inlier count during RANSAC for homography (`n=4`).
   - Compute: minimum total iterations for target confidence.
   - Pipeline: plug into `N` formula.
   - Why likely: exact 2024 Q19 and 2023 Q14 style.
   - Template priority: High

2. Variation:
   - Given: Gaussian keypoint noise and desired inlier rate (95%/99%).
   - Compute: squared distance threshold.
   - Pipeline: `tau^2 = chi2_quantile(dof)*sigma^2` (or per-axis variant).
   - Why likely: appears in 2024 Q20 and lecture thresholding section.
   - Template priority: High

## Topic: structured light (phase-shift computational core)

### Base question type
- Source(s): `notebooks/ex13.ipynb` (Ex 13.3-13.6), `notebooks/exam2024.ipynb` (Q21), `stoney/Exam24.pdf` (Q21), lecture structured-light slides.
- Given: primary/secondary phase-shift intensities and pattern periods.
- Compute: unwrapped phase (and in full pipeline, stereo matches + 3D points).
- Pipeline: FFT first harmonic -> wrapped phases -> heterodyne cue -> phase order -> unwrap.
- Needed helper functions: `unwrap`/`unwrap_phase_pixel`, `phase_match_row`, `triangulate_points`.
- Common traps: wrong period assignment (`n1`,`n2`) or modulo wrapping mistakes.

### Likely variations
1. Variation:
   - Given: one pixel’s primary and secondary intensity arrays.
   - Compute: unwrapped phase value.
   - Pipeline: exactly as Q21.
   - Why likely: exact 2024 computational pattern.
   - Template priority: High

2. Variation:
   - Given: rectified phase maps and masks from two cameras.
   - Compute: disparity/phase matches and triangulated 3D cloud.
   - Pipeline: row-wise phase nearest match -> `cv2.triangulatePoints` -> remove negative depth.
   - Why likely: direct exercise-13 pipeline and likely extension question.
   - Template priority: Medium

## Topic: camera calibration / Zhang (computational)

### Base question type
- Source(s): `notebooks/ex4.ipynb` (Ex 4.5-4.10), `notebooks/ex5.ipynb` (Ex 5.8-5.10), `notebooks/exam2024.ipynb` (Q10), `stoney/Exam24.pdf` (Q10), lectures on Zhang (2000).
- Given: checkerboard correspondences across views.
- Compute: intrinsics/extrinsics, focal length, reprojection error.
- Pipeline: estimate per-view homographies -> solve `Vb=0` with SVD -> recover `K` -> recover `(R,t)` -> evaluate RMSE.
- Needed helper functions: `checkerboard_points`, `estimate_homographies`, `estimate_b`, `estimate_intrinsics`, `estimate_extrinsics`, `calibrate_camera`, `compute_rmse`.
- Common traps: checkerboard point ordering mismatch with OpenCV corner order.

### Likely variations
1. Variation:
   - Given: 3-5 board images and corner detections.
   - Compute: focal length candidate and/or complete `K`.
   - Pipeline: Zhang calibration workflow; compare options.
   - Why likely: exact 2024 Q10 style.
   - Template priority: High

2. Variation:
   - Given: calibrated `K,Rs,ts` and observed points.
   - Compute: per-image reprojection RMSE and worst frame index.
   - Pipeline: project known 3D pattern points -> RMSE per frame.
   - Why likely: calibration quality assessment is emphasized in exercises/slides.
   - Template priority: Medium

## 5) Prioritized Implementation Roadmap

### Priority A (Exact 2023/2024 computational exam patterns)
- `project_3d_point_template`
- `estimate_homography_dlt_template`
- `map_undistorted_to_distorted_pixel_template`
- `harris_from_tensor_with_nms_template`
- `rootsift_ratio_match_count_template`
- `zhang_focal_from_checkerboards_template`
- `epipolar_distance_template`
- `triangulate_point_nonlinear_template`
- `camera_center_from_rt_template`
- `transform_point_between_camera_frames_template`
- `ransac_iterations_template`
- `ransac_threshold_from_sigma_template`
- `structured_light_phase_unwrap_pixel_template`
- `line_ransac_inlier_count_template`

### Priority B (Very likely variations from exercises and slides)
- `normalized_homography_template`
- `fundamental_from_extrinsics_template`
- `essential_from_relative_pose_template`
- `triangulate_point_linear_template`
- `reprojection_rmse_template`
- `camera_intrinsics_scaling_after_resize_template`
- `calibrate_camera_full_zhang_template`
- `ransac_fundamental_with_sampson_template`
- `ransac_homography_with_bidirectional_distance_template`
- `structured_light_rowwise_phase_match_and_triangulate_template`

### Priority C (Lower-probability or advanced variations)
- `blob_detector_dog_multiscale_nms_template`
- `adaptive_ransac_stopping_template`
- `pose_from_E_then_PnP_chain_template`
- `calibration_cross_validation_template`
- `bundle_adjustment_init_template`

## 6) Final Compact Build Table

| Priority | Problem type | Template name to build later | Helper functions needed | Source evidence | Notes |
|---|---|---|---|---|---|
| A | 3D projection with `K,R,t` | `project_3d_point_template` | `camera_intrinsic`, `project_points`, `Pi`, `PiInv` | 2024 Q1, 2023 Q3, Ex1/Ex2 | Highest reuse across exams |
| A | 4-point homography (DLT) | `estimate_homography_dlt_template` | `hest`, `Pi`, `PiInv` | 2024 Q2, Ex2 | Normalize comparison by scale |
| A | Radial distortion mapping | `map_undistorted_to_distorted_pixel_template` | `distort`, `Pi`, `PiInv` | 2024 Q4, Ex2 | Direction matters |
| A | Harris+NMS from tensor tables | `harris_from_tensor_with_nms_template` | `harris_measure`, NMS helper | 2024 Q6, 2023 Q12, Ex6 | Include both 4- and 8-neighborhood modes |
| A | SIFT/RootSIFT ratio count | `rootsift_ratio_match_count_template` | matcher wrapper, RootSIFT transform | 2024 Q9 | RootSIFT preprocessing ambiguous in notebook |
| A | Checkerboard focal estimation | `zhang_focal_from_checkerboards_template` | `checkerboard_points`, `calibrate_camera` | 2024 Q10, Ex4/Ex5 | Handle corner ordering conversion |
| A | Epipolar line distance | `epipolar_distance_template` | `fundamental_matrix`, `point_line_distance`, `PiInv` | 2024 Q15, 2023 Q5, Ex3 | Must encode `F` direction convention |
| A | Nonlinear triangulation | `triangulate_point_nonlinear_template` | `projection_matrix`, `triangulate`, `triangulate_nonlin` | 2024 Q16, Ex5 | Use linear init before least squares |
| A | Camera center from extrinsics | `camera_center_from_rt_template` | small algebra helper | 2024 Q17, lecture pose slides | `C = -R^T t` |
| A | Point transform cam3->cam2 | `transform_point_between_camera_frames_template` | `rt_to_T`, `Pi`, `PiInv` | 2024 Q18 | Transform chain order is critical |
| A | RANSAC iteration count | `ransac_iterations_template` | scalar math helper | 2024 Q19, 2023 Q14, Ex7 | Include ceil handling |
| A | RANSAC threshold from sigma | `ransac_threshold_from_sigma_template` | chi-square lookup helper | 2024 Q20, Ex9, lecture threshold slides | Include DOF switch (`1` vs `2`) |
| A | Structured-light pixel phase | `structured_light_phase_unwrap_pixel_template` | `unwrap` core | 2024 Q21, Ex13 | FFT bin and modulo pitfalls |
| A | Line inlier count from two points | `line_ransac_inlier_count_template` | `fit_line`, `point_line_distance` | 2023 Q13, Ex7 | Minimal but common |
| B | Normalized homography DLT | `estimate_homography_normalized_template` | `normalize2d`, `hest` | Ex2, lecture normalization | Improves numerical stability |
| B | Fundamental from camera poses | `fundamental_from_extrinsics_template` | `CrossOp`, `essential_matrix`, `fundamental_matrix` | Ex3, lecture epipolar | Reused in distance questions |
| B | Essential matrix from relative pose | `essential_from_relative_pose_template` | `essential_matrix`, relative pose helper | Ex3, Ex11 | Key multi-view primitive |
| B | Linear triangulation | `triangulate_point_linear_template` | `triangulate` | 2023 Q6, Ex3 | Fast initialization template |
| B | Reprojection RMSE | `reprojection_rmse_template` | `project_points`, `compute_rmse` | Ex4/Ex5 | Needed for calibration diagnostics |
| B | Scale intrinsics after resizing | `camera_intrinsics_scaling_after_resize_template` | `scale_intrinsics` | 2023 Q2, Ex5 notes | Important practical correction |
| B | Full Zhang calibration | `calibrate_camera_full_zhang_template` | `estimate_homographies`, `estimate_b`, `estimate_intrinsics`, `estimate_extrinsics` | Ex4, lectures | End-to-end reusable block |
| B | F-matrix RANSAC with Sampson | `ransac_fundamental_with_sampson_template` | `Fest_8point`, `sampsons_distance` | Ex9, lecture RANSAC-F | High practical value |
| B | Homography RANSAC with symmetric transfer distance | `ransac_homography_bidirectional_template` | `ransac_homography`, `Hest_dist` | Ex10, 2024 Q12 | Matches exam-like image pair tasks |
| B | Structured-light matching + triangulation | `structured_light_phase_match_and_triangulate_template` | `unwrap`, `cv2.triangulatePoints` wrapper | Ex13 | Likely extension beyond Q21 |
| C | DoG blob detector | `detect_blobs_dog_template` | `difference_of_gaussians`, NMS helper | Ex8, lectures | Lower direct exam frequency than core geometry |
| C | Adaptive RANSAC stop | `adaptive_ransac_stopping_template` | inlier-ratio estimator | Ex7.13, lecture | Useful optimization, less likely direct MCQ |
| C | Relative pose chaining and PnP | `pose_chain_with_pnp_template` | `cv2.findEssentialMat`, `cv2.recoverPose`, `cv2.solvePnPRansac` | Ex11, lectures | More advanced pipeline integration |
| C | Calibration cross-validation | `calibration_crossval_template` | reprojection split helpers | lecture calibration cautionary section | Advanced quality-control workflow |
| C | BA initializer | `bundle_adjustment_init_template` | residual/Jacobian stubs | lectures (nonlinear optimization) | lower-probability exam coding depth |

---

### Ambiguities Explicitly Marked
- 2023 Q15 in `notebooks/exam2023.ipynb` is only a placeholder comment (`DoG`) with no worked code; treated as likely computational topic but marked `uncertain`.
- 2024 Q20 thresholding has a DOF interpretation ambiguity between notebook practice (`3.84*sigma^2`) and common 2D squared reprojection usage (`5.99*sigma^2`). Keep both variants in template design.
