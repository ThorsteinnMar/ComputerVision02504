import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import chi2

def crossOp(p):
    x, y, z = p.reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

# Converts inhomogenous coordinates to homogeneous coordinates
def Pi(coords):
    return np.vstack((coords, np.ones(coords.shape[1])))

# Converts homogeneous coordinates to inhomogeneous coordinates
def PiInv(coords):
    return coords[:-1]/coords[-1]

# Q are the 3d points to project to undistorted 2d pixel space
def projectpoints(K, R, t, Q, distCoeffs):
    p_cam = np.concatenate((R, t), axis=1)@Pi(Q)
    p_cam_inv = PiInv(p_cam)
    if len(distCoeffs) > 0:
        p_cam_dist = p_cam_inv * (1+(distCoeffs[0]*(np.linalg.norm(p_cam_inv, axis=0))**2)+(distCoeffs[1]*(np.linalg.norm(p_cam_inv, axis=0))**4)+(distCoeffs[2]*(np.linalg.norm(p_cam_inv, axis=0))**6))
        p_h = K@Pi(p_cam_dist)
        return PiInv(p_h)
    p_h = K@Pi(p_cam_inv)
    return PiInv(p_h)

# p is undistorted image point (2, 1), distCoeffs is [k3, k5, k7]
def undistort_image_point(p, K, distCoeffs):
    q = PiInv(np.linalg.inv(K)@Pi(p))
    p_cam_distorted = q * (1+(distCoeffs[0]*(np.linalg.norm(q, axis=0))**2)+(distCoeffs[1]*(np.linalg.norm(q, axis=0))**4)+(distCoeffs[2]*(np.linalg.norm(q, axis=0))**6))
    return PiInv(K@Pi(p_cam_distorted))

# Estimates homogrophy between 2 sets of point correspondances
# pts1 = H*pts2, Solves for H
# points are image plane points in homogenous coords on the format (3, N)
def hest(pts1, pts2):
    A = []
    for i in range(pts1.shape[1]):
        pts1i = pts1[:, i]
        pts2i = pts2[:, i]
        B_i = np.kron(pts2i.T, crossOp(pts1i))
        A.append(B_i)
    A = np.vstack(A)

    # Solve Ah = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3).T

    return H

# Returns an epipolar line in Camera 1 induced by point in the other camera
# point in mage plane, format (2, 1)
def calculate_epipolar_line(K, R1, t1, R2, t2, point):

    R_12 = R1 @ R2.T
    t_12 = t1 - R_12 @ t2

    E = crossOp(t_12) @ R_12
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    l = F@Pi(point)
    return l

# point in image plane, format (2, 1)
def calculate_distance_from_line(l, point):
    dist = Pi(point).T@l
    dist = dist / np.sqrt(l[0]**2+l[1]**2)
    return dist

# p1 = R21p2 + t21
def calculate_3d_point_from_reference_frame(R1, t1, R2, t2, p2):
    R_21 = R2 @ R1.T
    t_21 = t2 - R_21 @ t1 
    p1 = R_21@p2 + t_21.reshape(3, 1)
    return p1
    

# Q multiple points, P multiple Pmat, returns point in inhomogenous coords, Q and P on the following format
# Q = np.concatenate((p1, p2, p3), axis=1)
# P = np.stack((projection_matrix1, projection_matrix2, projection_matrix3), axis=0)
def triangulate(Q, P):
    A = []
    for i in range(Q.shape[1]):
        x = Q[0, i]
        y = Q[1, i]

        Pi = P[i]           # shape (3, 4)
        p1 = Pi[0, :]       # row 1
        p2 = Pi[1, :]       # row 2
        p3 = Pi[2, :]       # row 3

        A.append(x * p3 - p1)
        A.append(y * p3 - p2)

    A = np.vstack(A) 


    # Solve using SVD: solution is the right singular vector
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]

    if np.isclose(X_h[-1], 0):
        raise np.linalg.LinAlgError("Triangulation failed: point at infinity or degenerate configuration.")

    return PiInv(X_h.reshape(4,1))

# Creates projection matrix, to project from 3d to 2d image plane
def projectionmatrix(K, R, t):
    t = np.asarray(t).reshape(3, 1)
    return K@np.concatenate((R,t), axis=1)

# Estimate projection matrix, with or without normalization
# Q are 3d points and q their 2d image plain correspondances, all in homogenous coordinates
# see camera calibration file for format
def pest(Q, q, normal = False):
    A = []

    # Normalize using inhomogenous coordinates
    Q = PiInv(Q)
    q = PiInv(q)
    # T for normalization
    TQ = np.linalg.inv(np.array([[Q[0].std(), 0, 0, Q[0].mean()], [0, Q[1].std(), 0, Q[1].mean()], [0, 0, Q[2].std(), Q[2].mean()], [0, 0, 0, 1]]))
    Tq = np.linalg.inv(np.array([[q[0].std(), 0, q[0].mean()], [0, q[1].std(), q[1].mean()], [0, 0, 1]]))

    Q = Pi(Q)
    q = Pi(q)
    for i in range(Q.shape[1]):
        Qi = Q[:, i]
        qi = q[:, i]

        if normal:
            Qi = TQ@Qi
            qi = Tq@qi

        B_i = np.kron(Qi, crossOp(qi))   # shape (3, 12)
        A.append(B_i)

    A = np.vstack(A)   # shape (3N, 12)

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]

    P = h.reshape(4, 3).T

    if normal:
        P = np.linalg.inv(Tq)@P@TQ

    return P

# Calibrates camera based on
# K_est, Rs, ts = calibrateCamera([q_a_noisy, q_b_noisy, q_c_noisy], Q)
def calibrateCamera(qs, Q):
    Hs = estimateHomographies(Q, qs)
    K = estimateIntrinsics(Hs)
    Rs, ts = estimateExtrinsics(K, Hs)
    return K, Rs, ts

# Triangulates points non-linearly, Q and P are as follows
# Q = np.concatenate((p1, p2, p3), axis=1)
# P = np.stack((projection_matrix1, projection_matrix2, projection_matrix3), axis=0)
def triangulate_nonlin(Q, P):
    x0 = triangulate(Q, P)

    def compute_residuals(x):
        residuals = []
        x = Pi(x.reshape(3, 1))
        for i in range(Q.shape[1]):
            projected_point = PiInv(P[i] @ x)
            observed_point = Q[:, i].reshape(2, 1)
            residuals.append((projected_point - observed_point).ravel())
        return np.concatenate(residuals)

    result = least_squares(compute_residuals, x0.reshape(3,))
    return result.x.reshape(3,1)


def harrisMeasure(im, sigma, epsilon, k, useC=False, in_C=None):
    C = in_C
    detC = np.linalg.det(C)
    traceC = np.trace(C, axis1=-2, axis2=-1)
    r = detC - k * traceC**2
    return r

# If not C not given look at simple_features.py, else:
# im="", sigma="", epsilon="", use_C = True, use_tau_as_threshold=True
# Gives corner in format (row, column)
# C on the following format:
# C = C_x_y = np.stack([
#     np.stack([g_x,   g_x_y], axis=-1),
#     np.stack([g_x_y, g_y],   axis=-1)
# ], axis=-2)
def cornerDetector(im, sigma, epsilon, k, tau, useC=False, in_C=None, use_tau_as_threshold=False):
    r = harrisMeasure(im, sigma, epsilon, k, useC=useC, in_C=in_C)

    threshold = tau * np.max(r)
    if use_tau_as_threshold:
        threshold = tau
    local_max = np.zeros_like(r, dtype=bool)
    center = r[1:-1, 1:-1]
    local_max[1:-1, 1:-1] = (
        (center > r[:-2, :-2]) &   # upper-left
        (center > r[:-2, 1:-1]) &  # up
        (center > r[:-2, 2:]) &    # upper-right
        (center > r[1:-1, :-2]) &  # left
        (center > r[1:-1, 2:]) &   # right
        (center > r[2:, :-2]) &    # lower-left
        (center > r[2:, 1:-1]) &   # down
        (center > r[2:, 2:])       # lower-right
    )

    ys, xs = np.where(local_max & (r > threshold))
    c = [(int(x), int(y)) for x, y in zip(ys, xs)]

    return c

# Fits line through 2 points
# point p on the form (1, 2)
def fit_line(p1, p2):
    p1_h = np.array([p1[0], p1[1], 1])
    p2_h = np.array([p2[0], p2[1], 1])

    l = np.cross(p1_h, p2_h)

    # optional normalization
    l = l / np.sqrt(l[0]**2 + l[1]**2)

    return l

# finds inliers that are points within threshold distance from line
# Points on the form (N, 2)
def find_inliers(points, l, threshold):
    a, b, c = l

    norm = np.sqrt(a**2 + b**2)
    l = l / norm
    a, b, c = l

    x = points[:, 0]
    y = points[:, 1]

    errors = np.abs(a*x + b*y + c)

    inliers = errors < threshold

    return inliers, errors

# Returns the count of inliers in an inlier matrix
def inlier_count(inliers):
    return np.sum(inliers)

# Fundemental matrix 8-point algorithm
# Points on format (2, N), should be image plane point correspondances
def Fest_8point(q1, q2):
    if q1.shape[1] != q2.shape[1]:
        raise ValueError("q1 and q2 must contain the same number of points")
    if q1.shape[1] < 8:
        raise ValueError("Need at least 8 correspondences")

    # Step 1: build A
    x1, y1 = q1[0], q1[1]
    x2, y2 = q2[0], q2[1]  

    A = np.column_stack([
        x2 * x1,
        x2 * y1,
        x2,
        y2 * x1,
        y2 * y1,
        y2,
        x1,
        y1,
        np.ones_like(x1)
    ])

    # Step 2: solve Af = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    f = Vt[-1]

    # Step 3: reshape into F
    F = f.reshape(3, 3)

    # Step 4: enforce rank 2
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    return F

### TODO RANSAC for fundemental matrix

# Hægt að finna annað version af þessu falli í image_stitching_RANSAC_Homography.py
# kp1, kp2 og matches fundið eftir að keyra SIFT
# Reverse means finding homography between image 2 and image 1
def ransac_homography(kp1, kp2, matches, num_iterations=2000, threshold=4.0, reverse=False):
    if len(matches) < 4:
        raise ValueError("Need at least 4 matches for homography estimation.")

    # Extract matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])  # from image 1
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])  # to image 2

    if reverse:
        temp_pts1 = pts1
        pts1 = pts2
        pts2 = temp_pts1

    best_H = None
    best_inliers = None
    best_num_inliers = 0

    n = len(matches)

    for _ in range(num_iterations):
        # Randomly sample 4 correspondences
        idx = np.random.choice(n, 4, replace=False)
        sample_pts1 = pts1[idx]
        sample_pts2 = pts2[idx]

        try:
            H = compute_homography(sample_pts1, sample_pts2)
        except np.linalg.LinAlgError:
            continue

        # Project all pts1 into image 2
        projected_pts2 = apply_homography(H, pts1)

        # Reprojection error in image 2
        errors = np.linalg.norm(projected_pts2 - pts2, axis=1)

        inliers = errors < threshold
        num_inliers = np.sum(inliers)

        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inliers = inliers
            best_H = H

    if best_H is None:
        raise RuntimeError("RANSAC failed to find a valid homography.")

    # Recompute H using all inliers
    inlier_pts1 = pts1[best_inliers]
    inlier_pts2 = pts2[best_inliers]
    best_H = compute_homography(inlier_pts1, inlier_pts2)

    best_matches = [m for i, m in enumerate(matches) if best_inliers[i]]
    return best_H

# Helper function for Ransac homography
def apply_homography(H, pts):
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])   # (N, 3)
    proj_h = (H @ pts_h.T).T                               # (N, 3)
    proj_h = proj_h / proj_h[:, [2]]
    return proj_h[:, :2]

# Calculates the number of iterations RANSAC needs
def ransac_iterations(s, m, n, p):
    e = s/m
    N = np.log(1-p) / np.log(1 - (1-e)**n)
    return N

# Applies rootsift to descriptors
def rootsift(des, eps=1e-7):
    # L1 normalize each descriptor
    des /= (np.sum(des, axis=1, keepdims=True) + eps)
    
    # Take elementwise square root
    des = np.sqrt(des)
    
    return des

# Calculates camera position in the world based on it's rotation and translation
def calculate_camera_position(R, t):
    return -R.T@t

# Hægt að finna annað version í file Structured_light_w13, ef þetta dugir ekki
def unwrap(in_primary, in_secondary, primary_periods, secondary_periods):
    primary = in_primary
    secondary = in_secondary 

    fft_primary = np.fft.rfft(primary, axis=0)
    theta_primary = np.mod(np.angle(fft_primary[1]), np.pi * 2)

    fft_secondary = np.fft.rfft(secondary, axis=0)
    theta_secondary = np.mod(np.angle(fft_secondary[1]), np.pi * 2)

    # Það sem er með hærra periods - lægra periods
    theta_c = None
    if primary_periods > secondary_periods:
        theta_c = np.mod(theta_primary - theta_secondary, np.pi * 2)
    else:
        theta_c = np.mod(theta_secondary - theta_primary, np.pi * 2)
    
    cue_periods = abs(secondary_periods - primary_periods) 
    o_primary = np.round(
        (primary_periods / cue_periods * theta_c - theta_primary) / (np.pi*2)
    )

    return (theta_primary + np.pi * 2 * o_primary) / primary_periods

def calculate_RANSAC_threshold(std, df, confidence):
    return (std ** 2) * chi2.ppf(confidence, df)

##### EXTRA CODE FOR HELPER FUNCTIONS
def estimateHomographies(Q_omega, qs):
    Q_tilde = Pi(Q_omega[:2, :])
    
    homographies = []
    for q in qs:
        H = hest(q, Q_tilde)
        homographies.append(H)
    return homographies

def v_ij(H, i, j):
    hi = H[:, i]
    hj = H[:, j]
    return np.array([
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[2] * hj[0] + hi[0] * hj[2],
        hi[2] * hj[1] + hi[1] * hj[2],
        hi[2] * hj[2],
    ])

def estimate_b(Hs):
    V = []
    for H in Hs:
        V.append(v_ij(H, 0, 1))
        V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))

    V = np.vstack(V)
    _, _, Vt = np.linalg.svd(V)
    return Vt[-1]

def estimateIntrinsics(Hs):
    b = estimate_b(Hs)
    if b[0] < 0:
        b = -b

    B11, B12, B22, B13, B23, B33 = b

    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lamb = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lamb / B11)
    beta = np.sqrt(lamb * B11 / (B11 * B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lamb
    u0 = gamma * v0 / beta - B13 * alpha**2 / lamb

    return np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1],
    ])

def estimateExtrinsics(K, Hs):
    Rs = []
    ts = []

    for H in Hs:
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]

        Kinv_h1 = np.linalg.solve(K, h1)
        Kinv_h2 = np.linalg.solve(K, h2)
        Kinv_h3 = np.linalg.solve(K, h3)

        lam = 1.0 / np.linalg.norm(Kinv_h1)

        r1 = lam * Kinv_h1
        r2 = lam * Kinv_h2
        r3 = np.cross(r1, r2)
        t = lam * Kinv_h3

        R = np.column_stack((r1, r2, r3))

        Rs.append(R)
        ts.append(t)

    return Rs, ts

def compute_homography(pts1, pts2):
    if pts1.shape[0] < 4:
        raise ValueError("Need at least 4 correspondences to compute homography.")

    A = []
    for (x, y), (xp, yp) in zip(pts1, pts2):
        A.append([-x, -y, -1,  0,  0,  0, x * xp, y * xp, xp])
        A.append([ 0,  0,  0, -x, -y, -1, x * yp, y * yp, yp])

    A = np.asarray(A)

    # Solve Ah = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3)

    return H