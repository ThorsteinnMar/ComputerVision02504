"""Feature and detector helpers (Harris, DoG, RootSIFT)."""

from __future__ import annotations

import numpy as np


def _to_gray_float(im: np.ndarray) -> np.ndarray:
    img = np.asarray(im)
    if img.ndim == 3:
        import cv2

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(float)


def gaussian_1d_kernel(sigma: float, radius_factor: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return 1D Gaussian and derivative kernels."""
    if sigma <= 0:
        return np.array([1.0]), np.array([0.0])
    radius = int(np.ceil(radius_factor * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    g = np.exp(-(x**2) / (2.0 * sigma**2))
    g = g / np.sum(g)
    gd = -x / (sigma**2) * g
    return g, gd


def gaussian_smoothing(im: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth image and compute first derivatives Ix, Iy."""
    import cv2

    img = _to_gray_float(im)
    g, gd = gaussian_1d_kernel(sigma)
    I = cv2.sepFilter2D(img, -1, g, g)
    Ix = cv2.sepFilter2D(img, -1, gd, g)
    Iy = cv2.sepFilter2D(img, -1, g, gd)
    return I, Ix, Iy


def structure_tensor(im: np.ndarray, sigma: float, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute smoothed structure tensor entries g*(Ix^2), g*(Iy^2), g*(IxIy)."""
    import cv2

    _, Ix, Iy = gaussian_smoothing(im, sigma)
    g_eps, _ = gaussian_1d_kernel(epsilon)
    gxx = cv2.sepFilter2D(Ix**2, -1, g_eps, g_eps)
    gyy = cv2.sepFilter2D(Iy**2, -1, g_eps, g_eps)
    gxy = cv2.sepFilter2D(Ix * Iy, -1, g_eps, g_eps)
    return gxx, gyy, gxy


def harris_response_from_tensor(gxx: np.ndarray, gyy: np.ndarray, gxy: np.ndarray, k: float = 0.06) -> np.ndarray:
    """Harris response R = det(C) - k trace(C)^2."""
    a = np.asarray(gxx, dtype=float)
    b = np.asarray(gyy, dtype=float)
    c = np.asarray(gxy, dtype=float)
    return a * b - c**2 - k * (a + b) ** 2


def harris_response(im: np.ndarray, sigma: float, epsilon: float, k: float = 0.06) -> np.ndarray:
    """Compute Harris response directly from image."""
    gxx, gyy, gxy = structure_tensor(im, sigma, epsilon)
    return harris_response_from_tensor(gxx, gyy, gxy, k=k)


def non_max_suppression_2d(
    response: np.ndarray,
    threshold: float,
    neighborhood: int = 1,
    connectivity: int = 8,
) -> np.ndarray:
    """Return corner coordinates after threshold + local NMS.

    Returns array of (row, col) coordinates with shape (M, 2).
    """
    r = np.asarray(response, dtype=float)
    if r.ndim != 2:
        raise ValueError("response must be 2D")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    coords = []
    h, w = r.shape
    for i in range(neighborhood, h - neighborhood):
        for j in range(neighborhood, w - neighborhood):
            if r[i, j] <= threshold:
                continue
            val = r[i, j]
            patch = r[
                i - neighborhood : i + neighborhood + 1,
                j - neighborhood : j + neighborhood + 1,
            ]
            if connectivity == 8:
                if val >= np.max(patch):
                    coords.append((i, j))
            else:  # 4-connectivity
                if (
                    val > r[i + 1, j]
                    and val >= r[i - 1, j]
                    and val > r[i, j + 1]
                    and val >= r[i, j - 1]
                ):
                    coords.append((i, j))

    if len(coords) == 0:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(coords, dtype=int)


def harris_corners(
    im: np.ndarray,
    sigma: float,
    epsilon: float,
    k: float,
    threshold: float,
    connectivity: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Harris response and NMS corner coordinates."""
    r = harris_response(im, sigma=sigma, epsilon=epsilon, k=k)
    corners = non_max_suppression_2d(r, threshold=threshold, connectivity=connectivity)
    return r, corners


def gaussian_scale_space(
    im: np.ndarray,
    sigma0: float,
    num_scales: int,
    scale_factor: float = 2.0,
) -> tuple[list[np.ndarray], list[float]]:
    """Build Gaussian scale space without downsampling."""
    import cv2

    img = _to_gray_float(im)
    scales = [sigma0 * (scale_factor**i) for i in range(num_scales)]
    gaussians = []
    for sigma in scales:
        g, _ = gaussian_1d_kernel(sigma)
        gaussians.append(cv2.sepFilter2D(img, -1, g, g))
    return gaussians, scales


def difference_of_gaussians(
    im: np.ndarray,
    sigma0: float,
    num_scales: int,
    scale_factor: float = 2.0,
) -> tuple[list[np.ndarray], list[float]]:
    """Build DoG pyramid and associated Gaussian scales."""
    gaussians, scales = gaussian_scale_space(im, sigma0, num_scales, scale_factor)
    dogs = [gaussians[i + 1] - gaussians[i] for i in range(len(gaussians) - 1)]
    return dogs, scales


def detect_blobs_dog(
    im: np.ndarray,
    sigma0: float,
    num_scales: int,
    threshold: float,
    scale_factor: float = 2.0,
) -> list[tuple[int, int, float]]:
    """Detect DoG blob candidates via 3D non-maximum suppression.

    Returns list of (row, col, sigma).
    """
    import cv2

    dogs, scales = difference_of_gaussians(im, sigma0, num_scales, scale_factor)
    abs_dogs = [np.abs(d) for d in dogs]
    dilated = [cv2.dilate(d, np.ones((3, 3), np.uint8)) for d in abs_dogs]

    blobs: list[tuple[int, int, float]] = []
    for s_idx, dog in enumerate(abs_dogs):
        prev_d = abs_dogs[s_idx - 1] if s_idx > 0 else np.zeros_like(dog)
        next_d = abs_dogs[s_idx + 1] if s_idx < len(abs_dogs) - 1 else np.zeros_like(dog)
        max2d = dilated[s_idx]
        h, w = dog.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                v = dog[i, j]
                if (
                    v > threshold
                    and np.isclose(v, max2d[i, j])
                    and v >= prev_d[i, j]
                    and v >= next_d[i, j]
                ):
                    blobs.append((i, j, scales[s_idx]))
    return blobs


def rootsift_descriptors(des: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert SIFT descriptors to RootSIFT."""
    d = np.asarray(des, dtype=float)
    d = d / (np.sum(d, axis=1, keepdims=True) + eps)
    d = np.sqrt(d)
    return d


def match_descriptors_ratio(
    des1: np.ndarray,
    des2: np.ndarray,
    ratio: float = 0.8,
    use_root_sift: bool = True,
) -> list:
    """KNN + Lowe ratio test matching for descriptors."""
    import cv2

    d1 = np.asarray(des1, dtype=np.float32)
    d2 = np.asarray(des2, dtype=np.float32)

    if use_root_sift:
        d1 = rootsift_descriptors(d1).astype(np.float32)
        d2 = rootsift_descriptors(d2).astype(np.float32)

    bf = cv2.BFMatcher()
    knn = bf.knnMatch(d1, d2, k=2)
    passed = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            passed.append(m)
    return passed
