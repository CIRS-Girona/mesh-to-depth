import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform


def compute_distortion_maps(height, width, cameras_info, max_iter=1000, tol=1e-3):
    """
    Compute mapping from distorted pixels to undistorted coordinates.
    :return: map_x, map_y for cv2.remap()

    Source: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    cx = cameras_info.cx + (width - cameras_info.width) // 2
    cy = cameras_info.cy + (height - cameras_info.height) // 2

    u_d, v_d = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    
    # Normalize coordinates (distorted)
    x_prime = (u_d - cx) / cameras_info.fx
    y_prime = (v_d - cy) / cameras_info.fy

    # Iteratively solve for undistorted (x, y)
    x, y = x_prime.copy(), y_prime.copy()
    for _ in range(max_iter):
        r2 = x**2 + y**2
        radial = 1 + cameras_info.k1*r2 + cameras_info.k2*r2**2 + cameras_info.k3*r2**3

        xd = x * radial + 2*cameras_info.p1*x*y + cameras_info.p2*(r2 + 2*x**2)
        yd = y * radial + cameras_info.p1*(r2 + 2*y**2) + 2*cameras_info.p2*x*y

        x_new = x - (xd - x_prime)
        y_new = y - (yd - y_prime)

        if np.linalg.norm((x - x_new, y - y_new)) <= tol:
            break

        x, y = x_new, y_new

    # Convert back to pixel coordinates
    map_x = (x * cameras_info.fx + cx).astype(np.float32)
    map_y = (y * cameras_info.fy + cy).astype(np.float32)

    return map_x, map_y


def compute_homography(img1, img2, transform='affine', min_match_count=10, distance_ratio=0.75, ransac_threshold=3.0, max_iterations=10000):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initiate ORB detector and Brute Force Matcher
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    # find the keypoints and descriptors with ORB
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # Apply ratio test
    matches = []
    for m, n in bf.knnMatch(des1, des2, k=2):
        if m.distance < distance_ratio * n.distance:
            matches.append([m])

    if len(matches) < min_match_count:
        return None

    src_pts = []
    dst_pts = []
    for m in matches:
        src_pts.append(kp1[m[0].queryIdx].pt)
        dst_pts.append(kp2[m[0].trainIdx].pt)

    src_pts = np.float32(src_pts).reshape(-1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 2)

    model_type = AffineTransform
    if transform == 'projective':
        model_type = ProjectiveTransform

    # RANSAC
    model, inliers = ransac(
        (src_pts, dst_pts),
        model_type,
        min_samples=min_match_count,
        residual_threshold=ransac_threshold,
        max_trials=max_iterations
    )

    inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
    inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
    placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(np.sum(inliers))]

    matched_img = cv2.drawMatches(
        img1, inlier_keypoints_left,
        img2, inlier_keypoints_right,
        placeholder_matches,
        None,
        matchColor=(0, 0, 255)
    )

    return model.params, model.inverse.params, matched_img
