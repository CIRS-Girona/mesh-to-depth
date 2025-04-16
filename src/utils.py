import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform


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

    return model.params, matched_img


