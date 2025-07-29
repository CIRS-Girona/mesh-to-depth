import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
from typing import List

from .pose import Pose


class Sensor:
    def __init__(self, padding: int = 0):
        self.id: str = None
        self.poses: List[Pose] = []

        # Intrinsic Parameters
        self.fx: float = None    # Focal length X-axis
        self.fy: float = None    # Focal length Y-axis

        self.cx: float = None    # Principal point X-axis
        self.cy: float = None    # Principal point Y-axis

        self.fovx: float = None  # Field of View X-axis (radians)
        self.fovy: float = None  # Field of View Y-axis (radians)

        self.width: int = None   # Resolution width
        self.height: int = None  # Resolution height

        # Distortion Parameters (Brown-Conrady)
        self.k1: float = None  # 1st Radial coefficient
        self.k2: float = None  # 2nd Radial coefficient
        self.k3: float = None  # 3rd Radial coefficient

        self.p1: float = None  # 1st Tangential coefficient
        self.p2: float = None  # 2nd Tangential coefficient

        # Distortion Mappings
        self.map_x: np.ndarray = None
        self.map_y: np.ndarray = None

        # Perspective Correction
        self.padding: int = padding

        self.H: np.ndarray = None
        self.H_inv: np.ndarray = None

    def compute_distortion_maps(self, max_iter: int = 1000, tol: float = 1e-3, eta: float = 0.1, dtype=np.float32) -> None:
        """
        Compute mapping from distorted pixels to undistorted coordinates.
        :return: map_x, map_y for cv2.remap()

        Source: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        """
        cx = self.cx + self.padding // 2
        cy = self.cy + self.padding // 2

        u_d, v_d = np.meshgrid(
            np.arange(self.width + self.padding, dtype=dtype),
            np.arange(self.height + self.padding, dtype=dtype)
        )
        
        # Normalize coordinates (distorted)
        x_prime = (u_d - cx) / self.fx
        y_prime = (v_d - cy) / self.fy

        # Iteratively solve for undistorted (x, y)
        x, y = x_prime.copy(), y_prime.copy()
        for _ in range(max_iter):
            r2 = x**2 + y**2
            radial = 1 + self.k1*r2 + self.k2*r2**2 + self.k3*r2**3

            xd = x * radial + 2*self.p1*x*y + self.p2*(r2 + 2*x**2)
            yd = y * radial + self.p1*(r2 + 2*y**2) + 2*self.p2*x*y

            x_new = x - eta * (xd - x_prime)
            y_new = y - eta * (yd - y_prime)

            if np.linalg.norm((x - x_new, y - y_new)) <= tol:
                break

            x, y = x_new, y_new

        # Convert back to pixel coordinates
        self.map_x = (x * self.fx + cx).astype(dtype)
        self.map_y = (y * self.fy + cy).astype(dtype)

    def correct_perspective(self, image: np.ndarray) -> np.ndarray:
        # Get the position of the top-left pixel in the padded image
        top_left = self.H_inv @ np.array((0, 0, 1), dtype=np.float32)
        top_left /= top_left[2]

        # Get the position of the top-left pixel in the warped image
        top_left_warped = self.H @ top_left
        top_left_warped /= top_left_warped[2]

        # Crop image to desired resolution
        start_y = int(round(top_left_warped[1]))
        start_x = int(round(top_left_warped[0]))
        end_y = start_y + self.height
        end_x = start_x + self.width

        return image[start_y:end_y, start_x: end_x]

    def compute_homography(self, img1: np.ndarray, img2: np.ndarray, model: str = 'affine', min_match_count: int = 10, distance_ratio: float = 0.70, ransac_threshold: float = 5.0, max_iterations: int = 10000) -> np.ndarray:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Initiate SIFT detector and Brute Force Matcher
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
        if model == 'projective':
            model_type = ProjectiveTransform

        # RANSAC
        model, inliers = ransac(
            (src_pts, dst_pts),
            model_type,
            min_samples=min_match_count,
            residual_threshold=ransac_threshold,
            max_trials=max_iterations
        )

        if model is None:
            raise RuntimeError("Failed to compute the homography matrix.")

        self.H = model.params
        self.H_inv = model.inverse.params

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

        return matched_img