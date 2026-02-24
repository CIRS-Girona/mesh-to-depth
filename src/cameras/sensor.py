import numpy as np
from typing import List

from .pose import Pose


class Sensor:
    def __init__(self) -> None:
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

        # Coordinate grids for ray tracing (undistorted)
        self.u, self.v = np.meshgrid(
            np.arange(self.width, dtype=np.float32),
            np.arange(self.height, dtype=np.float32)
        )

        self.x = (self.u - self.cx) / self.fx
        self.y = (self.v - self.cy) / self.fy


    def compute_distortion_maps(self, max_iter: int = 100, tol: float = 1e-3, eta: float = 1.0) -> None:
        # Iteratively solve for distorted (x, y)
        x, y = self.x.copy(), self.y.copy()
        for _ in range(max_iter):
            r2 = x**2 + y**2
            radial = 1 + self.k1*r2 + self.k2*r2**2 + self.k3*r2**3

            xd = x * radial + 2*self.p1*x*y + self.p2*(r2 + 2*x**2)
            yd = y * radial + self.p1*(r2 + 2*y**2) + 2*self.p2*x*y

            x_new = x - eta * (xd - self.x)
            y_new = y - eta * (yd - self.y)

            if np.linalg.norm((x - x_new, y - y_new)) <= tol:
                break

            x, y = x_new, y_new

        # Store the final distorted coordinates
        self.x, self.y = x, y

        # Convert back to pixel coordinates (to be used in remapping)
        self.u = (x * self.fx + self.cx)
        self.v = (y * self.fy + self.cy)
