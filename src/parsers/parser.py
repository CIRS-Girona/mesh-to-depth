import numpy as np
from typing import List

from ..cameras import Sensor


def rotation_matrix_3d(angle: float, axis: tuple) -> np.ndarray:
    """
    Returns a 4x4 homogenous rotation matrix.
    Replaces trimesh.transformations.rotation_matrix.
    """
    axis = np.asarray(axis) / np.linalg.norm(axis)
    s = np.sin(angle)
    c = np.cos(angle)
    t = 1 - c
    x, y, z = axis
    
    # Rotation matrix using Rodrigues' rotation formula logic
    R = np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
    ])
    
    T = np.eye(4)
    T[:3, :3] = R
    return T


class Parser:
    def __init__(self) -> None:
        pass

    def parse(self, file_path: str) -> List[Sensor]:
        raise NotImplementedError("This method must be implemented in the child class.")