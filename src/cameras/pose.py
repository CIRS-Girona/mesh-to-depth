import numpy as np


class Pose:
    def __init__(self):
        self.T: np.ndarray = None  # World to camera pose transformation
        self.label: str = None     # File name without extension
