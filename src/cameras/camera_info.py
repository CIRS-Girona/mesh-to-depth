class CameraInfo:
    def __init__(self):
        # Intrinsic Parameters
        self.fx, self.fy = None, None
        self.cx, self.cy = None, None
        self.width, self.height = None, None

        # Distortion Parameters
        self.p1, self.p2 = None, None
        self.k1, self.k2, self.k3 = None, None, None

        # World to camera transformation matrices and camera labels
        self.Ts = []
        self.labels = []

    def parse(self, file_path: str):
        raise NotImplementedError("This method needs to be implemented in the child class")