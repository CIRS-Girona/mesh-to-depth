from typing import List

from ..cameras import Sensor, Pose


class Parser:
    def __init__(self):
        pass

    def parse(self, file_path: str) -> List[Sensor]:
        raise NotImplementedError("This method must be implemented in the child class.")