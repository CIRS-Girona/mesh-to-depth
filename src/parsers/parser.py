from typing import List

from ..cameras import Sensor


class Parser:
    def __init__(self) -> None:
        pass

    def parse(self, file_path: str) -> List[Sensor]:
        raise NotImplementedError("This method must be implemented in the child class.")