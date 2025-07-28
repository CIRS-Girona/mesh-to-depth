import numpy as np
import json
import trimesh.transformations
from typing import Dict, List

from .parser import Parser
from ..cameras import Sensor, Pose

TRIMESH_T_MESHROOM = np.eye(4)


class Meshroom(Parser):
    def parse(self, file_path: str) -> List[Sensor]:
        data = None
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Parse calibration parameters
        sensors: Dict[str, Sensor] = {}
        for sensor in data['intrinsics']:
            s = Sensor()

            s.width = int(sensor['width'])
            s.height = int(sensor['height'])

            s.fx = float(sensor['focalLength'])
            s.fy = s.fx

            s.fovx = 2 * np.arctan(s.width / (2 * s.fx))
            s.fovy = 2 * np.arctan(s.height / (2 * s.fy))

            s.cx = s.width / 2.0 + float(sensor['principalPoint'][0])
            s.cy = s.height / 2.0 + float(sensor['principalPoint'][1])

            if len(sensor['distortionParams']) < 5:
                raise ValueError("Distortion provided doesn't follow Brown-Conrady model.")

            s.k1 = float(sensor['distortionParams'][0])
            s.k2 = float(sensor['distortionParams'][1])
            s.k3 = float(sensor['distortionParams'][2])

            s.p1 = float(sensor['distortionParams'][3])
            s.p2 = float(sensor['distortionParams'][4])

            sensors[sensor['intrinsicId']] = s

        # Parse camera views
        views = {
            views[view['poseId']]: (view['intrinsicId'], view['path'])
            for view in data['views']
        }

        for camera in data['poses']:
            transform = camera['pose']['transform']
            rotation = np.array([float(x) for x in transform['rotation']]).reshape((3, 3))
            translation = np.array([float(x) for x in transform['rotation']]).reshape((3, 1))

            c_T_m = np.eye(4)
            c_T_m[:3, :3] = rotation
            c_T_m[:3, 3] = translation

            pose = Pose()
            pose.T = c_T_m @ TRIMESH_T_MESHROOM
            pose.label = "".join(views[camera['poseId']][1].split("/")[-1].split('.')[:-1])

            sensors[views[camera['poseId']][0]].poses.append(pose)

        return list(sensors.values())

