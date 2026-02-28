import numpy as np
import json
from typing import Dict, List

from .parser import Parser, rotation_matrix_3d
from ..cameras import Sensor, Pose

MESH_T_MESHROOM_PRE = rotation_matrix_3d(np.pi, (1, 0, 0))
MESH_T_MESHROOM_POST = rotation_matrix_3d(np.pi, (1, 0, 0)) @ rotation_matrix_3d(-np.pi / 2, (0, 0, 1))


class Meshroom(Parser):
    def extractTransformation(self, transform: Dict[str, List[str]]) -> np.ndarray:
        rotation = np.array([float(x) for x in transform['rotation']]).reshape((3, 3))
        translation = np.array([float(x) for x in transform['center']]).reshape((3, ))

        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = translation

        return T

    def parse(self, file_path: str) -> List[Sensor]:
        extension = file_path.split('.')[-1]
        if extension.lower() != 'sfm':
            raise NameError(f"Invalid filename extension '{extension}', expected an SFM file.")

        data = None
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Parse calibration parameters
        sensors: Dict[str, Sensor] = {}
        for sensor in data['intrinsics']:
            width = int(sensor['width'])
            height = int(sensor['height'])

            s = Sensor(width, height)

            focal_length = float(sensor['focalLength'])  # mm

            px = float(sensor['sensorWidth']) / s.width  # mm / pixels
            py = float(sensor['sensorHeight']) / s.height # mm / pixels

            s.fx = focal_length / px  # Focal length in pixels
            s.fy = focal_length / py  # Focal length in pixels

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

            s.id = sensor['intrinsicId']

            sensors[s.id] = s

        # Parse camera views
        rigs = {}
        if data.get('rigs', None) is not None:
            rigs = {
                rig['rigId']: rig['subPoses']
                for rig in data['rigs']
            }

        poses = {
            pose['poseId']: pose['pose']['transform']
            for pose in data['poses']
        }

        for view in data['views']:
            if poses.get(view['poseId'], None) is None:
                continue

            c_T_m = self.extractTransformation(poses[view['poseId']])

            m_T_r = np.eye(4)
            if view.get('rigId', None) is not None:
                m_T_r = self.extractTransformation(rigs[view['rigId']][int(view['subPoseId'])]['pose'])
                m_T_r = m_T_r

            pose = Pose()
            pose.T = MESH_T_MESHROOM_PRE @ c_T_m @ m_T_r @ MESH_T_MESHROOM_POST
            pose.label = "".join(view['path'].split("/")[-1].split('.')[:-1])

            sensors[view['intrinsicId']].poses.append(pose)

        return list(sensors.values())

