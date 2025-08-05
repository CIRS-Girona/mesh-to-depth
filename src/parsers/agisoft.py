import numpy as np
import xml.etree.ElementTree as ET
import trimesh.transformations
from typing import Dict, List

from .parser import Parser
from ..cameras import Sensor, Pose

TRIMESH_T_AGISOFT = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]) @\
    trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1])


class Agisoft(Parser):
    def parse(self, file_path: str) -> List[Sensor]:
        extension = file_path.split('.')[-1]
        if extension.lower() != 'xml':
            raise NameError(f"Invalid filename extension '{extension}', expected an XML file.")

        # Parse XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Parse chunk transform
        chunk = root.find('chunk')
        transform = chunk.find('transform')
        rotation = np.array([float(x) for x in transform.find('rotation').text.split()]).reshape(3, 3)
        translation = np.array([float(x) for x in transform.find('translation').text.split()])
        scale = float(transform.find('scale').text)

        # Build chunk transformation matrix
        m_T_c = np.eye(4)  # Camera to mesh
        m_T_c[:3, :3] = rotation * scale  # Transpose for proper rotation matrix
        m_T_c[:3, 3] = translation

        # Parse calibration parameters
        sensors: Dict[str, Sensor] = {}
        for sensor in chunk.find('sensors').findall('sensor'):
            s = Sensor()

            calibration = sensor.find('calibration')
            if calibration is None:
                continue

            s.width = int(calibration.find('resolution').attrib['width'])
            s.height = int(calibration.find('resolution').attrib['height'])

            s.fx = float(calibration.find('f').text)
            s.fy = s.fx

            s.fovx = 2 * np.arctan(s.width / (2 * s.fx))
            s.fovy = 2 * np.arctan(s.height / (2 * s.fy))

            s.cx = s.width / 2.0 + float(calibration.find('cx').text)
            s.cy = s.height / 2.0 + float(calibration.find('cy').text)

            s.p1 = float(calibration.find('p1').text)
            s.p2 = float(calibration.find('p2').text)

            s.k1 = float(calibration.find('k1').text)
            s.k2 = float(calibration.find('k2').text)
            s.k3 = float(calibration.find('k3').text)

            s.id = sensor.attrib['id']

            sensors[s.id] = s

        # Parse camera views
        cameras = []
        [cameras.extend(g.findall('camera')) for g in chunk.find('cameras').findall('group')]

        for cam in cameras:
            transform_elem = cam.find('transform')
            if transform_elem is None:
                continue

            c_T_trimesh = np.array([float(x) for x in transform_elem.text.split()]).reshape(4,4)

            pose = Pose()
            pose.T = m_T_c @ c_T_trimesh @ TRIMESH_T_AGISOFT
            pose.label = cam.attrib['label']

            sensors[cam.attrib['sensor_id']].poses.append(pose)

        return list(sensors.values())

