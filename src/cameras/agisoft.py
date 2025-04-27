import numpy as np
import xml.etree.ElementTree as ET
import trimesh.transformations

from .camera_info import CameraInfo


AGISOFT_T = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]) @\
    trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1])


class Agisoft(CameraInfo):
    def parse(self, file_path):
        self.Ts.clear()
        self.labels.clear()

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
        mesh_transform = np.eye(4)
        mesh_transform[:3, :3] = rotation * scale  # Transpose for proper rotation matrix
        mesh_transform[:3, 3] = translation

        # Parse cameras
        cameras = []
        [cameras.extend(g.findall('camera')) for g in chunk.find('cameras').findall('group')]

        for cam in cameras:
            transform_elem = cam.find('transform')

            if transform_elem is None:
                continue

            transform = np.array([float(x) for x in transform_elem.text.split()]).reshape(4,4)
            self.Ts.append(mesh_transform @ transform @ AGISOFT_T)

            self.labels.append(cam.attrib['label'])

        # Parse calibration parameters
        sensor = chunk.find('sensors').find('sensor')
        calibration = sensor.find('calibration')

        self.width = int(calibration.find('resolution').attrib['width'])
        self.height = int(calibration.find('resolution').attrib['height'])

        self.fx = float(calibration.find('f').text)
        self.fy = self.fx

        self.fovx = 2 * np.arctan(self.width / (2 * self.fx))
        self.fovy = 2 * np.arctan(self.height / (2 * self.fy))

        self.cx = self.width / 2.0 + float(calibration.find('cx').text)
        self.cy = self.height / 2.0 + float(calibration.find('cy').text)

        self.p1 = float(calibration.find('p1').text)
        self.p2 = float(calibration.find('p2').text)

        self.k1 = float(calibration.find('k1').text)
        self.k2 = float(calibration.find('k2').text)
        self.k3 = float(calibration.find('k3').text)
