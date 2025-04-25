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

        self.cx = float(calibration.find('cx').text)
        self.cy = float(calibration.find('cy').text)

        self.p1 = float(calibration.find('p1').text)
        self.p2 = float(calibration.find('p2').text)

        self.k1 = float(calibration.find('k1').text)
        self.k2 = float(calibration.find('k2').text)
        self.k3 = float(calibration.find('k3').text)

        # Generate grid of (u, v) coordinates for the distorted image
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        u = u.astype(np.float32)
        v = v.astype(np.float32)

        # Convert to normalized coordinates
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy

        # Apply distortion
        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r4 * r2
        x_dist = x * (1 + self.k1*r2 + self.k2*r4 + self.k3*r6) + 2*self.p1*x*y + self.p2*(r2 + 2*x**2)
        y_dist = y * (1 + self.k1*r2 + self.k2*r4 + self.k3*r6) + self.p1*(r2 + 2*y**2) + 2*self.p2*x*y

        # Convert back to pixel coordinates
        u_undist = x_dist * self.fx + self.cx
        v_undist = y_dist * self.fy + self.cy

        # Create the maps for remap distortion
        self.mapx = u_undist.astype(np.float32)
        self.mapy = v_undist.astype(np.float32)
