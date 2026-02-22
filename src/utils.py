import numpy as np
import trimesh, io, cv2
from PIL import Image

from trimesh.ray.ray_pyembree import RayMeshIntersector

from src.cameras import Sensor, Pose


def setup_camera_scene(mesh, sensor: Sensor, pose: Pose):
    # Construct camera with parameters specified
    camera = trimesh.scene.Camera(
        resolution=(
            sensor.height,
            sensor.width
        ),
        focal=(
            sensor.fx,
            sensor.fy,
        )
    )

    # Create scene with proper camera transform
    scene = trimesh.Scene(
        geometry=mesh,
        camera=camera,
        camera_transform=pose.T
    )

    return camera, scene


def capture_scene(camera, scene):
    img_mesh = scene.save_image(resolution=camera.resolution, visible=True)
    img_mesh = np.array(Image.open(io.BytesIO(img_mesh)))
    return cv2.cvtColor(cv2.rotate(img_mesh, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_RGB2BGR)


def raytrace(
        ray_caster: RayMeshIntersector,
        sensor: Sensor,
        pose: Pose,
        scale: float = 1.0,
        distort: bool = False,
):
    # Create scene with proper camera transform
    camera, scene = setup_camera_scene(
        ray_caster.mesh,
        sensor,
        pose
    )

    # Generate rays and calculate intersections
    ray_origins, ray_vectors, ray_pixels = scene.camera_rays()

    # Distort ray vectors so that depth map is pixel accurate with original image
    if distort and sensor.x is not None and sensor.y is not None:
        # Trimesh convention is (y, x) for pixel coordinates, so swap the order when indexing
        x_cam = sensor.y[ray_pixels[:, 0], ray_pixels[:, 1]]
        y_cam = sensor.x[ray_pixels[:, 0], ray_pixels[:, 1]]
        z_cam = np.full_like(x_cam, -1.0)

        # Normalize the vectors to length 1
        ray_vectors = np.stack((x_cam, y_cam, z_cam), axis=-1)
        ray_vectors /= np.linalg.norm(ray_vectors, axis=1, keepdims=True)

        # Rotate the local vectors into world space
        R = pose.T[:3, :3]
        ray_vectors = (R @ ray_vectors.T).T

    valid_rays = ray_caster.intersects_any(ray_origins, ray_vectors)

    # Find intersections for valid rays
    hits = ray_caster.intersects_location(
        ray_origins[valid_rays],
        ray_vectors[valid_rays],
        multiple_hits=False
    )

    # Create depth map
    depth = np.full(camera.resolution, 0, dtype=np.float32)
    if hits:
        positions, pixels = hits[0], hits[1]
        depth_coords = ray_pixels[valid_rays][pixels]
        depth[depth_coords[:, 0], depth_coords[:, 1]] = positions[:, 2]

    depth = (1000 * scale * np.abs(depth)).astype(np.uint16)  # Scale to correct units and then convert to millimeters

    # Apply an exponential scale to the heatmap to amplify variations
    heatmap = np.power(10, depth.astype(np.float32) / (np.max(depth) + 1)) - 1
    if np.any(heatmap != 0):
        heatmap[heatmap == 0] = np.min(heatmap[heatmap != 0]) - 1
        heatmap -= np.min(heatmap)
        heatmap /= np.maximum(np.max(heatmap), 1)

    heatmap = cv2.applyColorMap((255 * heatmap).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return depth, heatmap