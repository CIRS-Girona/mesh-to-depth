import numpy as np
import trimesh, io, gc, cv2
from PIL import Image

from trimesh.ray.ray_pyembree import RayMeshIntersector

from src.cameras import Sensor, Pose


def setup_camera_scene(mesh, sensor: Sensor, pose: Pose):
    # Construct camera with parameters specified
    camera = trimesh.scene.Camera(
        resolution=(
            sensor.height + sensor.padding,
            sensor.width + sensor.padding
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
        correct_perspective: bool = False,
        capture_mesh: bool = False
):
    # Create scene with proper camera transform
    camera, scene = setup_camera_scene(
        ray_caster.mesh,
        sensor,
        pose
    )

    # Generate rays and calculate intersections
    ray_origins, ray_vectors, ray_pixels = scene.camera_rays()
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

    depth = np.astype(1000 * scale * np.abs(depth), np.uint16)  # Scale to correct units and then convert to millimeters
    if distort:
        depth = cv2.remap(depth, sensor.map_x, sensor.map_y, interpolation=cv2.INTER_LINEAR)

    if correct_perspective and sensor.H is not None:  # Correct perspective if enabled and possible
        depth = cv2.warpPerspective(depth, sensor.H, camera.resolution[::-1])
        depth = sensor.correct_perspective(depth)

    # Apply an exponential scale to the heatmap to amplify variations
    heatmap = np.power(10, np.astype(depth, np.float32) / (np.max(depth) + 1)) - 1
    if np.any(heatmap != 0):
        heatmap[heatmap == 0] = np.min(heatmap[heatmap != 0]) - 1
        heatmap -= np.min(heatmap)
        heatmap /= np.maximum(np.max(heatmap), 1)

    heatmap = cv2.applyColorMap(np.astype(255 * heatmap, np.uint8), cv2.COLORMAP_INFERNO)

    img_mesh = None
    if capture_mesh:
        img_mesh = capture_scene(camera, scene)
        if distort:
            img_mesh = cv2.remap(img_mesh, sensor.map_x, sensor.map_y, interpolation=cv2.INTER_LINEAR)

        if correct_perspective and sensor.H is not None:
            img_mesh = cv2.warpPerspective(img_mesh, sensor.H, camera.resolution[::-1])
            img_mesh = sensor.correct_perspective(img_mesh)

    # Memory cleanup
    del camera, scene, ray_origins, ray_vectors, ray_pixels, valid_rays, hits
    gc.collect()

    return depth, heatmap, img_mesh