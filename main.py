import numpy as np
import trimesh, yaml, io, os, gc, cv2
from PIL import Image

from src.cameras import Agisoft
from src.utils import compute_homography


def setup_camera_scene(mesh, cameras_info, cam_T):
    camera = trimesh.scene.Camera(
        resolution=(
            cameras_info.height,
            cameras_info.width
        ),
        focal=(
            cameras_info.fx,
            cameras_info.fy
        )
    )

    # Create scene with proper camera transform
    scene = trimesh.Scene(
        geometry=mesh,
        camera=camera,
        camera_transform=cam_T
    )

    return camera, scene


def capture_scene(camera, scene):
    img_mesh = scene.save_image(resolution=camera.resolution, visible=False)
    img_mesh = np.array(Image.open(io.BytesIO(img_mesh)))
    return cv2.rotate(img_mesh, cv2.ROTATE_90_CLOCKWISE)


if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if not os.path.exists(config['output_folder']) or not os.path.isdir(config['output_folder']):
        os.mkdir(config['output_folder'])

    cameras_info = None
    if config['camera_format'] == 'agisoft':
        cameras_info = Agisoft()
        cameras_info.parse(config['cameras_path'])
    else:
        raise ValueError(f"Unknown camera format {config['camera_format']}.")

    mesh = trimesh.load_mesh(config['mesh_path'])
    ray_caster = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    H = None  # Initialize the homography matrix
    if config['perspective_correction']['enabled']:
        reference_label = config['perspective_correction']['reference_image']
        reference_label = reference_label.split('.')[-2].split('/')[-1]

        reference_index = cameras_info.labels.index(reference_label)
        reference_T = cameras_info.Ts[reference_index]

        camera, scene = setup_camera_scene(mesh, cameras_info, reference_T)

        img_mesh = capture_scene(camera, scene)
        img_orig = cv2.imread(config['perspective_correction']['reference_image'])

        del scene, camera
        gc.collect()

        H = compute_homography(
            img_mesh,
            img_orig,
            config['perspective_correction']['minimum_match_count'],
            config['perspective_correction']['match_count'],
            config['perspective_correction']['ransac_threshold'],
        )

    print("Loaded mesh. Will begin raytracing.")

    for i, (label, T) in enumerate(zip(cameras_info.labels, cameras_info.Ts)):
        # Create scene with proper camera transform
        camera, scene = setup_camera_scene(mesh, cameras_info, T)

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

        depth = np.astype(1000 * np.abs(depth), np.uint16)  # Convert to millimeters
        if config['perspective_correction']['enabled'] and H is not None:  # Correct perspective if enabled and possible
            depth = cv2.warpPerspective(depth, H, camera.resolution[::-1])

        img_file = os.path.join(config['output_folder'], f"{label}.png")
        cv2.imwrite(img_file, depth, (cv2.IMWRITE_PNG_COMPRESSION, 9))

        # Save scene image
        if config['save_scene']:
            img_mesh = capture_scene(camera, scene)
            if config['perspective_correction']['enabled'] and H is not None:
                img_mesh = cv2.warpPerspective(img_mesh, H, camera.resolution[::-1])

            img_file = os.path.join(config['output_folder'], f"{label}_scene.png")
            cv2.imwrite(img_file, img_mesh)

        print(f"Finished processing: {label}")

        # Clean up
        del scene, ray_origins, ray_vectors, ray_pixels, valid_rays, hits, depth
        gc.collect()
