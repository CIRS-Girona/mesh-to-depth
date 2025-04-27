import numpy as np
import trimesh, yaml, io, os, gc, cv2
from PIL import Image

from src.cameras import Agisoft
from src.utils import compute_distortion_maps, compute_homography


def setup_camera_scene(mesh, cameras_info, cam_T, padding=0):
    # Construct camera with parameters specified
    camera = trimesh.scene.Camera(
        resolution=(
            cameras_info.height + padding,
            cameras_info.width + padding
        ),
        focal=(
            cameras_info.fx,
            cameras_info.fy,
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
    img_mesh = scene.save_image(resolution=camera.resolution, visible=True)
    img_mesh = np.array(Image.open(io.BytesIO(img_mesh)))
    return cv2.rotate(img_mesh, cv2.ROTATE_90_CLOCKWISE)


def adjust_warping(image, H, H_inv, width, height):
    # Get the position of the top-left pixel in the padded image
    top_left = H_inv @ np.array((0, 0, 1), dtype=np.float32)
    top_left /= top_left[2]

    # Get the position of the top-left pixel in the warped image
    top_left_warped = H @ top_left
    top_left_warped /= top_left_warped[2]

    # Crop image to desired resolution
    start_y = int(round(top_left_warped[1]))
    start_x = int(round(top_left_warped[0]))
    end_y = start_y + height
    end_x = start_x + width

    return image[start_y:end_y, start_x: end_x]


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

    pixel_padding = config['perspective_correction']['padding'] if config['perspective_correction']['enabled'] else 0

    if config['apply_distortion']:
        map_x, map_y = compute_distortion_maps(
            height=cameras_info.height + pixel_padding,
            width=cameras_info.width + pixel_padding,
            cameras_info=cameras_info
        )

    mesh = trimesh.load_mesh(config['mesh_path'])
    ray_caster = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    H = None  # Initialize the homography matrix
    if config['perspective_correction']['enabled']:
        reference_label = config['perspective_correction']['reference_image']
        reference_label = reference_label.split('.')[-2].split('/')[-1]

        reference_index = cameras_info.labels.index(reference_label)
        reference_T = cameras_info.Ts[reference_index]

        camera, scene = setup_camera_scene(
            mesh,
            cameras_info,
            reference_T,
            padding=pixel_padding
        )

        img_mesh = capture_scene(camera, scene)
        if config['apply_distortion']:
            img_mesh = cv2.remap(img_mesh, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        img_orig = cv2.imread(config['perspective_correction']['reference_image'])

        del scene, camera
        gc.collect()

        H, H_inv, matched_img = compute_homography(
            img_mesh,
            img_orig,
            transform=config['perspective_correction']['transform'],
            min_match_count=config['perspective_correction']['minimum_match_count'],
            distance_ratio=config['perspective_correction']['distance_ratio'],
            ransac_threshold=config['perspective_correction']['ransac_threshold'],
            max_iterations=config['perspective_correction']['max_iterations'],
        )

        img_file = os.path.join(config['output_folder'], "matches.png")
        cv2.imwrite(img_file, matched_img)

    print("Loaded mesh. Will begin raytracing.")

    for i, (label, T) in enumerate(zip(cameras_info.labels, cameras_info.Ts)):
        # Create scene with proper camera transform
        camera, scene = setup_camera_scene(
            mesh,
            cameras_info,
            T,
            padding=pixel_padding
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

        depth = np.astype(1000 * np.abs(depth), np.uint16)  # Convert to millimeters
        if config['apply_distortion']:
            depth = cv2.remap(depth, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        if config['perspective_correction']['enabled'] and H is not None:  # Correct perspective if enabled and possible
            depth = cv2.warpPerspective(depth, H, camera.resolution[::-1])
            depth = adjust_warping(depth, H, H_inv, cameras_info.width, cameras_info.height)

        img_file = os.path.join(config['output_folder'], f"{label}.png")
        cv2.imwrite(img_file, depth, (cv2.IMWRITE_PNG_COMPRESSION, 9))

        # Save scene image
        if config['save_scene']:
            img_mesh = capture_scene(camera, scene)
            if config['apply_distortion']:
                img_mesh = cv2.remap(img_mesh, map_x, map_y, interpolation=cv2.INTER_LINEAR)

            if config['perspective_correction']['enabled'] and H is not None:
                img_mesh = cv2.warpPerspective(img_mesh, H, camera.resolution[::-1])
                img_mesh = adjust_warping(img_mesh, H, H_inv, cameras_info.width, cameras_info.height)

            img_file = os.path.join(config['output_folder'], f"{label}_scene.png")
            cv2.imwrite(img_file, img_mesh)

        print(f"Finished processing: {label}")

        # Clean up
        del scene, ray_origins, ray_vectors, ray_pixels, valid_rays, hits, depth
        gc.collect()
