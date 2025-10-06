import numpy as np
import trimesh, yaml, os, gc, cv2, random
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple

from trimesh.transformations import quaternion_matrix
from trimesh.ray.ray_pyembree import RayMeshIntersector

from src.utils import raytrace
from src.parsers import Agisoft, Meshroom
from src.cameras import Sensor, Pose

# Disable upper limit for image pixels in Pillow library (Important for loading large texture maps)
Image.MAX_IMAGE_PIXELS = None


if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if not os.path.exists(config['output_folder']) or not os.path.isdir(config['output_folder']):
        os.mkdir(config['output_folder'])

    sensors = None
    if config['camera_format'] == 'agisoft':
        sensors = Agisoft().parse(config['cameras_path'])
    elif config['camera_format'] == 'meshroom':
        sensors = Meshroom().parse(config['cameras_path'])
    else:
        raise ValueError(f"Unknown camera info format {config['camera_format']}.")

    views: List[Tuple[Sensor, Pose]] = []
    [views.extend([(s, p) for p in s.poses]) for s in sensors]

    print("Parsed camera info successfully")

    print("Loading mesh...")
    mesh = trimesh.load_mesh(config['mesh_path'])
    ray_caster = RayMeshIntersector(mesh)

    scale = 1.0
    if config['scale_mesh']['enabled']:
        pose_1, pose_2 = None, None

        for (sensor, pose) in views:
            if config['scale_mesh']['pose_1'] == pose.label:
                pose_1 = pose
            elif config['scale_mesh']['pose_2'] == pose.label:
                pose_2 = pose
            elif pose_1 is not None and pose_2 is not None:
                break

        if pose_1 is None:
            raise LookupError(f"Pose couldn't be found: {config['scale_mesh']['pose_1']}")
        elif pose_2 is None:
            raise LookupError(f"Pose couldn't be found: {config['scale_mesh']['pose_2']}")

        scale = config['scale_mesh']['distance'] / np.linalg.norm(pose_1.T[:3, 3] - pose_2.T[:3, 3])

    if config['manual_view']['enabled']:
        print("Computing manual view...")

        sensor = Sensor()

        sensor.width = config['manual_view']['width']
        sensor.height = config['manual_view']['height']

        sensor.fx = config['manual_view']['fx']
        sensor.fy = config['manual_view']['fy']
     
        center = np.array((0, 0, 0))
        if config['manual_view']['use_center']:
            # Get mesh center and apply position relative to center
            center = (mesh.bounds[1, :] - mesh.bounds[0, :]) / 2 + mesh.bounds[0, :]

        pose = Pose()
        pose.T = quaternion_matrix(config['manual_view']['orientation'])
        pose.T[:3, 3] = center + config['manual_view']['position']

        depth, heatmap, img_mesh = raytrace(
            ray_caster,
            sensor,
            pose,
            scale=scale,
            capture_mesh=True
        )

        img_file = os.path.join(config['output_folder'], "depth.png")
        cv2.imwrite(img_file, depth, (cv2.IMWRITE_PNG_COMPRESSION, 9))

        img_file = os.path.join(config['output_folder'], "heatmap.jpg")
        cv2.imwrite(img_file, heatmap)

        img_file = os.path.join(config['output_folder'], "scene.jpg")
        cv2.imwrite(img_file, img_mesh)

        exit()

    if config['distortion']['enabled']:
        print("Computing distortion mappings...")

        for sensor in sensors:
            sensor.padding = config['perspective_correction']['padding'] if config['perspective_correction']['enabled'] else 0

            sensor.compute_distortion_maps(
                max_iter=config['distortion']['max_iterations'],
                tol=config['distortion']['tolerance'],
                eta=config['distortion']['damping']
            )

    if config['perspective_correction']['enabled']:
        print("Computing homography matrix...")

        clahe = cv2.createCLAHE(
            clipLimit=config['perspective_correction']['clahe_limit'],
            tileGridSize=(config['perspective_correction']['clahe_grid'], config['perspective_correction']['clahe_grid'])
        )

        for sensor in sensors:
            pose = random.choice(sensor.poses)  # Pick a random pose for feature matching

            img_file = None
            for ext in config['extensions']:
                img_path = os.path.join(config['images_path'], f"{pose.label}.{ext}")
                if os.path.exists(img_path):
                    img_file = img_path
                    break

            if img_file is None:
                raise FileNotFoundError(f"Couldn't find {pose.label} for any of the given extensions: {config['extensions']}")

            print(f"For camera {sensor.id}, {pose.label} will be used for feature matching.")

            img_orig = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2HSV)
            img_orig[:, :, 2] = clahe.apply(img_orig[:, :, 2])
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_HSV2BGR)

            _, _, img_mesh = raytrace(
                ray_caster,
                sensor,
                pose,
                distort=config['distortion']['enabled'],
                capture_mesh=True
            )

            img_mesh = cv2.cvtColor(img_mesh, cv2.COLOR_BGR2HSV)
            img_mesh[:, :, 2] = clahe.apply(img_mesh[:, :, 2])
            img_mesh = cv2.cvtColor(img_mesh, cv2.COLOR_HSV2BGR)

            matched_img = sensor.compute_homography(
                img_mesh,
                img_orig,
                model=config['perspective_correction']['model'],
                min_match_count=config['perspective_correction']['minimum_match_count'],
                distance_ratio=config['perspective_correction']['distance_ratio'],
                ransac_threshold=config['perspective_correction']['ransac_threshold'],
                max_iterations=config['perspective_correction']['max_iterations'],
            )

            img_file = os.path.join(config['output_folder'], f"matches_{sensor.id}.jpg")
            cv2.imwrite(img_file, matched_img)

    print("Will begin raytracing.")

    completed = os.listdir(config['output_folder'])
    for i, (sensor, pose) in enumerate(tqdm(views)):
        if f"{pose.label}.png" in completed:  # Skip already processed poses
            continue

        depth, heatmap, img_mesh = raytrace(
            ray_caster,
            sensor,
            pose,
            scale=scale,
            distort=config['distortion']['enabled'],
            correct_perspective=config['perspective_correction']['enabled'],
            capture_mesh=config['save_scene']
        )

        img_file = os.path.join(config['output_folder'], f"{pose.label}.png")
        cv2.imwrite(img_file, depth, (cv2.IMWRITE_PNG_COMPRESSION, 9))

        img_file = os.path.join(config['output_folder'], f"{pose.label}_heatmap.jpg")
        cv2.imwrite(img_file, heatmap)

        # Save scene image
        if img_mesh is not None:
            img_file = os.path.join(config['output_folder'], f"{pose.label}_scene.jpg")
            cv2.imwrite(img_file, img_mesh)

        # Clean up
        del img_mesh, depth
        gc.collect()
