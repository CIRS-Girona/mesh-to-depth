import numpy as np
import yaml, os, cv2
import open3d as o3d
from tqdm import tqdm
from typing import List, Tuple

from scipy.spatial.transform import Rotation as R

from src.utils import raytrace
from src.parsers import Agisoft, Meshroom
from src.cameras import Sensor, Pose


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

    views_to_process = []
    if not config['manual_view']['enabled']:
        print("Filtering completed views...")
        completed = set(os.listdir(config['output_folder']))
        views_to_process = [
            (s, p) for (s, p) in views
            if f"{p.label}.png" not in completed
        ]

        if not views_to_process:
            print("All views have already been processed.")
            exit(0)

    # Load the mesh and build the BVH tree ONCE
    print("Loading mesh...")
    main_mesh = o3d.t.io.read_triangle_mesh(config['mesh_path'])
    RAY_CASTER = o3d.t.geometry.RaycastingScene()
    RAY_CASTER.add_triangles(main_mesh)

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
            min_bound = main_mesh.vertex.positions.min(0).numpy()
            max_bound = main_mesh.vertex.positions.max(0).numpy()
            center = (max_bound - min_bound) / 2.0 + min_bound

        pose = Pose()
        pose.T = R.from_quat(config['manual_view']['orientation']).as_matrix()
        pose.T[:3, 3] = center + config['manual_view']['position']

        depth, heatmap = raytrace(
            RAY_CASTER,
            sensor,
            pose,
            scale=scale,
        )

        img_file = os.path.join(config['output_folder'], "depth.png")
        cv2.imwrite(img_file, depth, (cv2.IMWRITE_PNG_COMPRESSION, 9))

        img_file = os.path.join(config['output_folder'], "heatmap.jpg")
        cv2.imwrite(img_file, heatmap)

        exit(0)

    if config['distortion']['enabled']:
        print("Computing distortion mappings...")
        for sensor in sensors:
            sensor.compute_distortion_maps(
                max_iter=config['distortion']['max_iterations'],
                tol=config['distortion']['tolerance'],
                eta=config['distortion']['damping']
            )

    print(f"Processing {len(views_to_process)} views...")
    for view_data in tqdm(views_to_process, desc="Raytracing"):
        sensor, pose = view_data
        
        try:
            depth, heatmap = raytrace(
                RAY_CASTER,
                sensor,
                pose,
                scale=scale
            )

            img_file_depth = os.path.join(config['output_folder'], f"{pose.label}.png")
            cv2.imwrite(img_file_depth, depth)

            img_file_heatmap = os.path.join(config['output_folder'], f"{pose.label}_heatmap.jpg")
            cv2.imwrite(img_file_heatmap, heatmap)

        except Exception as exc:
            print(f"\nView '{pose.label}' generated an exception: {exc}")
