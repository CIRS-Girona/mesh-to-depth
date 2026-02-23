import numpy as np
import trimesh, yaml, os, cv2, gc
import multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from trimesh.transformations import quaternion_matrix
from trimesh.ray.ray_pyembree import RayMeshIntersector

from src.utils import raytrace
from src.parsers import Agisoft, Meshroom
from src.cameras import Sensor, Pose

# Disable upper limit for image pixels in Pillow library
Image.MAX_IMAGE_PIXELS = None

# Prevent OpenCV from deadlocking when combined with os.fork()
cv2.setNumThreads(0)

# This will be initialized in the main process and inherited by the workers
RAY_CASTER = None


def process_view_forked(view_data: Tuple[Sensor, Pose], output_folder: str, scale: float, distort: bool):
    """
    Worker function. Thanks to 'fork', it can freely read global_ray_caster 
    without duplicating the memory or pickling C-pointers.
    """
    sensor, pose = view_data
    
    # Access the shared C-level raycaster
    depth, heatmap = raytrace(
        RAY_CASTER,
        sensor,
        pose,
        scale=scale,
        distort=distort,
    )

    img_file_depth = os.path.join(output_folder, f"{pose.label}.png")
    cv2.imwrite(img_file_depth, depth, (cv2.IMWRITE_PNG_COMPRESSION, 9))

    img_file_heatmap = os.path.join(output_folder, f"{pose.label}_heatmap.jpg")
    cv2.imwrite(img_file_heatmap, heatmap)

    return pose.label


if __name__ == "__main__":
    # Force the fork method immediately
    mp.set_start_method('fork', force=True)

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

    # Load the mesh and build the BVH tree ONCE
    print("Loading mesh and building BVH tree in main memory...")
    main_mesh = trimesh.load_mesh(config['mesh_path'])
    RAY_CASTER = RayMeshIntersector(main_mesh)

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
            center = (main_mesh.bounds[1, :] - main_mesh.bounds[0, :]) / 2 + main_mesh.bounds[0, :]

        pose = Pose()
        pose.T = quaternion_matrix(config['manual_view']['orientation'])
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

    print("Filtering completed views...")
    completed = set(os.listdir(config['output_folder']))
    views_to_process = [
        (s, p) for (s, p) in views 
        if f"{p.label}.png" not in completed
    ]

    if not views_to_process:
        print("All views have already been processed.")
        exit(0)

    # Free the main process's reference to the mesh and BVH tree, allowing them to be shared via COW
    gc.freeze()

    max_workers = max(1, mp.cpu_count() - 1)
    print(f"Forking {max_workers} processes. Mesh memory will be shared via COW.")

    # Explicitly use the fork context for the ProcessPoolExecutor
    fork_context = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=fork_context) as executor:
        futures = {
            executor.submit(
                process_view_forked,
                view_data,
                config['output_folder'],
                scale,
                config['distortion']['enabled']
            ): view_data for view_data in views_to_process
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Raytracing"):
            try:
                processed_label = future.result() 
            except Exception as exc:
                print(f"\nView generated an exception: {exc}")