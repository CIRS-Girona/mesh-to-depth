import numpy as np
import cv2
import open3d as o3d

from src.cameras import Sensor, Pose


def raytrace(
        ray_caster: o3d.t.geometry.RaycastingScene,
        sensor: Sensor,
        pose: Pose,
        scale: float = 1.0,
):
    x_cam = sensor.y
    y_cam = sensor.x
    z_cam = np.full_like(x_cam, -1.0)

    # Stack into an (H, W, 3) array
    ray_vectors = np.stack((x_cam, y_cam, z_cam), axis=-1)

    # Rotate local ray vectors to world space
    R = pose.T[:3, :3]
    ray_vectors = np.einsum('ij,hwj->hwi', R, ray_vectors)
    
    # Normalize local rays to a length of 1
    ray_vectors /= np.linalg.norm(ray_vectors, axis=-1, keepdims=True)

    # Origins are simply the camera translation, broadcasted to the grid size
    origins = np.broadcast_to(pose.T[:3, 3], ray_vectors.shape)

    # Cast rays with Open3D
    rays = np.concatenate([origins, ray_vectors], axis=-1).astype(np.float32)
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    
    ans = ray_caster.cast_rays(rays_tensor)
    
    # Extract euclidean hit distances
    depth = ans['t_hit'].numpy()
    depth[np.isinf(depth)] = 0.0
    depth = (1000 * scale * depth).astype(np.uint16)

    # Apply an exponential scale to the heatmap to amplify variations
    heatmap = np.power(10, depth.astype(np.float32) / (np.max(depth) + 1)) - 1
    if np.any(heatmap != 0):
        heatmap[heatmap == 0] = np.min(heatmap[heatmap != 0]) - 1
        heatmap -= np.min(heatmap)
        heatmap /= np.maximum(np.max(heatmap), 1)

    heatmap = cv2.applyColorMap((255 * heatmap).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return depth, heatmap