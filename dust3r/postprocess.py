# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities for interpreting the DUST3R output
# --------------------------------------------------------
import torch
import numpy as np
import cv2

def postprocess(points: torch.Tensor,
                confidences: torch.Tensor,
                threshold: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Convert tensors to numpy arrays
    points = points.cpu().detach().numpy().squeeze()
    confidences = confidences.cpu().detach().numpy().squeeze()

    depth_map = points[..., 2]

    # Apply threshold
    mask = confidences > threshold
    points = points[mask, :]

    return points.reshape(-1, 3), confidences, depth_map, mask

def postprocess_with_color(points: torch.Tensor,
                           confidences: torch.Tensor,
                           img: np.ndarray,
                           threshold: float = 3.0,
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Convert tensors to numpy arrays
    points, confidences, depth_map, mask = postprocess(points, confidences, threshold)
    colors = img[mask, :].reshape(-1, 3)

    return points, colors, confidences, depth_map, mask

def estimate_intrinsics(pts3d: np.ndarray,
                        mask: np.ndarray,
                        iterations: int = 10) -> np.ndarray:

    width, height = mask.shape[::-1]
    pixels = np.mgrid[-width//2:width//2, -height//2:height//2].T.astype(np.float32)
    pixels = pixels[mask, :].reshape(-1, 2)

    # Compute normalized image plane coordinates (x/z, y/z)
    xy_over_z = np.divide(pts3d[:, :2], pts3d[:, 2:3], where=pts3d[:, 2:3] != 0)
    xy_over_z[np.isnan(xy_over_z) | np.isinf(xy_over_z)] = 0  # Handle invalid values

    # Initial least squares estimate of focal length
    dot_xy_px = np.sum(xy_over_z * pixels, axis=-1)
    dot_xy_xy = np.sum(xy_over_z**2, axis=-1)
    focal = np.mean(dot_xy_px) / np.mean(dot_xy_xy)

    # Iterative re-weighted least squares refinement
    for _ in range(iterations):
        residuals = np.linalg.norm(pixels - focal * xy_over_z, axis=-1)
        weights = np.reciprocal(np.clip(residuals, 1e-8, None))  # Avoid division by zero
        focal = np.sum(weights * dot_xy_px) / np.sum(weights * dot_xy_xy)

    K = np.array([[focal, 0, width//2],
                  [0, focal, height//2],
                  [0, 0, 1]], dtype=np.float32)

    return K

def estimate_camera_pose(pts3d: np.ndarray,
                         K: np.ndarray,
                         mask: np.ndarray,
                         iterations=100,
                         reprojection_error=5):

    width, height = mask.shape[::-1]
    pixels = np.mgrid[:width, :height].T.astype(np.float32).reshape(-1, 2)
    pixels_valid = pixels[mask.flatten()]

    try:
        # Solve PnP using RANSAC
        success, R_vec, T, inliers = cv2.solvePnPRansac(
            pts3d, pixels_valid, K, None,
            iterationsCount=iterations, reprojectionError=reprojection_error,
            flags=cv2.SOLVEPNP_SQPNP
        )

        if not success:
            raise ValueError("PnP failed")

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(R_vec)  # Converts Rodrigues rotation vector to matrix

        # Construct 4x4 transformation matrix (camera to world)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = T.flatten()

        # Invert to get world-to-camera transform
        pose = np.linalg.inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world
        # pose = np.linalg.inv(pose)

    except:
        # Return identity matrix if PnP fails
        pose = np.eye(4)

    return pose