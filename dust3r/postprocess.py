import torch
import numpy as np

def postprocess(points: torch.Tensor,
                confidences: torch.Tensor,
                threshold: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Convert tensors to numpy arrays
    points = points.cpu().detach().numpy().squeeze()
    confidences = confidences.cpu().detach().numpy().squeeze()

    depth_map = points[..., 2]

    # Apply threshold
    points = points[confidences > threshold, :]

    return points.reshape(-1, 3), confidences, depth_map

def postprocess_with_color(points: torch.Tensor,
                            confidences: torch.Tensor,
                            img: np.ndarray,
                            threshold: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Convert tensors to numpy arrays
    points = points.cpu().detach().numpy().squeeze()
    confidences = confidences.cpu().detach().numpy().squeeze()

    depth_map = points[..., 2]

    # Apply threshold
    points = points[confidences > threshold, :]
    colors = img[confidences > threshold, :]

    return points.reshape(-1, 3), colors.reshape(-1, 3), confidences, depth_map