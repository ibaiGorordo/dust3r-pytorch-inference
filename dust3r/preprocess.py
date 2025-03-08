import numpy as np
import cv2
import torch

import torchvision.transforms as T


ImgNorm = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def preprocess(img: np.ndarray, width: int, height: int, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (width, height))
    return ImgNorm(frame).unsqueeze(0).to(device), frame
