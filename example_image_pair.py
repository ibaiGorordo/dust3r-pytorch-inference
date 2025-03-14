import torch
import cv2

from dust3r import Dust3r
from dust3r.visualization import visualize_output

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

conf_threshold = 3.0
width, height = 512, 352
model_path = "models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
dust3r = Dust3r(model_path, width, height, symmetric=True, device=device, conf_threshold=conf_threshold)

# Read input images
frame1 = cv2.imread("data/test2.jpg")
frame2 = cv2.imread("data/test1.jpg")

# Run Dust3r model
output1, output2 = dust3r(frame1, frame2)

# Visualize the output
visualize_output(output1, output2, visualize_depth=False)