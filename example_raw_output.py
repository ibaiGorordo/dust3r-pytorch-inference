import torch
import cv2
import rerun as rr

from dust3r import Dust3r

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

conf_threshold = 3.0
width, height = 512, 352
model_path = "models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
dust3r = Dust3r(model_path, width, height, device=device, conf_threshold=conf_threshold)

frame1 = cv2.imread("data/test1.jpg")
frame2 = cv2.imread("data/test2.jpg")

output1, output2 = dust3r(frame1, frame2)

rr.init("Dust3r Visualizer", spawn=True)
rr.log("pts3d1", rr.Points3D(output1.pts3d))
rr.log("pts3d2", rr.Points3D(output2.pts3d))