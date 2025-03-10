import torch
import cv2
import rerun as rr

from dust3r import Dust3r

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

conf_threshold = 3.0
width, height = 512, 352
model_path = "models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
dust3r = Dust3r(model_path, width, height, conf_threshold, device)

frame1 = cv2.imread("data/test1.jpg")
frame2 = cv2.imread("data/test2.jpg")

pts1, colors1, depth_map1, mask1, pts2, colors2, mask2 = dust3r(frame1, frame2)

rr.init("Dust3r Visualizer", spawn=True)
rr.log("pts3d1", rr.Points3D(pts1))
rr.log("pts3d2", rr.Points3D(pts2))