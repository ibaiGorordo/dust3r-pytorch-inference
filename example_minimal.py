import torch
import cv2
import rerun as rr

from dust3r import Dust3r, preprocess, postprocess_with_color
from dust3r.postprocess import estimate_intrinsics, estimate_camera_pose

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

width, height = 512, 288
model_path = "models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
dust3r = Dust3r(model_path, width, height, device)

frame1 = cv2.imread("data/test1.jpg")
frame2 = cv2.imread("data/test2.jpg")
input1, frame1 = preprocess(frame1, width, height, device)
input2, frame2 = preprocess(frame2, width, height, device)

pt1, cf1, pt2, cf2 = dust3r(input1, input2)

pts1, colors1, conf_map1, depth_map1, mask1 = postprocess_with_color(pt1, cf1, frame1)
pts2, colors2, conf_map2, depth_map2, mask2 = postprocess_with_color(pt2, cf2, frame2)

intrinsics1 = estimate_intrinsics(pts1, mask1)
cam_pose1 = estimate_camera_pose(pts1, intrinsics1, mask1)

rr.init("Dust3r Visualizer", spawn=True)
rr.log("pts3d1", rr.Points3D(pts1, colors=colors1))
rr.log("pts3d2", rr.Points3D(pts2, colors=colors2))