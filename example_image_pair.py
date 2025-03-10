import torch
import cv2
import rerun as rr

from dust3r import Dust3r, preprocess, postprocess_with_color
from dust3r.postprocess import estimate_intrinsics, estimate_camera_pose

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

conf_threshold = 3.0
width, height = 512, 352
model_path = "models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
dust3r = Dust3r(model_path, width, height, conf_threshold, device)

# Read input images
frame1 = cv2.imread("data/test4.jpg")
frame2 = cv2.imread("data/test3.jpg")

# Run Dust3r model
pts1, colors1, depth_map1, mask1, pts2, colors2, mask2 = dust3r(frame1, frame2)

# Estimate intrinsics
intrinsics1 = estimate_intrinsics(pts1, mask1)
intrinsics2 = estimate_intrinsics(pts2, mask2)
print(intrinsics1)

# Estimate camera pose (the first one is the origin)
cam_pose = estimate_camera_pose(pts2, intrinsics2, mask2)
print(cam_pose)

# Visualize the output
rr.init("Dust3r Visualizer", spawn=True)
rr.log("pts3d1", rr.Points3D(pts1, colors=colors1))
rr.log("pts3d2", rr.Points3D(pts2, colors=colors2))

# rr.log("cam1/rgb", rr.Image(frame1))
# rr.log("cam1/depth", rr.DepthImage(depth_map1))
# rr.log("cam1", rr.Pinhole(image_from_camera=intrinsics1, width=width, height=height))
#
# rr.log("cam2/rgb", rr.Image(frame2))
# rr.log("cam2/depth", rr.DepthImage(depth_map2))
# rr.log("cam2", rr.Pinhole(image_from_camera=intrinsics2, width=width, height=height))
# rr.log("cam2", rr.Transform3D(mat3x3=cam_pose[:3, :3], translation=cam_pose[:3, 3]))
