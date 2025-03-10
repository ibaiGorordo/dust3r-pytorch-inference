import torch
import cv2
import rerun as rr

from dust3r import Dust3r

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

conf_threshold = 3.0
width, height = 512, 352
model_path = "models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
dust3r = Dust3r(model_path, width, height, conf_threshold, device)

# Read input images
frame1 = cv2.imread("data/test1.jpg")
frame2 = cv2.imread("data/test2.jpg")

# Run Dust3r model
output1, output2 = dust3r(frame1, frame2)

# Visualize the output
rr.init("Dust3r Visualizer", spawn=True)
rr.log("pts3d1", rr.Points3D(output1.pts3d, colors=output1.colors))
rr.log("pts3d2", rr.Points3D(output2.pts3d, colors=output2.colors))

rr.log("cam1/rgb", rr.Image(output1.input))
rr.log("cam1/depth", rr.DepthImage(output1.depth_map))
rr.log("cam1", rr.Pinhole(image_from_camera=output1.intrinsic, width=width, height=height))

rr.log("cam2/rgb", rr.Image(output2.input))
rr.log("cam2/depth", rr.DepthImage(output2.depth_map))
rr.log("cam2", rr.Pinhole(image_from_camera=output2.intrinsic, width=width, height=height))
rr.log("cam2", rr.Transform3D(mat3x3=output2.pose[:3, :3], translation=output2.pose[:3, 3]))
