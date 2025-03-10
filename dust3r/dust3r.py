from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .encoder import Dust3rEncoder
from .decoder import Dust3rDecoder
from .head import Dust3rHead
from .preprocess import preprocess
from .postprocess import postprocess_with_color, estimate_intrinsics, estimate_camera_pose, get_transformed_depth


@dataclass
class Output:
    input: np.ndarray
    pts3d: np.ndarray
    colors: np.ndarray
    conf_map: np.ndarray
    depth_map: np.ndarray
    intrinsic: np.ndarray
    pose: np.ndarray


class Dust3r(nn.Module):
    def __init__(self,
                 model_path: str,
                 width: int = 512,
                 height: int = 512,
                 symmetric: bool = False,
                 device: torch.device = torch.device('cuda'),
                 conf_threshold: float = 3.0,
                 ):
        super().__init__()

        self.width = width
        self.height = height
        self.symmetric = symmetric
        self.device = device
        self.conf_threshold = conf_threshold

        ckpt_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        self.encoder = Dust3rEncoder(ckpt_dict, width=width, height=height, device=device, batch=2)
        self.decoder = Dust3rDecoder(ckpt_dict, width=width, height=height, device=device)
        self.head = Dust3rHead(ckpt_dict, width=width, height=height, device=device)

    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> tuple[Output, Output]:
        return self.forward(img1, img2)

    @torch.inference_mode()
    def forward(self, img1: np.ndarray, img2: np.ndarray) -> tuple[Output, Output]:

        input1, frame1 = preprocess(img1, self.width, self.height, self.device)
        input2, frame2 = preprocess(img2, self.width, self.height, self.device)

        input = torch.cat((input1, input2), dim=0)
        feat = self.encoder(input)
        feat1, feat2 = feat.chunk(2, dim=0)

        pt1_1, cf1_1, pt2_1, cf2_1 = self.decoder_head(feat1, feat2)
        # if self.symmetric:
        #     pt2_2, cf2_2, pt1_2, cf1_2 = self.decoder_head(feat2, feat1)
        #
        #     output1, output2 = self.postprocess_symmetric(frame1, pt1_1, cf1_1, pt1_2, cf1_2, frame2, pt2_1, cf2_1, pt2_2, cf2_2)
        # else:
        output1, output2 = self.postprocess(frame1, pt1_1, cf1_1, frame2, pt2_1, cf2_1)

        return output1, output2

    def decoder_head(self, feat1, feat2):
        d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12 = self.decoder(feat1, feat2)
        pt1, cf1, pt2, cf2 = self.head(d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12)
        return pt1, cf1, pt2, cf2

    def postprocess(self,
                    frame1: np.ndarray,
                    pt1: torch.Tensor,
                    cf1: torch.Tensor,
                    frame2: np.ndarray,
                    pt2: torch.Tensor,
                    cf2: torch.Tensor,
                    ) -> tuple[Output, Output]:

        pts1, colors1, conf_map1, depth_map1, mask1 = postprocess_with_color(pt1, cf1, frame1, threshold=self.conf_threshold)
        pts2, colors2, conf_map2, depth_map2, mask2 = postprocess_with_color(pt2, cf2, frame2, threshold=self.conf_threshold)

        # Estimate intrinsics
        intrinsics1 = estimate_intrinsics(pts1, mask1)
        intrinsics2 = intrinsics1 # estimate_intrinsics(pts2, mask2)

        # Estimate camera pose (the first one is the origin)
        cam_pose1 = np.eye(4)
        cam_pose2 = estimate_camera_pose(pts2, intrinsics1, mask2)

        depth_map2 = get_transformed_depth(pts2, mask2, cam_pose2)

        output1 = Output(frame1, pts1, colors1, conf_map1, depth_map1, intrinsics1, cam_pose1)
        output2 = Output(frame2, pts2, colors2, conf_map2, depth_map2, intrinsics2, cam_pose2)

        return output1, output2

    # def postprocess_symmetric(self,
    #                           frame1: np.ndarray,
    #                           pt1_1: torch.Tensor,
    #                           cf1_1: torch.Tensor,
    #                           pt1_2: torch.Tensor,
    #                           cf1_2: torch.Tensor,
    #                           frame2: np.ndarray,
    #                           pt2_1: torch.Tensor,
    #                           cf2_1: torch.Tensor,
    #                           pt2_2: torch.Tensor,
    #                           cf2_2: torch.Tensor,
    #                           ) -> tuple[Output, Output]:
    #
    #     pts1_1, colors1_1, conf_map1_1, depth_map1, mask1_1 = postprocess_with_color(pt1_1, cf1_1, frame1, threshold=self.conf_threshold)
    #     pts1_2, colors1_2, conf_map1_2, depth_map1_2, mask1_2 = postprocess_with_color(pt1_2, cf1_2, frame1, threshold=self.conf_threshold)
    #     pts2_1, colors2_1, conf_map2_1, depth_map2_1, mask2_1 = postprocess_with_color(pt2_1, cf2_1, frame2, threshold=self.conf_threshold)
    #     pts2_2, colors2_2, conf_map2_2, depth_map2, mask2_2 = postprocess_with_color(pt2_2, cf2_2, frame2, threshold=self.conf_threshold)
    #
    #     # Estimate intrinsics
    #     intrinsics1 = estimate_intrinsics(pts1_1, mask1_1)
    #     intrinsics2 = estimate_intrinsics(pts2_2, mask2_2)
    #
    #     # Estimate camera pose (the first one is the origin)
    #     cam_pose1 = np.eye(4)
    #     cam_pose2 = estimate_camera_pose(pts2_1, intrinsics2, mask2_1)
    #
    #     if conf_map1_2.mean() > conf_map1_1.mean():
    #         temp_cam_pose1 = estimate_camera_pose(pts1_2, intrinsics1, mask1_2)
    #         depth_map1 = get_transformed_depth(pts1_2, mask1_2, temp_cam_pose1)
    #     if conf_map2_1.mean() > conf_map2_2.mean():
    #         print("Image2: ", conf_map2_1.mean(), conf_map2_2.mean())
    #         depth_map2 = get_transformed_depth(pts2_1, mask2_1, cam_pose2)
    #
    #     output1 = Output(frame1, pts1_1, colors1_1, conf_map1_1, depth_map1, intrinsics1, cam_pose1)
    #     output2 = Output(frame2, pts2_1, colors2_1, conf_map2_1, depth_map2, intrinsics2, cam_pose2)
    #
    #     return output1, output2


# class Dust3rAllToOne(Dust3r):
#     def __init__(self,
#                  model_path: str,
#                  origin_img: torch.Tensor,
#                  device: torch.device = torch.device('cuda'),
#                  ):
#         super().__init__(model_path, origin_img.shape[-1], origin_img.shape[-2], device)
#         self.origin_feat = self.encoder(origin_img)
#
#     def __call__(self, img: np.ndarray) -> tuple[Output, Output]:
#         return self.forward_single(img)
#
#     def update_origin(self, origin_img):
#         self.origin_feat = self.encoder(origin_img)
#
#     @torch.inference_mode()
#     def forward_single(self, img: np.ndarray):
#         input, frame = preprocess(img, self.width, self.height, self.device)
#
#         current_feat = self.encoder(input)
#
#         d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12 = self.decoder(self.origin_feat, current_feat)
#         pt1, cf1, pt2, cf2 = self.head(d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12)
#
#         return pt1, cf1, pt2, cf2