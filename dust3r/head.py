import torch
import torch.nn as nn


from .heads import DPTHead, LinearPts3d


class Dust3rHead(nn.Module):
    def __init__(self,
                 ckpt_dict,
                 width=512,
                 height=512,
                 device=torch.device('cuda'),
                 ):
        super().__init__()

        self.downstream_head1 = DPTHead(width, height) if self._is_dpt(ckpt_dict) else LinearPts3d(width, height)
        self.downstream_head2 = DPTHead(width, height) if self._is_dpt(ckpt_dict) else LinearPts3d(width, height)

        self._load_checkpoint(ckpt_dict)
        self.to(device)


    @torch.inference_mode()
    def forward(self, d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12):
        pts3d1, conf1 = self.downstream_head1(d1_0, d1_6, d1_9, d1_12)
        pts3d2, conf2 = self.downstream_head2(d2_0, d2_6, d2_9, d2_12)

        return pts3d1, conf1, pts3d2, conf2

    def _load_checkpoint(self, ckpt_dict):
        head_state_dict = {
            k.replace(".dpt", ""): v
            for k, v in ckpt_dict['model'].items()
            if "head" in k
        }
        self.load_state_dict(head_state_dict, strict=True)

    def _is_dpt(self, ckpt_dict):
        return any("dpt" in k for k in ckpt_dict['model'].keys())