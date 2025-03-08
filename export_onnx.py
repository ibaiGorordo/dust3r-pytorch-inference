import torch
import torch.nn as nn
from dust3r import Dust3r


class Dust3rDecoderHead(nn.Module):
    def __init__(self, model: Dust3r):
        super().__init__()
        self.model = model

    def forward(self, f1, f2):
        f1_0, f1_6, f1_9, f1_12, f2_0, f2_6, f2_9, f2_12 = self.model.decoder(f1, f2)
        pts3d1, conf1, pts3d2, conf2 = self.model.head(f1_0, f1_6, f1_9, f1_12, f2_0, f2_6, f2_9, f2_12)
        return pts3d1, conf1, pts3d2, conf2

if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    width, height = 512, 288
    model_path = "models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    dust3r = Dust3r(model_path, width, height, device)
    decoder_head = Dust3rDecoderHead(dust3r).to(device) # Combined decoder and head

    img1 = torch.randn(1, 3, height, width).to(device)
    img2 = torch.randn(1, 3, height, width).to(device)
    feat = dust3r.encoder(torch.cat((img1, img2)))
    f1, f2 = feat.chunk(2)

    torch.onnx.export(
        dust3r.encoder,
        (torch.cat((img1, img2)),),
        "model/dust3r_encoder.onnx",
        input_names=["img"],
        output_names=["feats"],
        opset_version=13,
    )

    torch.onnx.export(
        decoder_head,
        (f1, f2),
        "model/dust3r_decoder_head.onnx",
        input_names=["f1", "f2"],
        output_names=["pts3d1", "conf1", "pts3d2", "conf2"],
        opset_version=13,
    )