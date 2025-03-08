import os.path

import torch
import pickle

from dust3r import Dust3rEncoder, Dust3rDecoder, Dust3rHead

def test_dust3r(model_path, test_data_path):
    print(f"Testing {model_path} with {test_data_path}")

    ckpt_dict = torch.load(model_path, map_location='cpu', weights_only=False)

    filename = os.path.basename(test_data_path)
    width, height = 512, 288
    if "224" in filename:
        width = height = 224

    encoder = Dust3rEncoder(ckpt_dict, width=width, height=height, device=torch.device('cuda'))
    decoder = Dust3rDecoder(ckpt_dict, width=width, height=height, device=torch.device('cuda'))
    head = Dust3rHead(ckpt_dict, width=width, height=height, device=torch.device('cuda'))


    # load the "img1_img2.pkl" file
    with open(test_data_path, "rb") as f:
        img1, img2, dec1, dec2, feat1, feat2, pos1, pos2, pts3d1, pts3d2, conf1, conf2 = pickle.load(f)


    feat = encoder(torch.cat((img1, img2)))
    f1, f2 = feat.chunk(2, dim=0)
    d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12 = decoder(f1, f2)
    pt1, cf1, pt2, cf2 = head(d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12)
    assert torch.max(torch.abs(f1 - feat1)) == 0
    assert torch.max(torch.abs(f2 - feat2)) == 0
    assert torch.max(torch.abs(d1_0 - dec1[0])) == 0
    assert torch.max(torch.abs(d2_0 - dec2[0])) == 0
    assert torch.max(torch.abs(d1_6 - dec1[6])) == 0
    assert torch.max(torch.abs(d2_6 - dec2[6])) == 0
    assert torch.max(torch.abs(d1_9 - dec1[9])) == 0
    assert torch.max(torch.abs(d2_9 - dec2[9])) == 0
    assert torch.max(torch.abs(d1_12 - dec1[12])) == 0
    assert torch.max(torch.abs(d2_12 - dec2[12])) == 0
    assert torch.max(torch.abs(pts3d1 - pt1)) < 1e-5 # Because we use (exp(x) - 1) instead of expm1(x)
    assert torch.max(torch.abs(pts3d2 - pt2)) < 1e-5 # Because we use (exp(x) - 1) instead of expm1(x)
    assert torch.max(torch.abs(conf1 - cf1)) == 0
    assert torch.max(torch.abs(conf2 - cf2)) == 0

if __name__ == '__main__':
    model_dir = "../models"
    test_data_dir = "assets"

    test_dust3r(os.path.join(model_dir, "DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"), os.path.join(test_data_dir, "test_dust3r_224_linear.pkl"))
    test_dust3r(os.path.join(model_dir, "DUSt3R_ViTLarge_BaseDecoder_512_linear.pth"), os.path.join(test_data_dir, "test_dust3r_512_linear.pkl"))
    test_dust3r(os.path.join(model_dir, "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"), os.path.join(test_data_dir, "test_dust3r_512_dpt.pkl"))
