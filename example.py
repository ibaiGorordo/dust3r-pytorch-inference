import torch
from torch.xpu import device

from dust3r.dust3r import Dust3rEncoder, Dust3rDecoder, Dust3rHead

if __name__ == '__main__':

    from PIL import Image
    import rerun as rr
    import torchvision.transforms as tvf

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_path = "models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    img1 = Image.open("data/test1.jpg").convert("RGB")
    img2 = Image.open("data/test2.jpg").convert("RGB")

    # Resize to 512x288
    img1 = img1.resize((512, 288))
    img2 = img2.resize((512, 288))

    # transform to tensor
    ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img1 = ImgNorm(img1).unsqueeze(0).to(device)
    img2 = ImgNorm(img2).unsqueeze(0).to(device)

    encoder = Dust3rEncoder(ckpt, width=512, height=288, device=device)
    decoder = Dust3rDecoder(ckpt, width=512, height=288, device=device)
    head = Dust3rHead(ckpt, width=512, height=288, device=device)

    feat = encoder(torch.cat((img1, img2)))
    f1, f2 = feat.chunk(2, dim=0)
    d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12 = decoder(f1, f2)
    pt1, cf1, pt2, cf2 = head(d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12)

    pt1 = pt1.cpu().detach().numpy().reshape(-1, 3)
    pt2 = pt2.cpu().detach().numpy().reshape(-1, 3)
    cf1 = cf1.cpu().detach().numpy().reshape(-1)
    cf2 = cf2.cpu().detach().numpy().reshape(-1)

    threshold = 3
    mask1 = cf1 > threshold
    mask2 = cf2 > threshold
    pt1 = pt1[mask1, :]
    pt2 = pt2[mask2, :]

    rr.init("Dust3r Visualizer", spawn=True)
    rr.log("pts3d1", rr.Points3D(pt1))
    rr.log("pts3d2", rr.Points3D(pt2))

    # torch.onnx.export(
    #     encoder,
    #     (torch.cat((img1, img2)),),
    #     "encoder.onnx",
    #     input_names=["img"],
    #     output_names=["feats"],
    #     opset_version=13,  # or whichever opset you need
    # )
    #
    # torch.onnx.export(
    #     decoder,
    #     (f1, f2),
    #     "decoder.onnx",
    #     input_names=["f1", "f2"],
    #     output_names=["d1_0", "d1_6", "d1_9", "d1_12", "d2_0", "d2_6", "d2_9", "d2_12"],
    #     opset_version=13,  # or whichever opset you need
    # )
    #
    # torch.onnx.export(
    #     head,
    #     (d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12),
    #     "head.onnx",
    #     input_names=["d1_0", "d1_6", "d1_9", "d1_12", "d2_0", "d2_6", "d2_9", "d2_12"],
    #     output_names=["pts3d1", "conf1", "pts3d2", "conf2"],
    #     opset_version=13,  # or whichever opset you need
    # )