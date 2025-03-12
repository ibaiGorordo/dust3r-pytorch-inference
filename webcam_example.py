import torch

from dust3r.dust3r import Dust3rAllToOne
from dust3r.preprocess import preprocess
from dust3r.postprocess import parse_output, postprocess_with_color

if __name__ == '__main__':

    import cv2
    import rerun as rr

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    rr.init("Dust3r Visualizer", spawn=True)

    cap = cv2.VideoCapture(0)

    # Prepare first frame
    ret, original_frame = cap.read()
    width, height = 512, 384
    input, original_frame = preprocess(original_frame, width, height, device)

    # Initialize Dust3r model
    model_path = "models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    model = Dust3rAllToOne(model_path, input, device=device)


    num_frames = 100
    wait_time = 0.1 # second between frames
    for i in range(num_frames):

        ret, frame = cap.read()
        if not ret:
            break

        pts1, conf1, pts2, conf2 = model(frame)
        pts1, colors1, pts2, colors2 = postprocess_with_color(pts1, conf1, original_frame, pts2, conf2, frame)

        rr.log(f"pcd/{i}", rr.Points3D(pts2, colors=colors1), static=True)
        if i == 0:
            rr.log("pcd/origin", rr.Points3D(pts1, colors=colors2), static=True)

        # Wait for a while
        cv2.waitKey(int(wait_time * 1000))

