import torch
import cv2
import rerun as rr

from dust3r.dust3r import Dust3rAllToOne
from dust3r.visualization import init_visualizer

if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    cap = cv2.VideoCapture(0)

    # Prepare first frame
    ret, original_frame = cap.read()

    # Initialize Dust3r model
    conf_threshold = 3.0
    width, height = 512, 352
    model_path = "models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    model = Dust3rAllToOne(model_path, original_frame, width, height, device=device, conf_threshold=conf_threshold)

    num_frames = 20
    wait_time = 0.5 # second between frames
    init_visualizer()
    for i in range(1, num_frames):

        ret, frame = cap.read()
        if not ret:
            break

        output1, output2 = model(frame)

        if i == 0:
            rr.log(f"/output0/pts3d", rr.Points3D(output1.pts3d, colors=output1.colors), static=True)
        rr.log(f"/output{i}/pts3d", rr.Points3D(output2.pts3d, colors=output2.colors), static=True)

        # Wait for a while
        cv2.waitKey(int(wait_time * 1000))

