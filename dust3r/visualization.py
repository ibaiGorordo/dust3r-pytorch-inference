import rerun as rr
from .dust3r import Output


is_visualizer_init = False

def init_visualizer():
    global is_visualizer_init
    if is_visualizer_init:
        return
    rr.init("Dust3r Visualizer", spawn=True)
    is_visualizer_init = True

def visualize_output(output1: Output, output2: Output):
    init_visualizer()

    rr.log("pts3d1", rr.Points3D(output1.pts3d, colors=output1.colors))
    rr.log("pts3d2", rr.Points3D(output2.pts3d, colors=output2.colors))

    rr.log("cam1/rgb", rr.Image(output1.input))
    rr.log("cam1/depth", rr.DepthImage(output1.depth_map))
    rr.log("cam1", rr.Pinhole(image_from_camera=output1.intrinsic, width=output1.width, height=output1.height))
    rr.log("cam1", rr.Transform3D(mat3x3=output1.pose[:3, :3], translation=output1.pose[:3, 3]))

    rr.log("cam2/rgb", rr.Image(output2.input))
    rr.log("cam2/depth", rr.DepthImage(output2.depth_map))
    rr.log("cam2", rr.Pinhole(image_from_camera=output2.intrinsic, width=output2.width, height=output2.height))
    rr.log("cam2", rr.Transform3D(mat3x3=output2.pose[:3, :3], translation=output2.pose[:3, 3]))
