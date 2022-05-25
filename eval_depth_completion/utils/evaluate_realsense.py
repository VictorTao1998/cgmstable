import numpy as np
import torch
from path import Path
import argparse
import cv2
import os

from utils.io import load_pickle
from utils.render_utils import visualize_depth


from tqdm import tqdm
import torch
import copy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate real realsense performance")
    parser.add_argument("-d", "--data-folder", type=str, required=True)
    parser.add_argument(
        "-s",
        "--split-file",
        type=str,
        metavar="FILE",
        required=True,
    )

    args = parser.parse_args()
    return args


def register_depth(view_folder, meta_folder):
    view_folder = Path(view_folder)
    if (view_folder / "1024_depthL_real.png").exists():
        depth_u16 = cv2.imread(view_folder / "1024_depthL_real.png", cv2.IMREAD_UNCHANGED)
        assert depth_u16.shape == (540, 960)
        depth = (depth_u16.astype(np.float32)) / 1000.0
    else:
        img_meta = load_pickle(meta_folder)
        extrinsic_l = img_meta["extrinsic_l"]
        extrinsic_r = img_meta["extrinsic_r"]
        intrinsic_l = img_meta["intrinsic_l"]
        # intrinsic_l[:2] /= 2
        intrinsic_l[2] = np.array([0.0, 0.0, 1.0])

        rgb_cam_depth = cv2.imread(view_folder / "1024_depth_real.png", cv2.IMREAD_UNCHANGED)
        rgb_cam_depth = rgb_cam_depth.astype(np.float32) / 1000.0
        w, h = 1920, 1080
        rt_mainl = img_meta["extrinsic"] @ np.linalg.inv(img_meta["extrinsic_l"])
        rt_lmain = np.linalg.inv(rt_mainl)
        depth = cv2.rgbd.registerDepth(
            img_meta["intrinsic"], intrinsic_l, None, rt_lmain, rgb_cam_depth, (w, h), depthDilation=True
        )
        depth[np.isnan(depth)] = 0
        depth[np.isinf(depth)] = 0
        depth[depth < 0] = 0

        depth = cv2.resize(depth, (960, 540), interpolation=cv2.INTER_NEAREST)
        depth_u16 = copy.deepcopy(depth)
        depth_u16 = (depth_u16 * 1000.0).astype(np.uint16)

        cv2.imwrite(view_folder / "1024_depthL_real.png", depth_u16)
        vis_depth = visualize_depth(depth_u16)
        cv2.imwrite(view_folder / "1024_depthL_real_colored.png", vis_depth)

    return depth

