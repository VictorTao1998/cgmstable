
import os
import pickle
import random


import cv2
import matplotlib.pyplot as plt
import numpy as np

from path import Path


MAX_DEPTH = 2.0

RANDOM_SCALE = 0.5
SCALING_MIN = 0.5
SCALING_MAX = 1.5
METALLIC_MIN = 0.0
METALLIC_MAX = 0.8
ROUGHNESS_MIN = 0.0
ROUGHNESS_MAX = 0.8
SPECULAR_MIN = 0.0
SPECULAR_MAX = 0.8
TRANSMISSION_MIN = 0.0
TRANSMISSION_MAX = 1.0

PRIMITIVE_MIN = 25
PRIMITIVE_MAX = 50



def visualize_depth(depth):
    cmap = plt.get_cmap("rainbow")
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0
    if len(depth.shape) == 3:
        depth = depth[..., 0]
    depth = np.clip(depth / MAX_DEPTH, 0.0, 1.0)
    vis_depth = cmap(depth)
    vis_depth = (vis_depth[:, :, :3] * 255.0).astype(np.uint8)
    vis_depth = cv2.cvtColor(vis_depth, cv2.COLOR_RGB2BGR)
    return vis_depth
