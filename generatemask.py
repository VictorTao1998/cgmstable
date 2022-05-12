import torch
import numpy as np
from PIL import Image
import os
import copy
import matplotlib.pyplot as plt

labelpath = "/messytable-slow-vol/messy-table-dataset/obj_fixang_fixpat/"
imagepath = "/messytable-slow-vol/messy-table-dataset/obj_fixang_fixpat/"
trasparent_id = [3,7,9,13,14,16]

scene_id = os.listdir(imagepath)

for s_id in scene_id:
    if len(s_id.split('-')) != 3:
        continue
    label_pwd = os.path.join(labelpath, s_id, "labelL.png")
    label = np.array(Image.open(label_pwd))
    label_rgb = copy.deepcopy(label)
    label[label == 1] = 0
    label_rgb[label_rgb == 1] = 0
    for t_label in trasparent_id:
        label[label == t_label] = 1
        label_rgb[label_rgb == t_label] = 255
    label[label != 1] = 0
    label_rgb[label_rgb != 255] = 0
    output = Image.fromarray(label)
    output_rgb = Image.fromarray(label_rgb)
    savepath = os.path.join(imagepath, s_id, "transparent_mask.png")
    savepath_rgb = os.path.join(imagepath, s_id, "transparent_mask_rgb.png")
    output.save(savepath)
    output_rgb.save(savepath_rgb)
    print(savepath)
    break