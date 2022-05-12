import torch
import numpy as np
from PIL import Image
import os
import copy
import matplotlib.pyplot as plt

import scipy.ndimage as ndimage
import copy

labelpath = "/messytable-slow-vol/messy-table-dataset/obj_fixang_fixpat/"
imagepath = "/messytable-slow-vol/messy-table-dataset/obj_fixang_fixpat/"
trasparent_id = [3,7,9,13,14,16]

scene_id = os.listdir(imagepath)

thres = 30
for id in scene_id:
    if len(id.split('-')) != 3:
        continue
    label_pwd = os.path.join(labelpath, id, "labelL.png")
    label = Image.open(label_pwd)
    label = label.resize((960,540), resample=Image.NEAREST)
    label = np.array(label)

    bgmask = label == 17
    label[bgmask] = 0
    objmask = label != 0
    label[objmask] = 1

    depth_pwd = os.path.join(imagepath, id, "depthL.png")
    depth = Image.open(depth_pwd)
    #print(label.shape)
    depth = depth.resize((960,540), resample=Image.NEAREST)
    depth = np.array(depth)

    edge = label - ndimage.morphology.binary_erosion(label) 

    edge = ndimage.morphology.binary_dilation(edge, iterations=2) 
    


    edge = edge.astype(np.uint8)


    mask = edge == 1

  

    diff_v = np.diff(depth, axis=0)
    diff_h = np.diff(depth, axis=1)

    pad_v = np.zeros((1,960)).astype(int)
    pad_h = np.zeros((540,1)).astype(int)

    #print(diff_v.shape, diff_h.shape)

    diff_v_f = np.concatenate((pad_v, diff_v), axis=0)
    diff_v_b = np.concatenate((diff_v, pad_v), axis=0)

    diff_h_f = np.concatenate((pad_h, diff_h), axis=1)
    diff_h_b = np.concatenate((diff_h, pad_h), axis=1)

    contact_v = np.logical_and(np.abs(diff_v_f[mask]) < thres, np.abs(diff_v_b[mask]) < thres)
    contact_h = np.logical_and(np.abs(diff_h_f[mask]) < thres, np.abs(diff_h_b[mask]) < thres)
    contact = np.logical_and(contact_v, contact_h)
    #print(np.sum(contact))
    #print(edge.dtype)

    contact_edge = edge[mask]
    contact_edge[contact] = 2

    edge[mask] = contact_edge

    #print(contact, edge[mask])

    #print(diff_h_b)

    edge_2 = copy.deepcopy(edge)
    mask_2 = edge == 2
    edge_2[mask_2] = 1
    mask_2_n = edge != 2
    edge_2[mask_2_n] = 0
    edge_2 = ndimage.morphology.binary_erosion(edge_2, iterations=2) 
    
    edge_2 = ndimage.morphology.binary_dilation(edge_2, iterations=6) 


    #plt.imshow(edge_img)

    contact_edge_mask = edge_2[mask] == 1
    contact_edge_mask_n = edge_2[mask] == 0
    contact_edge = edge[mask]
    contact_edge[contact_edge_mask] = 2
    contact_edge[contact_edge_mask_n] = 1

    edge[mask] = contact_edge

    edge_img = Image.fromarray(edge,mode='L')

    savepath = os.path.join(imagepath, id, "outline.png")
    edge_img.save(savepath)
    print(savepath)
    break