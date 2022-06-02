import torch
import numpy as np
import os
import cv2

pred_dir = '/media/jianyu/dataset/eval/cleargrasp/messy-table/sim-noir/exp-016/normal'
gt_dir = '/media/jianyu/dataset/messy-table-dataset/real_data_v10'

prefix = os.listdir(pred_dir)
prefix = [x.split('.')[0] for x in prefix]

print(prefix)

sum_angle = 0
num_angle = 0

for pre in prefix:
    label_path = os.path.join(gt_dir, pre, 'normalL.png')
    gt_normal = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    gt_normal = (gt_normal.astype(float)) / 1000 - 1
    gt_normal = cv2.resize(gt_normal, (960, 540), interpolation=cv2.INTER_NEAREST)
    gt_normal = gt_normal.transpose((2, 0, 1))

    pred_label_path = os.path.join(pred_dir, pre + '.npy')
    pred_normal = np.load(pred_label_path)
    print(gt_normal.shape, pred_normal.shape)

    for x in range(gt_normal.shape[0]):
        for y in range(gt_normal.shape[1]):
            cur_pred_n = pred_normal[:,x,y]
            cur_gt_n = gt_normal[:,x,y]
            dot_out = np.dot(cur_pred_n, cur_gt_n)
            norm_out = np.linalg.norm(cur_pred_n)*np.linalg.norm(cur_gt_n)
            cos = dot_out/norm_out
            angle = np.arccos(cos)/np.pi*180
            sum_angle += angle
            num_angle += 1
            #print(angle)

print(sum_angle/num_angle)