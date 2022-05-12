import torch
import numpy as np
from PIL import Image
import os
import copy
import matplotlib.pyplot as plt
import pandas
import open3d as o3d
import numpy as np

db = pandas.read_pickle('~/haosulab/cam_db_full.pkl')
cam_intrinsic = []
cam_extrinsic = []
cam_intrinsic_np = []
for i in range(21):
    cur_intr = db[i]['intrinsic_l']/2
    cam_intrinsic_np.append(np.array(cur_intr))
    cam_intrinsic.append(o3d.camera.PinholeCameraIntrinsic(960,540,cur_intr[0,0],cur_intr[1,1],cur_intr[0,2],cur_intr[1,2]))
    cam_extrinsic.append(np.array(db[i]['extrinsic_l']))

root_path = '/messytable-slow-vol/messy-table-dataset/obj_fixang_fixpat/'
gt_root_path = '/messytable-slow-vol/messy-table-dataset/obj_fixang_fixpat/'

scene_id = os.listdir(root_path)

cam_extrin = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
for id in scene_id:
    if len(id.split('-')) != 3:
        continue
    angle_id = int(id.split('-')[2])
    #print(angle_id)
    #label_path = os.path.join(root_path, scene_id+'-'+str(angle_id), 'irL_label_image.png')
    #label_file = Image.open(label_path)
    #label = np.array(label_file.resize((960,540), resample=Image.NEAREST))
    
    gt_depth_file_path = os.path.join(gt_root_path, id, 'depthL.png')
    gt_depth_save_path = os.path.join(gt_root_path, id, 'depthL_half.png')
    gt_depth_half_path = os.path.join(gt_root_path, id, 'depthL_half.png')
    gt_depth = Image.open(gt_depth_file_path)
    gt_depth = gt_depth.resize((960,540), resample=Image.NEAREST)
    gt_depth.save(gt_depth_save_path)
    
    #color_path = os.path.join(root_path, scene_id+'-'+str(angle_id), '1024_irL_real_half.png')
    #color = Image.open(color_path)
    #color = np.array(color)

    
    gt_depth_o3d = o3d.io.read_image(gt_depth_half_path)
    #print(np.array(gt_depth_o3d).shape)
    gt_depth_np = np.array(Image.open(gt_depth_half_path))
    gt_pc = o3d.geometry.PointCloud.create_from_depth_image(gt_depth_o3d, intrinsic=cam_intrinsic[angle_id], depth_trunc=100000000, project_valid_depth_only=False)
    gt_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30),fast_normal_computation=False)
    gt_pc_p = np.asarray(gt_pc.points)
    
    cam_pos = -np.matmul(cam_extrin[:3,:3].T,cam_extrin[:3,3])
    #print(cam_pos)
    gt_pc.orient_normals_towards_camera_location(cam_pos)
    gt_normal = np.asarray(gt_pc.normals)
    #o3d.visualization.draw_geometries([gt_pc])
    #assert gt_normal.shape[0] == 518400
    #print(np.array(gt_depth_o3d).shape, gt_normal.shape, np.amax(np.array(gt_depth_o3d)))
    #cam_pos = -np.matmul(cam_extrinsic[angle_id][:3,:3].T,cam_extrinsic[angle_id][:3,3])

    pc_save_path = os.path.join(gt_root_path, id, 'gt_normal.txt')
    np.savetxt(pc_save_path, gt_normal)