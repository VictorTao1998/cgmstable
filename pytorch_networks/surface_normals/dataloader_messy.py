"""
Author: Isabella Liu 7/18/21
Feature: Load data from messy-table-dataset
"""

import os
import numpy as np
import random
from PIL import Image
import torch
import torchvision.transforms as Transforms
from torch.utils.data import Dataset
import cv2
import pickle
import torch.nn.functional as F

import open3d as o3d


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def __data_augmentation__():
    """
    :param gaussian_blur: Whether apply gaussian blur in data augmentation
    :param color_jitter: Whether apply color jitter in data augmentation
    Note:
        If you want to change the parameters of each augmentation, you need to go to config files,
        e.g. configs/remote_train_config.yaml
    """
    transform_list = [
        Transforms.ToTensor()
    ]

    # Normalization
    transform_list += [
        Transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ]
    custom_augmentation = Transforms.Compose(transform_list)
    return custom_augmentation


def __get_split_files__(cfg):
    """
    :param split_file: Path to the split .txt file, e.g. train.txt
    :param debug: Debug mode, load less data
    :param sub: If debug mode is enabled, sub will be the number of data loaded
    :param onReal: Whether test on real dataset, folder and file names are different
    :return: Lists of paths to the entries listed in split file
    """
    with open(cfg.split_file, 'r') as f:
        prefix = [line.strip() for line in f]
        np.random.shuffle(prefix)

    img_L = [os.path.join(cfg.images, p, cfg.image_name) for p in prefix]
    img_depth_l = [os.path.join(cfg.depth, p, cfg.depth_name) for p in prefix]

    img_meta = [os.path.join(cfg.depth, p, cfg.meta_name) for p in prefix]
    img_label = [os.path.join(cfg.images, p, cfg.label_name) for p in prefix]


    return img_L, img_depth_l, img_meta, img_label


class MessytableDataset(Dataset):
    def __init__(self, cfg):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param gaussian_blur: Whether apply gaussian blur in data augmentation
        :param color_jitter: Whether apply color jitter in data augmentation
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        """
        self.img_L, self.img_depth_l, self.img_meta, self.img_label = __get_split_files__(cfg)


    def __len__(self):
        return len(self.img_L)

    def __getitem__(self, idx):
        img_L = np.array(Image.open(self.img_L[idx]).convert(mode='L')) / 255  # [H, W]


        img_L_rgb = np.repeat(img_L[:, :, None], 3, axis=-1)

        img_depth_l = np.array(Image.open(self.img_depth_l[idx])) / 1000  # convert from mm to m
        #image_pcd = o3d.io.read_point_cloud(self.img_label[idx])
        #poin = np.array(image_pcd.points)
        surface_normal = np.loadtxt(self.img_label[idx])
        #if poin.shape[0] != 518400:
        #    print(self.img_label[idx])
        surface_normal = np.reshape(surface_normal,[540,960,3])
        surface_normal = surface_normal.transpose((2, 0, 1))  # To Shape: (3, H, W)
        img_meta = load_pickle(self.img_meta[idx])

        # Convert depth map to disparity map
        extrinsic_l = img_meta['extrinsic_l']
        extrinsic_r = img_meta['extrinsic_r']
        intrinsic_l = img_meta['intrinsic_l']
        baseline = np.linalg.norm(extrinsic_l[:, -1] - extrinsic_r[:, -1])
        focal_length = intrinsic_l[0, 0] / 2

        mask = img_depth_l > 0
        img_disp_l = np.zeros_like(img_depth_l)
        img_disp_l[mask] = focal_length * baseline / img_depth_l[mask]

        # Get data augmentation
        custom_augmentation = __data_augmentation__()
        #normalization = __data_augmentation__(gaussian_blur=False, color_jitter=False)

        

        item = {}
        item['img_L'] = custom_augmentation(img_L_rgb).type(torch.FloatTensor)
        item['img_disp_l'] = torch.tensor(img_disp_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W] in dataloader
        item['img_depth_l'] = torch.tensor(img_depth_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['prefix'] = self.img_L[idx].split('/')[-2]
        item['focal_length'] = torch.tensor(focal_length, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        item['baseline'] = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        item['img_label'] = torch.tensor(surface_normal, dtype=torch.float32)  # [bs, 1, H, W]

        #print(item['img_depth_l'].shape)
        depth_l = F.interpolate(item['img_depth_l'].unsqueeze(0), (540, 960), mode='nearest',
                                        recompute_scale_factor=False)
        img_ground_mask = torch.clone((depth_l > 0) & (depth_l < 1.25)).detach().squeeze(0)
        #print(img_ground_mask.shape)
        #_mask_tensor = torch.ones((1, item['img_L'].shape[1], item['img_L'].shape[2]), dtype=torch.float32)

        return item['img_L'], item['img_label'], img_ground_mask


if __name__ == '__main__':
    cdataset = MessytableDataset(cfg.SPLIT.TRAIN)
    item = cdataset.__getitem__(0)
    print(item['img_L'].shape)
    print(item['img_R'].shape)
    print(item['img_disp_l'].shape)
    print(item['prefix'])
    print(item['img_real_L'].shape)
    print(item['img_L_ir_pattern'].shape)
    print(item['img_real_L_ir_pattern'].shape)
