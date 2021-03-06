#!/usr/bin/env python3

from __future__ import print_function, division
import os
import glob
from PIL import Image
import numpy as np
import imageio
import sys

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
import imgaug as ia

from utils import utils
sys.path.append('../..')


class SurfaceNormalsDataset(Dataset):
    """
    Dataset class for training model on estimation of surface normals.
    Uses imgaug for image augmentations.

    If a label_dir is blank ( None, ''), it will assume labels do not exist and return a tensor of zeros
    for the label.

    Args:
        input_dir (str): Path to folder containing the input images (.png format).
        label_dir (str): (Optional) Path to folder containing the labels (.png format). If no labels exists, pass empty string.
        transform (imgaug transforms): imgaug Transforms to be applied to the imgs
        input_only (list, str): List of transforms that are to be applied only to the input img

    """

    def __init__(self,
                 cfg,
                 input_dir='data/datasets/train/milk-bottles-train/resized-files/preprocessed-rgb-imgs',
                 label_dir='',
                 transform=None,
                 input_only=None,
                 ):

        super().__init__()

        self.images_dir = input_dir
        self.labels_dir = label_dir
        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self._datalist_input = []  # Variable containing list of all input images filenames in dataset
        self._datalist_label = []  # Variable containing list of all ground truth filenames in dataset
        self._extension_input = ['-transparent-rgb-img.jpg', '-rgb.jpg']  # The file extension of input images
        self._extension_label = '-mask.png'  # The file extension of labels
        self.img_L, self.img_depth_l, self.img_meta, self.img_label = self._create_lists_filenames(cfg)
        #print(len(self.img_L))

    def __len__(self):
        return len(self.img_L)

    def __getitem__(self, index):
        '''Returns an item from the dataset at the given index. If no labels directory has been specified,
        then a tensor of zeroes will be returned as the label.

        Args:
            index (int): index of the item required from dataset.

        Returns:
            torch.Tensor: Tensor of input image
            torch.Tensor: Tensor of label
        '''

        # Open input imgs
        image_path = self.img_L[index]
        # for surface normals as input
        # _img = api_utils.exr_loader(image_path, ndim=3)  # (3, H, W)
        # _img = (_img + 1) / 2
        # _img = _img.transpose(1,2,0)
        # for rgb images as input
        _img = Image.open(image_path).convert('RGB')
        _img = np.array(_img)

        # Open labels
        if self.labels_dir:
            label_path = self.img_label[index]
            # _label = Image.open(label_path).convert('L')
            mask = imageio.imread(label_path)
            # _label = np.array(_label)[..., np.newaxis]
            _label = np.zeros(mask.shape, dtype=np.uint8)
            _label[mask > 0] = 1


        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()
            _img = det_tf.augment_image(_img)
            _img = np.ascontiguousarray(_img)  # To prevent errors from negative stride, as caused by fliplr()
        if self.labels_dir:
            _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))

        # Return Tensors
        _img_tensor = transforms.ToTensor()(_img)
        if self.labels_dir:
            _label_tensor = torch.from_numpy(_label.astype(np.float32))
            _label_tensor = torch.unsqueeze(_label_tensor, 0)
            # _label_tensor = transforms.ToTensor()(_label.astype(np.float))
        else:
            _label_tensor = torch.zeros((1, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        return _img_tensor, _label_tensor

    def _create_lists_filenames(self, cfg):
        '''Creates a list of filenames of images and labels each in dataset
        The label at index N will match the image at index N.

        Args:
            images_dir (str): Path to the dir where images are stored
            labels_dir (str): Path to the dir where labels are stored

        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
            ValueError: Number of images and labels do not match
        '''

        with open(cfg.split_file, 'r') as f:
            prefix = [line.strip() for line in f]

        np.random.shuffle(prefix)

        img_L = [os.path.join(cfg.images, p, cfg.image_name) for p in prefix]
        img_depth_l = [os.path.join(cfg.depth, p, cfg.depth_name) for p in prefix]

        img_meta = [os.path.join(cfg.depth, p, cfg.meta_name) for p in prefix]
        img_label = [os.path.join(cfg.images, p, cfg.label_name) for p in prefix]


        return img_L, img_depth_l, img_meta, img_label

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torchvision

    # Example Augmentations using imgaug
    # imsize = 512
    # augs_train = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0), # Resize image
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5),
    #     iaa.Rot90((0, 4)),
    #     # Blur and Noise
    #     #iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),
    #     #iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB", name="grayscale")),
    #     iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255), per_channel=True, name="gaus-noise")),
    #     # Color, Contrast, etc.
    #     #iaa.Sometimes(0.2, iaa.Multiply((0.75, 1.25), per_channel=0.1, name="brightness")),
    #     iaa.Sometimes(0.2, iaa.GammaContrast((0.7, 1.3), per_channel=0.1, name="contrast")),
    #     iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20), name="hue-sat")),
    #     #iaa.Sometimes(0.3, iaa.Add((-20, 20), per_channel=0.5, name="color-jitter")),
    # ])
    # augs_test = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0),
    # ])

    augs = None  # augs_train, augs_test, None
    input_only = None  # ["gaus-blur", "grayscale", "gaus-noise", "brightness", "contrast", "hue-sat", "color-jitter"]

    db_test = SurfaceNormalsDataset(
        input_dir='data/datasets/milk-bottles/resized-files/preprocessed-rgb-imgs',
        label_dir='data/datasets/milk-bottles/resized-files/preprocessed-camera-normals',
        transform=augs,
        input_only=input_only
    )

    batch_size = 16
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=32, drop_last=True)

    # Show 1 Shuffled Batch of Images
    for ii, batch in enumerate(testloader):
        # Get Batch
        img, label = batch
        print('image shape, type: ', img.shape, img.dtype)
        print('label shape, type: ', label.shape, label.dtype)

        # Show Batch
        sample = torch.cat((img, label), 2)
        im_vis = torchvision.utils.make_grid(sample, nrow=batch_size // 4, padding=2, normalize=True, scale_each=True)
        plt.imshow(im_vis.numpy().transpose(1, 2, 0))
        plt.show()

        break
