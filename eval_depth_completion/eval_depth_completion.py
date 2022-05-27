#!/usr/bin/env python3
'''Script to run Depth Completion on Synthetic and Real datasets, visualizing the results and computing the error metrics.
This will save all intermediate outputs like surface normals, etc, create a collage of all the inputs and outputs and
create pointclouds from the input, modified input and output depth images.
'''
import argparse
import csv
import glob
import os
import shutil
import sys
import itertools


# Importing Pytorch before Open3D can cause unknown "invalid pointer" error
import open3d as o3d
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from api import depth_completion_api
from api import utils as api_utils
from tensorboardX import SummaryWriter

import attrdict
import imageio
import termcolor
import yaml
import torch
import cv2
import numpy as np

from utils.io import load_pickle
from utils.metrics import ErrorMetric
from utils.evaluate_realsense import register_depth
from utils import eval_utils
#print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api import utils
#assert 1==0

from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run eval of depth completion on synthetic data')
    parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
    parser.add_argument('-m', '--maskInputDepth', action="store_true", help='Whether we should mask out objects in input depth')
    args = parser.parse_args()

    # Load Config File
    CONFIG_FILE_PATH = args.configFile
    with open(CONFIG_FILE_PATH) as fd:
        config_yaml = yaml.safe_load(fd)
    config = attrdict.AttrDict(config_yaml)

    # Create directory to save results
    RESULTS_ROOT_DIR = config.resultsDir
    runs = sorted(glob.glob(os.path.join(RESULTS_ROOT_DIR, 'exp-*')))
    prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
    results_dir = os.path.join(RESULTS_ROOT_DIR, 'exp-{:03d}'.format(prev_run_id))
    if os.path.isdir(results_dir):
        if len(os.listdir(results_dir)) > 1:
            # Min 1 file always in folder: copy of config file
            results_dir = os.path.join(RESULTS_ROOT_DIR, 'exp-{:03d}'.format(prev_run_id + 1))
            os.makedirs(results_dir)
    else:
        os.makedirs(results_dir)
    shutil.copy2(CONFIG_FILE_PATH, os.path.join(results_dir, 'config.yaml'))
    print('\nSaving results to folder: ' + termcolor.colored('"{}"\n'.format(results_dir), 'green'))

    # Init depth completion API
    outputImgHeight = int(config.depth2depth.yres)
    outputImgWidth = int(config.depth2depth.xres)

    logger = SummaryWriter(results_dir, comment='create-graph')
    # if config.files.camera_intrinsics is not None or config.files.camera_intrinsics is not '':
    #     print(colored('\nUsing camera intrinsics from yaml file of real camera!\n'), 'green')
    #     CONFIG_FILE_PATH = config.files.camera_intrinsics
    #     if not os.path.isfile(CONFIG_FILE_PATH):
    #         print('\nError: Camera Intrinsics yaml does not exist: {}\n'.format(CONFIG_FILE_PATH))
    #         exit()
    #     with open(CONFIG_FILE_PATH) as fd:
    #         config_intr_yaml = yaml.safe_load(fd)
    #         camera_params = attrdict.AttrDict(config_intr_yaml)
    #     fx = (float(config.depth2depth.xres) / camera_params.xres) * camera_params.fx
    #     fy = (float(config.depth2depth.yres) / camera_params.yres) * camera_params.fy
    #     cx = (float(config.depth2depth.xres) / camera_params.xres) * camera_params.cx
    #     cy = (float(config.depth2depth.yres) / camera_params.yres) * camera_params.cy
    # else:
    #     fx = int(config.depth2depth.fx)
    #     fy = int(config.depth2depth.fy)
    #     cx = int(config.depth2depth.cx)
    #     cy = int(config.depth2depth.cy)

    depthcomplete = depth_completion_api.DepthToDepthCompletion(normalsWeightsFile=config.normals.pathWeightsFile,
                                                                outlinesWeightsFile=config.outlines.pathWeightsFile,
                                                                masksWeightsFile=config.masks.pathWeightsFile,
                                                                normalsModel=config.normals.model,
                                                                outlinesModel=config.outlines.model,
                                                                masksModel=config.masks.model,
                                                                depth2depthExecutable=config.depth2depth.pathExecutable,
                                                                outputImgHeight=outputImgHeight,
                                                                outputImgWidth=outputImgWidth,
                                                                fx=int(config.depth2depth.fx),
                                                                fy=int(config.depth2depth.fy),
                                                                cx=int(config.depth2depth.cx),
                                                                cy=int(config.depth2depth.cy),
                                                                filter_d=config.outputDepthFilter.d,
                                                                filter_sigmaColor=config.outputDepthFilter.sigmaColor,
                                                                filter_sigmaSpace=config.outputDepthFilter.sigmaSpace,
                                                                maskinferenceHeight=config.masks.inferenceHeight,
                                                                maskinferenceWidth=config.masks.inferenceWidth,
                                                                normalsInferenceHeight=config.normals.inferenceHeight,
                                                                normalsInferenceWidth=config.normals.inferenceWidth,
                                                                outlinesInferenceHeight=config.normals.inferenceHeight,
                                                                outlinesInferenceWidth=config.normals.inferenceWidth,
                                                                min_depth=config.depthVisualization.minDepth,
                                                                max_depth=config.depthVisualization.maxDepth,
                                                                tmp_dir=results_dir)

    # Create lists of input data
    rgb_file_list = []
    depth_file_list = []
    segmentation_masks_list = []
    gt_depth_file_list = []
    """
    for dataset in config.files:
        EXT_COLOR_IMG = ['-transparent-rgb-img.jpg', '-rgb.jpg']  #'-rgb.jpg' - includes normals-rgb.jpg
        EXT_DEPTH_IMG = ['-depth-rectified.exr', '-transparent-depth-img.exr']
        EXT_DEPTH_GT = ['-depth-rectified.exr', '-opaque-depth-img.exr']
        EXT_MASK = ['-mask.png']
        for ext in EXT_COLOR_IMG:
            rgb_file_list += (sorted(glob.glob(os.path.join(dataset.image, '*' + ext))))
        for ext in EXT_DEPTH_IMG:
            depth_file_list += (sorted(glob.glob(os.path.join(dataset.depth, '*' + ext))))
        assert len(rgb_file_list) == len(depth_file_list), (
            'number of rgb ({}) and depth images ({}) are not equal'.format(len(rgb_file_list), len(depth_file_list)))

        if dataset.masks is not None and dataset.masks != '':
            for ext in EXT_MASK:
                segmentation_masks_list += (sorted(glob.glob(os.path.join(dataset.masks, '*' + ext))))
            assert len(rgb_file_list) == len(segmentation_masks_list), (
                'number of rgb ({}) and masks ({}) are not equal'.format(len(rgb_file_list),
                                                                         len(segmentation_masks_list)))
        if dataset.gt_depth is not None and dataset.gt_depth != '':
            for ext in EXT_DEPTH_GT:
                gt_depth_file_list += (sorted(glob.glob(os.path.join(dataset.gt_depth, '*' + ext))))
            assert len(rgb_file_list) == len(gt_depth_file_list), (
                'number of rgb ({}) and gt depth ({}) are not equal'.format(len(rgb_file_list),
                                                                            len(gt_depth_file_list)))
    """

    with open(config.files.split_file, 'r') as f:
        prefix = [line.strip() for line in f]
    
    rgb_file = config.files.image
    depth_file = config.files.depth
    normal_file = config.files.normal

    gt_depth_file = config.files.gt_depth
    label_file = config.files.label

    rgb_file_list = [os.path.join(rgb_file, p, config.files.imagename) for p in prefix]
    depth_file_list = [os.path.join(depth_file, p) for p in prefix]

    gt_depth_file_list = [os.path.join(gt_depth_file, p, config.files.gt_depthname) for p in prefix]
    meta_list = [os.path.join(depth_file, p, 'meta.pkl') for p in prefix]
    label_list = [os.path.join(label_file, p, config.files.labelname) for p in prefix]
    normal_list = [os.path.join(normal_file, p, config.files.normalname) for p in prefix]

    print('Total Num of rgb_files:', len(rgb_file_list))
    print('Total Num of depth_files:', len(depth_file_list))
    print('Total Num of gt_depth_files:', len(gt_depth_file_list))

    assert len(rgb_file_list) > 0, ('No files found in given directories')

    # Create CSV File to store error metrics
    csv_filename = 'computed_errors.csv'
    field_names = ["Image Num", "RMSE", "REL", "MAE", "Delta 1.25", "Delta 1.25^2", "Delta 1.25^3"]
    with open(os.path.join(results_dir, csv_filename), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
        writer.writeheader()

    # Iterate over all the files and run depth completion
    a1_mean = 0.0
    a2_mean = 0.0
    a3_mean = 0.0
    rmse_mean = 0.0
    abs_rel_mean = 0.0
    mae_mean = 0.0
    sq_rel_mean = 0.0

    metric = ErrorMetric()
    metric.reset()
    file_dir = results_dir

    for i in range(len(rgb_file_list)):

        # Run Depth Completion
        color_img = Image.open(rgb_file_list[i]).convert('RGB')
        color_img = color_img.resize([960,540], resample=Image.BILINEAR)
        color_img = np.array(color_img)
        #color_img = cv2.resize(color_img, (960, 540), interpolation=cv2.INTER_CUBIC)
        #input_depth = api_utils.exr_loader(depth_file_list[i], ndim=1)
        
        input_depth = register_depth(depth_file_list[i], meta_list[i])


        obj_label = cv2.imread(label_list[i], cv2.IMREAD_UNCHANGED).astype(int)
        obj_label = cv2.resize(obj_label, (960, 540), interpolation=cv2.INTER_NEAREST)

        depth_gt = (
                cv2.imread(gt_depth_file_list[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
            ) 
        depth_gt = cv2.resize(depth_gt, (960, 540), interpolation=cv2.INTER_NEAREST)

        img_normal_l = cv2.imread(normal_list[i], cv2.IMREAD_UNCHANGED)
        img_normal_l = (img_normal_l.astype(float)) / 1000 - 1
        img_normal_l = cv2.resize(img_normal_l, (960, 540), interpolation=cv2.INTER_NEAREST)

        img_meta = load_pickle(meta_list[i])
        extrinsic_l = img_meta["extrinsic_l"]
        extrinsic_r = img_meta["extrinsic_r"]
        intrinsic_l = img_meta["intrinsic_l"]
        intrinsic_l[:2] /= 2
        intrinsic_l[2] = np.array([0.0, 0.0, 1.0])
        baseline = np.linalg.norm(extrinsic_l[:, -1] - extrinsic_r[:, -1])
        focal_length = intrinsic_l[0, 0]

        mask = input_depth > 0
        img_disp_l = np.zeros_like(input_depth)
        img_disp_l[mask] = focal_length * baseline / depth_gt[mask]


        #input_depth = (np.array(Image.open(depth_file_list[i]).resize((960,540),Image.NEAREST)) / 1000).astype(np.float32)
        #obj_label = np.array(Image.open(label_list[i]).convert('RGB').resize((960,540),Image.NEAREST))
        #print(color_img.shape, input_depth.shape, obj_label.shape, depth_gt.shape)

        #print(color_img.dtype, input_depth.dtype, input_depth.shape, color_img.shape)
        # NOTE: If no gt_depth present, it means the depth itself is gt_depth (syn data). We mask out all objects in input depth so depthcomplete can't cheat.

        try:
            output_depth, filtered_output_depth = depthcomplete.depth_completion(
                color_img,
                input_depth,
                inertia_weight=float(config.depth2depth.inertia_weight),
                smoothness_weight=float(config.depth2depth.smoothness_weight),
                tangent_weight=float(config.depth2depth.tangent_weight),
                mode_modify_input_depth=config.modifyInputDepth.mode,
                dilate_mask=True)
        except depth_completion_api.DepthCompletionError as e:
            print('Depth Completion Failed:\n  {}\n  ...skipping image {}'.format(e, i))
            continue

        #print(output_depth.shape, filtered_output_depth.shape, type(output_depth))
        data_batch = {}
        data_batch['img_depth_l'] = depth_gt
        data_batch['intrinsic_l'] = intrinsic_l
        data_batch['img_label_l'] = obj_label
        data_batch['img_disp_l'] = img_disp_l
        data_batch['img_normal_l'] = img_normal_l

        data_dir = prefix[i]
        metric.compute(
                    data_batch,
                    output_depth,
                    save_folder=os.path.join(file_dir, data_dir),
                    real_data=False,
                )
        min_depth=0.0
        max_depth=1.5
        COLOR_MAP = cv2.COLORMAP_JET
        input_depth_rgb = utils.depth2rgb(depthcomplete.input_depth,
                                          min_depth=min_depth,
                                          max_depth=max_depth,
                                          color_mode=COLOR_MAP)
        orig_input_depth_rgb = utils.depth2rgb(depthcomplete.orig_input_depth,
                                               min_depth=min_depth,
                                               max_depth=max_depth,
                                               color_mode=COLOR_MAP)
        output_depth_rgb = utils.depth2rgb(depthcomplete.output_depth,
                                           min_depth=min_depth,
                                           max_depth=max_depth,
                                           color_mode=COLOR_MAP)
        gt_depth_rgb = utils.depth2rgb(depthcomplete.depth_gt, min_depth=min_depth, max_depth=max_depth, color_mode=COLOR_MAP)

        
        mask_rgb = depthcomplete.input_image.copy()
        mask_rgb[depthcomplete.mask_predicted > 0, 0] = 255
        mask_rgb[depthcomplete.outlines_rgb[:, :, 1] > 0, 1] = 255
        masked_img = cv2.addWeighted(mask_rgb, 0.6, depthcomplete.input_image, 0.4, 0)

        grid_image1 = np.concatenate(
            (depthcomplete.input_image, depthcomplete.surface_normals_rgb, depthcomplete.outlines_rgb, masked_img), 1)
        grid_image2 = np.concatenate(
            (orig_input_depth_rgb, input_depth_rgb, output_depth_rgb, gt_depth_rgb), 1)
        grid_image = np.concatenate((grid_image1, grid_image2), 0)
        img_normal_l

        path_result_viz = os.path.join(os.path.join(file_dir, data_dir),
                                       prefix[i] + '-vis.png')
        #print(grid_image.dtype)
        cv2.imwrite(path_result_viz, grid_image)
        #imageio.imwrite(path_result_viz, grid_image)

        grid_image = eval_utils.create_grid_image(depthcomplete.input_tensor, depthcomplete.surface_normals_tensor.float(), max_num_images_to_save=1)
        logger.add_image('Train', grid_image, i)



        
        # Compute Errors in Depth Estimation over the Masked Area
        # If a folder of masks is given, use it to calc error only over masked regions, else over entire image
        """
        if segmentation_masks_list:
            seg_mask = imageio.imread(segmentation_masks_list[i])
            seg_mask = cv2.resize(seg_mask, (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)
            seg_mask = (seg_mask > 0)
        else:
            seg_mask = np.full((outputImgHeight, outputImgWidth), True, dtype=float)

        # If Ground Truth depth folder is given, use that to compute errors. In case of Synthetic data, input depth is GT depth.

        depth_gt[np.isnan(depth_gt)] = 0
        depth_gt[np.isinf(depth_gt)] = 0
        mask_valid_region = (depth_gt > 0)
        mask_valid_region = np.logical_and(mask_valid_region, seg_mask)
        mask_valid_region = (mask_valid_region.astype(np.uint8) * 255)

        metrics = depthcomplete.compute_errors(depth_gt, output_depth, mask_valid_region)
        """
        print('\nImage {:09d} / {}:'.format(i, len(rgb_file_list) - 1))
        #print('{:>15}:'.format('rmse'), metrics['rmse'])
        #print('{:>15}:'.format('abs_rel'), metrics['abs_rel'])
        #print('{:>15}:'.format('mae'), metrics['mae'])
        #print('{:>15}:'.format('a1.05'), metrics['a1'])
        #print('{:>15}:'.format('a1.10'), metrics['a2'])
        #print('{:>15}:'.format('a1.25'), metrics['a3'])

        # Write the data into a csv file
        #with open(os.path.join(results_dir, csv_filename), 'a', newline='') as csvfile:
        #    writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
        #    row_data = [
        #        i, metrics["rmse"], metrics["abs_rel"], metrics["mae"], metrics["a1"], metrics["a2"], metrics["a3"]
        #    ]
        #    writer.writerow(dict(zip(field_names, row_data)))

        #a1_mean += metrics['a1']
        #a2_mean += metrics['a2']
        #a3_mean += metrics['a3']
        #rmse_mean += metrics['rmse']
        #abs_rel_mean += metrics['abs_rel']
        #mae_mean += metrics['mae']
        #sq_rel_mean += metrics['sq_rel']


        # Save Results of Depth Completion
        #print("success")
        #error_output_depth, error_filtered_output_depth = depthcomplete.store_depth_completion_outputs(
        #    root_dir=results_dir,
        #    files_prefix=prefix[i],
        #    min_depth=config.depthVisualization.minDepth,
        #    max_depth=config.depthVisualization.maxDepth)
        # print('    Mean Absolute Error in output depth (if Synthetic Data)   = {:.4f} cm'.format(error_output_depth))
        # print('    Mean Absolute Error in filtered depth (if Synthetic Data) = {:.4f} cm'.format(error_filtered_output_depth))

    """
    # Calculate Mean Errors over entire Dataset
    a1_mean = round(a1_mean / len(rgb_file_list), 2)
    a2_mean = round(a2_mean / len(rgb_file_list), 2)
    a3_mean = round(a3_mean / len(rgb_file_list), 2)
    rmse_mean = round(rmse_mean / len(rgb_file_list), 3)
    abs_rel_mean = round(abs_rel_mean / len(rgb_file_list), 3)
    mae_mean = round(mae_mean / len(rgb_file_list), 3)
    sq_rel_mean = round(sq_rel_mean / len(rgb_file_list), 3)

    print('\n\nMean Error Stats for Entire Dataset:')
    print('{:>15}:'.format('rmse_mean'), rmse_mean)
    print('{:>15}:'.format('abs_rel_mean'), abs_rel_mean)
    print('{:>15}:'.format('mae_mean'), mae_mean)
    print('{:>15}:'.format('a1.05_mean'), a1_mean)
    print('{:>15}:'.format('a1.10_mean'), a2_mean)
    print('{:>15}:'.format('a1.25_mean'), a3_mean)

    # Write the data into a csv file
    with open(os.path.join(results_dir, csv_filename), 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
        row_data = ['MEAN', rmse_mean, abs_rel_mean, mae_mean, a1_mean, a2_mean, a3_mean]
        writer.writerow(dict(zip(field_names, row_data)))
    """
    print(("Sim Eval Metric: \n" + metric.summary()))