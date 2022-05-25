import os
import os.path as osp

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), "../.."))

import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from tabulate import tabulate

from utils.geometry import cal_normal_map, depth2pts_np

with open("utils/objects.csv", "r") as f:
    OBJECT_INFO = csv.reader(f)
    OBJECT_INFO = list(OBJECT_INFO)[1:]
    OBJECT_NAMES = [_[0] for _ in OBJECT_INFO]

REAL_OBJECTS = [
    "coca_cola",
    "coffee_cup",
    "gold_ball",
    "jack_daniels",
    "spellegrino",
    "steel_ball",
    "tennis_ball",
    "voss",
]


class ErrorMetric(object):
    def __init__(
        self,
        use_mask: bool = True,
        max_disp: int = 192,
        depth_range=(0.2, 2.0),
        num_classes: int = 17,
        is_depth: bool = False,
    ):

        self.use_mask = use_mask
        self.max_disp = max_disp
        self.is_depth = is_depth
        self.depth_range = depth_range
        self.num_classes = num_classes
        assert len(OBJECT_NAMES) == num_classes


        self.cmap = plt.get_cmap("jet")

        # cutoff threshold
        self.disp_diff_threshold = 8
        self.depth_diff_threshold = 32e-3

    def reset(self):
        self.epe = []
        self.bad1 = []
        self.bad2 = []
        self.depth_abs_err = []
        self.depth_err2 = []
        self.depth_err4 = []
        self.depth_err8 = []
        self.normal_err = []
        self.normal_err10 = []
        self.normal_err20 = []
        self.obj_disp_err = np.zeros(self.num_classes)
        self.obj_depth_err = np.zeros(self.num_classes)
        self.obj_depth_err4 = np.zeros(self.num_classes)
        self.obj_normal_err = np.zeros(self.num_classes)
        self.obj_normal_err10 = np.zeros(self.num_classes)
        self.obj_count = np.zeros(self.num_classes)
        self.real_disp_err = 0.0
        self.real_depth_err = 0.0
        self.real_depth_err4 = 0.0
        self.real_normal_err = 0.0
        self.real_normal_err10 = 0.0
        self.real_count = 0
        self.print_disp_err = 0.0
        self.print_depth_err = 0.0
        self.print_depth_err4 = 0.0
        self.print_normal_err = 0.0
        self.print_normal_err10 = 0.0
        self.print_count = 0

    def compute(self, data_batch, pred_dict, save_folder="", real_data=False):
        """
        Compute the error metrics for predicted disparity map or depth map
        """
        #focal_length = data_batch["focal_length"][0].cpu().numpy()
        #baseline = data_batch["baseline"][0].cpu().numpy()

        prediction = pred_dict

        depth_gt = data_batch["img_depth_l"]
        disp_gt = data_batch["img_disp_l"]
        #disp_pred = focal_length * baseline / (prediction + 1e-7)
        depth_pred = prediction


        if self.use_mask:
            mask = np.logical_and(disp_gt > 1e-1, disp_gt < self.max_disp)
            x_base = np.arange(0, disp_gt.shape[1]).reshape(1, -1)
            mask = np.logical_and(mask, x_base > disp_gt)
            mask = np.logical_and(mask, depth_gt > self.depth_range[0])
            mask = np.logical_and(mask, depth_gt < self.depth_range[1])

        else:
            mask = np.ones_like(disp_gt).astype(np.bool)


        depth_diff = depth_gt - depth_pred
        #disp_diff = np.clip(disp_diff, -self.disp_diff_threshold, self.disp_diff_threshold)
        depth_diff = np.clip(depth_diff, -self.depth_diff_threshold, self.depth_diff_threshold)


        depth_abs_err = np.abs(depth_diff[mask]).mean()
        depth_err2 = (np.abs(depth_diff[mask]) > 2e-3).sum() / mask.sum()
        depth_err4 = (np.abs(depth_diff[mask]) > 4e-3).sum() / mask.sum()
        depth_err8 = (np.abs(depth_diff[mask]) > 8e-3).sum() / mask.sum()

        if "img_normal_l" in data_batch and "intrinsic_l" in data_batch:
            normal_gt = data_batch["img_normal_l"]
            #print(normal_gt.shape)
            valid_mask = np.abs(normal_gt).sum(-1) > 0
            if self.use_mask:
                valid_mask = np.logical_and(valid_mask, mask)
            invalid_mask = np.logical_not(valid_mask)
            intrinsic_l = data_batch["intrinsic_l"]
            intrinsic_l[1, 2] -= 2
            normal_pred = cal_normal_map(depth_pred, intrinsic_l)
            normal_err = np.arccos(np.clip(np.sum(normal_gt * normal_pred, axis=-1), -1, 1))
            normal_err[invalid_mask] = 0
            self.normal_err.append(normal_err.sum() / valid_mask.sum())
            self.normal_err10.append((normal_err > 10 / 180 * np.pi).sum() / valid_mask.sum())
            self.normal_err20.append((normal_err > 20 / 180 * np.pi).sum() / valid_mask.sum())

        if "img_label_l" in data_batch:
            label_l = data_batch["img_label_l"]
            for i in range(self.num_classes):
                obj_mask = label_l == i
                if self.use_mask:
                    obj_mask = np.logical_and(obj_mask, mask)
                if obj_mask.sum() > 0:
                    self.obj_count[i] += 1
                    self.obj_depth_err[i] += np.abs(depth_diff[obj_mask]).mean()
                    self.obj_depth_err4[i] += (np.abs(depth_diff[obj_mask]) > 4e-3).sum() / obj_mask.sum()

                    if OBJECT_NAMES[i] in REAL_OBJECTS:
                        self.real_count += 1
                        self.real_depth_err += np.abs(depth_diff[obj_mask]).mean()
                        self.real_depth_err4 += (np.abs(depth_diff[obj_mask]) > 4e-3).sum() / obj_mask.sum()
                    else:
                        self.print_count += 1
                        self.print_depth_err += np.abs(depth_diff[obj_mask]).mean()
                        self.print_depth_err4 += (np.abs(depth_diff[obj_mask]) > 4e-3).sum() / obj_mask.sum()

                    if "img_normal_l" in data_batch and "intrinsic_l" in data_batch:
                        self.obj_normal_err[i] += (normal_err[obj_mask]).mean()
                        self.obj_normal_err10[i] += (normal_err[obj_mask] > 10 / 180 * np.pi).sum() / obj_mask.sum()
                        if OBJECT_NAMES[i] in REAL_OBJECTS:
                            self.real_normal_err += (normal_err[obj_mask]).mean()
                            self.real_normal_err10 += (normal_err[obj_mask] > 10 / 180 * np.pi).sum() / obj_mask.sum()
                        else:
                            self.print_normal_err += (normal_err[obj_mask]).mean()
                            self.print_normal_err10 += (normal_err[obj_mask] > 10 / 180 * np.pi).sum() / obj_mask.sum()

        if save_folder:
            os.makedirs(save_folder, exist_ok=True)


            plt.imsave(
                os.path.join(save_folder, "depth_err.png"),
                depth_diff - 1e5 * (1 - mask),
                vmin=-16e-3,
                vmax=16e-3,
                cmap="jet",
            )
            if self.use_mask:
                plt.imsave(os.path.join(save_folder, "mask.png"), mask.astype(float), vmin=0.0, vmax=1.0, cmap="jet")
            if "intrinsic_l" in data_batch:
                intrinsic_l = data_batch["intrinsic_l"]
                intrinsic_l[1, 2] -= 2
                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(depth2pts_np(depth_gt, intrinsic_l))
                pcd_gt = pcd_gt.crop(
                    o3d.geometry.AxisAlignedBoundingBox(
                        min_bound=np.array([-10, -10, 0.1]), max_bound=np.array([10, 10, 1.8])
                    )
                )
                o3d.io.write_point_cloud(os.path.join(save_folder, "gt.pcd"), pcd_gt)
                pcd_pred = o3d.geometry.PointCloud()
                pcd_pred.points = o3d.utility.Vector3dVector(depth2pts_np(depth_pred, intrinsic_l))
                pcd_pred.colors = o3d.utility.Vector3dVector(
                    self.cmap(np.clip((depth_diff + 16e-3 - 1e5 * (1 - mask)) / 32e-3, 0, 1))[..., :3].reshape(-1, 3)
                )
                pcd_pred = pcd_pred.crop(
                    o3d.geometry.AxisAlignedBoundingBox(
                        min_bound=np.array([-10, -10, 0.1]), max_bound=np.array([10, 10, 1.8])
                    )
                )
                o3d.io.write_point_cloud(os.path.join(save_folder, "pred.pcd"), pcd_pred)

            if "img_normal_l" in data_batch and "intrinsic_l" in data_batch:
                cv2.imwrite(os.path.join(save_folder, "normal_gt.png"), ((normal_gt + 1) * 127.5).astype(np.uint8))
                cv2.imwrite(os.path.join(save_folder, "normal_pred.png"), ((normal_pred + 1) * 255).astype(np.uint8))
                plt.imsave(
                    os.path.join(save_folder, "normal_err.png"),
                    normal_err * mask,
                    vmin=0.0,
                    vmax=np.pi,
                    cmap="jet",
                )



            # 2D Discriminator

 
        self.depth_abs_err.append(depth_abs_err)
        self.depth_err2.append(depth_err2)
        self.depth_err4.append(depth_err4)
        self.depth_err8.append(depth_err8)

    def summary(self):
        s = ""
        headers = ["epe", "bad1", "bad2", "depth_abs_err", "depth_err2", "depth_err4", "depth_err8"]
        table = [
            [
                np.mean(self.epe),
                np.mean(self.bad1),
                np.mean(self.bad2),
                np.mean(self.depth_abs_err),
                np.mean(self.depth_err2),
                np.mean(self.depth_err4),
                np.mean(self.depth_err4),
            ]
        ]
        if self.normal_err:
            headers += ["norm_err", "norm_err10", "norm_err20"]
            table[0] += [np.mean(self.normal_err), np.mean(self.normal_err10), np.mean(self.normal_err20)]
        s += tabulate(table, headers=headers, tablefmt="fancy_grid", floatfmt=".4f")

        if self.obj_count.sum() > 0:
            headers = ["class_id", "name", "count", "disp_err", "depth_err", "depth_err4"]
            if self.normal_err:
                headers += ["obj_norm_err", "obj_norm_err10"]
            table = []
            for i in range(self.num_classes):
                t = [
                    i,
                    OBJECT_NAMES[i],
                    self.obj_count[i],
                    self.obj_disp_err[i] / (self.obj_count[i] + 1e-7),
                    self.obj_depth_err[i] / (self.obj_count[i] + 1e-7),
                    self.obj_depth_err4[i] / (self.obj_count[i] + 1e-7),
                ]
                if self.normal_err:
                    t += [
                        self.obj_normal_err[i] / (self.obj_count[i] + 1e-7),
                        self.obj_normal_err10[i] / (self.obj_count[i] + 1e-7),
                    ]
                table.append(t)
            t = [
                "-",
                "REAL",
                self.real_count,
                self.real_disp_err / (self.real_count + 1e-7),
                self.real_depth_err / (self.real_count + 1e-7),
                self.real_depth_err4 / (self.real_count + 1e-7),
            ]
            if self.normal_err:
                t += [
                    self.real_normal_err / (self.real_count + 1e-7),
                    self.real_normal_err10 / (self.real_count + 1e-7),
                ]
            table.append(t)
            t = [
                "-",
                "PRINT",
                self.print_count,
                self.print_disp_err / (self.print_count + 1e-7),
                self.print_depth_err / (self.print_count + 1e-7),
                self.print_depth_err4 / (self.print_count + 1e-7),
            ]
            if self.normal_err:
                t += [
                    self.print_normal_err / (self.print_count + 1e-7),
                    self.print_normal_err10 / (self.print_count + 1e-7),
                ]
            table.append(t)
            t = [
                "-",
                "ALL",
                self.print_count + self.real_count,
                (self.print_disp_err + self.real_disp_err) / (self.print_count + self.real_count + 1e-7),
                (self.print_depth_err + self.real_depth_err) / (self.print_count + self.real_count + 1e-7),
                (self.print_depth_err4 + self.real_depth_err4) / (self.print_count + self.real_count + 1e-7),
            ]
            if self.normal_err:
                t += [
                    (self.print_normal_err + self.real_normal_err) / (self.print_count + self.real_count + 1e-7),
                    (self.print_normal_err10 + self.real_normal_err10) / (self.print_count + self.real_count + 1e-7),
                ]
            table.append(t)
            s += "\n"
            s += tabulate(table, headers=headers, tablefmt="fancy_grid", floatfmt=".4f")

        return s


# Error metric for messy-table-dataset object error
def compute_obj_err(disp_gt, depth_gt, disp_pred, focal_length, baseline, label, mask, obj_total_num=17):
    """
    Compute error for each object instance in the scene
    :param disp_gt: GT disparity map, [bs, 1, H, W]
    :param depth_gt: GT depth map, [bs, 1, H, W]
    :param disp_pred: Predicted disparity map, [bs, 1, H, W]
    :param focal_length: Focal length, [bs, 1]
    :param baseline: Baseline of the camera, [bs, 1]
    :param label: Label of the image [bs, 1, H, W]
    :param obj_total_num: Total number of objects in the dataset
    :return: obj_disp_err, obj_depth_err - List of error of each object
             obj_count - List of each object appear count
    """
    depth_pred = focal_length * baseline / disp_pred  # in meters

    obj_list = label.unique()  # TODO this will cause bug if bs > 1, currently only for testing
    obj_num = obj_list.shape[0]

    # Array to store error and count for each object
    total_obj_disp_err = np.zeros(obj_total_num)
    total_obj_depth_err = np.zeros(obj_total_num)
    total_obj_depth_4_err = np.zeros(obj_total_num)
    total_obj_count = np.zeros(obj_total_num)

    for i in range(obj_num):
        obj_id = int(obj_list[i].item())
        obj_mask = label == obj_id
        obj_disp_err = F.l1_loss(disp_gt[obj_mask * mask], disp_pred[obj_mask * mask], reduction="mean").item()
        obj_depth_err = torch.clip(
            torch.abs(depth_gt[obj_mask * mask] * 1000 - depth_pred[obj_mask * mask] * 1000),
            min=0,
            max=100,
        )
        obj_depth_err = torch.mean(obj_depth_err).item()
        obj_depth_diff = torch.abs(depth_gt[obj_mask * mask] - depth_pred[obj_mask * mask])
        obj_depth_err4 = obj_depth_diff[obj_depth_diff > 4e-3].numel() / obj_depth_diff.numel()

        total_obj_disp_err[obj_id] += obj_disp_err
        total_obj_depth_err[obj_id] += obj_depth_err
        total_obj_depth_4_err[obj_id] += obj_depth_err4
        total_obj_count[obj_id] += 1
    return (
        total_obj_disp_err,
        total_obj_depth_err,
        total_obj_depth_4_err,
        total_obj_count,
    )


def save_prob_volume(
    prob_volume,
    file_path,
    threshold=0.01,
):
    d, h, w = prob_volume.shape
    custom_cmap = plt.get_cmap("jet")
    mask = (prob_volume > threshold).reshape(-1)

    color = custom_cmap(prob_volume)[..., :3].reshape(-1, 3)

    coor = np.zeros((d, h, w, 3))
    for i in range(h):
        coor[:, i, :, 0] = i
    for i in range(w):
        coor[:, :, i, 1] = i
    for i in range(d):
        coor[i, :, :, 2] = i

    coor = coor.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coor[mask])
    pcd.colors = o3d.utility.Vector3dVector(color[mask])

    o3d.io.write_point_cloud(file_path, pcd)
