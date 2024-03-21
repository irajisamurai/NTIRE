import sys
import torch as th
import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
import cv2
import csv
import tqdm
sys.path.insert(0, "../../")
from utils.metrics import PSNR, SSIM, LPIPS
from utils import logger

def write_scores_to_csv(out_path, start_step, pre_model, scores):
    header = ['pre-model', 'start step'] + list(scores.keys())
    mode = 'w' if not os.path.isfile(out_path) else 'a'
    with open(out_path, mode, newline='') as f:
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(header)
        writer.writerow([pre_model] + [start_step] + [i[0] for i in scores.values()])

def reverseChannel(img):
  # 画像コピー
  dst = img.copy()
  # チャネル入れ替え
  dst[:, :, 0] = img[:, :, 2]
  dst[:, :, 2] = img[:, :, 0]

  return dst

def compute_score(all_images_gt_raw, all_images_pred_raw, start_step, pre_model):
    metrics = ('psnr', 'ssim', 'lpips')
    device = 'cpu'
    boundary_ignore = 40
    metrics_all = {}
    scores = {}
    for m in ["lpips"]:
        if m == 'psnr':
            loss_fn = PSNR(boundary_ignore=boundary_ignore)
        elif m == 'ssim':
            loss_fn = SSIM(boundary_ignore=boundary_ignore, use_for_loss=False)
        elif m == 'lpips':
            loss_fn = LPIPS(boundary_ignore=boundary_ignore)
            loss_fn.to(device)
        else:
            raise ValueError(f"Unknown metric: {m}")
        metrics_all[m] = loss_fn
        scores[m] = []

    scores = {k: [] for k, v in scores.items()}
    all_images_gt_raw = th.cat(all_images_gt_raw)
    all_images_pred_raw = th.cat(all_images_pred_raw)

    for m, m_fn in metrics_all.items():
        metric_value = m_fn(all_images_pred_raw, all_images_gt_raw).cpu().item()
        scores[m].append(metric_value)
        logger.log(f"{m} is {metric_value}")

    out_path = os.path.join("official/lpips", f"score.csv")
    write_scores_to_csv(out_path, start_step, pre_model, scores)

gt_path = "/home/yokoyama/image-restoration-sde/codes/config/deshadow/official/gt"
pre_path = "/home/yokoyama/image-restoration-sde/codes/config/deshadow/official/pred"
device = "cuda:2" if th.cuda.is_available() else "cpu"
all_images_gt_raw = []
all_images_pred_raw = []
#img_list = os.listdir(gt_path)
img_list = os.listdir(pre_path)
for img in img_list:
    gt = os.path.join(gt_path,img)
    pre = os.path.join(pre_path,img)
    gt_img = cv2.imread(gt, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    pre_img = cv2.imread(pre, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    gt_img = th.unsqueeze(th.from_numpy(np.ascontiguousarray(np.transpose(gt_img, (2, 0, 1)))).float(), dim=0)[:, [2, 1, 0]]
    pre_img = th.unsqueeze(th.from_numpy(np.ascontiguousarray(np.transpose(pre_img, (2, 0, 1)))).float(), dim=0)[:, [2, 1, 0]]
    all_images_gt_raw.append(gt_img)
    all_images_pred_raw.append(pre_img)
compute_score(all_images_gt_raw, all_images_pred_raw, start_step="unknown", pre_model="unknown")