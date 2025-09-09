import torch
import torch.nn.functional as F
import numpy as np
import os, argparse, time
import imageio

from model.SeaNet_models import SeaNet
from data import test_dataset

# ----------------------------
# Metrics
# ----------------------------
def mae_metric(pred, gt):
    return np.mean(np.abs(pred - gt))

def iou_metric(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-8)

def f_measure(pred, gt, beta2=0.3):
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)

# S-measure (structural similarity for saliency maps)
def s_measure(pred, gt):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    alpha = 0.5
    # object-aware similarity
    fg = pred[gt == 1]
    bg = pred[gt == 0]
    o_fg = np.mean(fg) if fg.size > 0 else 0
    o_bg = np.mean(bg) if bg.size > 0 else 0
    object_score = alpha * o_fg + (1 - alpha) * (1 - o_bg)
    # region-aware similarity (divide into 4 regions)
    h, w = gt.shape
    y, x = h // 2, w // 2
    gt_quads = [gt[:y, :x], gt[:y, x:], gt[y:, :x], gt[y:, x:]]
    pr_quads = [pred[:y, :x], pred[:y, x:], pred[y:, :x], pred[y:, x:]]
    region_score = 0
    for gq, pq in zip(gt_quads, pr_quads):
        region_score += np.mean(1 - np.abs(pq - gq))
    region_score /= 4.0
    return 0.5 * (object_score + region_score)

# E-measure (enhanced alignment measure)
def e_measure(pred, gt):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    fm = np.mean(pred)
    gt_mean = np.mean(gt)
    align_matrix = 2 * (pred - fm) * (gt - gt_mean) / (
        (pred - fm) ** 2 + (gt - gt_mean) ** 2 + 1e-8
    )
    return np.mean((align_matrix + 1) ** 2 / 4)


# ----------------------------
# Main
# ----------------------------
torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=288, help='testing size')
opt = parser.parse_args()

# dataset paths
dataset_path = '/kaggle/input/eorssd'
image_root = os.path.join(dataset_path, 'test-images/')
gt_root = os.path.join(dataset_path, 'test-labels/')

# model
model = SeaNet()
model.load_state_dict(torch.load('/kaggle/working/SeaNet-main/models/SeaNet_EORSSD.pth.50'))
model.cuda()
model.eval()

save_path = './models/SeaNet/EORSSD/'
os.makedirs(save_path, exist_ok=True)

test_loader = test_dataset(image_root, gt_root, opt.testsize)

# store metrics
all_mae, all_iou, all_f, all_s, all_e = [], [], [], [], []
time_sum = 0

for i in range(test_loader.size):
    image, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)  # normalize GT to [0,1]

    image = image.cuda()
    time_start = time.time()
    res, *_ = model(image)
    time_end = time.time()
    time_sum += (time_end - time_start)

    res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)

    # -------- Metrics --------
    mae = mae_metric(res, gt)
    all_mae.append(mae)

    pred_mask = (res >= 0.5).astype(np.uint8)
    gt_mask = (gt >= 0.5).astype(np.uint8)

    all_iou.append(iou_metric(pred_mask, gt_mask))
    all_f.append(f_measure(pred_mask, gt_mask))
    all_s.append(s_measure(res, gt))
    all_e.append(e_measure(res, gt))

    # -------- Save --------
    save_img = (res * 255).astype(np.uint8)
    imageio.imsave(save_path + name, save_img)

    if i == test_loader.size - 1:
        print('Running time {:.5f}'.format(time_sum / test_loader.size))
        print('FPS {:.5f}'.format(test_loader.size / time_sum))

# ----------------------------
# Final Results
# ----------------------------
print("\n==== Evaluation Results on EORSSD ====")
print(f"MAE:       {np.mean(all_mae):.4f}")
print(f"IoU:       {np.mean(all_iou):.4f}")
print(f"F-measure: {np.mean(all_f):.4f}")
print(f"S-measure: {np.mean(all_s):.4f}")
print(f"E-measure: {np.mean(all_e):.4f}")
