import os
from pathlib import Path

import numpy as np
from joblib import Parallel
from joblib.parallel import delayed
from PIL import Image
from scipy.optimize import linear_sum_assignment


def hungarian_match(flat_preds, flat_targets, preds_k, targets_k, metric='acc', n_jobs=16):
    assert (preds_k == targets_k)  # one to one
    num_k = preds_k

    # perform hungarian matching
    print('Using iou as metric')
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(get_iou)(
        flat_preds, flat_targets, c1, c2) for c2 in range(num_k) for c1 in range(num_k))
    results = np.array(results)
    results = results.reshape((num_k, num_k)).T
    match = linear_sum_assignment(flat_targets.shape[0] - results)
    match = np.array(list(zip(*match)))
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res


def majority_vote(flat_preds, flat_targets, preds_k, targets_k, n_jobs=16):
    iou_mat = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(get_iou)(
        flat_preds, flat_targets, c1, c2) for c2 in range(targets_k) for c1 in range(preds_k))
    iou_mat = np.array(iou_mat)
    results = iou_mat.reshape((targets_k, preds_k)).T
    results = np.argmax(results, axis=1)
    match = np.array(list(zip(range(preds_k), results)))
    return match


def get_iou(flat_preds, flat_targets, c1, c2):
    tp = 0
    fn = 0
    fp = 0
    tmp_all_gt = (flat_preds == c1)
    tmp_pred = (flat_targets == c2)
    tp += np.sum(tmp_all_gt & tmp_pred)
    fp += np.sum(~tmp_all_gt & tmp_pred)
    fn += np.sum(tmp_all_gt & ~tmp_pred)
    jac = float(tp) / max(float(tp + fp + fn), 1e-8)
    return jac


def eval_predictions(pred_dir: str, gt_dir: str, num_classes: int = 21, ignore_index: int = 255) -> float:
    """Compute mIoU between predicted semantic segmaps and VOC-style ground truth."""
    pred_files = sorted(Path(pred_dir).glob('*.png'))
    iou_per_class = np.zeros(num_classes)
    count_per_class = np.zeros(num_classes)

    for pred_path in pred_files:
        image_id = pred_path.stem
        gt_path = Path(gt_dir) / f'{image_id}.png'
        if not gt_path.exists():
            continue

        pred = np.array(Image.open(pred_path).convert('L'))
        gt = np.array(Image.open(gt_path))

        # Resize pred to GT resolution if they differ (pred is at patch resolution)
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred).resize(
                (gt.shape[1], gt.shape[0]), resample=Image.NEAREST))

        valid = gt != ignore_index
        flat_pred = pred[valid].astype(np.int32)
        flat_gt = gt[valid].astype(np.int32)

        for cls in range(num_classes):
            tp = np.sum((flat_pred == cls) & (flat_gt == cls))
            fp = np.sum((flat_pred == cls) & (flat_gt != cls))
            fn = np.sum((flat_pred != cls) & (flat_gt == cls))
            denom = tp + fp + fn
            if denom > 0:
                iou_per_class[cls] += float(tp) / denom
                count_per_class[cls] += 1

    valid_classes = count_per_class > 0
    miou = np.mean(iou_per_class[valid_classes] / count_per_class[valid_classes]) * 100
    return miou
