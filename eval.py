"""
eval.py — Step 2: benchmarks + visualization.

Reads segmentations and bboxes written by segment.py and evaluates:
  - CorLoc   (object localization, IoU >= 0.5)
  - Jaccard  (object segmentation)
  - mIoU     (semantic segmentation, 21 VOC classes)

CPU-only, ~10 min (plus ~30-60 min for mIoU if --semantic is passed).

──────────────────────────────────────────────────────────────────────────────
USAGE:
──────────────────────────────────────────────────────────────────────────────

    # After running segment.py:
    python eval.py --eigs_dir ~/Downloads/eigs --output_dir ./output --voc_dir /path/to/VOC2012

    # Skip slow mIoU and just get CorLoc + Jaccard:
    python eval.py --eigs_dir ~/Downloads/eigs --output_dir ./output --voc_dir /path/to/VOC2012 --no_semantic

    # Skip visualization (headless / server):
    python eval.py --eigs_dir ~/Downloads/eigs --output_dir ./output --voc_dir /path/to/VOC2012 --no_vis
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as PILImage
from skimage.color import label2rgb

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO / 'extract'))


def parse_args():
    p = argparse.ArgumentParser(description='Benchmark evaluation (~10 min + optional mIoU)')
    p.add_argument('--eigs_dir',    required=True, help='Directory of .pth eigenvector files')
    p.add_argument('--output_dir',  required=True, help='output_dir used in segment.py')
    p.add_argument('--voc_dir',     required=True, help='VOC2012 root (contains Annotations/, SegmentationClass/, etc.)')
    p.add_argument('--no_semantic', action='store_true', help='Skip mIoU eval (~30-60 min extra)')
    p.add_argument('--no_vis',      action='store_true', help='Skip visualization')
    p.add_argument('--n_vis',       type=int, default=6, help='Number of images to visualize (default: 6)')
    return p.parse_args()


# ── CorLoc ────────────────────────────────────────────────────────────────────

def eval_corloc(bbox_file, anno_root):
    print('\n[CorLoc] Object localization (IoU >= 0.5)...')

    def iou(b1, b2):
        xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def gt_boxes(anno_path):
        root = ET.parse(anno_path).getroot()
        return [
            [int(obj.find('bndbox').find(t).text) for t in ('xmin', 'ymin', 'xmax', 'ymax')]
            for obj in root.findall('object')
        ]

    bbox_list = torch.load(bbox_file, weights_only=False)
    correct, total = 0, 0
    for item in bbox_list:
        if not item['bboxes_original_resolution']:
            continue
        pred = item['bboxes_original_resolution'][0]
        anno = f'{anno_root}/{item["id"]}.xml'
        if not Path(anno).exists():
            continue
        gts = gt_boxes(anno)
        if not gts:
            continue
        correct += int(max(iou(pred, g) for g in gts) >= 0.5)
        total += 1

    corloc = 100.0 * correct / total if total > 0 else 0.0
    print(f'  CorLoc: {corloc:.1f}%  ({correct}/{total})  |  baseline ~50-55%')
    return corloc


# ── Jaccard ───────────────────────────────────────────────────────────────────

def eval_jaccard(single_seg_dir, seg_obj_root):
    if not Path(seg_obj_root).exists():
        print('[Jaccard] SegmentationObject GT not found — skipping.')
        return None
    sys.path.insert(0, str(REPO / 'object-segmentation'))
    from metrics import compute_jaccard
    print('\n[Jaccard] Object segmentation...')
    jacc = compute_jaccard(pred_dir=single_seg_dir, gt_dir=seg_obj_root, threshold=0.5)
    print(f'  Jaccard: {jacc:.1f}%  |  baseline ~35-40%')
    return jacc


# ── mIoU ─────────────────────────────────────────────────────────────────────

def eval_miou(eigs_dir, segs_dir, bbox_file, output_dir, images_root, seg_class_root):
    if not Path(seg_class_root).exists():
        print('[mIoU] SegmentationClass GT not found — skipping.')
        return None

    from extract import extract_bbox_features, extract_bbox_clusters, extract_semantic_segmentations

    bbox_feat_file    = f'{output_dir}/bbox_features.pth'
    bbox_cluster_file = f'{output_dir}/bbox_clusters.pth'
    sem_segs_dir      = f'{output_dir}/semantic_segmentations'
    os.makedirs(sem_segs_dir, exist_ok=True)

    print('\n[mIoU] Semantic segmentation...')
    print('  Extracting bbox features (DINO CLS token)...')
    extract_bbox_features(
        images_root = images_root,
        bbox_file   = bbox_file,
        model_name  = 'dino_vits16',
        output_file = bbox_feat_file,
    )

    print('  Clustering into 21 semantic categories...')
    extract_bbox_clusters(
        bbox_features_file = bbox_feat_file,
        output_file        = bbox_cluster_file,
        num_clusters       = 21,
        pca_dim            = 32,
        seed               = 0,
    )

    print('  Assigning semantic labels to segmentation maps...')
    extract_semantic_segmentations(
        segmentations_dir  = segs_dir,
        bbox_clusters_file = bbox_cluster_file,
        output_dir         = sem_segs_dir,
    )

    sys.path.insert(0, str(REPO / 'semantic-segmentation'))
    from eval_utils import eval_predictions
    miou = eval_predictions(pred_dir=sem_segs_dir, gt_dir=seg_class_root, num_classes=21)
    print(f'  mIoU: {miou:.1f}%  |  baseline ~15-18%')
    return miou


# ── visualization ─────────────────────────────────────────────────────────────

def visualize(eigs_dir, segs_dir, images_root, n_vis):
    seg_files = sorted(Path(segs_dir).glob('*.png'))[:n_vis]
    if not seg_files:
        print('No segmentation files found.')
        return

    print(f'\n[Vis] Showing {len(seg_files)} images...')
    for seg_file in seg_files:
        image_id = seg_file.stem
        image    = np.array(PILImage.open(f'{images_root}/{image_id}.jpg').convert('RGB'))
        segmap   = np.array(PILImage.open(seg_file))
        seg_full = cv2.resize(segmap, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        labels  = np.unique(seg_full)
        colors  = [plt.cm.tab10.colors[i % 10] for i in labels if i != 0]
        overlay = label2rgb(seg_full, image=image, colors=colors, bg_label=0, alpha=0.45)

        eig_data = torch.load(f'{eigs_dir}/{image_id}.pth', map_location='cpu', weights_only=False)
        eigvecs  = eig_data['eigenvectors']
        H_p, W_p = segmap.shape[:2]
        n_show   = min(4, eigvecs.shape[0] - 1)

        fig, axes = plt.subplots(1, 2 + n_show, figsize=(4 * (2 + n_show), 4))
        axes[0].imshow(image);   axes[0].set_title('Image');        axes[0].axis('off')
        axes[1].imshow(overlay); axes[1].set_title('Segmentation'); axes[1].axis('off')
        for j in range(n_show):
            ev = eigvecs[j + 1].numpy().reshape(H_p, W_p)
            axes[2 + j].imshow(ev, cmap='RdBu_r')
            axes[2 + j].set_title(f'Eigvec {j + 1}')
            axes[2 + j].axis('off')
        plt.suptitle(image_id, fontsize=12)
        plt.tight_layout()
        plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    eigs_dir   = str(Path(args.eigs_dir).expanduser())
    output_dir = str(Path(args.output_dir).expanduser())
    voc_dir    = str(Path(args.voc_dir).expanduser())

    images_root    = f'{voc_dir}/JPEGImages'
    anno_root      = f'{voc_dir}/Annotations'
    seg_obj_root   = f'{voc_dir}/SegmentationObject'
    seg_class_root = f'{voc_dir}/SegmentationClass'

    segs_dir       = f'{output_dir}/segmentations'
    single_seg_dir = f'{output_dir}/segmentations_single'
    bbox_file      = f'{output_dir}/bboxes.pth'

    assert Path(bbox_file).exists(),  f'bboxes.pth not found — run segment.py first'
    assert Path(segs_dir).exists(),   f'segmentations/ not found — run segment.py first'

    corloc = eval_corloc(bbox_file, anno_root)
    jacc   = eval_jaccard(single_seg_dir, seg_obj_root)
    miou   = None
    if not args.no_semantic:
        miou = eval_miou(eigs_dir, segs_dir, bbox_file, output_dir, images_root, seg_class_root)

    print('\n── Results ─────────────────────────────────────────────────────────')
    if corloc is not None: print(f'  CorLoc : {corloc:.1f}%  (baseline ~50-55%)')
    if jacc   is not None: print(f'  Jaccard: {jacc:.1f}%   (baseline ~35-40%)')
    if miou   is not None: print(f'  mIoU   : {miou:.1f}%   (baseline ~15-18%)')

    if not args.no_vis:
        visualize(eigs_dir, segs_dir, images_root, args.n_vis)


if __name__ == '__main__':
    main()
