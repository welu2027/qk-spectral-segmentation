"""
segment.py — Step 1: segmentation + bounding box extraction.

Reads eigenvectors from --eigs_dir and writes segmentation masks + bboxes to
--output_dir. CPU-only, ~30-50 min for full VOC2012.

Run this first, then run eval.py.

──────────────────────────────────────────────────────────────────────────────
HOW THE EIGENVECTORS WERE CREATED (Kaggle, GPU required, already done):
──────────────────────────────────────────────────────────────────────────────

    from extract_multilayer_qk import extract_features_and_eigs

    # Hooks into DINO ViT blocks 2,5,8,11 (= layers 3,6,9,12).
    # For each block: computes Q+K (element-wise sum), concatenates across blocks.
    # Feature shape per image: (N, 4×384)  where N = H_patch × W_patch.
    #
    # SVD-based eigensolver — never builds N×N Laplacian:
    #   s = feats.sum(dim=0)                       # (D,)
    #   d = (feats @ s).clamp(min=1e-6)            # degree vector (N,)
    #   feats_tilde = feats * d.pow(-0.5)          # D^{-1/2} F
    #   U, S, _ = torch.svd_lowrank(feats_tilde, q=K+10, niter=4)
    #   eigenvalues  = 1 - S[:K]**2               # (K,)
    #   eigenvectors = U[:, :K].T                  # (K, N)
    #
    # vs. baseline: K-only last layer + eigsh on N×N Laplacian (O(N²·K))

    extract_features_and_eigs(
        images_list     = 'data/VOC2012/lists/images.txt',
        images_root     = '/path/to/VOC2012/JPEGImages',
        model_name      = 'dino_vits16',
        batch_size      = 1,
        eigs_output_dir = 'data/VOC2012/eigs/multilayer_svd',
        K               = 20,
        which_blocks    = '2,5,8,11',
    )

    # Zipped for download (split to stay under Kaggle 500MB limit):
    #   import zipfile, os
    #   files = sorted([f for f in os.listdir(EIGS_DIR) if f.endswith('.pth')])
    #   mid = len(files) // 2
    #   with zipfile.ZipFile('eigs_part1.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    #       for f in files[:mid]: z.write(f'{EIGS_DIR}/{f}', f)
    #   with zipfile.ZipFile('eigs_part2.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    #       for f in files[mid:]: z.write(f'{EIGS_DIR}/{f}', f)

──────────────────────────────────────────────────────────────────────────────
USAGE:
──────────────────────────────────────────────────────────────────────────────

    # 1. Unzip eigs from Google Drive:
    #       unzip eigs_part1.zip -d ~/Downloads/eigs
    #       unzip eigs_part2.zip -d ~/Downloads/eigs
    # 2. Run:
    python segment.py --eigs_dir ~/Downloads/eigs --voc_dir /path/to/VOC2012
"""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO / 'extract'))


def parse_args():
    p = argparse.ArgumentParser(description='Segmentation + bbox extraction (~30-50 min)')
    p.add_argument('--eigs_dir',     required=True, help='Directory of .pth eigenvector files')
    p.add_argument('--voc_dir',      required=True, help='VOC2012 root (contains JPEGImages/)')
    p.add_argument('--output_dir',   default='./output', help='Where to save outputs (default: ./output)')
    p.add_argument('--num_segments', type=int, default=4, help='Segments per image (default: 4)')
    return p.parse_args()


def main():
    args = parse_args()

    eigs_dir   = str(Path(args.eigs_dir).expanduser())
    output_dir = str(Path(args.output_dir).expanduser())
    images_root = str(Path(args.voc_dir).expanduser() / 'JPEGImages')

    assert Path(eigs_dir).exists(),    f'eigs_dir not found: {eigs_dir}'
    assert Path(images_root).exists(), f'JPEGImages not found: {images_root}'

    eig_files = list(Path(eigs_dir).glob('*.pth'))
    print(f'Found {len(eig_files)} eigenvector files')

    from extract import extract_multi_region_segmentations, extract_single_region_segmentations, extract_bboxes

    segs_dir       = f'{output_dir}/segmentations'
    single_seg_dir = f'{output_dir}/segmentations_single'
    bbox_file      = f'{output_dir}/bboxes.pth'

    os.makedirs(segs_dir, exist_ok=True)
    os.makedirs(single_seg_dir, exist_ok=True)

    print('\n[1/3] Multi-region segmentation...')
    extract_multi_region_segmentations(
        features_dir              = eigs_dir,
        eigs_dir                  = eigs_dir,
        output_dir                = segs_dir,
        adaptive                  = False,
        non_adaptive_num_segments = args.num_segments,
        infer_bg_index            = True,
        multiprocessing           = 0,
    )

    print('\n[2/3] Single-region segmentation...')
    extract_single_region_segmentations(
        features_dir    = eigs_dir,
        eigs_dir        = eigs_dir,
        output_dir      = single_seg_dir,
        threshold       = 0.0,
        multiprocessing = 0,
    )

    print('\n[3/3] Bounding box extraction...')
    extract_bboxes(
        features_dir      = eigs_dir,
        segmentations_dir = single_seg_dir,
        output_file       = bbox_file,
        num_erode         = 2,
        num_dilate        = 3,
        skip_bg_index     = True,
    )

    print(f'\nDone. Outputs saved to {output_dir}')
    print('Next: python eval.py --output_dir {output_dir} --voc_dir {voc_dir}')


if __name__ == '__main__':
    main()
