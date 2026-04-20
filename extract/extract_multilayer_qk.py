"""
Multi-Layer QK Spectral Segmentation

Improvements over deep-spectral-segmentation baseline:
  Features  : Q+K from layers 3,6,9,12  vs  K only from last layer
  Eigensolver: SVD on N×D matrix O(N·D·K)  vs  eigsh on N×N Laplacian O(N²·K)
  No color affinity (λ_knn) needed
  No CRF needed for clean results

Typical runtime on Kaggle 2×T4: ~5 min for full VOC2012
vs ~10 days on 100 CPUs with the baseline.
"""

from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
import extract_utils as utils


# ---------------------------------------------------------------------------
# Step 1: feature extraction
# ---------------------------------------------------------------------------

def extract_multilayer_features(
    images_list: str,
    images_root: Optional[str],
    model_name: str,
    batch_size: int,
    output_dir: str,
    which_blocks: str = '2,5,8,11',
):
    """
    Extract multi-layer Q+K features from DINO ViT.

    Hooks into QKV projections at which_blocks (0-indexed: 2,5,8,11 = layers
    3,6,9,12).  For each block, computes Q+K (element-wise sum to capture
    mutual attention signal), then concatenates across blocks.
    Output shape per image: (N, num_blocks × D) where N = H_patch × W_patch.

    Example:
        python extract_multilayer_qk.py extract_multilayer_features \\
            --images_list "./data/VOC2012/lists/images.txt" \\
            --images_root "./data/VOC2012/images" \\
            --output_dir "./data/VOC2012/features/dino_vits16_multilayer" \\
            --model_name dino_vits16 \\
            --batch_size 1
    """
    which_blocks = [int(b) for b in str(which_blocks).split(',')]

    utils.make_output_dir(output_dir)
    model_name = model_name.lower()
    model, val_transform, patch_size, num_heads = utils.get_model(model_name)

    # Hooks on each requested block
    feat_out = {b: {} for b in which_blocks}

    def make_hook(b):
        def _hook(module, input, output):
            feat_out[b]['qkv'] = output
        return _hook

    for b in which_blocks:
        (model._modules['blocks'][b]
              ._modules['attn']
              ._modules['qkv']
              .register_forward_hook(make_hook(b)))

    filenames = Path(images_list).read_text().splitlines()
    dataset = utils.ImagesDataset(
        filenames=filenames, images_root=images_root, transform=val_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=4)
    print(f'Dataset size: {len(dataset)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for images, files, indices in tqdm(dataloader, desc='Extracting features'):
        id_ = Path(files[0]).stem
        output_file = Path(output_dir) / f'{id_}.pth'
        if output_file.is_file():
            continue

        B, C, H, W = images.shape
        P = patch_size
        H_patch, W_patch = H // P, W // P
        H_pad, W_pad = H_patch * P, W_patch * P
        T = H_patch * W_patch + 1  # patches + CLS
        images = images[:, :, :H_pad, :W_pad].to(device)

        with torch.no_grad():
            model(images)  # forward pass — hooks fire for all blocks

        # Q+K per block, concat across blocks → (B, N, num_blocks * D)
        qk_layers = []
        for b in which_blocks:
            qkv = feat_out[b]['qkv']  # (B, T, 3*D)
            qkv = qkv.reshape(B, T, 3, num_heads, -1 // num_heads).permute(2, 0, 3, 1, 4)
            q = qkv[0].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]  # (B, N, D)
            k = qkv[1].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]  # (B, N, D)
            qk_layers.append(q + k)  # sum captures Q·K attention signal

        feats = torch.cat(qk_layers, dim=-1).cpu().half()  # (B, N, num_blocks*D) — float16 halves disk usage

        torch.save({
            'k': feats,        # 'k' key keeps downstream code compatible
            'k_multi': feats,
            'indices': indices[0],
            'file': files[0],
            'id': id_,
            'model_name': model_name,
            'patch_size': patch_size,
            'shape': (B, C, H, W),
        }, str(output_file))

    print(f'Saved features to {output_dir}')


# ---------------------------------------------------------------------------
# Step 2: SVD-based eigenvector extraction
# ---------------------------------------------------------------------------

def _extract_eig_svd(
    inp: Tuple[int, str],
    K: int,
    output_dir: str,
    normalize: bool = True,
):
    """
    Compute normalized-Laplacian eigenvectors via randomized SVD.

    Never materializes the N×N affinity / Laplacian matrix.

    Math:
      W = F·Fᵀ  (N×N, never built)
      d_i = Σ_j W_ij = F_i · s,  s = Σ_j F_j   [O(N·D)]
      F̃ = D^{-1/2} · F
      eigvecs(L_sym) = left singular vectors of F̃  [randomized SVD, O(N·D·K)]
    """
    index, features_file = inp
    data_dict = torch.load(features_file, map_location='cpu')
    image_id = data_dict['id']

    output_file = Path(output_dir) / f'{image_id}.pth'
    if output_file.is_file():
        return

    feats = data_dict['k_multi'].squeeze().float()  # (N, D) — cast back to float32 for SVD
    if normalize:
        feats = F.normalize(feats, p=2, dim=-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feats = feats.to(device)

    # Degree: d_i = F_i · (Σ_j F_j)  — no N×N matrix
    s = feats.sum(dim=0)                      # (D,)
    d = (feats @ s).clamp(min=1e-6)          # (N,)

    # Symmetric normalized features: rows of F̃ are D^{-1/2}_i · F_i
    feats_tilde = feats * d.pow(-0.5).unsqueeze(1)  # (N, D)

    # Randomized SVD: top-K left singular vectors of F̃
    # These are eigenvectors of L_sym = I - D^{-1/2}·W·D^{-1/2}
    U, S, _ = torch.svd_lowrank(feats_tilde, q=K + 10, niter=4)

    eigenvalues = (1.0 - S[:K] ** 2).cpu()   # λ_k = 1 − σ_k²
    eigenvectors = U[:, :K].T.cpu()           # (K, N)

    # Sign correction: majority of entries positive
    for i in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[i] > 0).float()).item() < 1.0:
            eigenvectors[i] = -eigenvectors[i]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors},
               str(output_file))


def extract_eigs_svd(
    features_dir: str,
    output_dir: str,
    K: int = 20,
    normalize: bool = True,
    multiprocessing: int = 0,
):
    """
    Run SVD-based eigenvector extraction on all feature files.

    Replaces eigsh(N×N Laplacian) with randomized SVD on (N×D) features.

    Example:
        python extract_multilayer_qk.py extract_eigs_svd \\
            --features_dir "./data/VOC2012/features/dino_vits16_multilayer" \\
            --output_dir "./data/VOC2012/eigs/multilayer_svd" \\
            --K 20
    """
    utils.make_output_dir(output_dir)
    fn = partial(_extract_eig_svd, K=K, output_dir=output_dir, normalize=normalize)
    inputs = list(enumerate(sorted(Path(features_dir).iterdir())))
    utils.parallel_process(inputs, fn, multiprocessing)


if __name__ == '__main__':
    import fire
    torch.set_grad_enabled(False)
    fire.Fire({
        'extract_multilayer_features': extract_multilayer_features,
        'extract_eigs_svd': extract_eigs_svd,
    })
