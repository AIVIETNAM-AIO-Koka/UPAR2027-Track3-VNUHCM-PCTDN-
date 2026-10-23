from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Multi-threading for CPU operations
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "assets", "vitpose.torchscript")
META_PATH = os.path.join(HERE, "assets", "preprocess.json")
PRIOR_PATH = os.path.join(HERE, "assets", "mean_pose.json")

# Hyperparameters
BATCH_SIZE = 32
IN_WIDTH = 288
IN_HEIGHT = 384
HM_WIDTH = 72
HM_HEIGHT = 96
TARGET_ASPECT = float(IN_HEIGHT) / float(IN_WIDTH)  # 1.333333

# PoseTrack18 17-keypoint horizontal flip indices
# 0:nose, 1:head_bottom, 2:head_top, 3:l_ear, 4:r_ear, 5:l_sho, 6:r_sho,
# 7:l_elb, 8:r_elb, 9:l_wri, 10:r_wri, 11:l_hip, 12:r_hip, 13:l_knee,
# 14:r_knee, 15:l_ank, 16:r_ank
FLIP_INDEX = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# 15 active evaluation keypoints (ears 3, 4 have 0 ground truth annotations in UPAR)
EVAL_INDICES = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Global cache
_MODEL = None
_DEVICE = None
_META: dict[str, Any] = {}
_PRIOR: list[list[float]] | None = None
_MEAN: torch.Tensor | None = None
_STD: torch.Tensor | None = None
_BASE_GRID: torch.Tensor | None = None
_MOD_GX: torch.Tensor | None = None
_MOD_GY: torch.Tensor | None = None
_RESCORE_WEIGHTS: torch.Tensor | None = None


def load_model() -> None:
    global _MODEL, _DEVICE, _META, _PRIOR, _MEAN, _STD, _BASE_GRID, _MOD_GX, _MOD_GY, _RESCORE_WEIGHTS
    if _MODEL is not None:
        return

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} is missing.")

    with open(META_PATH, "r", encoding="utf-8") as fh:
        _META = json.load(fh)
    with open(PRIOR_PATH, "r", encoding="utf-8") as fh:
        _PRIOR = json.load(fh)["mean_pose_normalized"]

    _DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Precomputed normalization tensors
    mean_vals = _META.get("image_mean", [0.485, 0.456, 0.406])
    std_vals = _META.get("image_std", [0.229, 0.224, 0.225])
    _MEAN = torch.tensor(mean_vals, device=_DEVICE, dtype=torch.float32).view(1, 3, 1, 1)
    _STD = torch.tensor(std_vals, device=_DEVICE, dtype=torch.float32).view(1, 3, 1, 1)

    # Base grid for UDP-unbiased affine crop
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, IN_HEIGHT, device=_DEVICE, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, IN_WIDTH, device=_DEVICE, dtype=torch.float32),
        indexing="ij",
    )
    _BASE_GRID = torch.stack([gx, gy], dim=-1)  # [384, 288, 2]

    # Precomputed 1D Gaussian kernel for separable Re-Modulation (k=9, sigma=2.0)
    kx = torch.arange(9, device=_DEVICE, dtype=torch.float32) - 4.0
    kg = torch.exp(-kx**2 / (2.0 * 2.0**2))
    _MOD_GX = (kg / kg.sum()).view(1, 1, 1, 9)
    _MOD_GY = _MOD_GX.view(1, 1, 9, 1)

    # OKS-sensitivity instance rescoring weights for 15 active evaluation keypoints
    # Head (0, 1, 2): 1.35x, Wrists (9, 10): 1.15x, Torso: 0.80x, Limbs: 1.00x
    weights_15 = [
        1.35, 1.35, 1.35,  # 0: nose, 1: head_bottom, 2: head_top
        0.80, 0.80,        # 5: l_sho, 6: r_sho
        1.00, 1.00,        # 7: l_elb, 8: r_elb
        1.15, 1.15,        # 9: l_wri, 10: r_wri
        0.80, 0.80,        # 11: l_hip, 12: r_hip
        1.00, 1.00,        # 13: l_knee, 14: r_knee
        1.15, 1.15,        # 15: l_ank, 16: r_ank
    ]
    _RESCORE_WEIGHTS = torch.tensor(weights_15, device=_DEVICE, dtype=torch.float32).unsqueeze(0)

    # Load TorchScript model
    _MODEL = torch.jit.load(MODEL_PATH, map_location=_DEVICE)
    _MODEL.eval()
    print(f"[Champion-V6-Rescorer] ViTPose-Large 384x288 ready on {_DEVICE}", flush=True)


def _prior_pose(box: List[float]) -> List[List[float]]:
    bx, by, bw, bh = [float(v) for v in box[:4]]
    return [[bx + u * bw, by + v * bh, 0.05] for u, v in _PRIOR]


def _resolve_image_path(sample: Dict[str, Any]) -> str:
    image_path = sample.get("image_path")
    if image_path and os.path.isfile(image_path):
        return image_path
    file_name = sample.get("file_name")
    if file_name and os.path.isfile(file_name):
        return file_name
    raise FileNotFoundError("Could not resolve image path for sample.")


def _compute_crop_geometry(
    box: List[float], img_w: int, img_h: int, custom_pad: float | None = None
) -> Tuple[float, float, float, float]:
    """
    Computes Top-Down expansion parameters (cx, cy, box_w, box_h).
    Scale-adaptive padding with vertical border re-centering for truncated heads/ankles.
    """
    bx, by, bw, bh = [float(v) for v in box[:4]]
    bw = max(bw, 1.0)
    bh = max(bh, 1.0)
    cx = bx + bw * 0.5
    cy = by + bh * 0.5

    # Match target aspect ratio: 288 / 384 = 0.75 (TARGET_ASPECT = 384 / 288 = 1.333333)
    aspect = float(IN_WIDTH) / float(IN_HEIGHT)  # 0.75
    if bw > aspect * bh:
        bh = bw / aspect
    else:
        bw = bh * aspect

    area = bw * bh
    is_large = (area >= 9216.0) or (bh >= 220.0)

    # Vertical border re-centering: ensures border joints (head apex, feet) have margin
    if is_large:
        if (by + bh) >= float(img_h) - 4.0:
            cy = by + bh * 0.52
        elif by <= 4.0:
            cy = by + bh * 0.48

    if custom_pad is not None:
        padding_factor = custom_pad
    elif is_large:
        padding_factor = 1.10
    else:
        padding_factor = 1.25

    box_w = bw * padding_factor
    box_h = bh * padding_factor

    cx = max(0.0, min(float(cx), float(img_w)))
    cy = max(0.0, min(float(cy), float(img_h)))

    return cx, cy, box_w, box_h


def _preprocess_batch_gpu(
    img_tensor: torch.Tensor,
    boxes: List[List[float]],
    img_w: int,
    img_h: int,
    custom_pads: List[float] | None = None,
) -> Tuple[torch.Tensor, List[Tuple[float, float, float, float]]]:
    """
    Executes UDP-unbiased affine crop with zero padding on GPU in <0.3ms matching training distribution.
    """
    B = len(boxes)
    meta_list: List[Tuple[float, float, float, float]] = []
    grids = []

    for idx, box in enumerate(boxes):
        cpad = custom_pads[idx] if custom_pads is not None else None
        cx, cy, bw, bh = _compute_crop_geometry(box, img_w, img_h, custom_pad=cpad)
        meta_list.append((cx, cy, bw, bh))

        src_x = cx + _BASE_GRID[..., 0] * (bw * 0.5)
        src_y = cy + _BASE_GRID[..., 1] * (bh * 0.5)

        norm_x = (2.0 * src_x / max(img_w - 1.0, 1.0)) - 1.0
        norm_y = (2.0 * src_y / max(img_h - 1.0, 1.0)) - 1.0
        grids.append(torch.stack([norm_x, norm_y], dim=-1))

    batch_grid = torch.stack(grids, dim=0)  # [B, 384, 288, 2]

    # Bilinear grid_sample with zero padding matching training distribution
    crops = F.grid_sample(
        img_tensor.expand(B, -1, -1, -1),
        batch_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    # Standardize
    crops = (crops - _MEAN) / _STD
    return crops, meta_list


def _modulate_heatmaps(hm: torch.Tensor) -> torch.Tensor:
    """
    Separable 2D Gaussian Re-Modulation (Pillar 3) to restore ideal Gaussian curvature
    after bilateral Flip TTA averaging. Eliminates sub-pixel Hessian drift on head & wrists.
    """
    B, K, H, W = hm.shape
    x = hm.view(B * K, 1, H, W)
    m0 = x.amax(dim=(2, 3), keepdim=True)
    x = F.pad(x, (4, 4, 4, 4), mode="replicate")
    x = F.conv2d(x, _MOD_GX)
    x = F.conv2d(x, _MOD_GY)
    m1 = x.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    return (x * (m0 / m1)).view(B, K, H, W)


def _compute_flip_agreement(hm_orig: torch.Tensor, hm_flip: torch.Tensor) -> torch.Tensor:
    """
    Computes bilateral flip spatial consistency (Pillar 3).
    High agreement on wrists and cranial apex boosts instance confidence.
    """
    B, K, H, W = hm_orig.shape
    idx_orig = hm_orig.view(B, K, -1).argmax(dim=-1)
    idx_flip = hm_flip.view(B, K, -1).argmax(dim=-1)
    ox, oy = (idx_orig % W).float(), (idx_orig // W).float()
    fx, fy = (idx_flip % W).float(), (idx_flip // W).float()
    dist_sq = (ox - fx)**2 + (oy - fy)**2
    # Spatial agreement with sigma = 2.0 heatmap pixels (~8 image pixels)
    return torch.exp(-dist_sq / (2.0 * 4.0))  # [B, K]


def _decode_heatmaps_dark_hessian(
    heatmaps: torch.Tensor,
    meta_list: List[Tuple[float, float, float, float]],
    agreement: torch.Tensor,
) -> Tuple[np.ndarray, List[float]]:
    """
    Sub-Pixel Decoding via Re-Modulation + 2D Log-Space DARK Hessian Refinement +
    Truncation-Protected OKS-Calibrated Instance Rescoring.
    """
    B, num_kpts, H, W = heatmaps.shape

    # 1. Re-Modulate heatmaps to restore ideal Gaussian curvature (Pillar 3)
    hm_smooth = _modulate_heatmaps(heatmaps)

    # Find coarse mode coordinates
    flat = hm_smooth.view(B, num_kpts, -1)
    max_scores, flat_idx = flat.max(dim=2)
    my = flat_idx // W
    mx = flat_idx % W

    # 2. 2D Log-Space DARK Hessian Refinement
    padded_smooth = F.pad(hm_smooth, (1, 1, 1, 1), mode="replicate")
    log_hm = torch.log(torch.clamp(padded_smooth, min=1e-7))

    b_idx = torch.arange(B, device=_DEVICE)[:, None]
    k_idx = torch.arange(num_kpts, device=_DEVICE)[None, :]
    px = mx + 1
    py = my + 1

    L_c = log_hm[b_idx, k_idx, py, px]
    L_xp = log_hm[b_idx, k_idx, py, px + 1]
    L_xm = log_hm[b_idx, k_idx, py, px - 1]
    L_yp = log_hm[b_idx, k_idx, py + 1, px]
    L_ym = log_hm[b_idx, k_idx, py - 1, px]

    L_xp_yp = log_hm[b_idx, k_idx, py + 1, px + 1]
    L_xp_ym = log_hm[b_idx, k_idx, py - 1, px + 1]
    L_xm_yp = log_hm[b_idx, k_idx, py + 1, px - 1]
    L_xm_ym = log_hm[b_idx, k_idx, py - 1, px - 1]

    dx = 0.5 * (L_xp - L_xm)
    dy = 0.5 * (L_yp - L_ym)
    dxx = (L_xp - 2.0 * L_c + L_xm).clamp(max=-1e-4)
    dyy = (L_yp - 2.0 * L_c + L_ym).clamp(max=-1e-4)
    dxy = 0.25 * (L_xp_yp - L_xp_ym - L_xm_yp + L_xm_ym)

    det = (dxx * dyy - dxy * dxy).clamp(min=1e-6)
    offset_x = (-(dyy * dx - dxy * dy) / det).clamp(-0.5, 0.5)
    offset_y = (-(-dxy * dx + dxx * dy) / det).clamp(-0.5, 0.5)

    final_x = mx.float() + offset_x
    final_y = my.float() + offset_y

    # 3. Peak Sharpness modulated by Bilateral Flip Agreement
    m3 = F.avg_pool2d(heatmaps, kernel_size=3, stride=1, padding=1)[b_idx, k_idx, my, mx] * 9.0
    m7 = F.avg_pool2d(heatmaps, kernel_size=7, stride=1, padding=3)[b_idx, k_idx, my, mx] * 49.0
    sharpness = (m3 / torch.clamp(m7, min=1e-6)).clamp(0.1, 1.0).pow(0.25)
    kpt_scores = (max_scores * sharpness * (0.85 + 0.15 * agreement)).clamp(0.0, 1.0)

    # Strictly mask ears (3, 4) to eliminate noise on unannotated UPAR keypoints
    kpt_scores[:, 3] = 0.0
    kpt_scores[:, 4] = 0.0

    # 4. Map back to original image space via continuous coordinates
    coords_out = np.zeros((B, num_kpts, 3), dtype=np.float32)
    final_x_cpu = final_x.cpu().numpy()
    final_y_cpu = final_y.cpu().numpy()
    kpt_scores_cpu = kpt_scores.cpu().numpy()

    for i in range(B):
        cx, cy, bw, bh = meta_list[i]
        coords_out[i, :, 0] = cx + (final_x_cpu[i] / (HM_WIDTH - 1.0) - 0.5) * bw
        coords_out[i, :, 1] = cy + (final_y_cpu[i] / (HM_HEIGHT - 1.0) - 0.5) * bh
        coords_out[i, :, 2] = kpt_scores_cpu[i]

    # 5. OKS-Calibrated Instance Rescoring with Truncation Protection
    eval_scores = kpt_scores[:, EVAL_INDICES]  # [B, 15]
    vis_mask = (eval_scores > 0.20).float()
    vis_weights = _RESCORE_WEIGHTS * vis_mask
    vis_sum = vis_weights.sum(dim=1).clamp(min=1e-5)
    vis_term = (vis_weights * eval_scores).sum(dim=1) / vis_sum
    power_term = ((vis_weights * eval_scores.pow(1.5)).sum(dim=1) / vis_sum).pow(1.0 / 1.5)

    has_vis = (vis_mask.sum(dim=1) > 0).float()
    fallback = torch.topk(eval_scores, 8, dim=1).values.mean(dim=1)
    # Calibrated coverage threshold: avoids over-penalizing valid half-body crops
    coverage = (vis_mask.sum(dim=1) / 6.0).clamp(0.85, 1.0)
    person_scores_tensor = (has_vis * (0.65 * vis_term + 0.35 * power_term) + (1.0 - has_vis) * fallback) * coverage
    person_scores = [float(s) for s in person_scores_tensor.cpu().numpy()]

    return coords_out, person_scores


RESCORER_PATH = os.path.join(HERE, "assets", "rescorer.json")
_RESCORER = None
_RESCORER_TRIED = False


def _ensure_rescorer() -> None:
    """Loads the optional learned rescorer. Without assets/rescorer.json the v5 score is used unchanged."""
    global _RESCORER, _RESCORER_TRIED
    if _RESCORER_TRIED:
        return
    _RESCORER_TRIED = True
    if os.path.isfile(RESCORER_PATH):
        with open(RESCORER_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        _RESCORER = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else v) for k, v in raw.items()}
        print("[rescorer] learned rescorer loaded", flush=True)


def _rescore(feats: np.ndarray) -> np.ndarray:
    r = _RESCORER
    x = (feats - r["mu"]) / r["sd"]
    h = np.maximum(x @ r["W1"] + r["b1"], 0.0)
    h = np.maximum(h @ r["W2"] + r["b2"], 0.0)
    z = h @ r["W3"] + r["b3"]
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    return p.mean(axis=1)


def _is_large(box: List[float]) -> bool:
    bw, bh = float(box[2]), float(box[3])
    return (bw * bh >= 9216.0) or (bh >= 220.0)


@torch.no_grad()
def _extract(items: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[float], np.ndarray]:
    """
    Shared by inference and by the Kaggle calibration notebook so features are identical.
    items: dicts with img (1,3,H,W float tensor in [0,1] on _DEVICE), box [x,y,w,h], W, H.
    Returns fused keypoints (B,17,3), v5 person scores, and an (B,84) feature matrix.
    Keypoints and v5 scores are exactly what v5 produced.
    """
    B = len(items)
    flip_idx = torch.tensor(FLIP_INDEX, device=_DEVICE, dtype=torch.long)
    large = [_is_large(it["box"]) for it in items]

    crops, metas = [], []
    for it, lg in zip(items, large):
        c, m = _preprocess_batch_gpu(it["img"], [it["box"]], it["W"], it["H"],
                                     custom_pads=[1.10 if lg else 1.25])
        crops.append(c)
        metas.append(m[0])
    crops = torch.cat(crops, dim=0)

    hm = _MODEL(torch.cat([crops, torch.flip(crops, dims=[-1])], dim=0)).float()
    ho = hm[:B]
    hf = torch.flip(hm[B:], dims=[-1])[:, flip_idx]
    agr = _compute_flip_agreement(ho, hf)
    c1, s1 = _decode_heatmaps_dark_hessian(0.5 * (ho + hf), metas, agr)
    co, _ = _decode_heatmaps_dark_hessian(ho, metas, agr)
    cf, _ = _decode_heatmaps_dark_hessian(hf, metas, agr)

    final = c1.copy()
    base = [float(v) for v in s1]
    s2_kpt = c1[:, :, 2].copy()
    dpass = np.zeros((B, c1.shape[1]), dtype=np.float32)

    li = [i for i, lg in enumerate(large) if lg]
    if li:
        crops2, metas2 = [], []
        for i in li:
            c, m = _preprocess_batch_gpu(items[i]["img"], [items[i]["box"]], items[i]["W"], items[i]["H"],
                                         custom_pads=[1.30])
            crops2.append(c)
            metas2.append(m[0])
        crops2 = torch.cat(crops2, dim=0)
        n2 = len(li)
        hm2 = _MODEL(torch.cat([crops2, torch.flip(crops2, dims=[-1])], dim=0)).float()
        ho2 = hm2[:n2]
        hf2 = torch.flip(hm2[n2:], dims=[-1])[:, flip_idx]
        agr2 = _compute_flip_agreement(ho2, hf2)
        c2, sc2 = _decode_heatmaps_dark_hessian(0.5 * (ho2 + hf2), metas2, agr2)
        for k, j in enumerate(li):
            w1 = np.maximum(c1[j, :, 2:3], 0.05)
            w2 = np.maximum(c2[k, :, 2:3], 0.05)
            final[j, :, :2] = (w1 * c1[j, :, :2] + w2 * c2[k, :, :2]) / (w1 + w2)
            final[j, :, 2] = np.maximum(c1[j, :, 2], c2[k, :, 2])
            base[j] = max(base[j], float(sc2[k]))
            s2_kpt[j] = c2[k, :, 2]
            norm = math.sqrt(metas[j][2] * metas[j][3])
            dpass[j] = np.linalg.norm(c1[j, :, :2] - c2[k, :, :2], axis=1) / norm

    norm_all = np.sqrt(np.array([m[2] * m[3] for m in metas], dtype=np.float32))[:, None]
    dflip = np.linalg.norm(co[:, :, :2] - cf[:, :, :2], axis=2) / norm_all

    sel = EVAL_INDICES
    s1k = c1[:, sel, 2]
    s2k = s2_kpt[:, sel]
    agk = agr.cpu().numpy()[:, sel]
    dfk = dflip[:, sel]
    dpk = dpass[:, sel]

    geom = np.zeros((B, 9), dtype=np.float32)
    for j, it in enumerate(items):
        bx, by, bw, bh = [float(v) for v in it["box"][:4]]
        bw, bh = max(bw, 1.0), max(bh, 1.0)
        border = (bx <= 4.0) or (by <= 4.0) or (bx + bw >= it["W"] - 4.0) or (by + bh >= it["H"] - 4.0)
        geom[j] = [math.log(bw), math.log(bh), math.log(bh / bw), float(large[j]), float(border),
                   base[j], float(s1k[j].mean()), float(dfk[j].mean()), float(dpk[j].mean())]

    feats = np.concatenate([s1k, s2k, agk, dfk, dpk, geom], axis=1).astype(np.float32)
    return final, base, np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def predict_image(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Official Codabench entry point. Keypoints are identical to v5 (Dual-Scale TTA for large pedestrians).
    If assets/rescorer.json exists the instance score is replaced by a learned OKS predictor.
    """
    if _MODEL is None:
        load_model()
    _ensure_rescorer()

    boxes = sample["boxes"]
    if not boxes:
        return {"keypoints": [], "scores": []}

    pil_img = Image.open(_resolve_image_path(sample)).convert("RGB")
    img_w, img_h = pil_img.size
    np_img = np.frombuffer(pil_img.tobytes(), dtype=np.uint8).copy().reshape(img_h, img_w, 3)
    img_tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).to(_DEVICE, dtype=torch.float32).div_(255.0)

    keypoints: List[Any] = [None] * len(boxes)
    scores: List[float] = [0.0] * len(boxes)

    usable = [i for i, b in enumerate(boxes) if float(b[2]) > 0 and float(b[3]) > 0]
    for i in set(range(len(boxes))) - set(usable):
        keypoints[i] = _prior_pose(boxes[i])
        scores[i] = 0.01

    for start in range(0, len(usable), BATCH_SIZE):
        chunk = usable[start:start + BATCH_SIZE]
        items = [{"img": img_tensor, "box": boxes[i], "W": img_w, "H": img_h} for i in chunk]
        final, base, feats = _extract(items)
        if _RESCORER is not None:
            blend = float(_RESCORER.get("blend", 1.0))
            learned = _rescore(feats)
            chunk_scores = [float(blend * learned[j] + (1.0 - blend) * base[j]) for j in range(len(chunk))]
        else:
            chunk_scores = base
        for j, i in enumerate(chunk):
            keypoints[i] = final[j].tolist()
            scores[i] = chunk_scores[j]

    return {"keypoints": keypoints, "scores": scores}
