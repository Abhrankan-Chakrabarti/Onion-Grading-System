"""
Instance segmentation of onions with Segment Anything (SAM).

Usage:
    from src.sam_segmenter import segment_onions
    items = segment_onions(bgr_image)          # bgr_image = cv2.imread(...) result
    for it in items:
        it["mask"], it["bbox"], it["pixel_area"], it["max_diameter_px"]

Setup (once):
    pip install torch torchvision
    pip install git+https://github.com/facebookresearch/segment-anything.git
    download https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    put it at <project root>/models/sam_vit_b_01ec64.pth   (or set env var SAM_CHECKPOINT)
"""
import os
from typing import Any, Dict, List

import cv2
import numpy as np

# Default: <project root>/models/sam_vit_b_01ec64.pth  (project root = folder that contains src/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.environ.get("SAM_CHECKPOINT",
                            os.path.join(_PROJECT_ROOT, "models", "sam_vit_b_01ec64.pth"))
POINTS_PER_SIDE = int(os.environ.get("SAM_POINTS_PER_SIDE", "32"))   # lower (16-24) = faster on CPU
_generator = None   # loaded once, reused for every request


def _get_generator():
    global _generator
    if _generator is None:
        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
        sam.to("cuda" if torch.cuda.is_available() else "cpu")
        _generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=POINTS_PER_SIDE,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            min_mask_region_area=800,
        )
    return _generator


def postprocess_masks(raw_masks: List[Dict[str, Any]], img_shape,
                      min_area_frac: float = 0.002,   # ignore specks (< 0.2% of image)
                      max_area_frac: float = 0.15,    # ignore whole-heap / background masks
                      min_solidity: float = 0.90,     # reject merged blobs (two touching onions score ~0.88)
                      max_overlap: float = 0.6) -> List[Dict[str, Any]]:
    """SAM returns many overlapping masks (whole, parts, background). Keep one clean mask per onion."""
    h_img, w_img = img_shape[:2]
    img_area = float(h_img * w_img)
    candidates = []

    for m in raw_masks:
        seg = m["segmentation"].astype(np.uint8)
        area = float(seg.sum())
        if not (min_area_frac * img_area <= area <= max_area_frac * img_area):
            continue
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = cv2.contourArea(c) / hull_area if hull_area > 0 else 0.0
        if solidity < min_solidity:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        candidates.append({
            "seg": seg.astype(bool),
            "area": area,
            "score": float(m.get("predicted_iou", 0.0)) + float(m.get("stability_score", 0.0)) + solidity,
            "bbox": cv2.boundingRect(c),
            "max_diameter_px": float(max(rw, rh)),
        })

    # best-scoring masks first; drop any mask that mostly overlaps one already kept
    candidates.sort(key=lambda c: c["score"], reverse=True)
    kept: List[Dict[str, Any]] = []
    for c in candidates:
        dup = False
        for k in kept:
            inter = np.logical_and(c["seg"], k["seg"]).sum()
            if inter / min(c["area"], k["area"]) > max_overlap:
                dup = True
                break
        if not dup:
            kept.append(c)

    # number them top-to-bottom, left-to-right so "Item #1, #2..." is stable
    kept.sort(key=lambda c: (c["bbox"][1] // 60, c["bbox"][0]))
    return [{
        "id": i + 1,
        "mask": (c["seg"].astype(np.uint8) * 255),
        "bbox": c["bbox"],                       # (x, y, w, h)
        "pixel_area": c["area"],
        "max_diameter_px": c["max_diameter_px"],
    } for i, c in enumerate(kept)]


def segment_onions(bgr: np.ndarray) -> List[Dict[str, Any]]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raw = _get_generator().generate(rgb)
    return postprocess_masks(raw, bgr.shape)