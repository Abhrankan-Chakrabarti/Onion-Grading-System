"""
Visual defect analysis for onions (OpenCV, no training needed).

Looks at the actual pixels of the segmented onion and measures:
  - rot / dark patches   (dark brown-black areas, mould)
  - sprout               (green shoots)
  - colour score         (how close the skin is to healthy golden/red/white onion)
  - overall defect %     (rot + sprout + bruise/wet patches)

Output keys match what OnionAGMARKClassifier.classify_onion() expects.
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional


def _auto_mask(bgr: np.ndarray) -> np.ndarray:
    """Fallback mask when the segmenter's mask isn't passed: anything not near-white."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] > 25) | (hsv[..., 2] < 200)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask


def analyze_onion_defects(bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    bgr  : cropped onion image (BGR, as read by cv2)
    mask : optional uint8 mask (255 = onion pixel) from your segmentation step
    """
    if mask is None:
        mask = _auto_mask(bgr)
    mask = (mask > 0)
    total = int(mask.sum())
    if total < 50:
        return {"rot_percentage": 0.0, "sprout_percentage": 0.0, "color_score": 0.5,
                "defect_percentage": 0.0, "primary_defect": "Not enough pixels to analyse"}

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Rot / mould: dark pixels (low brightness), inside the onion
    dark = mask & (v < 95)

    # Wet / bruised brown patches: mid-dark, saturated brown
    bruise = mask & (v >= 95) & (v < 140) & (s > 90) & (h >= 5) & (h <= 25)

    # Sprout: green hues
    green = mask & (h >= 35) & (h <= 85) & (s > 60) & (v > 50)

    # Ignore tiny specks (noise, dry-skin shadows): keep only blobs of meaningful size
    def _keep_blobs(binary: np.ndarray, min_frac: float) -> np.ndarray:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        out = np.zeros_like(binary, dtype=bool)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_frac * total:
                out |= labels == i
        return out

    dark = _keep_blobs(dark, 0.004)
    bruise = _keep_blobs(bruise, 0.01)
    green = _keep_blobs(green, 0.003)

    rot_pct = float(100.0 * dark.sum() / total)
    sprout_pct = float(100.0 * green.sum() / total)
    bruise_pct = float(100.0 * bruise.sum() / total)
    defect_pct = min(100.0, rot_pct + sprout_pct + bruise_pct)

    # Colour score: 1.0 = clean uniform skin, lower = patchy / dull / dark
    onion_v = v[mask].astype(np.float32)
    onion_s = s[mask].astype(np.float32)
    uniformity = 1.0 - min(1.0, float(onion_v.std()) / 80.0)
    brightness = min(1.0, float(onion_v.mean()) / 170.0)
    color_score = float(np.clip(0.5 * uniformity + 0.5 * brightness - defect_pct / 200.0, 0.0, 1.0))

    # Primary defect label (largest contributor above a small floor)
    candidates = {"Rot / Dark Decay": rot_pct, "Sprouting": sprout_pct, "Bruise / Wet Patch": bruise_pct}
    name, val = max(candidates.items(), key=lambda kv: kv[1])
    primary = name if val >= 2.0 else "Sound Produce (No Defect)"

    return {
        "rot_percentage": round(rot_pct, 1),
        "sprout_percentage": round(sprout_pct, 1),
        "color_score": round(color_score, 2),
        "defect_percentage": round(defect_pct, 1),
        "primary_defect": primary,
    }
