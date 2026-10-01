import cv2
import numpy as np
from typing import Dict, Any, List
from src.preprocessing import get_onion_color_mask


def _extract_defect_boxes(defect_mask: np.ndarray, defect_type: str, min_area: int = 12) -> List[Dict[str, Any]]:
    """
    Turns a binary defect mask (rot pixels, sprout pixels, ...) into one or
    more bounding boxes in full-image pixel coordinates, so the caller can
    draw a rectangle around the actual defective patch rather than only
    around the whole onion. A light morphological close merges scattered
    single-pixel noise into one region before contouring; min_area drops
    leftover speckle that isn't worth drawing a box around.
    """
    if defect_mask is None or np.count_nonzero(defect_mask) == 0:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append({
            "type": defect_type,
            "bbox": [int(x), int(y), int(w), int(h)],
            "area_px": int(area)
        })

    # Largest patches first, so a caller that only wants the top N still
    # gets the most visually significant defect region.
    boxes.sort(key=lambda b: b["area_px"], reverse=True)
    return boxes


def _refine_mask(hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Removes plain bright background that sits INSIDE the item's mask.

    Detectors such as the Roboflow tier return a rectangular box, so the
    mask also covers the white backdrop around a round onion. Every
    percentage below is "pixels / mask pixels", so that background silently
    dilutes rot, sprout and colour results. Near-white pixels that are
    connected to the edge of the mask's bounding rectangle are background;
    near-white pixels enclosed by the onion (highlights, white skin) are
    kept. If this would remove most of the mask (e.g. a white onion filling
    its box) the original mask is returned unchanged.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return mask
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub_mask = mask[y0:y1, x0:x1] > 0
    sub_hsv = hsv[y0:y1, x0:x1]
    near_white = (sub_hsv[..., 2] >= 235) & (sub_hsv[..., 1] <= 30) & sub_mask

    n, labels = cv2.connectedComponents(near_white.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    border = np.zeros_like(near_white)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    touching = np.unique(labels[border & near_white])
    touching = touching[touching > 0]
    background = np.isin(labels, touching)

    refined_sub = sub_mask & ~background
    if refined_sub.sum() < 0.30 * sub_mask.sum():
        return mask
    refined = np.zeros_like(mask)
    refined[y0:y1, x0:x1] = refined_sub.astype(np.uint8) * 255
    return refined


def _detect_decay(hsv: np.ndarray, mask: np.ndarray,
                  rel: float = 0.5, abs_cap: int = 110,
                  min_blob_frac: float = 0.006) -> np.ndarray:
    """
    Finds dark decayed patches RELATIVE to this onion's own skin.

    Real rot is usually dark brown (still saturated), not pure black, so the
    fixed "V < 25 and S < 30" black-mould rule misses it completely. Here a
    pixel counts as decay when it is much darker than the onion's own median
    brightness (below `rel` x median, capped at `abs_cap`). Judging against
    the onion's own colour also means a naturally dark red/purple onion is
    not flagged just for being dark. The rim is eroded away (edge shadow),
    and only blobs above `min_blob_frac` of the onion area are kept.
    """
    v = hsv[..., 2]
    inner = cv2.erode(mask, np.ones((7, 7), np.uint8)) > 0
    total = int(inner.sum())
    out = np.zeros(mask.shape, np.uint8)
    if total < 200:
        return out

    base = float(np.median(v[inner]))
    thr = min(rel * base, float(abs_cap))
    dark = (inner & (v < thr)).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    min_area = max(30.0, min_blob_frac * total)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255

    # Bridge the lighter flecks (exposed tissue) inside a decayed patch.
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return cv2.bitwise_and(out, mask)


def analyze_onion_defects(bgr_image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """
    Performs research-grade computer vision diagnostic analysis strictly on valid onion skin:
    Accurately distinguishes true Black Mold Rot (Aspergillus niger) from normal dim room lighting, 
    shadows, and healthy dark red/purple onion skin.
    - True Black Mold Rot: Low Value (V < 25) AND Low Saturation (S < 30)
    - Premature Top Sprouting: Vibrant Green Shoots (Hue 35-85, Saturation >= 50, Coverage > 12%)
    - Skin Peeling & Discoloration
    """
    if bgr_image is None or mask is None or np.count_nonzero(mask) == 0:
        return {
            "rot_percentage": 0.0,
            "sprout_percentage": 0.0,
            "color_score": 1.0,
            "defect_percentage": 0.0,
            "dried_patch_percentage": 0.0,
            "primary_defect": "Sound Produce (No Defect)",
            "defect_boxes": []
        }

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # IMPORTANT: rot/defect detection must run over the FULL produce mask,
    # not a mask pre-filtered down to "onion colored" pixels. True black
    # mold rot (very low V, low S) and severe discoloration are, by
    # definition, pixels that fail the onion-color test -- so intersecting
    # with the onion-color mask before looking for rot silently zeroes out
    # the very pixels being searched for. The onion-color mask is still
    # useful for the healthy-skin/color-uniformity score below, but rot and
    # sprout detection are evaluated against `mask` (the item's own
    # segmented silhouette) directly.
    mask = _refine_mask(hsv, mask)
    if np.count_nonzero(mask) == 0:
        return analyze_onion_defects(None, None)
    total_onion_pixels = float(np.count_nonzero(mask))

    # 1. True Black Mold Rot Detection (Aspergillus niger)
    # Requires VERY LOW brightness V < 25 AND LOW saturation S < 30 (decayed black mold)
    # Excludes healthy dark red onion skin (which has S >= 40) and dim room shadows
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    true_rot_mask = ((v_channel < 25) & (s_channel < 30) & (mask > 0)).astype(np.uint8) * 255
    # Plus dark-brown decay judged against this onion's own skin (see _detect_decay).
    true_rot_mask = cv2.bitwise_or(true_rot_mask, _detect_decay(hsv, mask))
    rot_pixel_count = float(np.count_nonzero(true_rot_mask))
    rot_percentage = min((rot_pixel_count / total_onion_pixels) * 100.0, 100.0)

    # 2. Sprout Detection (Vibrant green shoots Hue 35 to 85, Saturation >= 50)
    lower_green = np.array([35, 50, 45])
    upper_green = np.array([85, 255, 255])
    sprout_mask = cv2.inRange(hsv, lower_green, upper_green)
    sprout_mask = cv2.bitwise_and(sprout_mask, sprout_mask, mask=mask)
    sprout_pixel_count = float(np.count_nonzero(sprout_mask))
    sprout_percentage = min((sprout_pixel_count / total_onion_pixels) * 100.0, 100.0)

    # 3. Color Uniformity Score (healthy onion-skin coverage, for context only --
    # this one legitimately restricts to onion-colored pixels, since it is
    # measuring how much of the item looks like sound onion skin)
    valid_onion_mask = get_onion_color_mask(hsv)
    onion_skin_mask = cv2.bitwise_and(mask, valid_onion_mask)

    # 3b. Dried Root Plate / Neck Scar Detection.
    #
    # A large golden/tan/light-brown patch (dried roots, a withered neck,
    # a healed-over cut) sits in HSV territory that `get_onion_color_mask`
    # classifies as "yellow/golden onion" -- so it is (correctly) treated
    # as onion-colored, which means it gets excluded from the
    # discoloration_mask below by design (see that mask's own comment).
    # But that also means it silently contributes to NEITHER color_score
    # (it isn't one of the healthy pink/white bands) NOR discoloration --
    # it's invisible to both checks. On a red or white onion, a big patch
    # like this is exactly the kind of blemish a real inspector would
    # flag, so it needs its own check: only fire when the bulb's DOMINANT
    # skin tone is clearly pink/red or white (i.e. this clearly isn't a
    # yellow onion variety, where this same hue is just normal skin) and
    # a golden/tan patch still covers a meaningful share of the surface.
    lower_dried = np.array([8, 25, 60])
    upper_dried = np.array([32, 130, 210])
    dried_patch_mask = cv2.inRange(hsv, lower_dried, upper_dried)
    dried_patch_mask = cv2.bitwise_and(dried_patch_mask, dried_patch_mask, mask=mask)
    dried_patch_pixel_count = float(np.count_nonzero(dried_patch_mask))
    dried_patch_percentage = min((dried_patch_pixel_count / total_onion_pixels) * 100.0, 100.0)

    lower_pink1 = np.array([0, 20, 40])
    upper_pink1 = np.array([22, 255, 255])
    lower_pink2 = np.array([145, 20, 40])
    upper_pink2 = np.array([180, 255, 255])

    healthy_skin_mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
    healthy_skin_mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
    healthy_skin_mask = cv2.bitwise_or(healthy_skin_mask1, healthy_skin_mask2)
    healthy_skin_mask = cv2.bitwise_and(healthy_skin_mask, healthy_skin_mask, mask=onion_skin_mask)
    healthy_pixel_count = float(np.count_nonzero(healthy_skin_mask))
    healthy_pink_ratio = healthy_pixel_count / total_onion_pixels

    lower_white = np.array([0, 0, 175], dtype=np.uint8)
    upper_white = np.array([180, 35, 255], dtype=np.uint8)
    healthy_white_mask = cv2.inRange(hsv, lower_white, upper_white)
    healthy_white_mask = cv2.bitwise_and(healthy_white_mask, healthy_white_mask, mask=mask)
    healthy_white_ratio = float(np.count_nonzero(healthy_white_mask)) / total_onion_pixels

    color_score = min(max(healthy_pixel_count / total_onion_pixels, 0.1), 1.0)

    # Only treat the golden/tan patch as a blemish when the bulb's own
    # dominant skin is clearly pink/red or white -- i.e. this is not
    # naturally a yellow onion, where the same hue is just normal skin.
    is_dried_patch_defect = (
        (healthy_pink_ratio >= 0.20 or healthy_white_ratio >= 0.20)
        and dried_patch_percentage >= 15.0
    )
    dried_patch_pct_r = round(dried_patch_percentage, 1)

    true_sprout_pct = max(0.0, sprout_percentage - 12.0)
    dried_patch_penalty = dried_patch_percentage * 0.9 if is_dried_patch_defect else 0.0
    defect_percentage = min(
        rot_percentage * 1.5 + true_sprout_pct * 2.0 + (1.0 - color_score) * 10.0 + dried_patch_penalty,
        100.0
    )

    # 4. Discoloration / sunburn mask, for boxing.
    #
    # Not every visible problem is severe enough to hit the "true black
    # mold rot" thresholds (V < 25 and S < 30) -- a dried/withered neck,
    # sunburn, or general skin discoloration can still be exactly the
    # patch a person looking at the photo would call "the bad spot", even
    # though it's flagged as "Skin Discoloration" rather than "Rot" by the
    # primary_defect text below. Without its own mask, that patch never
    # gets a bounding box drawn on it at all. This is: pixels inside the
    # item's own silhouette that are NOT healthy onion skin color AND NOT
    # already counted as true rot (avoid double-boxing the same pixels).
    discoloration_mask = cv2.bitwise_and(mask, cv2.bitwise_not(valid_onion_mask))
    discoloration_mask = cv2.bitwise_and(discoloration_mask, cv2.bitwise_not(true_rot_mask))

    rot_r = round(rot_percentage, 1)
    sprout_r = round(sprout_percentage, 1)

    # Diagnostic Classification -- thresholds aligned with the actual
    # reject rule used downstream (rot_pct >= 8.0 or sprout_pct >= 8.0 in
    # classifier.py/ai_model.py), so a batch marked Reject never gets
    # labeled "Sound Produce" here.
    if rot_r >= 12.0:
        primary_defect = f"Rot / Decay [Severe: {rot_r}%]"
    elif rot_r >= 8.0:
        primary_defect = f"Rot / Decay [Moderate: {rot_r}%]"
    elif sprout_r >= 18.0:
        primary_defect = f"Top Sprouting (Growth Shoot) [Severe: {sprout_r}%]"
    elif sprout_r >= 8.0:
        primary_defect = f"Top Sprouting (Growth Shoot) [Moderate: {sprout_r}%]"
    elif rot_r >= 3.0:
        primary_defect = f"Rot / Decay [Minor: {rot_r}%]"
    elif color_score < 0.45:
        primary_defect = f"Skin Discoloration / Sunburn [{round(defect_percentage, 1)}%]"
    elif is_dried_patch_defect and dried_patch_pct_r >= 25.0:
        primary_defect = f"Dried Root Plate / Neck Scar [Severe: {dried_patch_pct_r}%]"
    elif is_dried_patch_defect:
        primary_defect = f"Dried Root Plate / Neck Scar [Moderate: {dried_patch_pct_r}%]"
    else:
        primary_defect = "Sound Produce (No Defect)"

    # Bounding boxes around the actual defective patches, in the same
    # full-image pixel coordinates as `mask`, so the caller can draw a
    # rectangle on the defect itself instead of only on the whole onion.
    # Discoloration boxes are only added when discoloration is actually
    # the flagged issue (color_score below the same 0.45 cutoff used for
    # primary_defect above) -- otherwise ordinary color variation on a
    # sound onion would get boxed for no reason. Same idea for the dried
    # patch: only box it when it's actually being counted as a defect.
    defect_boxes = (
        _extract_defect_boxes(true_rot_mask, "rot")
        + _extract_defect_boxes(sprout_mask, "sprout")
    )
    if color_score < 0.45:
        defect_boxes += _extract_defect_boxes(discoloration_mask, "discoloration", min_area=40)
    if is_dried_patch_defect:
        defect_boxes += _extract_defect_boxes(dried_patch_mask, "dried_neck_scar", min_area=40)

    return {
        "rot_percentage": round(rot_percentage, 2),
        "sprout_percentage": round(sprout_percentage, 2),
        "color_score": round(color_score, 2),
        "defect_percentage": round(defect_percentage, 2),
        "dried_patch_percentage": dried_patch_pct_r,
        "primary_defect": primary_defect,
        "defect_boxes": defect_boxes
    }


if __name__ == "__main__":
    print("Testing True Black Mold Rot vs Dim Lighting Shadow Rejection...")
    # Synthetic healthy dark red onion under dim lighting (BGR: 20, 30, 120 -> S=212, V=120)
    dim_red_onion = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(dim_red_onion, (100, 100), 60, (20, 30, 120), -1)

    gray = cv2.cvtColor(dim_red_onion, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    analysis = analyze_onion_defects(dim_red_onion, mask)
    print("Dim Red Onion Defect Result:", analysis["primary_defect"])
    assert "Sound Produce" in analysis["primary_defect"]
    print("Rot vs Dim Lighting Rejection verified successfully.")