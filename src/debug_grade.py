"""
Diagnostic for the onion grading pipeline.  Run from the project root:

    python debug_grade.py path/to/photo.jpg            # default pixels_per_mm = 2.5
    python debug_grade.py path/to/photo.jpg 4.1        # your calibrated pixels_per_mm

It (1) checks the right versions of your files are actually loaded, (2) runs the
same steps as /grade_image, (3) prints WHICH detector produced the items and one
row per item explaining its grade, and (4) writes debug_out.png:
left = what was graded (outline colour = grade, boxes = defect patches),
right = the background mask the fallback detector sees.
It never writes to the active-learning dataset.
"""
import inspect
import os
import sys

import cv2
import numpy as np

from src.preprocessing import normalize_lighting, subtract_background, resize_for_fast_processing
import src.detection as detection_mod
import src.defect_analysis as defect_mod
import src.agmark_classifier as clf_mod
from src.detection import OnionDetectorStub
from src.defect_analysis import analyze_onion_defects
from src.agmark_classifier import OnionAGMARKClassifier

if len(sys.argv) < 2:
    sys.exit(__doc__)
img_path = sys.argv[1]
ppm = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5

# ------------------------------------------------------------------ 1. versions
print("=" * 72)
print("1. WHICH FILES ARE LOADED")
detector = OnionDetectorStub()
api_src = open("api.py", encoding="utf-8").read() if os.path.exists("api.py") else ""
checks = [
    ("detection.py has SAM tier (new)", "raw_image" in inspect.signature(detector.detect_and_measure).parameters, detection_mod.__file__),
    ("defect_analysis.py has decay detector (new)", hasattr(defect_mod, "_detect_decay"), defect_mod.__file__),
    ("agmark_classifier.py is A/B/C/D (new)", hasattr(clf_mod, "GRADE_DISPLAY"), clf_mod.__file__),
    ("api.py uses src.agmark_classifier (new)", "src.agmark_classifier" in api_src, os.path.abspath("api.py")),
]
for label, ok, path in checks:
    print(f"   [{'OK ' if ok else 'OLD'}] {label}\n         {path}")
print(f"   Roboflow tier enabled: {detector.roboflow_enabled} | SAM tier enabled: {detector.sam_enabled if hasattr(detector, 'sam_enabled') else 'n/a (old detection.py)'} | YOLO loaded: {detector.model is not None}")

# ------------------------------------------------------------------ 2. pipeline
raw = cv2.imread(img_path)
if raw is None:
    sys.exit(f"Could not read image: {img_path}")
img, scale = resize_for_fast_processing(raw, max_dim=800)
adj_ppm = ppm * scale
norm = normalize_lighting(img)
fg, mask = subtract_background(norm)

# record which detection tier actually returned items
tier_log = []
for name in ("detect_with_roboflow", "detect_with_sam", "detect_with_yolo", "_fallback_watershed_measure"):
    if hasattr(detector, name):
        orig = getattr(detector, name)
        def make(orig=orig, name=name):
            def wrapped(*a, **k):
                out = orig(*a, **k)
                tier_log.append((name, len(out)))
                return out
            return wrapped
        setattr(detector, name, make())

if "raw_image" in inspect.signature(detector.detect_and_measure).parameters:
    items = detector.detect_and_measure(fg, mask, raw_image=norm)
else:
    items = detector.detect_and_measure(fg, mask)

print("\n" + "=" * 72)
print(f"2. DETECTION  (image {raw.shape[1]}x{raw.shape[0]}, working size {img.shape[1]}x{img.shape[0]}, pixels_per_mm={adj_ppm:.2f})")
for name, n in tier_log:
    print(f"   tier {name:30s} returned {n} item(s)")
print(f"   => {len(items)} onion(s) will be graded")

# ------------------------------------------------------------------ 3. per item
clf = OnionAGMARKClassifier(pixels_per_mm=adj_ppm)
colors = {"Grade A": (0, 220, 0), "Grade B": (0, 215, 255), "Grade C": (0, 140, 255), "Grade D": (0, 0, 255)}
out = img.copy()
print("\n" + "=" * 72)
print("3. PER-ITEM RESULT")
print(f"   {'#':>2} {'mm':>6} {'rot%':>5} {'spr%':>5} {'def%':>5} {'colr':>4}  {'size/visual/AI':<14} {'FINAL':<8} why")
for it in items:
    m = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(m, [it["contour"]], -1, 255, -1)
    d = analyze_onion_defects(img, m)
    r = clf.classify_onion(it["pixel_area"], it["max_diameter"], d, 42.0)
    b = r["grade_breakdown"]
    print(f"   {it['item_id']:>2} {r['metrics']['diameter_mm']:>6} {d['rot_percentage']:>5} {d['sprout_percentage']:>5} "
          f"{d['defect_percentage']:>5} {d['color_score']:>4}  {str(b['size'])+'/'+str(b['visual'])+'/'+str(b['ai']):<14} "
          f"{r['grade']:<8} {r['limited_by']}")
    print(f"      defect: {d['primary_defect']}")
    col = colors[r["grade"]]
    cv2.drawContours(out, [it["contour"]], -1, col, 3)
    x, y, w, h = cv2.boundingRect(it["contour"])
    cv2.putText(out, f"#{it['item_id']} {r['grade'][-1]} {r['metrics']['diameter_mm']}mm", (x, max(y - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    for db in d.get("defect_boxes", []):
        dx, dy, dw, dh = db["bbox"]
        cv2.rectangle(out, (dx, dy), (dx + dw, dy + dh), (255, 0, 255), 1)
        cv2.putText(out, db["type"], (dx, max(dy - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
cv2.imwrite("debug_out.png", np.hstack([out, mask_vis]))
print("\nWrote debug_out.png  (left: graded result | right: background mask)")
print("Send me this printout + debug_out.png, plus what you expected for the photo.")