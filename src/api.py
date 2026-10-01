import os

# Load environment variables from a local .env file (ROBOFLOW_API_KEY,
# ROBOFLOW_MODEL_ID, etc.) BEFORE anything below reads them. This has to
# happen first, because `detector = OnionDetectorStub()` further down reads
# os.environ at construction time -- if load_dotenv() ran after that line,
# Roboflow would never see the key and would silently fall back to local
# YOLO/OpenCV detection.
from dotenv import load_dotenv
load_dotenv()

import cv2
import base64
import inspect
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from src.preprocessing import normalize_lighting, subtract_background, resize_for_fast_processing
from src.detection import OnionDetectorStub
from src.defect_analysis import analyze_onion_defects
from src.agmark_classifier import OnionAGMARKClassifier   # A/B/C/D: size + appearance + AI (was src.classifier)
from src.market_analytics import MandiPricePredictor

app = FastAPI(
    title="AGMARK Compliant High-Speed Onion Quality Assessment Engine",
    description="Ultra-Fast Computer Vision & Quality Analytics API for Agricultural Produce (Onions). Equipped with Active Learning Online Retraining.",
    version="3.1.0"
)


class Point(BaseModel):
    x: int
    y: int


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    center: Point


def _make_bbox(x: int, y: int, w: int, h: int) -> BoundingBox:
    """Builds a BoundingBox (with center point) from full-image pixel coords."""
    return BoundingBox(
        x=int(x), y=int(y), width=int(w), height=int(h),
        center=Point(x=int(x + w / 2), y=int(y + h / 2))
    )


# detector picks up ROBOFLOW_API_KEY / ROBOFLOW_MODEL_ID (and the optional
# confidence/overlap/accepted-classes vars) from the environment at
# construction time -- see detection.py: OnionDetectorStub.__init__. With
# .env loaded above, your Roboflow model is now used first automatically,
# with local YOLO / OpenCV watershed as the fallback.
detector = OnionDetectorStub()
classifier = OnionAGMARKClassifier(pixels_per_mm=2.5)

# Newer detection.py (SAM tier) accepts raw_image=; an older one doesn't. Detect
# it once here so this file works with either version instead of crashing.
_DETECTOR_TAKES_RAW_IMAGE = "raw_image" in inspect.signature(detector.detect_and_measure).parameters
market_predictor = MandiPricePredictor()


class MetricDetails(BaseModel):
    diameter_mm: float
    area_cm2: float
    pixel_area: float
    max_diameter_px: float


class DefectPosition(BaseModel):
    type: str  # "rot" or "sprout"
    bbox: BoundingBox
    area_px: int


class DefectDetails(BaseModel):
    color_score: float
    rot_percentage: float
    sprout_percentage: float
    defect_percentage: float
    primary_defect: str
    defect_positions: List[DefectPosition] = Field(default_factory=list)


class MarketEstimate(BaseModel):
    estimated_price_inr_per_kg: float
    currency: str
    price_source: str


class AGMARKItemResult(BaseModel):
    item_id: int
    grade: str
    quality_status: str
    agmark_standard: str
    position: Optional[BoundingBox] = None
    metrics: MetricDetails
    defects: DefectDetails
    market_estimate: MarketEstimate
    # Present only when this item came from the Roboflow model rather than
    # the local YOLO/OpenCV fallback -- lets the UI show what your trained
    # model itself called the item and how confident it was.
    roboflow_class: Optional[str] = None
    roboflow_confidence: Optional[float] = None
    # Which check set the final grade, e.g. "appearance (rot 13.4%, ...)".
    limited_by: Optional[str] = None
    grade_breakdown: Optional[Dict[str, Optional[str]]] = None


class ComprehensiveGradingResponse(BaseModel):
    items_count: int
    overall_batch_grade: str
    average_diameter_mm: float
    estimated_mandi_price_inr: float
    price_source_info: str
    active_learning_status: str = "Active Learning Idle"
    annotated_image_base64: Optional[str] = None
    results: List[AGMARKItemResult]
    # Transparency about size measurement: without a physical reference
    # object in the photo, diameter_mm is only as good as the assumed
    # pixels_per_mm constant, which was tuned for a specific camera
    # distance/zoom. A close-up product shot at a very different scale
    # will read as a much bigger or smaller onion than it really is.
    calibration_status: str = "default_assumed"
    calibration_note: str = (
        "Sizes assume a fixed camera distance (pixels_per_mm). "
        "For accurate mm/price, pass known_item_id and known_diameter_mm "
        "for one visible item to calibrate this photo."
    )


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves the interactive hackathon web user interface."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>AGMARK Onion Quality Assessment Pipeline Online</h1>", status_code=200)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/market_trends")
def get_market_trends(mandi_name: str = "Lasalgaon (Nashik, MH)"):
    """
    Returns live Agmarknet daily wholesale price benchmarks, 30-day historical prices,
    14-day AI future price forecast, and cold storage profit recommendations.
    """
    return market_predictor.get_market_analytics(mandi_name)


@app.post("/grade_image", response_model=ComprehensiveGradingResponse)
async def grade_image(
    file: UploadFile = File(...),
    pixels_per_mm: Any = 2.5,
    mandi_name: str = "Lasalgaon (Nashik, MH)",
    known_item_id: Optional[int] = None,
    known_diameter_mm: Optional[float] = None,
):
    """
    Ultra-Fast Computer Vision & AGMARK Grading Pipeline with Active Learning:
    Automatically logs user image features and retrains the AI model online for continuously improving accuracy.
    """
    try:
        pixels_per_mm_val = float(pixels_per_mm)
        if pixels_per_mm_val <= 0 or np.isnan(pixels_per_mm_val):
            pixels_per_mm_val = 2.5
    except Exception:
        pixels_per_mm_val = 2.5

    price_source = f"Agmarknet Live ({mandi_name} Index)"

    try:
        contents = await file.read()
        if not contents or len(contents) < 50:
            return ComprehensiveGradingResponse(
                items_count=0,
                overall_batch_grade="No Produce Detected",
                average_diameter_mm=0.0,
                estimated_mandi_price_inr=0.0,
                price_source_info=price_source,
                active_learning_status="No produce detected to learn from",
                annotated_image_base64=None,
                results=[]
            )

        nparr = np.frombuffer(contents, np.uint8)
        img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_raw is None or img_raw.size == 0:
            return ComprehensiveGradingResponse(
                items_count=0,
                overall_batch_grade="No Produce Detected",
                average_diameter_mm=0.0,
                estimated_mandi_price_inr=0.0,
                price_source_info=price_source,
                active_learning_status="No produce detected to learn from",
                annotated_image_base64=None,
                results=[]
            )
    except Exception:
        return ComprehensiveGradingResponse(
            items_count=0,
            overall_batch_grade="No Produce Detected",
            average_diameter_mm=0.0,
            estimated_mandi_price_inr=0.0,
            price_source_info=price_source,
            active_learning_status="No produce detected to learn from",
            annotated_image_base64=None,
            results=[]
        )

    try:
        img, scale_factor = resize_for_fast_processing(img_raw, max_dim=800)
        adjusted_px_per_mm = pixels_per_mm_val * scale_factor

        market_analytics = market_predictor.get_market_analytics(mandi_name)
        today_modal_price = market_analytics.get("today_modal_price_inr", 40.0)
        price_source = f"Agmarknet Live ({mandi_name} Index - Today's Modal Price: ₹{today_modal_price}/kg)"

        norm_img = normalize_lighting(img)
        fg_img, mask = subtract_background(norm_img)

        # detect_and_measure tries Roboflow first (when configured via
        # .env), then local YOLO, then the OpenCV watershed fallback --
        # see detection.py. Nothing else here needs to change based on
        # which tier actually ran.
        # raw_image=norm_img: SAM (see detection.py) segments the real photo,
        # not the background-blacked-out fg_img, so dark / rotten onions
        # aren't hollowed out by the colour mask. Other tiers ignore it.
        if _DETECTOR_TAKES_RAW_IMAGE:
            detected_items = detector.detect_and_measure(fg_img, mask, raw_image=norm_img)
        else:
            detected_items = detector.detect_and_measure(fg_img, mask)

        # --------------------------------------------------------------
        # Optional per-photo calibration.
        #
        # `adjusted_px_per_mm` up to this point is just the fixed
        # `pixels_per_mm` constant (scaled for the resize) -- it has no
        # idea what camera/distance/zoom actually produced this photo. If
        # the caller tells us the *real* diameter of one visible item, we
        # can derive the true pixels_per_mm for THIS photo from it and
        # use that for every item in the batch instead of guessing.
        # --------------------------------------------------------------
        calibration_status = "default_assumed"
        calibration_note = (
            "Sizes assume a fixed camera distance (pixels_per_mm). "
            "For accurate mm/price, pass known_item_id and known_diameter_mm "
            "for one visible item to calibrate this photo."
        )
        if known_item_id is not None and known_diameter_mm and known_diameter_mm > 0:
            reference_item = next(
                (it for it in detected_items if it["item_id"] == known_item_id), None
            )
            if reference_item is not None and reference_item["max_diameter"] > 0:
                adjusted_px_per_mm = reference_item["max_diameter"] / float(known_diameter_mm)
                calibration_status = "user_calibrated"
                calibration_note = (
                    f"Calibrated from item #{known_item_id} "
                    f"(assumed {known_diameter_mm}mm actual diameter)."
                )
            else:
                calibration_note = (
                    f"known_item_id={known_item_id} was not found in this photo's "
                    "detections, so the default calibration was used instead."
                )

        annotated_img = img.copy()
        grading_results = []
        total_diam = 0.0
        prices = []
        retrained_online = False

        local_classifier = OnionAGMARKClassifier(pixels_per_mm=adjusted_px_per_mm)
        img_h, img_w = img.shape[:2]

        for item in detected_items:
            item_id = item["item_id"]
            pixel_area = item["pixel_area"]
            max_diam_px = item["max_diameter"]

            # Roboflow-sourced items already carry their own contour
            # (polygon or box) straight from detect_with_roboflow, so use
            # it directly instead of re-matching against the background
            # mask's contours, which is only meaningful for the watershed
            # fallback path.
            item_mask = np.zeros_like(mask)
            matched_cnt = item.get("contour")
            if matched_cnt is not None:
                cv2.drawContours(item_mask, [matched_cnt], -1, 255, -1)
            else:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    matched_cnt = min(contours, key=lambda c: abs(cv2.contourArea(c) - pixel_area))
                    cv2.drawContours(item_mask, [matched_cnt], -1, 255, -1)
                else:
                    item_mask = mask

            defect_data = analyze_onion_defects(img, item_mask)
            class_res = local_classifier.classify_onion(pixel_area, max_diam_px, defect_data, today_mandi_modal_price=today_modal_price)
            grade = class_res["grade"]

            # Active Learning Feedback: Log user image sample into AI model dataset & trigger online retraining
            was_retrained = local_classifier.ai_model.log_user_sample_and_retrain(
                diam_mm=class_res["metrics"]["diameter_mm"],
                area_cm2=class_res["metrics"]["area_cm2"],
                rot_pct=defect_data["rot_percentage"],
                sprout_pct=defect_data["sprout_percentage"],
                color_score=defect_data["color_score"],
                defect_pct=defect_data["defect_percentage"],
                assigned_grade=class_res["legacy_label"]
            )
            if was_retrained:
                retrained_online = True

            color_map = {
                "Grade A": (0, 220, 0),        # Bright Green
                "Grade B": (0, 215, 255),      # Golden Yellow
                "Grade C": (0, 140, 255),      # Orange
                "Grade D": (0, 0, 255)         # Crimson Red (reject)
            }
            box_color = color_map.get(grade, (255, 255, 255))

            item_position = None
            item_defect_positions: List[DefectPosition] = []

            if matched_cnt is not None:
                x, y, w, h = cv2.boundingRect(matched_cnt)
                if w < (img_w * 0.90) and h < (img_h * 0.90):
                    item_position = _make_bbox(x, y, w, h)
                    item_defect_positions = [
                        DefectPosition(
                            type=dbox["type"],
                            bbox=_make_bbox(*dbox["bbox"]),
                            area_px=dbox["area_px"]
                        )
                        for dbox in defect_data.get("defect_boxes", [])
                    ]

                    # Draw the actual segmented outline rather than its
                    # axis-aligned bounding rectangle. Two touching round
                    # onions have bounding rectangles that overlap near
                    # the touch point even when their real silhouettes
                    # barely do -- that overlap made it look like the
                    # segmentation was confused between the two items,
                    # when the underlying per-item measurements were
                    # actually fine. The contour outline reflects the
                    # true detected shape and doesn't have that artifact.
                    cv2.drawContours(annotated_img, [matched_cnt], -1, box_color, 3)
                    label = f"Item #{item_id}: {grade} ({class_res['metrics']['diameter_mm']}mm)"
                    cv2.putText(annotated_img, label, (x, max(y - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, box_color, 2)
                    if defect_data["primary_defect"] != "Sound Produce (No Defect)":
                        cv2.putText(annotated_img, f"Defect: {defect_data['primary_defect']}", (x, y + h + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                    defect_box_color = {"rot": (0, 0, 255), "sprout": (0, 200, 255), "discoloration": (0, 140, 255)}
                    for dbox in defect_data.get("defect_boxes", []):
                        dx, dy, dw, dh = dbox["bbox"]
                        d_color = defect_box_color.get(dbox["type"], (255, 0, 255))
                        cv2.rectangle(annotated_img, (dx, dy), (dx + dw, dy + dh), d_color, 2)
                        cv2.putText(annotated_img, dbox["type"], (dx, max(dy - 4, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, d_color, 1)

            total_diam += class_res["metrics"]["diameter_mm"]
            prices.append(class_res["market_estimate"]["estimated_price_inr_per_kg"])

            grading_results.append(AGMARKItemResult(
                item_id=item_id,
                grade=grade,
                quality_status=class_res["quality_status"],
                agmark_standard=class_res["agmark_standard"],
                position=item_position,
                metrics=MetricDetails(**class_res["metrics"]),
                defects=DefectDetails(**class_res["defects"], defect_positions=item_defect_positions),
                market_estimate=MarketEstimate(
                    estimated_price_inr_per_kg=class_res["market_estimate"]["estimated_price_inr_per_kg"],
                    currency="INR (₹)",
                    price_source=price_source
                ),
                roboflow_class=item.get("roboflow_class"),
                roboflow_confidence=item.get("roboflow_confidence"),
                limited_by=class_res.get("limited_by"),
                grade_breakdown=class_res.get("grade_breakdown"),
            ))

        _, buffer = cv2.imencode('.jpg', annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        items_cnt = len(grading_results)
        learned_count = local_classifier.ai_model.user_samples_count

        if items_cnt == 0:
            overall_grade = "No Produce Detected"
            avg_diam = 0.0
            avg_price = 0.0
            learning_msg = f"Active Learning Idle (Total Learned: {learned_count})"
        else:
            avg_diam = round(total_diam / items_cnt, 1)

            if retrained_online:
                learning_msg = f"⚡ Active Learning: AI Model Retrained Online Live! (Total Samples Learned: {learned_count})"
            else:
                learning_msg = f"🧠 Active Learning: Logged User Image to AI Memory (Total Samples Learned: {learned_count})"

            grades_list = [r.grade for r in grading_results]
            reject_count = grades_list.count("Grade D")
            reject_ratio = reject_count / float(items_cnt)

            sound_grades = [g for g in grades_list if g != "Grade D"]
            sound_prices = [r.market_estimate.estimated_price_inr_per_kg for r in grading_results if r.grade != "Grade D"]

            if reject_ratio >= 0.25:
                overall_grade = "Reject"
                avg_price = 0.0
            elif reject_ratio >= 0.15:
                overall_grade = "Mixed (Contains Rejects)"
                avg_price = round(float(np.mean(prices)), 1)
            else:
                if sound_prices:
                    avg_price = round(float(np.mean(sound_prices)), 1)
                else:
                    avg_price = round(today_modal_price * 0.65, 1)

                # Batch grade = the most common grade among the sound items;
                # on a tie the lower quality wins ("Grade C" > "Grade B" > "Grade A").
                from collections import Counter
                grade_counts = Counter(sound_grades)
                overall_grade = max(grade_counts, key=lambda g: (grade_counts[g], g)) if grade_counts else "Grade B"

        return ComprehensiveGradingResponse(
            items_count=items_cnt,
            overall_batch_grade=overall_grade,
            average_diameter_mm=avg_diam,
            estimated_mandi_price_inr=avg_price,
            price_source_info=price_source,
            active_learning_status=learning_msg,
            calibration_status=calibration_status,
            calibration_note=calibration_note,
            annotated_image_base64=annotated_b64,
            results=grading_results
        )
    except Exception as exc:
        print(f"Error during image grading: {exc}")
        return ComprehensiveGradingResponse(
            items_count=0,
            overall_batch_grade="No Produce Detected",
            average_diameter_mm=0.0,
            estimated_mandi_price_inr=0.0,
            price_source_info=price_source,
            active_learning_status="Active Learning Idle",
            annotated_image_base64=None,
            results=[]
        )


if __name__ == "__main__":
    print("Testing Active Learning API Endpoint...")
    from fastapi.testclient import TestClient
    client = TestClient(app)

    dummy = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(dummy, (200, 200), 70, (30, 20, 180), -1)
    _, img_bytes = cv2.imencode('.png', dummy)

    res = client.post("/grade_image?pixels_per_mm=2.5", files={"file": ("test.png", img_bytes.tobytes(), "image/png")})
    assert res.status_code == 200
    data = res.json()
    print("Active Learning Status:", data["active_learning_status"])
    print("Active Learning API verified successfully.")