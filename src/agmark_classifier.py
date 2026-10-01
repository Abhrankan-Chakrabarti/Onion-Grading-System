from typing import Dict, Any
from src.ai_model import OnionQualityAIModel


# Price as a fraction of today's modal mandi price
GRADE_PRICE_FACTOR = {"A": 1.00, "B": 0.65, "C": 0.45, "D": 0.00}

GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}   # higher = worse

# Map the old AI model labels onto A/B/C/D
AI_LABEL_TO_GRADE = {"Grade A": "A", "Grade B": "B", "Small Grade": "C", "Reject": "D"}
# ...and back, for src/ai_model.py's active-learning log (it only knows the 4 old labels)
GRADE_TO_AI_LABEL = {v: k for k, v in AI_LABEL_TO_GRADE.items()}
# Names shown in the API / UI
GRADE_DISPLAY = {"A": "Grade A", "B": "Grade B", "C": "Grade C", "D": "Grade D"}


def _worst(*grades: str) -> str:
    return max(grades, key=lambda g: GRADE_ORDER[g])


class OnionAGMARKClassifier:
    """
    AGMARK-style A / B / C / D grading using BOTH size and how the onion looks.

    Three independent opinions are computed; the FINAL grade is the WORST of them,
    so a big onion that is rotten can never be Grade A.

      1. Size grade    (diameter):   A >= 55 mm | B 40-55 | C 30-40 | D < 30
      2. Visual grade  (from image): rot %, sprout %, total defect %, colour score
      3. AI grade      (Gradient Boosting model, optional)

    Visual grade rules (first match wins, checked from best to worst):
      A: rot < 1%  and sprout < 2%  and defects < 5%
      B: rot < 3%  and sprout < 5%  and defects < 15%
      C: rot < 8%  and sprout < 8%  and defects < 25%
      D: anything worse  (rejected, no commercial price)
    (colour is already inside defect_percentage, so it isn't a separate gate.)
    """

    def __init__(self, pixels_per_mm: float = 2.5, use_ai_model: bool = True):
        self.pixels_per_mm = pixels_per_mm
        self.use_ai_model = use_ai_model
        self.ai_model = OnionQualityAIModel() if use_ai_model else None

    # ---- 1. size -----------------------------------------------------------
    @staticmethod
    def size_grade(diameter_mm: float) -> str:
        if diameter_mm >= 55.0:
            return "A"
        if diameter_mm >= 40.0:
            return "B"
        if diameter_mm >= 30.0:
            return "C"
        return "D"

    # ---- 2. visual appearance ---------------------------------------------
    @staticmethod
    def visual_grade(rot: float, sprout: float, defect: float, color: float) -> str:
        if rot < 1.0 and sprout < 2.0 and defect < 5.0:
            return "A"
        if rot < 3.0 and sprout < 5.0 and defect < 15.0:
            return "B"
        if rot < 8.0 and sprout < 8.0 and defect < 25.0:
            return "C"
        return "D"

    # ---- main --------------------------------------------------------------
    def classify_onion(self, pixel_area: float, max_diameter_px: float,
                       defect_analysis: Dict[str, Any],
                       today_mandi_modal_price: float = 40.0) -> Dict[str, Any]:
        diameter_mm = float(max_diameter_px / self.pixels_per_mm)
        area_cm2 = float(pixel_area / (self.pixels_per_mm ** 2) / 100.0)

        rot = float(defect_analysis.get("rot_percentage", 0.0))
        sprout = float(defect_analysis.get("sprout_percentage", 0.0))
        color = float(defect_analysis.get("color_score", 0.8))
        defect = float(defect_analysis.get("defect_percentage", 0.0))
        primary_defect = defect_analysis.get("primary_defect", "Sound Produce (No Defect)")

        g_size = self.size_grade(diameter_mm)
        g_visual = self.visual_grade(rot, sprout, defect, color)

        g_ai, confidence = None, 0.0
        if self.ai_model is not None:
            ai_label, confidence = self.ai_model.predict_quality_grade(
                diam_mm=diameter_mm, area_cm2=area_cm2, rot_pct=rot,
                sprout_pct=sprout, color_score=color, defect_pct=defect)
            g_ai = AI_LABEL_TO_GRADE.get(ai_label, "C")

        votes = [g for g in (g_size, g_visual, g_ai) if g]
        final = _worst(*votes)

        # Why did it get this grade? (limiting factor = whichever vote equals the final grade)
        reasons = []
        if g_size == final:
            reasons.append(f"size {diameter_mm:.0f} mm")
        if g_visual == final:
            reasons.append(f"appearance (rot {rot:.1f}%, sprout {sprout:.1f}%, defects {defect:.1f}%)")
        if g_ai == final:
            reasons.append("AI model")
        limited_by = ", ".join(reasons)

        descriptions = {
            "A": "AGMARK Grade A - large, clean, export quality",
            "B": "AGMARK Grade B - medium / minor blemishes, standard commercial",
            "C": "AGMARK Grade C - small or noticeable defects, low-price / processing use",
            "D": "Grade D - Reject: severe rot/sprout or undersized, unfit for retail sale",
        }
        price = round(today_mandi_modal_price * GRADE_PRICE_FACTOR[final], 1)

        return {
            "grade": GRADE_DISPLAY[final],
            "legacy_label": GRADE_TO_AI_LABEL[final],   # for ai_model.log_user_sample_and_retrain
            "grade_breakdown": {"size": g_size, "visual": g_visual, "ai": g_ai},
            "limited_by": limited_by,
            "quality_status": f"{descriptions[final]} (AI confidence: {confidence * 100:.0f}%)",
            "agmark_standard": descriptions[final],
            "ai_confidence": round(confidence, 2),
            "metrics": {
                "diameter_mm": round(diameter_mm, 1),
                "area_cm2": round(area_cm2, 2),
                "pixel_area": round(pixel_area, 0),
                "max_diameter_px": round(max_diameter_px, 1),
            },
            "defects": {
                "color_score": color,
                "rot_percentage": rot,
                "sprout_percentage": sprout,
                "defect_percentage": defect,
                "primary_defect": primary_defect,
            },
            "market_estimate": {
                "estimated_price_inr_per_kg": price,
                "currency": "INR (₹)",
            },
        }


if __name__ == "__main__":
    clf = OnionAGMARKClassifier(pixels_per_mm=2.5, use_ai_model=False)  # rules only, no model file needed

    clean = {"rot_percentage": 0.0, "sprout_percentage": 0.0, "color_score": 0.9,
             "defect_percentage": 1.0, "primary_defect": "Sound Produce (No Defect)"}
    blemish = {"rot_percentage": 2.0, "sprout_percentage": 0.0, "color_score": 0.6,
               "defect_percentage": 9.0, "primary_defect": "Bruise / Wet Patch"}
    rotten = {"rot_percentage": 6.6, "sprout_percentage": 0.0, "color_score": 0.59,
              "defect_percentage": 11.0, "primary_defect": "Rot / Dark Decay"}
    very_bad = {"rot_percentage": 12.0, "sprout_percentage": 0.0, "color_score": 0.3,
                "defect_percentage": 30.0, "primary_defect": "Rot / Dark Decay"}

    assert clf.classify_onion(15000, 150, clean, 42)["grade"] == "Grade A"       # big + clean
    assert clf.classify_onion(15000, 150, blemish, 42)["grade"] == "Grade B"     # big but blemished
    assert clf.classify_onion(15000, 150, rotten, 42)["grade"] == "Grade C"      # big but visibly rotting
    r = clf.classify_onion(15000, 150, very_bad, 42)
    assert r["grade"] == "Grade D" and r["market_estimate"]["estimated_price_inr_per_kg"] == 0.0
    assert clf.classify_onion(8000, 100, clean, 42)["grade"] == "Grade B"        # clean but medium size
    print("All A/B/C/D checks passed.")