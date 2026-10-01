import os
import base64
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

try:
    import requests

    HAS_REQUESTS = True

except Exception:
    requests = None
    HAS_REQUESTS = False

try:
    from ultralytics import YOLO

    HAS_YOLO = True

except Exception as e:
    YOLO = None
    HAS_YOLO = False

    print(
        f"Notice: YOLO import bypassed due to system policy ({e}). "
        "Operating with OpenCV watershed segmentation."
    )

from src.preprocessing import get_onion_color_mask
from src.sam_segmenter import segment_onions, CHECKPOINT as SAM_CHECKPOINT


class OnionDetectorStub:
    """
    Onion detection and segmentation pipeline.

    Supports:
        - Red onions
        - Yellow onions
        - White onions
        - Individual onions in batches
        - Touching / partially overlapping onions
        - Webcam frames
        - Human hand / arm rejection

    Detection modes (tried in order, first usable result wins):
        1. Roboflow hosted model, when an API key + model ID are configured.
           This is the most accurate option since it's a real object
           detector trained on actual produce photos, rather than a
           colour/shape heuristic -- it's what should be used to get
           reliable detection instead of the OpenCV fallback below.
        2. Segment Anything (SAM) instance segmentation, when torch,
           segment_anything and the checkpoint file are present. Class-
           agnostic, so it separates touching onions in a heap far better
           than any colour/threshold method; each mask is then validated
           with the same onion-colour / skin / shape checks as every
           other tier. Runs BEFORE local YOLO because the default
           yolov8n-seg.pt is a COCO model with no onion class.
        3. Local YOLO segmentation weights, when available.
        4. OpenCV color-mask + watershed fallback (always available,
           least accurate on cluttered / touching onions).
    """

    def __init__(
        self,
        model_weights: str = "yolov8n-seg.pt",
        use_yolo: bool = True,
        roboflow_api_key: Optional[str] = None,
        roboflow_model_id: Optional[str] = None,
        roboflow_confidence: Optional[float] = None,
        roboflow_overlap: Optional[float] = None,
        use_roboflow: bool = True,
        use_sam: bool = True,
    ):
        self.model_weights = model_weights
        self.model: Optional[Any] = None

        if HAS_YOLO and use_yolo:
            self._load_model()

        # --------------------------------------------------------
        # Segment Anything (SAM) tier. Enabled only when everything it
        # needs is actually present; otherwise it is skipped silently
        # (with a one-line notice) and the older tiers run as before.
        # Set USE_SAM=0 in .env to force it off.
        # --------------------------------------------------------

        self.sam_enabled = False
        if use_sam and os.environ.get("USE_SAM", "1") != "0":
            import importlib.util

            if importlib.util.find_spec("torch") is None:
                sam_reason = "torch is not installed"
            elif importlib.util.find_spec("segment_anything") is None:
                sam_reason = "segment_anything is not installed"
            elif not os.path.exists(SAM_CHECKPOINT):
                sam_reason = f"checkpoint not found at {SAM_CHECKPOINT}"
            else:
                sam_reason = None
                self.sam_enabled = True
                print(f"SAM instance segmentation enabled ({SAM_CHECKPOINT}).")

            if sam_reason:
                print(f"Notice: SAM segmentation disabled ({sam_reason}).")

        # --------------------------------------------------------
        # Roboflow hosted inference configuration.
        #
        # NEVER hardcode the API key here. It's read from an
        # environment variable (or passed in explicitly by the
        # caller) so it never ends up committed to source control.
        #   export ROBOFLOW_API_KEY="your_private_api_key"
        #   export ROBOFLOW_MODEL_ID="your-project-id/1"
        # --------------------------------------------------------

        self.roboflow_api_key = roboflow_api_key or os.environ.get("ROBOFLOW_API_KEY")
        self.roboflow_model_id = roboflow_model_id or os.environ.get("ROBOFLOW_MODEL_ID")

        self.roboflow_confidence = (
            roboflow_confidence
            if roboflow_confidence is not None
            else float(os.environ.get("ROBOFLOW_CONFIDENCE", "0.40"))
        )
        self.roboflow_overlap = (
            roboflow_overlap
            if roboflow_overlap is not None
            else float(os.environ.get("ROBOFLOW_OVERLAP", "0.50"))
        )

        # Optional: comma-separated list of class names (as labeled in
        # your Roboflow project) that count as a valid onion detection,
        # e.g. "onion,red-onion,yellow-onion". Leave unset to accept any
        # returned class.
        accepted = os.environ.get("ROBOFLOW_ACCEPTED_CLASSES", "")
        self.roboflow_accepted_classes = (
            {c.strip().lower() for c in accepted.split(",") if c.strip()}
            if accepted
            else None
        )

        self.roboflow_enabled = bool(
            use_roboflow
            and HAS_REQUESTS
            and self.roboflow_api_key
            and self.roboflow_model_id
        )

        if use_roboflow and not self.roboflow_enabled:
            if not HAS_REQUESTS:
                reason = "the 'requests' package is not installed"
            elif not self.roboflow_api_key:
                reason = "ROBOFLOW_API_KEY is not set"
            elif not self.roboflow_model_id:
                reason = "ROBOFLOW_MODEL_ID is not set"
            else:
                reason = "configuration incomplete"

            print(
                f"Notice: Roboflow hosted detection disabled ({reason}). "
                "Falling back to local YOLO / OpenCV watershed detection."
            )
        elif self.roboflow_enabled:
            print(
                f"Roboflow hosted detection enabled: model '{self.roboflow_model_id}'."
            )

    # ============================================================
    # YOLO
    # ============================================================

    def _load_model(self):
        try:
            self.model = YOLO(self.model_weights)

            print(
                f"YOLO model loaded successfully: "
                f"{self.model_weights}"
            )

        except Exception as e:

            print(
                f"Notice: YOLO model loading failed ({e}). "
                "Using OpenCV watershed segmentation."
            )

            self.model = None

    # ============================================================
    # ROBOFLOW HOSTED DETECTION
    # ============================================================

    def detect_with_roboflow(
        self,
        image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Runs inference against a Roboflow hosted object-detection model.

        This calls the Serverless Hosted API V2 endpoint
        (https://serverless.roboflow.com/:model_id), which is what current
        Roboflow trainings need -- it accepts BOTH the legacy
        {project}/{version} ID format and the newer {workspace}/{model-slug}
        format used by recently trained models (e.g. YOLO26, Roboflow
        Instant). The older https://detect.roboflow.com endpoint only
        understands the legacy format, so a model trained today would
        silently 404 against it. Sending the image as base64 in the
        request body avoids a separate upload step or URL hosting.
        On any failure (network issue, bad key, model unavailable) this
        returns an empty list rather than raising, so the caller can fall
        through to the next detection tier instead of crashing the whole
        request.
        """
        items: List[Dict[str, Any]] = []

        if not self.roboflow_enabled:
            return items

        try:
            ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                print("Notice: Roboflow inference skipped (failed to encode image).")
                return items

            img_b64 = base64.b64encode(buffer).decode("ascii")

            url = f"https://serverless.roboflow.com/{self.roboflow_model_id}"
            params = {
                "api_key": self.roboflow_api_key,
                "confidence": self.roboflow_confidence * 100.0,  # API takes 0-100
                "overlap": self.roboflow_overlap * 100.0,
            }

            response = requests.post(
                url,
                params=params,
                data=img_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15,
            )
            response.raise_for_status()
            predictions = response.json().get("predictions", [])
            print(f"Roboflow raw predictions: {len(predictions)}")
            for p in predictions:
                print(f"  class={p.get('class')} conf={p.get('confidence'):.2f}")
        except Exception as e:
            print(f"Notice: Roboflow inference failed ({e}). Falling back to local detection.")
            return items

        img_h, img_w = image.shape[:2]
        item_id = 1

        for pred in predictions:
            try:
                confidence = float(pred.get("confidence", 0.0))
                class_name = str(pred.get("class", "")).strip().lower()

                # Belt-and-braces: the `confidence` query param already
                # asks Roboflow to filter server-side, but don't trust
                # that blindly -- re-check locally too.
                if confidence < self.roboflow_confidence:
                    continue

                if self.roboflow_accepted_classes and class_name not in self.roboflow_accepted_classes:
                    continue

                contour = self._roboflow_prediction_to_contour(pred, img_w, img_h)
                if contour is None:
                    continue

                if self.touches_frame_border(contour, image.shape):
                    continue

                area, diameter = self.measure_contour(contour)
                if area < 250.0:
                    continue

                items.append({
                    "item_id": item_id,
                    "pixel_area": area,
                    "max_diameter": diameter,
                    "contour": contour,
                    # Kept alongside the geometry so a caller (e.g. the
                    # API layer) can optionally cross-check or display
                    # what the trained model itself called this item,
                    # without it being required for grading to work.
                    "roboflow_class": pred.get("class"),
                    "roboflow_confidence": round(confidence, 3),
                })
                item_id += 1

            except Exception as e:
                print(f"Notice: Skipping malformed Roboflow prediction ({e}).")
                continue

        # Sort left -> right / top -> bottom, same convention as the
        # watershed fallback, and renumber accordingly.
        def item_center(item):
            x, y, w, h = cv2.boundingRect(item["contour"])
            return (y + h / 2.0, x + w / 2.0)

        items.sort(key=item_center)
        for index, item in enumerate(items, start=1):
            item["item_id"] = index

        return items

    def _roboflow_prediction_to_contour(
        self,
        pred: Dict[str, Any],
        img_w: int,
        img_h: int
    ) -> Optional[np.ndarray]:
        """
        Converts one Roboflow prediction into an OpenCV contour
        (Nx1x2 int32 array), in full-image pixel coordinates, so it can
        flow through the same measure_contour / bounding-box / defect
        analysis code as every other detection path.

        Instance-segmentation models return a `points` polygon, which is
        used directly for a tighter silhouette. Plain object-detection
        models only return a center + width/height box, which is turned
        into a rectangular contour instead.
        """
        points = pred.get("points")

        if points:
            poly = np.array(
                [[int(round(p["x"])), int(round(p["y"]))] for p in points],
                dtype=np.int32
            )
            if poly.shape[0] < 3:
                return None
            return poly.reshape(-1, 1, 2)

        try:
            cx = float(pred["x"])
            cy = float(pred["y"])
            w = float(pred["width"])
            h = float(pred["height"])
        except (KeyError, TypeError, ValueError):
            return None

        x0 = max(0, int(round(cx - w / 2.0)))
        y0 = max(0, int(round(cy - h / 2.0)))
        x1 = min(img_w - 1, int(round(cx + w / 2.0)))
        y1 = min(img_h - 1, int(round(cy + h / 2.0)))

        if x1 <= x0 or y1 <= y0:
            return None

        rect = np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            dtype=np.int32
        )
        return rect.reshape(-1, 1, 2)

    # ============================================================
    # SAM (SEGMENT ANYTHING) DETECTION
    # ============================================================

    def detect_with_sam(
        self,
        image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Segments every onion with SAM, then keeps only masks that pass the
        same validation every other tier uses (frame border, size, shape,
        human skin, onion colour). Returns items in the standard format
        (item_id / pixel_area / max_diameter / contour), so nothing
        downstream needs to know which tier produced them.

        `image` should be the normal (not background-blacked-out) BGR
        image: SAM finds object edges itself, and a colour-thresholded
        foreground can hollow out dark or rotten onions.
        """
        items: List[Dict[str, Any]] = []

        if not self.sam_enabled:
            return items

        try:
            segments = segment_onions(image)
        except Exception as e:
            print(f"Notice: SAM inference failed ({e}). Falling back to next detection tier.")
            return items

        print(f"SAM: {len(segments)} candidate masks after filtering.")

        for seg in segments:
            contours, _ = cv2.findContours(
                seg["mask"],
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)

            if not self.is_valid_produce_contour(image, contour):
                continue

            area, diameter = self.measure_contour(contour)

            items.append({
                "item_id": len(items) + 1,
                "pixel_area": area,
                "max_diameter": diameter,
                "contour": contour,
            })

        def item_center(item):
            x, y, w, h = cv2.boundingRect(item["contour"])
            return (y + h / 2.0, x + w / 2.0)

        items.sort(key=item_center)
        for index, item in enumerate(items, start=1):
            item["item_id"] = index

        print(f"SAM: {len(items)} onions kept after validation.")
        return items

    # ============================================================
    # FRAME BORDER REJECTION
    # ============================================================

    def touches_frame_border(
        self,
        contour: np.ndarray,
        img_shape: Tuple[int, int]
    ) -> bool:

        x, y, w, h = cv2.boundingRect(contour)

        img_h, img_w = img_shape[:2]

        # Reject extremely large regions.
        if (
            w >= img_w * 0.90 or
            h >= img_h * 0.90
        ):
            return True

        # Check whether contour actually touches the image border.
        points = contour.reshape(-1, 2)

        if (
            np.any(points[:, 0] <= 1) or
            np.any(points[:, 1] <= 1) or
            np.any(points[:, 0] >= img_w - 2) or
            np.any(points[:, 1] >= img_h - 2)
        ):
            return True

        return False

    # ============================================================
    # HUMAN SKIN / ARM DETECTION
    # ============================================================

    def is_human_hand_or_skin(
        self,
        bgr_crop: np.ndarray
    ) -> bool:

        if (
            bgr_crop is None or
            bgr_crop.size == 0
        ):
            return True

        ycrcb = cv2.cvtColor(
            bgr_crop,
            cv2.COLOR_BGR2YCrCb
        )

        lower_skin = np.array(
            [0, 133, 77],
            dtype=np.uint8
        )

        upper_skin = np.array(
            [255, 173, 127],
            dtype=np.uint8
        )

        skin_mask = cv2.inRange(
            ycrcb,
            lower_skin,
            upper_skin
        )


        hsv = cv2.cvtColor(
            bgr_crop,
            cv2.COLOR_BGR2HSV
        )

        lower_hsv_skin = np.array(
            [0, 20, 50],
            dtype=np.uint8
        )

        upper_hsv_skin = np.array(
            [25, 170, 255],
            dtype=np.uint8
        )

        hsv_skin_mask = cv2.inRange(
            hsv,
            lower_hsv_skin,
            upper_hsv_skin
        )


        total_pixels = float(
            bgr_crop.shape[0] *
            bgr_crop.shape[1]
        )

        skin_ratio = (
            np.count_nonzero(skin_mask) /
            total_pixels
        )

        hsv_skin_ratio = (
            np.count_nonzero(hsv_skin_mask) /
            total_pixels
        )


        # --------------------------------------------------------
        # Onion-color safeguard
        # --------------------------------------------------------

        onion_color_mask = get_onion_color_mask(hsv)

        onion_ratio = (
            np.count_nonzero(onion_color_mask) /
            total_pixels
        )

        # If most of the region looks like onion,
        # don't reject it as skin merely because some skin
        # threshold overlaps.
        if onion_ratio > 0.50:
            return False


        if (
            skin_ratio > 0.28 or
            hsv_skin_ratio > 0.28
        ):

            print(
                "Notice: Candidate rejected "
                "(human hand/arm/face detected)."
            )

            return True

        return False

    # ============================================================
    # ONION COLOR VALIDATION
    # ============================================================

    def verify_onion_color_match(
        self,
        bgr_crop: np.ndarray,
        contour: Optional[np.ndarray] = None
    ) -> bool:

        if (
            bgr_crop is None or
            bgr_crop.size == 0
        ):
            return False

        hsv = cv2.cvtColor(
            bgr_crop,
            cv2.COLOR_BGR2HSV
        )

        onion_mask = get_onion_color_mask(hsv)

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # If a contour is available, calculate onion coverage
        # only INSIDE the contour.
        #
        # Previously your bounding rectangle included background.
        # --------------------------------------------------------

        if contour is not None:

            contour_mask = np.zeros(
                bgr_crop.shape[:2],
                dtype=np.uint8
            )

            cv2.drawContours(
                contour_mask,
                [contour],
                -1,
                255,
                -1
            )

            valid_pixels = (
                contour_mask > 0
            )

            if not np.any(valid_pixels):
                return False

            onion_pixels = (
                onion_mask[valid_pixels] > 0
            )

            onion_ratio = (
                np.count_nonzero(onion_pixels) /
                np.count_nonzero(valid_pixels)
            )

        else:

            total_pixels = float(
                bgr_crop.shape[0] *
                bgr_crop.shape[1]
            )

            onion_ratio = (
                np.count_nonzero(onion_mask) /
                total_pixels
            )


        # Slightly lower than your original 35%.
        #
        # This is important for:
        #   - white onions
        #   - highlights
        #   - shadows
        #   - partially segmented onions
        #
        if onion_ratio < 0.20:

            print(
                "Notice: Candidate rejected "
                f"(onion color ratio: {onion_ratio:.2f})"
            )

            return False

        return True

    # ============================================================
    # CONTOUR VALIDATION
    # ============================================================

    def is_valid_produce_contour(
        self,
        bgr_image: np.ndarray,
        contour: np.ndarray
    ) -> bool:

        if self.touches_frame_border(
            contour,
            bgr_image.shape
        ):
            return False


        area = float(
            cv2.contourArea(contour)
        )

        # Lower than your original 800.
        #
        # Watershed markers can generate smaller contours,
        # especially for onions farther from the camera.
        if area < 250.0:
            return False


        perimeter = float(
            cv2.arcLength(
                contour,
                True
            )
        )

        if perimeter <= 0:
            return False


        circularity = (
            4.0 *
            np.pi *
            area /
            (perimeter ** 2)
        )


        x, y, w, h = cv2.boundingRect(
            contour
        )

        aspect_ratio = (
            float(w) / float(h)
            if h > 0
            else 0.0
        )


        # Onions don't need to be perfect circles.
        if circularity < 0.20:
            return False

        if (
            aspect_ratio < 0.30 or
            aspect_ratio > 3.0
        ):
            return False


        # --------------------------------------------------------
        # Crop
        # --------------------------------------------------------

        crop = bgr_image[
            y:y + h,
            x:x + w
        ]

        if (
            crop is None or
            crop.size == 0
        ):
            return False


        # --------------------------------------------------------
        # Human rejection
        # --------------------------------------------------------

        if self.is_human_hand_or_skin(crop):
            return False


        # --------------------------------------------------------
        # Onion color
        #
        # Convert contour coordinates to crop coordinates.
        # --------------------------------------------------------

        local_contour = (
            contour -
            np.array(
                [[x, y]],
                dtype=np.int32
            )
        )

        if not self.verify_onion_color_match(
            crop,
            local_contour
        ):
            return False


        return True

    # ============================================================
    # MEASURE CONTOUR
    # ============================================================

    def measure_contour(
        self,
        contour: np.ndarray
    ) -> Tuple[float, float]:

        area = float(
            cv2.contourArea(contour)
        )

        (_, _), radius = (
            cv2.minEnclosingCircle(contour)
        )

        max_diameter = float(
            radius * 2.0
        )

        return area, max_diameter

    # ============================================================
    # YOLO DETECTION
    # ============================================================

    def detect_with_yolo(
        self,
        image: np.ndarray
    ) -> List[Dict[str, Any]]:

        items = []

        if self.model is None:
            return items


        try:

            results = self.model(
                image,
                verbose=False
            )


            item_id = 1


            for result in results:

                if (
                    result.boxes is None or
                    result.masks is None
                ):
                    continue


                boxes = result.boxes

                masks = result.masks.data


                for box, mask_data in zip(
                    boxes,
                    masks
                ):

                    cls_id = int(
                        box.cls[0]
                        .cpu()
                        .numpy()
                    )


                    # ------------------------------------------------
                    # DO NOT blindly reject class 0.
                    #
                    # A custom onion model may use:
                    #   class 0 = onion
                    #
                    # COCO uses:
                    #   class 0 = person
                    #
                    # If using a custom onion model, don't use the
                    # COCO person assumption.
                    # ------------------------------------------------

                    class_name = None

                    try:
                        class_name = (
                            self.model.names.get(
                                cls_id,
                                str(cls_id)
                            )
                        )

                    except Exception:
                        pass


                    if class_name is not None:

                        name_lower = str(
                            class_name
                        ).lower()

                        # For a custom model, accept onion classes.
                        # If model is generic COCO, reject person.
                        if name_lower == "person":
                            continue


                    # ------------------------------------------------
                    # Mask
                    # ------------------------------------------------

                    mask_np = (
                        mask_data
                        .cpu()
                        .numpy()
                        * 255
                    ).astype(
                        np.uint8
                    )


                    # Resize mask if necessary.
                    if mask_np.shape[:2] != image.shape[:2]:

                        mask_np = cv2.resize(
                            mask_np,
                            (
                                image.shape[1],
                                image.shape[0]
                            ),
                            interpolation=cv2.INTER_NEAREST
                        )


                    _, binary_mask = cv2.threshold(
                        mask_np,
                        127,
                        255,
                        cv2.THRESH_BINARY
                    )


                    contours, _ = cv2.findContours(
                        binary_mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )


                    if not contours:
                        continue


                    contour = max(
                        contours,
                        key=cv2.contourArea
                    )


                    if not self.is_valid_produce_contour(
                        image,
                        contour
                    ):
                        continue


                    area, diameter = (
                        self.measure_contour(
                            contour
                        )
                    )


                    items.append({

                        "item_id": item_id,

                        "pixel_area": area,

                        "max_diameter": diameter,

                        "contour": contour

                    })


                    item_id += 1


            return items


        except Exception as e:

            print(
                f"YOLO inference failed: {e}"
            )

            return []

    # ============================================================
    # MAIN DETECTOR
    # ============================================================

    def detect_and_measure(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        raw_image: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        `image` / `mask` are the background-subtracted foreground image and
        its mask (used by Roboflow / YOLO / watershed, as before).
        `raw_image` is the un-masked BGR photo; only the SAM tier uses it.
        If omitted, SAM falls back to `image`.
        """

        if (
            image is None or
            image.size == 0
        ):
            return []


        # --------------------------------------------------------
        # Try Roboflow hosted detection first.
        #
        # This is a real object detector trained on actual produce
        # photos, so it's the most reliable tier -- notably it doesn't
        # depend on the white-background / color-mask assumptions the
        # OpenCV watershed fallback needs, and isn't fooled by touching
        # onions the way a pure color-mask blob can be.
        # --------------------------------------------------------

        if self.roboflow_enabled:

            items = self.detect_with_roboflow(
                image
            )

            if items:

                return items


        # --------------------------------------------------------
        # Try SAM (Segment Anything) next: class-agnostic instance
        # segmentation, the best local option for touching onions / heaps.
        # --------------------------------------------------------

        if self.sam_enabled:

            items = self.detect_with_sam(
                raw_image if raw_image is not None else image
            )

            if items:

                return items


        # --------------------------------------------------------
        # Try local YOLO weights next
        # --------------------------------------------------------

        if self.model is not None:

            items = self.detect_with_yolo(
                image
            )

            if items:

                return items


        # --------------------------------------------------------
        # OpenCV watershed fallback
        # --------------------------------------------------------

        return self._fallback_watershed_measure(
            image,
            mask
        )

    # ============================================================
    # WATERSHED FALLBACK
    # ============================================================

    def _fallback_watershed_measure(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:

        if (
            mask is None or
            np.count_nonzero(mask) == 0
        ):
            return []


        # --------------------------------------------------------
        # Normalize mask
        # --------------------------------------------------------

        binary = (
            mask > 0
        ).astype(
            np.uint8
        ) * 255


        # --------------------------------------------------------
        # Morphological cleanup
        # --------------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )


        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )


        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )


        # --------------------------------------------------------
        # Remove tiny connected components
        # --------------------------------------------------------

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )


        cleaned = np.zeros_like(
            binary
        )


        for label in range(
            1,
            num_labels
        ):

            area = stats[
                label,
                cv2.CC_STAT_AREA
            ]

            if area >= 300:

                cleaned[
                    labels == label
                ] = 255


        binary = cleaned


        if np.count_nonzero(binary) == 0:
            return []


        # ========================================================
        # DISTANCE TRANSFORM
        # ========================================================

        dist = cv2.distanceTransform(
            binary,
            cv2.DIST_L2,
            5
        )


        if dist.max() <= 0:
            return []


        # ========================================================
        # CREATE SURE FOREGROUND VIA LOCAL-MAXIMA PEAK SEEDS
        #
        # IMPORTANT:
        #
        # A single global threshold on dist.max() (the old "0.35 *
        # dist.max()" rule) only splits touching onions apart when the
        # connecting "neck" between them dips deep enough in the distance
        # transform. Two onions of similar size that are merely touching
        # (not overlapping much) often keep that neck ABOVE the cutoff, so
        # sure_fg stays ONE connected blob and the whole pair gets treated
        # as a single, oversized item -- this is why touching onions were
        # being under-counted.
        #
        # Instead, find local maxima of the distance transform: a pixel
        # whose distance value is the largest within its own neighborhood
        # is a candidate onion center. Each onion still has its own local
        # peak at its own center regardless of the neighboring onion's
        # peak, so this reliably yields one seed per onion even when their
        # footprints are fused together in `binary`.
        # ========================================================

        foreground_threshold = (
            0.35 *
            dist.max()
        )

        # Neighborhood radius for the peak search, sized relative to how
        # large onions actually appear in this image (dist.max() is a
        # proxy for the largest onion's radius). Kept smaller than a
        # typical onion radius so two touching onions still produce two
        # separate peaks, but large enough to ignore small noisy bumps.
        # `dist.max()` is a numpy scalar (np.float32). Multiplying/rounding
        # it can, depending on the installed numpy/opencv build, leave a
        # numpy float type (e.g. np.float64) sitting inside `peak_window`
        # even after int(round(...)) -- some OpenCV Python bindings only
        # accept plain Python ints for a ksize tuple and raise exactly
        # "Can't parse 'ksize'. Sequence item with index 0 has a wrong
        # type" if they see anything else. Force a genuine Python int here
        # (not just at assignment, but again explicitly where the tuple is
        # built) and clamp it to a sane range so a huge dist.max() on a
        # large image can't create an absurdly expensive kernel either.
        peak_window = int(round(float(max(9, dist.max() * 0.6))))
        if peak_window % 2 == 0:
            peak_window += 1
        peak_window = max(3, min(peak_window, 51))

        local_max_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (int(peak_window), int(peak_window))
        )

        local_max = cv2.dilate(dist, local_max_kernel)

        # A pixel is a peak if it equals its neighborhood's maximum
        # (allowing for float rounding) AND still clears the usual
        # "sure foreground" cutoff.
        sure_fg = np.zeros_like(binary)
        sure_fg[
            (dist >= local_max - 1e-3) &
            (dist >= foreground_threshold)
        ] = 255

        peak_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        # --------------------------------------------------------
        # Turn the sparse peak pixels into small solid seed blobs.
        #
        # IMPORTANT: a local-maxima peak is often only 1-3 pixels wide.
        # Running the old MORPH_OPEN (an erosion followed by a dilation)
        # on something that small erases it outright -- erosion needs the
        # full 3x3 neighborhood to already be foreground, which a lone
        # peak pixel never satisfies. That silently emptied sure_fg and
        # fell through to the "no markers" branch below, which is exactly
        # what was reverting touching onions back to a single merged blob.
        # Dilating (not opening) instead grows each peak into a compact
        # marker without needing pre-existing neighboring foreground.
        # --------------------------------------------------------

        sure_fg = cv2.dilate(
            sure_fg,
            peak_kernel,
            iterations=1
        )

        # Local-maxima search can occasionally wipe out every seed on a
        # perfectly flat / tiny blob (rare, but possible). Fall back to
        # the plain global-threshold behaviour (with the usual opening,
        # since that path yields a broad contiguous region that can
        # contain speckle noise) rather than losing the item entirely.
        if np.count_nonzero(sure_fg) == 0:
            sure_fg = np.zeros_like(binary)
            sure_fg[dist >= foreground_threshold] = 255
            sure_fg = cv2.morphologyEx(
                sure_fg, cv2.MORPH_OPEN, peak_kernel, iterations=1
            )


        # ========================================================
        # SURE BACKGROUND
        # ========================================================

        sure_bg = cv2.dilate(
            binary,
            kernel,
            iterations=2
        )


        # ========================================================
        # UNKNOWN REGION
        # ========================================================

        unknown = cv2.subtract(
            sure_bg,
            sure_fg
        )


        # ========================================================
        # CONNECTED COMPONENT MARKERS
        # ========================================================

        num_markers, markers = (
            cv2.connectedComponents(
                sure_fg
            )
        )


        if num_markers <= 1:

            # No useful foreground markers.
            # Fall back to normal contours.
            return self._simple_contour_fallback(
                image,
                binary
            )


        # ========================================================
        # WATERSHED
        # ========================================================

        markers = markers + 1

        markers[
            unknown == 255
        ] = 0


        image_for_watershed = (
            image.copy()
        )


        cv2.watershed(
            image_for_watershed,
            markers
        )


        # ========================================================
        # EXTRACT INDIVIDUAL REGIONS
        # ========================================================

        items = []

        item_id = 1


        unique_labels = np.unique(
            markers
        )


        for label in unique_labels:

            # 1 = background
            # -1 = watershed boundary

            if label <= 1:
                continue


            region_mask = np.zeros(
                binary.shape,
                dtype=np.uint8
            )


            region_mask[
                markers == label
            ] = 255


            contours, _ = cv2.findContours(
                region_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )


            if not contours:
                continue


            contour = max(
                contours,
                key=cv2.contourArea
            )


            area = cv2.contourArea(
                contour
            )


            if area < 250:
                continue


            if not self.is_valid_produce_contour(
                image,
                contour
            ):
                continue


            measured_area, diameter = (
                self.measure_contour(
                    contour
                )
            )


            items.append({

                "item_id": item_id,

                "pixel_area": measured_area,

                "max_diameter": diameter,

                "contour": contour

            })


            item_id += 1


        # ========================================================
        # SORT LEFT → RIGHT / TOP → BOTTOM
        # ========================================================

        def contour_center(item):

            contour = item["contour"]

            M = cv2.moments(
                contour
            )

            if M["m00"] == 0:

                return (
                    999999,
                    999999
                )

            cx = (
                M["m10"] /
                M["m00"]
            )

            cy = (
                M["m01"] /
                M["m00"]
            )

            return (
                cy,
                cx
            )


        items.sort(
            key=contour_center
        )


        # Re-number after sorting.

        for index, item in enumerate(
            items,
            start=1
        ):

            item["item_id"] = index


        return items

    # ============================================================
    # SIMPLE CONTOUR FALLBACK
    # ============================================================

    def _simple_contour_fallback(
        self,
        image: np.ndarray,
        binary: np.ndarray
    ) -> List[Dict[str, Any]]:

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        items = []


        for contour in contours:

            if not self.is_valid_produce_contour(
                image,
                contour
            ):
                continue


            area, diameter = (
                self.measure_contour(
                    contour
                )
            )


            items.append({

                "item_id": len(items) + 1,

                "pixel_area": area,

                "max_diameter": diameter,

                "contour": contour

            })


        return items


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":

    print(
        "Testing Onion Detector..."
    )


    detector = OnionDetectorStub()


    # ------------------------------------------------------------
    # Synthetic human arm
    # ------------------------------------------------------------

    arm_img = np.full(
        (400, 400, 3),
        (120, 140, 210),
        dtype=np.uint8
    )


    cv2.circle(
        arm_img,
        (200, 200),
        80,
        (120, 140, 210),
        -1
    )


    gray = cv2.cvtColor(
        arm_img,
        cv2.COLOR_BGR2GRAY
    )


    _, arm_mask = cv2.threshold(
        gray,
        1,
        255,
        cv2.THRESH_BINARY
    )


    results = detector.detect_and_measure(
        arm_img,
        arm_mask
    )


    print(
        f"Detected items in human arm image: "
        f"{len(results)}"
    )


    assert len(results) == 0, (
        "Human arm candidate contour "
        "was wrongly accepted!"
    )


    print(
        "Strict Human Arm Rejection "
        "verified successfully."
    )