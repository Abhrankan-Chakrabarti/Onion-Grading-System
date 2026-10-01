# 🧅 AGMARK AI Onion Quality Assessment & Dynamic Market Forecaster

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-red)
![AGMARK](https://img.shields.io/badge/AGMARK-Govt._of_India-yellow)

An end-to-end computer vision and market forecasting platform designed for **Smart India Hackathon (SIH)**. The system performs non-destructive quality grading of onions (Red, Yellow/Golden, and White varieties), extracts surface defects (Black Mold Rot, Top Sprouting, Sunburn Discoloration), enforces official **Government of India AGMARK millimeter standards**, and predicts 14-day wholesale Mandi price trends.

---

## ✨ Key Features

- **Multi-Variety Produce Vision Pipeline**:
  - CLAHE lighting normalization in LAB color space.
  - Multi-chromatic HSV masking for **Red/Purple**, **Yellow/Brown**, and **White/Ivory** onions.
  - Studio white background detection and Otsu inverse thresholding.
  - Distance Transform & Watershed segmentation for dense batch crates.
- **Anti-False Positive Filters**:
  - YCrCb & HSV human skin rejection ($Cr \in [133, 173], Cb \in [77, 127]$) to filter out hands, arms, and fists.
  - Inter-onion dark shadow gap rejection (evaluates rot strictly on valid onion skin).
  - Differentiates natural fresh green neck stems ($\le 12\%$) from actual post-harvest top sprouting ($> 12\%$).
- **AGMARK Millimeter Size Classification**:
  - **Grade A (Export Grade)**: $\ge 55\text{ mm}$ ($5.5\text{ cm}+$)
  - **Grade B (Domestic Wholesale)**: $40 - 55\text{ mm}$ ($4.0 - 5.5\text{ cm}$)
  - **Small Grade (Sambar / Pickle Size)**: $30 - 40\text{ mm}$ ($3.0 - 4.0\text{ cm}$)
  - **Reject**: $< 30\text{ mm}$ or severe rot/sprouting ($\ge 12\%$)
- **AGMARK 15% Bulk Lot Batch Tolerance**:
  - Enforces official bulk lot inspection standards so minor edge snippets do not cause false "Mixed Lot" rejects.
- **Dynamic Mandi Price Forecasting**:
  - Time-Series Ridge Regression 14-day price forecasting across major Indian trading hubs (Lasalgaon, Azadpur Delhi, Pune, Bengaluru, Kurnool).
  - Smart Cold Storage holding & selling advice.
- **Interactive Web Dashboard**:
  - Single-page dashboard built with Chart.js, webcam integration, camera distance calibration presets, and AGMARK reference tables.

---

## 📁 Repository Structure

```
onion_grading_system/
├── src/
│   ├── __init__.py
│   ├── api.py               # FastAPI web server & annotation pipeline
│   ├── classifier.py        # AGMARK millimeter classifier & dynamic pricing
│   ├── defect_analysis.py   # HSV defect diagnostic engine (Rot, Sprout, Color)
│   ├── detection.py         # YOLOv8 & OpenCV distance transform detector
│   ├── market_analytics.py # Time-series Mandi price predictor & cold storage advice
│   ├── preprocessing.py     # CLAHE & multi-variety HSV background subtractor
│   └── templates/
│       └── index.html       # Interactive web UI & Chart.js dashboard
├── requirements.txt         # Project dependencies
├── .gitignore               # Git ignore configuration
└── README.md                # System documentation
```

---

## 🚀 Quickstart Guide (Local Desktop Run)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application Server
```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Open in Browser
Navigate to **`http://127.0.0.1:8000`** in your browser.

---

## 📊 AGMARK Size Classification Table

| Grade | Diameter Range ($\text{mm}$) | Commercial Category | Wholesale Use |
| :--- | :--- | :--- | :--- |
| **Grade A** | **$\ge 55\text{ mm}$** | Large / Export Grade | Gulf & European Export |
| **Grade B** | **$40 - 55\text{ mm}$** | Medium / Wholesale | Domestic Wholesale |
| **Small Grade** | **$30 - 40\text{ mm}$** | Sambar / Pickle Size | Restaurants & Pickling |
| **Reject** | **$< 30\text{ mm}$** | Factory Dehydration | Processing Factories |

---

## 📜 License
This project is open-source under the MIT License.
