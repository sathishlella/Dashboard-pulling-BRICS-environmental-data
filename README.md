# Tourism Energy & CO₂ ML Dashboard

Interactive machine learning dashboard for analyzing tourism energy consumption and CO₂ emissions across China's 30 provinces (2008–2023).

## Features

- **Model Arena** — Train & compare 6 ML algorithms (XGBoost, LightGBM, Random Forest, Gradient Boosting, Neural Network, Ridge) in real-time
- **What-If Predictor** — Adjust policy levers and see live ensemble predictions
- **Future Forecast** — Multi-year iterative projections with growth assumptions
- **Spatial Explorer** — Geographic spillover analysis between neighboring provinces
- **Explainability** — SHAP-based feature importance and individual prediction breakdowns
- **Live Terminal** — Transparent training logs

## Tech Stack

- Python, Streamlit, Plotly, scikit-learn, XGBoost, LightGBM, SHAP

## Setup

```bash
pip install -r tourism_co2/requirements.txt
```

## Run

```bash
cd tourism_co2
streamlit run live_ml_dashboard.py
```

## Data

Place `Tourism carbon emissions data.xlsx` in the `tourism_co2/` directory (not included in repo).

## Report Generator

```bash
python tourism_co2/generate_ml_report_for_professor.py
```

Generates `ml_report_for_professor.html` — a static HTML research report with identical model configurations.

---

**Research:** Dr. Danish | **Built by:** Sathish Lella
