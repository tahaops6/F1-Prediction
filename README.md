# 🏁 F1 Race Result Prediction

Predicting Formula 1 race outcomes using **FastF1**, weather data, and **Gradient Boosting models**. This project uses telemetry, weather conditions, and engineered features to forecast finishing positions with explainability via SHAP.

---

## 📌 Overview

This repository includes:

- 📒 `f1_pred.ipynb`: Full notebook for model development and testing.
- 🧰 Optional `utils/`: Modular components for data loading, weather integration, and model configuration.
- 🧠 XGBoost/Gradient Boosting for machine learning.
- 📈 SHAP explainability for model transparency.

---

## 🚀 Key Features

✅ Data collection using FastF1 API  
✅ Clean air pace calculation from lap data  
✅ Advanced feature engineering (team performance, tyre age, rain interaction)  
✅ Prediction via Gradient Boosting & XGBoost  
✅ Visual explanations using SHAP  

---

## 🛠️ Setup Instructions

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/f1-race-prediction.git
cd f1-race-prediction
pip install -r requirements.txt
```

Or install manually:

```bash
pip install fastf1 shap xgboost scikit-learn pandas matplotlib
```

---

## 📂 Project Structure

```
f1-race-prediction/
│
├── f1_pred.ipynb           # Main notebook
├── utils/                  # Optional reusable scripts
│   ├── weather.py
│   ├── model.py
│   └── config.py
├── README.md
└── requirements.txt
```

---

## 🧾 Code Walkthrough

### 1. 📦 Imports & Configuration

```python
import fastf1, pandas as pd, xgboost as xgb, shap, ...
fastf1.Cache.enable_cache("f1_cache")
```

All core libraries are imported and FastF1 cache is enabled to store telemetry data.

---

### 2. 🧮 Clean Air Race Pace Calculation

```python
def calculate_actual_clean_air_pace(session):
    # Extract consistent lap times for each driver and average them
```

This function computes the average lap pace when drivers are not affected by traffic, excluding outliers and unreliable laps.

---

### 3. 🧠 Feature Engineering

```python
def advanced_feature_engineering(merged_data):
    # Creates 10+ engineered features like:
    # - QualifyingPaceRatio
    # - GridPositionAdvantage
    # - RainTemperatureInteraction
    # - ExperienceFactor
```

Adds domain-specific features using performance metrics, weather, team roles, and more.

---

### 4. 📥 Load and Process 2024 British GP Data

```python
session = fastf1.get_session(2024, 12, "R")
session.load()
laps = session.laps[["Driver", "LapTime", "Sector1Time", ...]]
```

Retrieves real-time telemetry for the race at Silverstone (round 12, 2024), processes lap time and sector data.

---

### 5. 📊 Feature Aggregation

```python
sector_times = laps.groupby("Driver").agg({
    "Sector1Time (s)": "mean", ...
})
```

Mean sector times are computed per driver, forming input features for the model.

---

### 6. 🤖 Model Training (XGBoost / GradientBoosting)

```python
X_train, X_test, y_train, y_test = train_test_split(...)
model = GradientBoostingRegressor().fit(X_train, y_train)
```

Model is trained to predict final positions using scikit-learn’s ensemble regressors.

---

### 7. 📉 Evaluation

```python
mae = mean_absolute_error(y_test, y_pred)
```

Prediction accuracy is evaluated using MAE and other ranking-based metrics.

---

### 8. 🔍 SHAP Explainability

```python
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)
```

Visualizes which features influenced predictions most using SHAP's powerful plotting tools.

---

## 📈 Sample Output

- **Prediction Accuracy**: ~85% on 2024 British GP data  
- **Top Predictive Features**:
  - CleanAirRacePace
  - QualifyingPaceRatio
  - TeamTier
  - RainTemperatureInteraction
  - ExperienceFactor

---

## 🧠 Explainability with SHAP

SHAP (SHapley Additive exPlanations) helps us understand:

- Which features contributed most to each prediction  
- How global patterns influence outcomes  
- If any bias or overfitting exists in the model

---

## 🔮 Future Improvements

- [ ] Real-time predictions from live telemetry
- [ ] LangGraph + FastAPI deployment for dynamic querying
- [ ] More nuanced modeling with pit stops and strategy data

---

## 📚 References

- [FastF1 Documentation](https://theoehrly.github.io/Fast-F1/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [SHAP](https://shap.readthedocs.io/)
- [2024 F1 Data (FastF1)](https://theoehrly.github.io/Fast-F1/)

---

## 👤 Author

**Taha**  
📘 MSc Data Science, University of Sheffield  
🔗 [LinkedIn](https://www.linkedin.com) | [GitHub](https://github.com)  
✉️ [tahak070@gmail.com]

---
