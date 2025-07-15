# 🏁 F1 Race Result Prediction

Predicting Formula 1 race outcomes using **FastF1**, weather data, and **Gradient Boosting models**. This project leverages live telemetry, session data, and machine learning techniques to forecast driver results with explainability via SHAP.

---

## 📌 Overview

This repository contains:

- 📒 `f1_pred.ipynb` — End-to-end notebook for F1 race prediction.
- 🧰 Utility scripts (optional) for modular expansion (e.g., weather, config, model loading).
- 📊 Machine Learning pipeline with feature engineering and explainability.

---

## 🚀 Key Features

✅ Data collection via FastF1 (telemetry, lap times, weather)  
✅ Advanced feature engineering for race context  
✅ Gradient Boosting (XGBoost) modeling  
✅ SHAP explainability for transparency  
✅ Future-ready for real-time prediction and API deployment

---

## 🛠️ Setup

Clone the repo and install dependencies:

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
├── utils/                  # Optional: reusable utilities
│   ├── weather.py
│   ├── model.py
│   └── config.py
├── README.md               # Project documentation
└── requirements.txt        # Dependency list
```

---

## 🔍 How It Works

1. **Data Extraction**: FastF1 APIs fetch race-specific data (drivers, telemetry, weather).
2. **Feature Engineering**: Tyre age, session status, track temperature, driver names, etc.
3. **Modeling**: Gradient Boosting/XGBoost used to train and evaluate.
4. **Evaluation**: Accuracy, SHAP values, and finish position metrics.
5. **Explainability**: SHAP plots for model insight and transparency.

---

## 📈 Sample Results

- **Prediction Accuracy**: ~85% on selected historical races  
- **Top Features**:  
  - Tyre Age  
  - Air Temperature  
  - Track Status  
  - Driver Identity

---

## 🧠 Explainability with SHAP

We use SHAP (SHapley Additive exPlanations) to visualize and interpret model outputs:

- Global feature importance  
- Individual prediction explanations  
- Insight into race-critical variables

---

## 📚 References

- [FastF1 Documentation](https://theoehrly.github.io/Fast-F1/)
- [SHAP for Explainable AI](https://shap.readthedocs.io/)
- [XGBoost](https://xgboost.readthedocs.io/en/stable/)
- [F1 Dataset (via FastF1)](https://theoehrly.github.io/Fast-F1/)

---

## 👤 Author

**Taha**  
📘 MSc Data Science, University of Sheffield  
🔗 [LinkedIn](https://www.linkedin.com) | [GitHub](https://github.com)  
✉️ [tahak070@gmail.com]

---
