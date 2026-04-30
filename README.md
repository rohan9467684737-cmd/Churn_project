# 📉 Customer Churn Intelligence Platform

An **industry-level Machine Learning web application** for predicting, explaining, and acting on customer churn risk — built with Scikit-learn and Streamlit.

---

## 🚀 Live Demo Features

| Page | Description |
|------|-------------|
| 🏠 **Overview** | KPI dashboard, confusion matrix, feature importance |
| 📊 **EDA Dashboard** | Distributions, churn drivers, correlation heatmap, segment analysis |
| 🔮 **Live Prediction** | Single-customer scoring with probability gauge & risk factors |
| 📁 **Batch Prediction** | CSV upload → score thousands of customers → download results |
| 📈 **Model Performance** | ROC curve, PR curve, threshold sweep, full model card |

---

## 🛠️ Tech Stack

- **ML Model:** Logistic Regression with `class_weight='balanced'` (ROC-AUC: **0.82**)
- **Feature Engineering:** 7 derived features + 2 binned categorical features
- **Preprocessing:** Sklearn `ColumnTransformer` → `StandardScaler` + `OrdinalEncoder`
- **App Framework:** Streamlit with custom dark-mode CSS
- **Visualizations:** Matplotlib + Seaborn
- **Data:** 440K training / 64K test records (Kaggle: Customer Churn Dataset)

---

## 📂 Project Structure

```
churn_project/
│
├── app.py                          # Main Streamlit application (5 pages)
├── train_model.py                  # Standalone training pipeline script
├── requirements.txt                # Python dependencies
│
├── data/
│   ├── train.csv                   # 440K training records
│   └── test.csv                    # 64K test records
│
└── models/
    ├── churn_pipeline.pkl          # Serialized sklearn Pipeline (prep + model)
    ├── metadata.json               # Metrics, curves, feature importances, EDA stats
    └── feature_importance.csv      # Feature importance ranking
```

---

## ⚙️ Setup & Run

### 1. Clone / Download the project
```bash
cd churn_project
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Retrain the model
```bash
python train_model.py
```

### 5. Launch the app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

---

## 🧠 ML Pipeline

```
Raw CSV
  │
  ▼
Feature Engineering (8 derived features)
  ├── SpendPerTenure   = Total Spend / (Tenure + 1)
  ├── CallsPerTenure   = Support Calls / (Tenure + 1)
  ├── DelayRatio       = Payment Delay / (Tenure + 1)
  ├── EngagementScore  = Usage Frequency × Tenure
  ├── SupportIntensity = Support Calls × Payment Delay
  ├── HighRiskFlag     = (Calls > 5) AND (Delay > 15)
  ├── AgeGroup         = binned into 5 age bands
  └── TenureGroup      = binned into 5 tenure bands
  │
  ▼
ColumnTransformer
  ├── 13 numeric features  → StandardScaler
  └──  5 categorical       → OrdinalEncoder
  │
  ▼
Logistic Regression (C=0.5, class_weight='balanced')
  │
  ▼
Threshold Tuning (F1-optimal on validation set)
  │
  ▼
Churn Probability + Risk Level (Low / Medium / High)
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.8203** |
| PR-AUC | **0.8468** |
| F1 Score | **0.6891** |
| Precision | **0.5311** |
| Recall | **0.9809** |
| Accuracy | **0.5807** |

> **Note:** The dataset has ~57% churn rate in train vs ~47% in test (distribution shift). The model is tuned for recall — catching churners is prioritized over precision. Adjust the threshold slider in the Live Prediction page to balance precision/recall per your business need.

---

## 🔑 Key Features for Resume

- ✅ **End-to-end ML pipeline** — ingestion, FE, preprocessing, training, evaluation
- ✅ **Class imbalance handling** — `class_weight='balanced'` + threshold tuning
- ✅ **Multi-page Streamlit app** with real-time predictions
- ✅ **Interactive gauge chart** for single-customer risk scoring
- ✅ **Batch prediction** with CSV upload + download
- ✅ **Model explainability** — feature importances, threshold sweep, ROC/PR curves
- ✅ **Production-ready** — cached model loading, error handling, modular code
- ✅ **Industry-standard metrics** — ROC-AUC, PR-AUC, F1, Precision, Recall

---

## 👤 Author

Built as a portfolio/resume project demonstrating end-to-end ML engineering skills.

**Dataset:** [Customer Churn Dataset — Kaggle](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)
# Churn_project
