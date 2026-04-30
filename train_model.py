"""
Customer Churn Prediction — Standalone Training Script
Run: python train_model.py
"""

import os, json, warnings, joblib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    average_precision_score, roc_curve, precision_recall_curve,
)

print("=" * 60)
print("  CUSTOMER CHURN — TRAINING PIPELINE")
print("=" * 60)

# 1. Load
print("\n[1/6] Loading data...")
train_raw = pd.read_csv("data/train.csv").dropna().reset_index(drop=True)
test_raw  = pd.read_csv("data/test.csv").dropna().reset_index(drop=True)
print(f"      Train: {len(train_raw):,} | Test: {len(test_raw):,}")

# 2. Feature Engineering
print("\n[2/6] Feature engineering...")
def feature_engineering(df):
    df = df.copy()
    df.drop(columns=["CustomerID"], errors="ignore", inplace=True)
    df["SpendPerTenure"]   = df["Total Spend"]   / (df["Tenure"] + 1)
    df["CallsPerTenure"]   = df["Support Calls"] / (df["Tenure"] + 1)
    df["DelayRatio"]       = df["Payment Delay"] / (df["Tenure"] + 1)
    df["EngagementScore"]  = df["Usage Frequency"] * df["Tenure"]
    df["SupportIntensity"] = df["Support Calls"]   * df["Payment Delay"]
    df["HighRiskFlag"]     = ((df["Support Calls"] > 5) & (df["Payment Delay"] > 15)).astype(int)
    df["AgeGroup"]    = pd.cut(df["Age"],    bins=[0,25,35,50,65,100],
                                labels=["lt25","25to35","35to50","50to65","gt65"]).astype(str)
    df["TenureGroup"] = pd.cut(df["Tenure"], bins=[0,12,24,36,60,200],
                                labels=["lt1yr","1to2yr","2to3yr","3to5yr","gt5yr"]).astype(str)
    return df

train_df = feature_engineering(train_raw)
test_df  = feature_engineering(test_raw)
TARGET   = "Churn"
_, train_s = train_test_split(train_df, test_size=100000/len(train_df),
                               stratify=train_df[TARGET], random_state=42)
X_train = train_s.drop(columns=[TARGET]);  y_train = train_s[TARGET].astype(int)
X_test  = test_df.drop(columns=[TARGET]);  y_test  = test_df[TARGET].astype(int)

CAT_COLS = ["Gender","Subscription Type","Contract Length","AgeGroup","TenureGroup"]
NUM_COLS = [c for c in X_train.columns if c not in CAT_COLS]
ALL_COLS = NUM_COLS + CAT_COLS
print(f"      {len(ALL_COLS)} total features")

# 3. Pipeline
print("\n[3/6] Building pipeline...")
prep = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
])
clf  = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42, n_jobs=-1)
pipe = Pipeline([("prep", prep), ("clf", clf)])

# 4. Train
print("\n[4/6] Training...")
pipe.fit(X_train, y_train)
print("      Done!")

# 5. Threshold tuning
print("\n[5/6] Threshold tuning...")
val_p = pipe.predict_proba(X_train)[:,1]
thresholds = np.arange(0.05, 0.95, 0.005)
f1s = [f1_score(y_train, (val_p>=t).astype(int)) for t in thresholds]
best_thresh = float(thresholds[np.argmax(f1s)])
print(f"      Optimal threshold: {best_thresh:.3f}")

# 6. Evaluate
print("\n[6/6] Evaluating on test set...")
test_proba = pipe.predict_proba(X_test)[:,1]
test_preds = (test_proba >= best_thresh).astype(int)
acc  = accuracy_score(y_test, test_preds)
auc  = roc_auc_score(y_test, test_proba)
f1   = f1_score(y_test, test_preds)
prec = precision_score(y_test, test_preds)
rec  = recall_score(y_test, test_preds)
ap   = average_precision_score(y_test, test_proba)
cm   = confusion_matrix(y_test, test_preds).tolist()
print(f"\n  Accuracy  : {acc:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  PR-AUC    : {ap:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")

# Curves
fpr, tpr, _ = roc_curve(y_test, test_proba)
st = max(1, len(fpr)//300)
roc_data = {"fpr": fpr[::st].tolist(), "tpr": tpr[::st].tolist()}
pc, rc, _ = precision_recall_curve(y_test, test_proba)
st = max(1, len(pc)//300)
pr_data = {"precision": pc[::st].tolist(), "recall": rc[::st].tolist()}

thresh_sweep = []
for t in np.arange(0.1, 0.91, 0.05):
    p = (test_proba >= t).astype(int)
    thresh_sweep.append({"threshold": round(float(t),2),
        "f1": round(f1_score(y_test,p,zero_division=0),4),
        "precision": round(precision_score(y_test,p,zero_division=0),4),
        "recall": round(recall_score(y_test,p,zero_division=0),4),
        "accuracy": round(accuracy_score(y_test,p),4)})

imp     = np.abs(clf.coef_[0])
feat_df = pd.DataFrame({"feature":ALL_COLS,"importance":imp}).sort_values("importance",ascending=False)

eda_stats = {
    "churn_dist": {str(int(k)):int(v) for k,v in train_raw["Churn"].value_counts().items()},
    "churn_by_contract":     {str(k):round(float(v),4) for k,v in train_raw.groupby("Contract Length")["Churn"].mean().items()},
    "churn_by_subscription": {str(k):round(float(v),4) for k,v in train_raw.groupby("Subscription Type")["Churn"].mean().items()},
    "churn_by_gender":       {str(k):round(float(v),4) for k,v in train_raw.groupby("Gender")["Churn"].mean().items()},
    "num_stats": {c: {"mean":round(float(train_raw[c].mean()),2),"std":round(float(train_raw[c].std()),2),
        "min":round(float(train_raw[c].min()),2),"max":round(float(train_raw[c].max()),2)}
        for c in ["Age","Tenure","Usage Frequency","Support Calls","Payment Delay","Total Spend","Last Interaction"]},
    "churn_avg": {c: {"churned":round(float(train_raw[train_raw["Churn"]==1][c].mean()),2),
        "retained":round(float(train_raw[train_raw["Churn"]==0][c].mean()),2)}
        for c in ["Age","Tenure","Usage Frequency","Support Calls","Payment Delay","Total Spend","Last Interaction"]},
}

os.makedirs("models", exist_ok=True)
joblib.dump(pipe, "models/churn_pipeline.pkl")
meta = {"model_name":"Logistic Regression","test_accuracy":round(acc,4),"test_roc_auc":round(auc,4),
    "test_pr_auc":round(ap,4),"test_f1":round(f1,4),"test_precision":round(prec,4),"test_recall":round(rec,4),
    "confusion_matrix":cm,"best_threshold":best_thresh,"roc_curve":roc_data,"pr_curve":pr_data,
    "threshold_sweep":thresh_sweep,"feature_importances":feat_df.to_dict("records"),
    "feature_names":ALL_COLS,"num_cols":NUM_COLS,"cat_cols":CAT_COLS,
    "train_size":int(len(train_s)),"test_size":int(len(test_df)),"total_train_rows":int(len(train_raw)),
    "churn_rate_train":round(float(y_train.mean()),4),"churn_rate_test":round(float(y_test.mean()),4),
    "eda_stats":eda_stats}
json.dump(meta, open("models/metadata.json","w"), indent=2)
feat_df.to_csv("models/feature_importance.csv", index=False)

print("\n" + "=" * 60)
print("  Artifacts saved to models/")
print("  Run app:  streamlit run app.py")
print("=" * 60)
