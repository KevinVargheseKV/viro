# =========================================================
# VIRO++ : XGBoost Model Training & Saving Script
# =========================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier


# =========================================================
# 1️⃣ LOAD DATASET
# =========================================================

DATA_FILE = "final_feature_dataset.csv"

df = pd.read_csv(DATA_FILE)

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# Separate features & labels
X = df.drop("Label", axis=1)
y = df["Label"]

print("Feature Matrix Shape:", X.shape)


# =========================================================
# 2️⃣ TRAIN–TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain Size:", X_train.shape)
print("Test Size :", X_test.shape)


# =========================================================
# 3️⃣ BUILD XGBOOST MODEL
# =========================================================

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost Model...")
model.fit(X_train, y_train)

print("Training Completed!")


# =========================================================
# 4️⃣ SAVE TRAINED MODEL
# =========================================================

MODEL_PATH = "xgboost_model.pkl"

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully at → {MODEL_PATH}")


# =========================================================
# 5️⃣ PREDICTIONS
# =========================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# =========================================================
# 6️⃣ PERFORMANCE METRICS
# =========================================================

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\n==============================")
print("   XGBOOST MODEL PERFORMANCE  ")
print("==============================")
print("Accuracy :", round(acc, 4))
print("Precision:", round(prec, 4))
print("Recall   :", round(rec, 4))
print("F1-score :", round(f1, 4))
print("ROC-AUC  :", round(roc, 4))


# =========================================================
# 7️⃣ CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Non-Interaction","Interaction"],
    yticklabels=["Non-Interaction","Interaction"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.tight_layout()
plt.show()


# =========================================================
# 8️⃣ ROC CURVE
# =========================================================

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc:.4f}")
plt.plot([0,1], [0,1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# END OF SCRIPT
# =========================================================
