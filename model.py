import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- LOAD DATA ----------

DATA_FILE = "final_feature_dataset.csv"

df = pd.read_csv(DATA_FILE)

X = df.drop("Label", axis=1)
y = df["Label"]

print("Dataset shape:", df.shape)
print("Feature matrix shape:", X.shape)

# ---------- TRAIN-TEST SPLIT ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ---------- RANDOM FOREST MODEL ----------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)
# ---------- RANDOM FOREST CREATION ----------
import joblib
joblib.dump(model, "random_forest.pkl")

print("Model saved successfully!")
# ---------- PREDICTIONS ----------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# ---------- METRICS ----------

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\nMODEL PERFORMANCE")
print("----------------------------")
print("Accuracy :", round(acc,4))
print("Precision:", round(prec,4))
print("Recall   :", round(rec,4))
print("F1-score :", round(f1,4))
print("ROC-AUC  :", round(roc,4))

# ---------- CONFUSION MATRIX ----------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Interaction","Interaction"],
            yticklabels=["Non-Interaction","Interaction"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.show()

# ---------- ROC CURVE ----------

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc:.4f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.tight_layout()
plt.show()
