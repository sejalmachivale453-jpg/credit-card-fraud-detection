import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import Counter

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("credit_card_fraud_dataset.csv")   # change your file path

print("First 5 rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# -------------------------------
# 2. Encode categorical data
# -------------------------------
df = pd.get_dummies(df, drop_first=True)

# -------------------------------
# 3. Split X and Y
# -------------------------------
X = df.drop("fraud", axis=1)
y = df["fraud"]

# -------------------------------
# 4. Train–Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 5. SMOTE for balancing
# -------------------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE:", Counter(y_train_res))

# -------------------------------
# 6. Scaling
# -------------------------------
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# -------------------------------
# 7. Baseline Logistic Regression
# -------------------------------
model = LogisticRegression()
model.fit(X_train_res, y_train_res)

# -------------------------------
# 8. Predictions (default threshold = 0.5)
# -------------------------------
y_pred_default = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy (default threshold):", accuracy_score(y_test, y_pred_default))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_default))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_default))

# -------------------------------
# 9. Precision–Recall Curve
# -------------------------------
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Find best threshold → maximize F1 Score
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]

print("\nBest Threshold Found:", best_threshold)

# -------------------------------
# 10. Predictions using best threshold
# -------------------------------
y_pred_best = (y_prob >= best_threshold).astype(int)

print("\nAccuracy (after threshold tuning):", accuracy_score(y_test, y_pred_best))

print("\nClassification Report (after threshold tuning):")
print(classification_report(y_test, y_pred_best))

print("\nConfusion Matrix (after threshold tuning):")
print(confusion_matrix(y_test, y_pred_best))

# -------------------------------
# 11. Plot Precision–Recall Curve
# -------------------------------
plt.figure(figsize=(7, 5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.grid()
plt.show()
