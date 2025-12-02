import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ML imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler

# sampling
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import numpy as np

# 1 Load data
df = pd.read_csv("credit_card_fraud_dataset.csv")

# 2 Encoding
df_encoded = pd.get_dummies(df, columns=['transaction_type'], dtype=int)

# 3 Features
X = df_encoded.drop('fraud', axis=1)
y = df_encoded['fraud']

# 4 Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5 SMOTE-Tomek (better than SMOTE)
smt = SMOTETomek(random_state=42)
X_res, y_res = smt.fit_resample(X_train, y_train)

print("After SMOTE-TOMEK:", Counter(y_res))

# 6 Scaling
scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)
X_test_scaled = scaler.transform(X_test)

# 7 XGBoost with Hyperparameter Search
params = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1],
    "colsample_bytree": [0.7, 0.8, 1],
    "gamma": [0, 1, 5]
}

xgb = XGBClassifier(
    objective="binary:logistic",
    random_state=42,
    eval_metric="logloss"
)

search = RandomizedSearchCV(
    xgb, params, n_iter=15, scoring='f1', cv=3, verbose=1, n_jobs=-1
)
search.fit(X_res_scaled, y_res)

best_model = search.best_estimator_
print("\nBest XGBoost Params:", search.best_params_)

# 8 Predict Probabilities
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

# 9 Best threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2*(precision*recall)/(precision+recall+1e-6)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\nBest Threshold: {best_threshold:.3f}")

# 10 Predictions
y_pred = (y_prob > best_threshold).astype(int)

# 11 Evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
