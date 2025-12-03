📝 README.md — Credit Card Fraud Detection (Logistic Regression + SMOTE + Threshold Tuning)
📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning.
Since fraud cases are very rare, the dataset is highly imbalanced.
To solve this, the project uses:

SMOTE Oversampling to balance classes

Logistic Regression as the main model

Threshold Tuning using Precision–Recall Curve to improve fraud detection performance

The goal is to reduce missed frauds (higher recall) while keeping false alarms controlled.

📂 Project Workflow
1️⃣ Data Preprocessing

Loaded dataset and checked missing values

Performed one-hot encoding for the transaction_type column

Split dataset into training and testing sets (80–20 split)

2️⃣ Handling Imbalanced Data

Used SMOTE (Synthetic Minority Oversampling Technique) to balance class distribution.

Before SMOTE:

Fraud cases were very few

After SMOTE:

Class distribution: {0: 3310, 1: 3310}

3️⃣ Model Training

Used Logistic Regression (simple + beginner-friendly)

Scaled features using StandardScaler

4️⃣ Threshold Tuning (Major Improvement)

Instead of using default threshold 0.5, the model finds the best threshold using the Precision–Recall curve.

This improves fraud detection performance.

Default threshold accuracy: 66%

Tuned threshold accuracy: 74%

Better precision & better recall balance for fraud class

📊 Model Performance
🔹 Before Threshold Tuning

Accuracy: 0.667

Recall (Fraud): 0.63

Precision (Fraud): 0.29

🔹 After Threshold Tuning

Accuracy: 0.743

Recall (Fraud): 0.51

Precision (Fraud): 0.34

✔ Model becomes more stable
✔ Fewer false positives
✔ Better fraud detection balance
📁 Project Files

credit_card_fraud_detection.py – main ML script

credit_card_fraud_dataset.csv – dataset (add your link here)

README.md – documentation

🔧 Technologies Used

Python

Pandas, NumPy

Scikit-Learn

Imbalanced-Learn

Matplotlib

SMOTE

Logistic Regression

🚀 How to Run the Project
pip install -r requirements.txt
python credit_card_fraud_detection.py

📌 Key Learnings

Understanding data imbalance

Using SMOTE for oversampling

Logistic Regression for classification

Using Precision-Recall Curve

Threshold tuning to improve minority class recall

Evaluating model using confusion matrix & reports
