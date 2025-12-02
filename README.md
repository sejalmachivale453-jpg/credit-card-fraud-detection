# Credit Card Fraud Detection – Machine Learning Project

This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques.  
It includes end-to-end data preprocessing, imbalance handling, model building, threshold optimization, and evaluation.

---

## 🚀 Project Highlights

- Built an XGBoost-based fraud detection model on imbalanced financial transaction data  
- Applied **SMOTE-Tomek** to handle class imbalance and improve minority fraud detection  
- Performed **hyperparameter tuning** using RandomizedSearchCV  
- Used **Precision–Recall curve** to find the best decision threshold  
- Achieved ~80% recall on fraud cases by reducing false negatives  
- Evaluated the model using classification report & confusion matrix  

---

## 🧠 Machine Learning Pipeline

1. Data Loading & Exploration  
2. Encoding categorical features  
3. Train-test split with stratification  
4. SMOTE-Tomek sampling  
5. Feature scaling (StandardScaler)  
6. XGBoost model training  
7. Hyperparameter tuning (RandomizedSearchCV)  
8. Threshold optimization  
9. Final evaluation  

---

## 🛠️ Tech Stack

- **Python**
- **NumPy**, **Pandas**
- **Scikit-Learn**
- **XGBoost**
- **Imbalanced-Learn**
- **Matplotlib**

---

## 📂 Project Structure

├── credit_card_fraud_detection.py # Main ML code
├── credit_card_fraud_dataset.csv # Dataset (optional)
├── README.md # Project documentation


---

## ▶️ How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt

2.run the script

python credit_card_fraud_detection.py

📊 Model Performance

Recall (Fraud class): ~80%

Improved fraud capture rate

Lower false negatives

Balanced results after SMOTE-Tomek

📘 Results Included

Classification Report

Confusion Matrix

Precision–Recall threshold

Best hyperparameters found using RandomizedSearchCV

dataset
This dataset is included in the repository.  
File: `credit_card_fraud_dataset.csv`

🙌 Author

Sejal Machivale
Machine Learning & Data Science Enthusiast
