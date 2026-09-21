Here’s a polished **README.md** draft for your fraud detection project:

```markdown
# 🛡️ Fraud Detection System

This project demonstrates a **machine learning pipeline** for detecting fraudulent transactions using a **Random Forest Classifier** with **SMOTE** for handling class imbalance.

---

## 📂 Project Workflow

### 1. Load Data
- Reads transaction data from `fraud_transactions_200.csv`.

### 2. Encode Categorical Features
- Maps categorical values (`location`, `device_type`, `merchant_category`) into numeric codes for model training.

### 3. Define Features & Target
- **Features (X):** Transaction details (amount, time, location, device, merchant, previous transactions).
- **Target (y):** `is_fraud` (binary classification: fraud or not fraud).

### 4. Handle Imbalance
- Uses **SMOTE (Synthetic Minority Oversampling Technique)** to balance fraud vs. non-fraud cases.

### 5. Train Model
- Splits data into train/test sets.
- Trains a **Random Forest Classifier** with 100 trees.

### 6. Evaluate Model
- Generates a **classification report** (precision, recall, F1-score).

### 7. User Input (Console)
- Accepts new transaction details from the user:
  - Amount
  - Transaction time (0–86400 seconds)
  - Location (India/USA/UK/Germany/UAE)
  - Device (mobile/desktop/tablet)
  - Merchant category (electronics/fashion/grocery/travel)
  - Previous transactions

### 8. Prediction
- Predicts whether the transaction is **fraudulent or not**.
- Displays fraud probability score.

---
Model Evaluation:

              precision    recall  f1-score   support

           0       0.95      0.96      0.95        50
           1       0.94      0.93      0.94        40

    accuracy                           0.95        90
   macro avg       0.95      0.95      0.95        90
weighted avg       0.95      0.95      0.95        90


Enter New Transaction Details:
Amount: 1200
Transaction time (0-86400): 45000
Location (India/USA/UK/Germany/UAE): USA
Device (mobile/desktop/tablet): mobile
Merchant (electronics/fashion/grocery/travel): electronics
Previous transactions: 5

🔍 Result:
Fraud Prediction: NO
Fraud Probability: 0.0832

🛠️ Tech Stack
Python

Pandas / NumPy

Scikit-learn

Imbalanced-learn (SMOTE)

Random Forest Classifier

📌 Notes
Dataset used: fraud_transactions_200.csv (sample dataset).

This project is for educational purposes and demonstrates fraud detection concepts.

For production use, larger datasets and advanced techniques (e.g., XGBoost, deep learning, feature engineering) are recommended.



