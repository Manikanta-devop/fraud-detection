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


