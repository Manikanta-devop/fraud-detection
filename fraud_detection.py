import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv("fraud_transactions_200.csv")

# -----------------------------
# 2. Encode Categorical Data
# -----------------------------
location_map = {"India": 0, "USA": 1, "UK": 2, "Germany": 3, "UAE": 4}
device_map = {"mobile": 0, "desktop": 1, "tablet": 2}
merchant_map = {"electronics": 0, "fashion": 1, "grocery": 2, "travel": 3}

df["location"] = df["location"].map(location_map)
df["device_type"] = df["device_type"].map(device_map)
df["merchant_category"] = df["merchant_category"].map(merchant_map)

# -----------------------------
# 3. Features & Target
# -----------------------------
X = df.drop(["is_fraud", "transaction_id"], axis=1)
y = df["is_fraud"]

# -----------------------------
# 4. Handle Imbalance
# -----------------------------
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# -----------------------------
# 5. Train Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = model.predict(X_test)
print("\nModel Evaluation:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 7. Take User Input (Console)
# -----------------------------
print("\nEnter New Transaction Details:")

amount = float(input("Amount: "))
transaction_time = int(input("Transaction time (0-86400): "))
location = input("Location (India/USA/UK/Germany/UAE): ")
device = input("Device (mobile/desktop/tablet): ")
merchant = input("Merchant (electronics/fashion/grocery/travel): ")
prev_trans = int(input("Previous transactions: "))

# Convert input using same mapping
input_data = pd.DataFrame([{
    "amount": amount,
    "transaction_time": transaction_time,
    "location": location_map[location],
    "device_type": device_map[device],
    "merchant_category": merchant_map[merchant],
    "previous_transactions": prev_trans
}])
# -----------------------------
# 8. Prediction
# -----------------------------
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

print("\n🔍 Result:")
print("Fraud Prediction:", "YES" if prediction == 1 else "NO")
print("Fraud Probability:", round(probability, 4))