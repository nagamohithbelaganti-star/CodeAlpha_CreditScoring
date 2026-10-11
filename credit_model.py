import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# 1. Generate Dataset
np.random.seed(42)
n_samples = 1000

data = {
    'income': np.random.randint(20000, 120000, size=n_samples),
    'total_debt': np.random.randint(1000, 50000, size=n_samples),
    'payment_history_score': np.random.randint(1, 10, size=n_samples),
    'credit_age_years': np.random.randint(1, 15, size=n_samples)
}

df = pd.DataFrame(data)

# Target variable: Creditworthiness (1 = Good Credit, 0 = High Risk)
df['creditworthiness'] = np.where(
    (df['income'] > 50000) & (df['payment_history_score'] > 5) & (df['total_debt'] < 30000), 1, 0
)

print("--- Dataset Successfully Created & Loaded ---")
print(df.head(), "\n")

# 2. Define Features (X) and Target (y)
X = df[['income', 'total_debt', 'payment_history_score', 'credit_age_years']]
y = df['creditworthiness']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training (Random Forest)
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Predictions & Evaluation
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")