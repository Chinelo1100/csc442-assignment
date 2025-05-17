import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load breast cancer dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split and scale
X = df.drop('target', axis=1)
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train initial model to get top features
model = LogisticRegression(max_iter=10000)
model.fit(X_scaled, y)
importance = pd.Series(abs(model.coef_[0]), index=data.feature_names)
top_10_features = importance.sort_values(ascending=False).head(10).index.tolist()

# Re-train with top 10 features
X_top = df[top_10_features]
X_top_scaled = scaler.fit_transform(X_top)
X_train, X_test, y_train, y_test = train_test_split(X_top_scaled, y, test_size=0.2, random_state=42)

best_model = LogisticRegression(max_iter=10000)
best_model.fit(X_train, y_train)

# Save model to pickle
with open("logistic_regression_top_features.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("✅ Model saved successfully as logistic_regression_top_features.pkl")
print(f"Top 10 features: {top_10_features}")
