# Internship-task-05

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

df = pd.DataFrame({
    'Age': np.random.randint(18, 60, 200),
    'Gender': np.random.choice(['Male', 'Female'], 200),
    'Income': np.random.randint(20000, 100000, 200),
    'BrowsingTime': np.random.randint(1, 20, 200),
    'PagesVisited': np.random.randint(1, 10, 200),
    'Purchased': np.random.choice([0, 1], 200)
})

for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop('Purchased', axis=1)
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

sample = X.iloc[0:1]
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)
print("\nSample Prediction:", "Yes" if prediction[0] == 1 else "No")
