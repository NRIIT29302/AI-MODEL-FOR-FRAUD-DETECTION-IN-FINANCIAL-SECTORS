# -*- coding: utf-8 -*-
"""pp1

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JdIvOTdFpGAfPsbue9JO6MF2DOOwWk0G
"""

import numpy as np
import IPython.display as display
from matplotlib import pyplot as plt
import io
import base64

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

fig = plt.figure(figsize=(4, 3), facecolor='w')
plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)
plt.title("Sample Visualization", fontsize=10)

data = io.BytesIO()
plt.savefig(data)
image = F"data:image/png;base64,{base64.b64encode(data.getvalue()).decode()}"
alt = "Sample Visualization"
display.display(display.Markdown(F"""![{alt}]({image})"""))
plt.close(fig)

"""To learn more about accelerating pandas on Colab, see the [10 minute guide](https://colab.research.google.com/github/rapidsai-community/showcase/blob/main/getting_started_tutorials/cudf_pandas_colab_demo.ipynb) or
 [US stock market data analysis demo](https://colab.research.google.com/github/rapidsai-community/showcase/blob/main/getting_started_tutorials/cudf_pandas_stocks_demo.ipynb).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
credit_card_data = pd.read_csv('/content/creditcard.csv.zip')

# Check for possible column name variations:
# 1. Check if 'class' exists (case-sensitive)
if 'class' in credit_card_data.columns:
    credit_card_data = credit_card_data.rename(columns={'class': 'Class'})
# 2. Check for leading/trailing spaces (e.g., ' Class '):
credit_card_data.columns = credit_card_data.columns.str.strip()
if 'Class' not in credit_card_data.columns:
    raise KeyError("'Class' column not found in the DataFrame. Please check your data source.")

# Display first and last 5 rows
print("First 5 rows of dataset:")
print(credit_card_data.head())

print("\nLast 5 rows of dataset:")
print(credit_card_data.tail())

# Dataset Info
print("\nDataset Information:")
credit_card_data.info()

# Checking for missing values
print("\nMissing values in dataset:")
print(credit_card_data.isnull().sum())

# Count of legit vs fraud transactions
print("\nTransaction Class Distribution:")
print(credit_card_data['Class'].value_counts())

# Separate fraud and legit transactions
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print("\nLegit Transactions Shape:", legit.shape)
print("Fraud Transactions Shape:", fraud.shape)

# Descriptive statistics
print("\nLegit Transaction Amount Stats:")
print(legit.Amount.describe())

print("\nFraud Transaction Amount Stats:")
print(fraud.Amount.describe())

# Compare feature means
print("\nFeature Mean Comparison:")
print(credit_card_data.groupby('Class').mean())

# Balance the dataset by undersampling legit transactions
legit_sample = legit.sample(n=len(fraud))
balanced_data = pd.concat([legit_sample, fraud], axis=0)

# Shuffle the data
balanced_data = balanced_data.sample(frac=1, random_state=42)

# Show new distribution
print("\nBalanced Dataset Distribution:")
print(balanced_data['Class'].value_counts())

# Feature mean comparison after balancing
print("\nBalanced Dataset Mean Comparison:")
print(balanced_data.groupby('Class').mean())

# Define features (X) and labels (Y)
X = balanced_data.drop(columns='Class', axis=1)
Y = balanced_data['Class']

# Split dataset (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

print("\nDataset Shapes:")
print("X:", X.shape)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Train Logistic Regression Model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, Y_train)

# Predictions
y_train_pred_log = log_model.predict(X_train)
y_test_pred_log = log_model.predict(X_test)

# Model Accuracy
print("\nLogistic Regression Model Performance:")
print("Training Accuracy:", accuracy_score(Y_train, y_train_pred_log))
print("Testing Accuracy:", accuracy_score(Y_test, y_test_pred_log))


import joblib
# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# Predictions
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

# Model Accuracy
print("\nRandom Forest Model Performance:")
print("Training Accuracy:", accuracy_score(Y_train, y_train_pred_rf))
print("Testing Accuracy:", accuracy_score(Y_test, y_test_pred_rf))

# Detailed classification report
print("\nClassification Report (Random Forest):")
print(classification_report(Y_test, y_test_pred_rf))

print("Training Random Forest complete.")
joblib.dump(rf_model, 'fraud_detection_random_forest_model.joblib')
print("Random Forest model saved to fraud_detection_random_forest_model.joblib")
feature_columns = X_train.columns.tolist()
joblib.dump(feature_columns, 'feature_columns.joblib')
print(f"Feature column list saved. Expected number of features: {len(feature_columns)}")

# Train Isolation Forest (Unsupervised Anomaly Detection)
iso_forest = IsolationForest(contamination=0.02, random_state=42)
iso_forest.fit(X)

# Predict anomalies (-1 = anomaly, 1 = normal)
balanced_data['anomaly_score'] = iso_forest.decision_function(X)
balanced_data['is_anomaly'] = iso_forest.predict(X)

# Convert results (-1 = Fraud, 1 = Legit)
balanced_data['is_anomaly'] = balanced_data['is_anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Show anomaly detection results
print("\nAnomaly Detection Results:")
print(balanced_data[['Amount', 'is_anomaly']].head(10))
