import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define file path
file_path = r"C:\Users\SHRAVINYA\Downloads\archive\predictive_maintenance_dataset.csv"

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at: {file_path}")

# Load dataset
df = pd.read_csv(file_path)

# Print column names for debugging
print("Dataset Columns:", df.columns.tolist())

# Drop non-numeric columns
if 'device' in df.columns:
    df.drop(columns=['device'], inplace=True)

# Convert 'date' column to datetime format (optional)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# Select numerical features
numerical_features = [col for col in df.columns if col.startswith('metric')]

# Print selected features
print("Using Features:", numerical_features)

# Check if we have numerical features
if not numerical_features:
    raise ValueError("No valid numerical features found in the dataset!")

# Normalize numerical features
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Check target column
if 'failure' not in df.columns:
    raise KeyError("Target column 'failure' is missing from the dataset!")

# Define features and target variable
X = df[numerical_features]
y = df['failure']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Training Complete! Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save trained model and scaler
joblib.dump(model, "predictive_maintenance_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("🔹 Model and scaler saved successfully.")
