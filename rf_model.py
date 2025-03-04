import arff
import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

### Load and Convert ARFF to CSV ###
def convert_arff_to_csv(arff_file, csv_file):
    with open(arff_file, "r") as f:
        data = arff.load(f)

    df = pd.DataFrame(data["data"], columns=[attr[0] for attr in data["attributes"]])
    df.to_csv(csv_file, index=False)
    print(f"File converted to {csv_file}")

# Convert ARFF to CSV
arff_file = "Training_Dataset.arff"  # Change this to your file name
csv_file = "Training_Dataset_CSV.csv"
convert_arff_to_csv(arff_file, csv_file)

# Load the converted CSV
df = pd.read_csv(csv_file)

print(df.columns)

# The dataset’s label is "Result"
X = df.drop("Result", axis=1)
y = df["Result"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print("\n🔹 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))


