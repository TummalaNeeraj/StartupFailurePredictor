import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load the dataset
data = pd.read_csv("dataset/startup.csv")

print("Dataset loaded successfully!")
print(data.head())


# 2. Select input features
X = data[
    [
        "Funding",
        "Employees",
        "FounderExperience",
        "CompanyAge",
        "Revenue",
        "Expenses",
        "MarketSize",
        "CustomerGrowth"
    ]
]


# 3. Select target
y = data["Status"]


# 4. Convert Success/Failure into numbers
y = y.map({
    "Failure": 0,
    "Success": 1
})


# 5. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))


# 6. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 7. Train the model
print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")


# 8. Test the model
y_pred = model.predict(X_test)


# 9. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# 10. Display detailed performance
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Failure", "Success"]
    )
)


# 11. Save trained model
joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")