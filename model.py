import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
train_data = pd.read_csv("train_titanic.csv")
test_data = pd.read_csv("test_titanic.csv")

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features],dtype = int)
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# Save model and its expected feature columns
joblib.dump(
    {
        "model": model,
        "features": list(X.columns)
    },
    "model.pkl"
)
print("Model trained and saved successfully!")