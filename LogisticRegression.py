import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

target_col = 'result'

X = train_df.drop(columns=[target_col])
y = train_df[target_col]

X = X.fillna(X.mean(numeric_only=True))
test_df = test_df.fillna(test_df.mean(numeric_only=True))

X = pd.get_dummies(X, drop_first=True)
test_df = pd.get_dummies(test_df, drop_first=True)

X, test_df = X.align(test_df, join='left', axis=1, fill_value=0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

test_scaled = scaler.transform(test_df)
test_predictions = model.predict(test_scaled)

submission = pd.DataFrame({"id": test_df.index,"result": test_predictions})

submission.to_csv("submission.csv", index=False)

