import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

target_col = 'result'

edu_features = [
    'education_status',
    'education_form',
    'graduation',
    'occupation_type'
]


X_train = train_df[edu_features].fillna("Unknown")
y_train = train_df[target_col]

X_test = test_df[edu_features].fillna("Unknown")

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

submission = pd.DataFrame({"id": test_df["id"],"result": predictions})
submission.to_csv("submission_model2.csv", index=False)