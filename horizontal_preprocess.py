import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import numpy as np

train_df = pd.read_csv("train_horizontal.csv")
test_df = pd.read_csv("test_horizontal.csv")


well = train_df["well_id"]

train_df = train_df.sort_values(["well_id", "MD"])

train_df["Delta_Z"] = train_df.groupby("well_id")["Z"].diff()
train_df["Delta_MD"] = train_df.groupby("well_id")["MD"].diff()

train_df["Delta_Z/Delta_MD"] = train_df["Delta_Z"]/train_df["Delta_MD"]

test_df = test_df.sort_values(["well_id", "MD"])

test_df["Delta_Z"] = test_df.groupby("well_id")["Z"].diff()
test_df["Delta_MD"] = test_df.groupby("well_id")["MD"].diff()

test_df["Delta_Z/Delta_MD"] = test_df["Delta_Z"]/test_df["Delta_MD"]

train_df = train_df.drop(columns = ["ANCC", "EGFDL", "ASTNU", "ASTNL", "BUDA","EGFDU", "TVT_input", "well_id"])
test_df = test_df.drop(columns=["TVT_input", "well_id"])

print(train_df.columns)
X = train_df.drop(columns= "TVT")

y = train_df["TVT"]

print(test_df.columns)


split = GroupShuffleSplit(
    n_splits = 1,
    test_size=.20,
    random_state=42
)

train_idx, val_idx = next(split.split(X,y, groups=well))

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]
y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

X_test = test_df


model = XGBRegressor(
    n_estimators = 850,
    max_depth = 4,
    subsample = 0.75,
    learning_rate = 0.05,
    random_state = 42,
)


model.fit(X_train,y_train)


train_preds = model.predict(X_train)

val_preds = model.predict(X_val)

test_preds = model.predict(X_test)

train_score = model.score(X_train, y_train)
val_score = model.score(X_val, y_val)

print("train score: ", train_score)
print("val_score: ", val_score)

test_df["predicted_score"] = test_preds

print("min: ", test_preds.min())
print("max: ", test_preds.max())
print("mean: ", test_preds.mean())
print("std: ", test_preds.std())

train_rmse = root_mean_squared_error(y_train, train_preds)

rmse = root_mean_squared_error(y_val, val_preds)

test_df["predicted_score"] = test_preds

print(test_df)

print("Train RMSE: ", train_rmse)
print("Validation RMSE:", rmse)

print(X_train.columns)

feature_importance = pd.Series(model.feature_importances_, index = X_train.columns)

print(feature_importance.head(10))

test_df.to_csv("final_horizontal.csv")

#print(test_df["predicted_score"].isna().sum())
print(len(test_df))
print(len(test_preds))

sample_sub = pd.read_csv("sample_submission.csv")
print(len(sample_sub))

