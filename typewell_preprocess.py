import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

train_df = pd.read_csv("train_typewell.csv")
test_df = pd.read_csv("test_typewell.csv")

well = train_df["well_id"]
train_df = train_df.drop(columns = ["Geology", "well_id"])
test_df = test_df.drop(columns=["TVT", "well_id"])

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
    max_depth = 6,
    #subsample = 0.75,
    learning_rate = 0.25,
    reg_lambda = 1
)

model.fit(X_train,y_train)

train_preds = model.predict(X_train)

val_preds = model.predict(X_val)

test_preds = model.predict(X_test)

train_score = model.score(X_train, y_train)
val_score = model.score(X_val, y_val)

print("train score: ", train_score)
print("val_score: ", val_score)

rmse = root_mean_squared_error(y_val, val_preds)

train_rmse = root_mean_squared_error(y_train, train_preds)

print("Train RMSE: ", train_rmse)
print("Validation RMSE:", rmse)

test_df["predicted_score"] = test_preds

print("min: ", test_preds.min())
print("max: ", test_preds.max())
print("mean: ", test_preds.mean())
print("std: ", test_preds.std())

print(test_df)

print(X_train.columns)

feature_importance = pd.Series(model.feature_importances_, index = X_train.columns)

print(feature_importance.head(10))

test_df.to_csv("final_typewell.csv")