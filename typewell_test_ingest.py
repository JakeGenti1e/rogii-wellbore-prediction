import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pandas as pd
import os

files = os.listdir("test")

typewell_files = [f for f in files if "__typewell.csv" in f]

print("typewell:", len(typewell_files))

typewell_dfs = []

for i in range(len(typewell_files)):
    curr_file = typewell_files[i]
    curr_path = "test/" + curr_file
    curr_file = typewell_files[i]
    curr_df = pd.read_csv(curr_path)
    curr_df["well_id"] = curr_file.split("__")[0]
    typewell_dfs.append(curr_df)

print(files)
print(typewell_files)
print(type(typewell_dfs[0]))

all_typewell = pd.concat(typewell_dfs)

print(all_typewell.shape)
print(all_typewell["well_id"].nunique())
print(all_typewell.isna().sum().sort_values(ascending = False))
print(all_typewell.dtypes)

all_typewell.to_csv("test_typewell.csv", index=False)