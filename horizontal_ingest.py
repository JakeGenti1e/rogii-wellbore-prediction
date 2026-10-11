import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pandas as pd
import os

files = os.listdir("train")

horizontal_files = [f for f in files if "__horizontal_well.csv" in f]
typewell_files = [f for f in files if "__typewell.csv" in f]

print("Horizontal:", len(horizontal_files))
print("Typewell:", len(typewell_files))

horizontal_dfs = []
for i in range(len(horizontal_files)):
    curr_file = horizontal_files[i]
    curr_path = "train/" + curr_file
    curr_file = horizontal_files[i]
    curr_df = pd.read_csv(curr_path)
    curr_df["well_id"] = curr_file.split("__")[0]
    horizontal_dfs.append(curr_df)


print(type(horizontal_dfs[0]))

all_horizontal = pd.concat(horizontal_dfs)

'''for i in range(5):
    curr_file = horizontal_files[i]
    well_id = curr_file.split("__")[0]

    print(curr_file, "->", well_id)'''

print(all_horizontal.shape)
print(all_horizontal["well_id"].nunique())
print("na values: ", all_horizontal.isna().sum().sort_values(ascending = False))
print(all_horizontal.dtypes)

all_horizontal.to_csv("train_horizontal.csv", index=False)