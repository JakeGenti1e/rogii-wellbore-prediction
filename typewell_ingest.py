import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pandas as pd
import os

files = os.listdir("train")

typewell_files = [f for f in files if "__typewell.csv" in f]

print("Typewell:", len(typewell_files))

typewell_dfs = []
for i in range(len(typewell_files)):
    curr_file = typewell_files[i]
    curr_path = "train/" + curr_file
    curr_file = typewell_files[i]
    curr_df = pd.read_csv(curr_path)
    curr_df["well_id"] = curr_file.split("__")[0]
    typewell_dfs.append(curr_df)


print(type(typewell_dfs[0]))

all_typewell = pd.concat(typewell_dfs)

'''for i in range(5):
    curr_file = horizontal_files[i]
    well_id = curr_file.split("__")[0]

    print(curr_file, "->", well_id)'''

print(all_typewell.shape)
print(all_typewell["well_id"].nunique())
print("na values: ", all_typewell.isna().sum().sort_values(ascending = False))
print(all_typewell.dtypes)

all_typewell.to_csv("train_typewell.csv", index=False)