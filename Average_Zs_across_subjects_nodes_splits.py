import pandas as pd
import numpy as np
import re
import os
from matplotlib import pyplot as plt

working_dir = os.getcwd()

metric="fa"

file_path = os.path.join(working_dir, "data", f"{metric}", f"Z_scores_by_region_validation_{metric}_set_all_splits.txt")

# Read file (treat blanks as NaN automatically)
df = pd.read_csv(file_path, sep=None, engine="python", na_values=["", " ", "NA", "NaN"])

data = df.drop(columns=["subject_id_val"])

tract_pattern = re.compile(r"(.+)_\d+$")

tract_groups = {}

for col in data.columns:
    m = tract_pattern.match(col)
    if m:
        tract = m.group(1)
        tract_groups.setdefault(tract, []).append(col)

summary = []

for tract, cols in tract_groups.items():
    values = data[cols].to_numpy().ravel()
    values = values[~np.isnan(values)]

    summary.append({
        "tract": tract,
        "mean_z": np.mean(values),
        "std_z": np.std(values, ddof=1)
    })

summary_df = pd.DataFrame(summary).sort_values("tract")
print(summary_df)
summary_df.to_csv(os.path.join(working_dir, f"Z_by_tract_mean_std_{metric}_100splits.csv"), index=False)

# vals = data[tract_groups["Callosum Forceps Minor"]].to_numpy().ravel()
# vals = vals[~np.isnan(vals)]
#
# print(np.mean(vals), np.std(vals))
#
# plt.figure(figsize=(8, 5))
# plt.hist(vals, bins=50)
# plt.xlabel("Z-score")
# plt.ylabel("Count")
# plt.title("Callosum Forceps Minor Z-score Distribution")
# plt.axvline(0, linestyle="--")
# plt.tight_layout()
# plt.show()
#
# cols = tract_groups["Callosum Forceps Minor"]
#
# node_means = data[cols].mean(axis=0)
#
# print(node_means.describe())
#
# cols = tract_groups["Callosum Forceps Minor"]
#
# node_means = data[cols].mean(axis=0)
#
# print(node_means.describe())
#
# plt.figure(figsize=(10,4))
# plt.plot(node_means.values)
# plt.axhline(0, linestyle="--")
# plt.xlabel("Node")
# plt.ylabel("Mean Z-score")
# plt.title("Callosum Forceps Minor Node Means")
# plt.show()