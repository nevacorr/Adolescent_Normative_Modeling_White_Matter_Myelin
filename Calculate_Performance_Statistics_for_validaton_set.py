import pandas as pd
import glob
import os
import re
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

metric = "md"

# path to current directory
working_dir = os.getcwd()
# path to your directory
data_dir = f"{working_dir}/validation_set_predictions_20splits_June/{metric}"

# grab all relevant files
files = glob.glob(os.path.join(data_dir, f"predictions_true_yhat_Z_{metric}_*_split*.csv"))

pattern = rf"predictions_true_yhat_Z_{metric}_(.*?)_(.*?)_split(\d+)\.csv"

results = []
z_by_tract = defaultdict(list)
z_all = []

for f in files:
    filename = os.path.basename(f)
    match = re.match(pattern, filename)

    if not match:
        continue

    tract = match.group(1)
    node = int(match.group(2))
    split = int(match.group(3))

    df = pd.read_csv(f)
    z = df["Z_score"].dropna().values

    if len(z) == 0:
        print(f'{tract}{node}split{split} {metric} has nans')
        continue

    y_true = df["y_true"]
    y_hat = df["yhat_te"]
    z = df["Z_score"]
    z_by_tract[tract].append(z)
    z_all.append(z)
    # nans_present = df.isnull().values.any()
    # if nans_present:
    #     print(f'{tract}{node}split{split} {metric} has nans')
    #     continue
    # else:
        # compute metrics
    results.append({
        "tract": tract,
        "node": node,
        "split": split,
        "EV":  1 - np.var(y_true - y_hat) / np.var(y_true),
        "R2": r2_score(y_true, y_hat),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
        "corr": np.corrcoef(y_true, y_hat)[0, 1],
        "z_mean": z.mean(),
        "z_std": z.std()
    })

metrics_df = pd.DataFrame(results)

z_by_tract = {k: np.concatenate(v) for k, v in z_by_tract.items()}
z_all = np.concatenate(z_all)

summary = metrics_df[['EV', 'R2','RMSE','corr','z_mean','z_std']].agg(['mean','std'])
print(summary)
# summary.to_csv(f"{data_dir}/summary_metrics_{metric}.csv")

tract_summary = metrics_df.groupby('tract')[['EV', 'R2','RMSE','corr', 'z_mean', 'z_std']].mean()
tract_summary.to_csv(f"{data_dir}/summary_by_tract_{metric}.csv")

plt.figure()
plt.hist(z_all, bins=100)
plt.xlim(-10, 10)
mu=np.mean(z_all)
sigma=np.std(z_all)
plt.title(f"{metric.upper()} Z-score Distribution: All Tracts\nmean={mu:.2f}, std={sigma:.2f}")
plt.xlabel("Z_score")
plt.ylabel("Frequency")
plt.show()
filepath = os.path.join(data_dir, f'{metric.upper()} Z_score Distribution All Tracts.png')
plt.savefig(filepath, dpi=300, bbox_inches='tight')

n = len(z_by_tract)
cols = 4
rows = 4

fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
axes = axes.flatten()

for i, (tract, z_vals) in enumerate(z_by_tract.items()):
    mu = np.mean(z_vals)
    sigma = np.std(z_vals)

    axes[i].hist(z_vals, bins=50)
    axes[i].set_xlim(-10, 10)
    axes[i].set_title(f"{metric.upper()} {tract}\nmean={mu:.2f}, std={sigma:.2f}")

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
filepath = os.path.join(data_dir, f'{metric} Z_score Distribution By Tract.png')
plt.savefig(filepath, dpi=300, bbox_inches='tight')

mystop=1
