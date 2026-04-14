import pandas as pd
import glob
import os
import re
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

metric = "fa"

# path to current directory
working_dir = os.getcwd()
# path to your directory
data_dir = f"{working_dir}/validation_set_predictions/{metric}"

# grab all relevant files
files = glob.glob(os.path.join(data_dir, f"predictions_true_yhat_Z_{metric}_*_split*.csv"))

pattern = rf"predictions_true_yhat_Z_{metric}_(.*?)_(.*?)_split(\d+)\.csv"

results = []

for f in files:
    filename = os.path.basename(f)
    match = re.match(pattern, filename)

    if not match:
        continue

    tract = match.group(1)
    node = int(match.group(2))
    split = int(match.group(3))

    df = pd.read_csv(f)

    y_true = df["y_true"]
    y_hat = df["yhat_te"]
    z = df["Z_score"]

    # compute metrics immediately
    results.append({
        "tract": tract,
        "node": node,
        "split": split,
        "R2": r2_score(y_true, y_hat),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
        "corr": np.corrcoef(y_true, y_hat)[0, 1],
        "z_mean": z.mean(),
        "z_std": z.std()
    })

metrics_df = pd.DataFrame(results)

mystop=1
