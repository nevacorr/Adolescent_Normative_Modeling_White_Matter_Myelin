import os
import pandas as pd

dti_metric='md'
workingdir = os.getcwd()

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(f"{workingdir}/data/{dti_metric}/blr_metrics_{dti_metric}_100_splits.txt")

# -------------------------
# convert metrics to numeric (important if empty strings exist)
# -------------------------
metric_cols = [
    "MSLL","EV","SMSE","RMSE","Rho"
]

df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce")

# drop invalid rows
df = df.dropna(subset=["EV"])

# -------------------------
# Extract tract name
# ROI format: "Some Name_20"
# Remove trailing underscore + 2 digits
# -------------------------
df["tract"] = df["ROI"].str.replace(r'_\d{2}$', '', regex=True)

# -------------------------
# Metrics to summarize
# -------------------------
metric_cols = [
    "MSLL",
    "EV",
    "SMSE",
    "RMSE",
    "Rho"
]

# -------------------------
# Group and compute summaries
# -------------------------
rows = []

for tract, g in df.groupby("tract"):

    summary = {"tract": tract}

    # mean and std for each metric
    for col in metric_cols:
        summary[f"{col}_mean"] = g[col].mean()
        summary[f"{col}_std"] = g[col].std()

    # fraction of positive EV values
    summary["fraction_EV_positive"] = (g["EV"] > 0).mean()

    # number of samples contributing
    summary["N"] = len(g)

    rows.append(summary)

summary_df = pd.DataFrame(rows)

# sort nicely
summary_df = summary_df.sort_values("tract").reset_index(drop=True)

# -------------------------
# Save output
# -------------------------
output_file = f"blr_metric_{dti_metric}_tract_summary.csv"
summary_df.to_csv(output_file, index=False)

print(f"Saved summary to: {output_file}")
print(summary_df)

