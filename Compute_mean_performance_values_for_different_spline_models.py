import pandas as pd
from matplotlib import pyplot as plt
from numpy.core.defchararray import capitalize

data_type='md'  #option "md", "fa"

# Load data
df = pd.read_csv(f"spline_cv_split0_md/mean_metrics_per_spline_{data_type}.csv")

# Extract region prefix (R1, R2, etc.)
df["region"] = df["roi"].str.split("_").str[0]

# Metrics to average
metrics = ["RMSE", "Rho", "pRho", "SMSE", "EXPV", "MSLL", "NLL", "BIC"]

# ---- 1. Mean per region AND spline settings ----
df_region_spline_mean = (
    df.groupby(["region", "spline_order", "spline_knots"])[metrics]
      .mean()
      .reset_index()
)

# ---- 2. Mean across all regions (still per spline settings) ----
df_overall_spline_mean = (
    df.groupby(["spline_order", "spline_knots"])[metrics]
      .mean()
      .reset_index()
)

df_region_spline_mean["order_knot"] = (
    df_region_spline_mean["spline_order"].astype(str) + "_" +
    df_region_spline_mean["spline_knots"].astype(str)
)

df_region_spline_mean = df_region_spline_mean.sort_values(
    ["spline_order", "spline_knots"]
)

# Create numeric x positions
order_knot_labels = df_region_spline_mean["order_knot"].unique()
x_map = {k: i for i, k in enumerate(order_knot_labels)}

plt.figure(figsize=(10, 6))

for region in df_region_spline_mean["region"].unique():
    sub = df_region_spline_mean[df_region_spline_mean["region"] == region]
    x = sub["order_knot"].map(x_map)

    plt.plot(x, sub["BIC"], marker='o', label=region)

plt.xticks(range(len(order_knot_labels)), order_knot_labels)
plt.xlabel("spline_order_knots")
plt.ylabel("BIC")
plt.title(f"{capitalize(data_type)} BIC across spline values (all regions)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

mystop=1