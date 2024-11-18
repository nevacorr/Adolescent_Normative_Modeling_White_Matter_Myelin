import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def compute_df_correlations(df1, df2):
    # Copy the dataframes to avoid modifying the originals
    df1 = df1.copy()
    df2 = df2.copy()

    # Drop participant_id column
    df1.drop(columns=['participant_id'], inplace=True)
    df2.drop(columns=['participant_id'], inplace=True)

    # Replace 'FA' and 'MD' in column names
    df1_columnnames = [col.replace('FA', '') for col in df1.columns.tolist()]
    df2_columnnames = [col.replace('MD', '') for col in df2.columns.tolist()]

    df1.columns = df1_columnnames
    df2.columns = df2_columnnames

    # Group by the 'split' column and compute correlations for each group
    unique_splits = df1['split'].unique()

    correlation_results = {}

    for split_value in unique_splits:
        # Filter the rows corresponding to this split value
        df1_split = df1[df1['split'] == split_value].drop(columns=['split'])
        df2_split = df2[df2['split'] == split_value].drop(columns=['split'])

        # Compute column-wise correlations for the filtered data
        column_correlations = df1_split.corrwith(df2_split)

        # Store the correlation results for each split value
        correlation_results[split_value] = column_correlations

    # Convert correlation results to a DataFrame
    correlation_df = pd.DataFrame(correlation_results)

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_df, fmt=".2f", cmap="coolwarm", cbar=True, vmin=-1, vmax=1)
    plt.title("Column-wise Correlations for Each Split Value")
    plt.xlabel("Columns")
    plt.ylabel("Split Value")
    plt.tight_layout()
    plt.show()

    mystop=1