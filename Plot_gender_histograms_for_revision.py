import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import seaborn as sns

n_splits = 100

palette = {
    "Male": [0.11, 0.62, 0.47],
    "Female": [0.46, 0.44, 0.70]
}

hue_order = ["Male", "Female"]

working_dir = os.getcwd()
data_dir = 'results_orig_ImageNeursci_submission'

for metric in ['fa']:

    Z_time2 = pd.read_csv(
        f'{working_dir}/{data_dir}/Z_time2_{metric}_{n_splits}_splits.csv.bak',
        usecols=lambda col: col != "Unnamed: 0"
    )

    roi_ids = pd.read_csv(f'{working_dir}/{data_dir}/roi_ids.txt', header=None)
    roi_ids = roi_ids.iloc[:, 0].tolist()

    z_cols = Z_time2.columns.difference(['participant_id', 'split'])
    Z_time2 = Z_time2.groupby('participant_id', as_index=False)[z_cols].mean()

    # gender
    Z_time2['gender'] = Z_time2['participant_id'].apply(lambda x: 0 if x % 2 == 0 else 1)

    Z_time2['gender'] = Z_time2['gender'].map({1: 'Male', 0: 'Female'})

    Z_long = Z_time2.melt(
        id_vars=['participant_id', 'gender'],
        var_name='tract_node',
        value_name='z'
    )

    Z_long[['tractID', 'nodeID']] = Z_long['tract_node'].str.rsplit('_', n=1, expand=True)

    Z_collapsed = (
        Z_long
        .groupby(['participant_id', 'gender', 'tractID'], as_index=False)['z']
        .mean()
    )

    keep_tracts = ["Callosum Forceps Minor", "Right IFOF"]

    Z_plot = Z_collapsed[Z_collapsed["tractID"].isin(keep_tracts)]

    # Make sure gender is readable
    Z_plot = Z_plot.copy()
    Z_plot["gender"] = pd.Categorical(
        Z_plot["gender"],
        categories=hue_order,
        ordered=True
    )

    # Z_plot['gender'] = Z_plot['gender'].map({0: 'Male', 1: 'Female'})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    x_min, x_max = -2, 2

    for ax, tract in zip(axes, keep_tracts):
        subset = Z_plot[Z_plot["tractID"] == tract]

        sns.histplot(
            data=subset,
            x="z",
            hue="gender",
            hue_order=hue_order,
            bins=12,
            stat="count",
            multiple="layer",
            # element="step",
            alpha=0.5,
            edgecolor=None,
            palette=palette,
            ax=ax
        )
        ax.set_xlim(x_min, x_max)
        ax.set_title(tract + ' FA')
        ax.set_xlabel("Z-score")
    #
    # # only one legend for whole figure
    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, title="Sex", loc="center right")

    handles = [
        mpatches.Patch(color=palette["Male"], label="Male", alpha=0.5),
        mpatches.Patch(color=palette["Female"], label="Female", alpha=0.5),
    ]

    fig.legend(
        handles=handles,
        title="Sex",
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        frameon=False
    )

    # fig.legend(handles=handles, title="Sex", loc="center right")

    # remove duplicate legends from axes
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    fig.subplots_adjust(right=0.85)
    # plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


    mystop=1