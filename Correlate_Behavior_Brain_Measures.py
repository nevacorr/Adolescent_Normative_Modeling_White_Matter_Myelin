
import os
import pandas as pd
from itertools import product
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define dti metric for calculations
diffusion_var = 'md'

# Add remove outliers flag
remove_outliers = 0

brain_regions_of_interest = ['Minor']
behaviors_of_interest = ['FlankerSU', 'DCSU']

# Get working directory
working_dir = os.getcwd()

# Load behavioral z scores
behav_zs = pd.read_csv('/home/toddr/neva/PycharmProjects/AdolNormativeModelingCOVID/'
                       'Z_scores_all_meltzoff_cogn_behav_visit2.csv', usecols=lambda column: column != 'Unnamed: 0')

# Load diffusion metric z scores
if diffusion_var == 'fa':
    dwi_zs = pd.read_csv(f'{working_dir}/Z_time2_fa_100_splits.csv', usecols=lambda column: column != 'Unnamed: 0')
elif diffusion_var == 'md':
    dwi_zs = pd.read_csv(f'{working_dir}/Z_time2_md_100_splits.csv', usecols=lambda column: column != 'Unnamed: 0')

# Average diffusion data across splits
dwi_zs = dwi_zs.drop(columns=['split'])
dwi_zs = dwi_zs.groupby('participant_id', as_index=False).mean()

# Remove rows where participant_id is odd
# behav_zs = behav_zs[behav_zs['participant_id'] % 2 == 0]

# Keep only behavior columns that contain substrings from behaviors_of_interest, plus 'participant_id'
behav_zs = behav_zs[[col for col in behav_zs.columns if any(sub in col for sub in behaviors_of_interest) or col == 'participant_id']]

# Keep only DWI columns that contain substrings from brain_regions_of_interest, plus 'participant_id'
dwi_zs = dwi_zs[[col for col in dwi_zs.columns if any(sub in col for sub in brain_regions_of_interest) or col == 'participant_id']]

# Merge behavior and brain data by participant
combined_df = behav_zs.merge(dwi_zs, on='participant_id')

# Remove all rows that are missing values
combined_df = combined_df.dropna(axis=0)

columns_to_keep = brain_regions_of_interest + behaviors_of_interest

# Create a list of columns to keep by filtering out the ones we don't need
columns_to_keep = [
    col for col in combined_df.columns if any(sub in col for sub in columns_to_keep)
]

# Reassign the DataFrame with the filtered columns
combined_df = combined_df[columns_to_keep]

# Expand brain_regions_of_interest to full column names using substring matching
matched_brain_columns = [col for col in combined_df.columns if any(sub in col for sub in behaviors_of_interest)]

if remove_outliers:
    # Drop rows where *any* of those columns has a value less than -2
    combined_df = combined_df[~(combined_df[matched_brain_columns] < -2).any(axis=1)]

# Expand behavior columns based on substrings
expanded_behaviors = [col for col in combined_df.columns if any(sub in col for sub in behaviors_of_interest)]

# Expand brain region columns based on substrings
expanded_brain_regions = [col for col in combined_df.columns if any(sub in col for sub in brain_regions_of_interest)]

results = []
# Average values for all brain regions
substring = brain_regions_of_interest[0]
columns_to_average = [col for col in combined_df.columns if substring in col]
combined_df['average_brain_val'] = combined_df[columns_to_average].mean(axis=1)

behav='FlankerSU'
single_brain_corr, single_brain_p = pearsonr(combined_df[behav], combined_df['average_brain_val'])
results.append({'Column1': behav, 'Column2': 'average_brain_val', 'Correlation': single_brain_corr, 'p_value': single_brain_p})
behav='DCSU'
single_brain_corr, single_brain_p = pearsonr(combined_df[behav], combined_df['average_brain_val'])
results.append({'Column1': behav, 'Column2': 'average_brain_val', 'Correlation': single_brain_corr, 'p_value': single_brain_p})
# behav='WMemorySU'
# single_brain_corr, single_brain_p = pearsonr(combined_df[behav], combined_df['average_brain_val'])
# results.append({'Column1': behav, 'Column2': 'average_brain_val', 'Correlation': single_brain_corr, 'p_value': single_brain_p})

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Perform FDR correction
_, pvals_corrected, _, _ = multipletests(results_df['p_value'], alpha=0.05, method='fdr_bh')

# Add corrected p-values to the DataFrame
results_df['p_value_corrected'] = pvals_corrected

# Determine significance after FDR correction
results_df['Significant'] = results_df['p_value_corrected'] < 0.05

filtered_df = results_df[results_df['Significant'] == True]

# Define a function that plots 2 columns as scatter plot
def plot_scatter(df, col1name, col2name, title):

    # Create a scatter plot
    plt.scatter(df[col1name], df[col2name])

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(combined_df[[col1name]], combined_df[[col2name]])

    # Plot the regression line
    plt.plot(combined_df[[col1name]], model.predict(combined_df[[col1name]]), color='red')

    # Add labels and title
    plt.xlabel(col1name)
    plt.ylabel(col2name)
    plt.title(title)

    # Show the plot
    plt.show(block=False)

title = f'Z Flanker SU vs Z Callosum Forceps Minor avg {diffusion_var.upper()} post-COVID'
plot_scatter(combined_df, 'FlankerSU', 'average_brain_val', title)

pd.set_option('display.max_columns', None)
print(results_df)

mystop=1


