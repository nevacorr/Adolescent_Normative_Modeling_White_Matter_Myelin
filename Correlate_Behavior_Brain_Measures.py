
import os
import pandas as pd
from itertools import product
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define dti metric for calculations
diffusion_var = 'md'

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

brain_regions_of_interest = ['Arcuate', 'IFOF', 'ILF', 'Thalamic', 'Callosum', 'Corticospinal']
behaviors_of_interest = ['Vocab', 'Anxiety', 'CDI', 'anxiety', 'SU', 'anger']

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

# Expand behavior columns based on substrings
expanded_behaviors = [col for col in combined_df.columns if any(sub in col for sub in behaviors_of_interest)]

# Expand brain region columns based on substrings
expanded_brain_regions = [col for col in combined_df.columns if any(sub in col for sub in brain_regions_of_interest)]

# Calculate correlations and p-values
results = []
for col1, col2 in product(expanded_behaviors, expanded_brain_regions):
    corr, p_value = pearsonr(combined_df[col1], combined_df[col2])  # Compute correlation and p-value
    results.append({'Column1': col1, 'Column2': col2, 'Correlation': corr, 'p_value': p_value})

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Perform FDR correction
_, pvals_corrected, _, _ = multipletests(results_df['p_value'], alpha=0.05, method='fdr_bh')

# Add corrected p-values to the DataFrame
# results_df['p_value_corrected'] = pvals_corrected
results_df['p_value_corrected'] = results_df['p_value']

# Determine significance after FDR correction
results_df['Significant'] = results_df['p_value_corrected'] < 0.05

a_filtered_df = results_df[results_df['Significant'] == True]

# Define a function that plots 2 columns as scatter plot
def plot_scatter(df, col1name, col2name):

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
    plt.title(f'{col1name} vs {col2name}')

    # Show the plot
    plt.show()

# plot_scatter(combined_df, 'FlankerSU', 'Callosum Forceps Minor MD')

mystop=1


