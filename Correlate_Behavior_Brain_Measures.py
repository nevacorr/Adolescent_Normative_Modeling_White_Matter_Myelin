
import os
import pandas as pd
from itertools import product
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

working_dir = os.getcwd()

behav_zs = pd.read_csv('/home/toddr/neva/PycharmProjects/AdolNormativeModelingCOVID/'
                       'Z_scores_all_meltzoff_cogn_behav_visit2.csv', usecols=lambda column: column != 'Unnamed: 0')

fa_zs = pd.read_csv(f'{working_dir}/Z_time2_fa_100_splits.csv', usecols=lambda column: column != 'Unnamed: 0')
md_zs = pd.read_csv(f'{working_dir}/Z_time2_md_100_splits.csv', usecols=lambda column: column != 'Unnamed: 0')

# behaviors_of_interest = ['participant_id', 'CDImean', 'RSQanxiety', 'RSQanger', 'StateAnxiety', 'TraitAnxiety', 'FlankerSU', 'DCSU']
behaviors_of_interest = ['participant_id', 'FlankerSU', 'DCSU']

FA_regions_of_interest = ['participant_id', 'Right ILF FA', 'Right IFOF FA']

# MD_regions_of_interest = ['participant_id', 'Callosum Forceps Major MD', 'Callosum Forceps Minor MD', 'Left Thalamic Radiation MD',
#                           'Right Thalamic Radiation MD']
MD_regions_of_interest = md_zs.columns.tolist()
MD_regions_of_interest.remove('gender')

# Remove rows where participant_id is odd
# behav_zs = behav_zs[behav_zs['participant_id'] % 2 == 0]

behav_zs.drop(columns = behav_zs.columns.difference(behaviors_of_interest), inplace=True)
fa_zs.drop(columns = fa_zs.columns.difference(FA_regions_of_interest), inplace=True)
md_zs.drop(columns = md_zs.columns.difference(MD_regions_of_interest), inplace=True)

combined_df = behav_zs.merge(md_zs, on='participant_id')

combined_df = combined_df.dropna(axis=0)

behaviors_of_interest.remove('participant_id')
FA_regions_of_interest.remove('participant_id')
MD_regions_of_interest.remove('participant_id')

# Calculate correlations and p-values
results = []
for col1, col2 in product(behaviors_of_interest, MD_regions_of_interest):
    corr, p_value = pearsonr(combined_df[col1], combined_df[col2])  # Compute correlation and p-value
    results.append({'Column1': col1, 'Column2': col2, 'Correlation': corr, 'p_value': p_value})

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Perform FDR correction
_, pvals_corrected, _, _ = multipletests(results_df['p_value'], alpha=0.05, method='fdr_bh')

# Add corrected p-values to the DataFrame
results_df['p_value_corrected'] = pvals_corrected

# Determine significance after FDR correction
results_df['Significant'] = results_df['p_value_corrected'] < 0.05

print(results_df)

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

plot_scatter(combined_df, 'FlankerSU', 'Callosum Forceps Minor MD')

mystop=1


