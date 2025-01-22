from enum import unique

import pandas as pd
import re

def load_genz_tract_profile_data(struct_var, visit, data_dir, datafilename):

    data_orig = pd.read_csv(f'{data_dir}/{datafilename}')

    # Define regex patterns for columns to remove (nodes 1-20 and 81-100)
    pattern_to_remove = r'_(?:[1-9]|1[0-9]|20|8[1-9]|9[0-9]|100)$'

    # Filter the dataframe to keep only columns that don't match the pattern
    data = data_orig[data_orig.columns[~data_orig.columns.str.contains(pattern_to_remove)]]

    subject_col = data['Subject'].copy()

    col_names = data.columns.to_list()

    col_names.remove('Subject')

    # make a list of tract names
    region_names = [re.sub('_[0-9]+$', '', s) for s in col_names]
    unique_region_names = list(set(region_names))

    if struct_var == 'md':
        unique_region_names = [re.sub('$', ' MD', s) for s in unique_region_names]
    elif struct_var == 'fa':
        unique_region_names = [re.sub('$', ' FA', s) for s in unique_region_names]

    avgdf = pd.DataFrame()

    avgdf.insert(0, 'Subject', subject_col)

    # Convert subject numbers to integers
    avgdf['Subject'] = avgdf['Subject'].str.replace('sub-genz', '', regex=False).astype(int)

    def assign_age(row):
        if 100 <= row['Subject'] <= 199:
            age = 9
        elif 200 <= row['Subject'] <= 299:
            age = 11
        elif 300 <= row['Subject'] <= 399:
            age = 13
        elif 400 <= row['Subject'] <= 499:
            age = 15
        elif 500 <= row['Subject'] <= 599:
            age = 17

        if visit == 2:
            age += 3

        return age

    avgdf['Age'] = avgdf.apply(assign_age, axis=1)

    avgdf.insert(2, 'Visit', visit)

    # Average values for all nodes in each region for each subject
    for region in unique_region_names:
        if struct_var == 'md':
            cols = [col for col in data.columns if col.startswith(region.rstrip(' MD')) ]
        elif struct_var == 'fa':
            cols = [col for col in data.columns if col.startswith(region.rstrip(' FA'))]
        avgdf[region] = data[cols].mean(axis=1)

    return avgdf
