from enum import unique

import pandas as pd
import re

def load_genz_tract_profile_data_noavg(visit, data_dir, datafilename):

    data_orig = pd.read_csv(f'{data_dir}/{datafilename}')

    # Define regex patterns for columns to remove (nodes 1-20 and 81-100)
    pattern_to_remove = r'_(?:[1-9]|1[0-9]|20|8[1-9]|9[0-9]|100)$'

    # Filter the dataframe to keep only columns that don't match the pattern
    data = data_orig[data_orig.columns[~data_orig.columns.str.contains(pattern_to_remove)]]

    subject_col = data['Subject'].copy()

    col_names = data.columns.to_list()

    col_names.remove('Subject')

    df = pd.DataFrame()

    df.insert(0, 'Subject', subject_col)

    # Convert subject numbers to integers
    df['Subject'] = df['Subject'].str.replace('sub-genz', '', regex=False).astype(int)

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

    df['Age'] = df.apply(assign_age, axis=1)

    df.insert(2, 'Visit', visit)

    # Convert subject numbers in data to integers
    data.loc[:,'Subject'] = data.loc[:, 'Subject'].str.replace('sub-genz', '', regex=False).astype(int)
    final_df = df.merge(data, on='Subject')

    return final_df
