from enum import unique

import pandas as pd
import re

def load_genz_tract_profile_data(visit, data_dir, datafilename):

    data_orig = pd.read_csv(f'{data_dir}/{datafilename}')

    # Define regex patterns for columns to remove (nodes 1-20 and 81-100)
    pattern_to_remove = r'_(?:[1-9]|1[0-9]|20|8[1-9]|9[0-9]|100)$'

    # Filter the dataframe to keep only columns that don't match the pattern
    data = data_orig[data_orig.columns[~data_orig.columns.str.contains(pattern_to_remove)]].copy()

    # Convert subject numbers to integers
    data['Subject'] = data['Subject'].str.replace('sub-genz', '', regex=False).astype(int)

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

    # Add age to dataframe
    data['Age'] = data.apply(assign_age, axis=1)
    agecol = data.pop('Age')
    data.insert(1, 'Age', agecol)

    # Add visit number
    data.insert(1, 'Visit', visit)

    return data
