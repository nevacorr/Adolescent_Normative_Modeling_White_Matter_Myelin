import pandas as pd
import numpy as np

def load_raw_data_wm_mpf(path, datafilename):

    # Load mpf tract data
    genz_data = pd.read_csv(f'{path}/{datafilename}')

    # Remove columns with FA
    cols_to_drop = [cols for cols in genz_data.columns if 'FA' in cols]
    genz_data.drop(columns=cols_to_drop, inplace=True)

    # convert subject numbers to integers
    genz_data['Subject'] = genz_data['Subject'].str.replace('sub-genz', '', regex=False).astype(int)

    return genz_data
