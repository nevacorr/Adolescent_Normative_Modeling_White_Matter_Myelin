import pandas as pd
import numpy as np

def load_raw_data_wm_mpf(struct_var, visit, path, datafilename):

    # Load mpf tract data
    genz_data = pd.read_csv(f'{path}/{datafilename}')

    # Remove columns with FA
    cols_to_drop = [cols for cols in genz_data.columns if 'FA' in cols]
    cols_to_drop.append('Visit')
    genz_data.drop(columns=cols_to_drop, inplace=True)

    return genz_data
