# This function loads adolescent visit 1 and visit 2 cortical thickness and quality control data. It then only keeps
# the data for the visit of interest and returns the cortical thickness values for all regions, the covariates,
# and a list of all region names
##########

import pandas as pd
from load_raw_data_WM_MPF import load_raw_data_wm_mpf

def load_genz_data_wm_mpf(struct_var, visit, path, braindatafilename, demographics_filename):

    # get brain data
    brain_data = load_raw_data_wm_mpf(struct_var, visit, path, braindatafilename)

    # convert subject numbers to integers
    brain_data['Subject'] = brain_data['Subject'].str.replace('sub-genz', '', regex=False).astype(int)

    # get demographic data
    demo_data = pd.read_csv(f'{path}/{demographics_filename}')
    demo_data=demo_data[demo_data['visit']==visit]
    demo_to_keep = ['subject', 'gender', 'agemonths', 'agedays', 'agegroup']
    cols_to_drop = [item for item in demo_data.columns if item not in demo_to_keep]
    demo_data.drop(columns=cols_to_drop, inplace=True)
    demo_data.dropna(inplace=True, ignore_index=True)
    columns_to_convert = demo_data.columns.difference(['subject'])
    demo_data[columns_to_convert] = demo_data[columns_to_convert].astype('int64')

    # merge demo data with brain data
    brain_data.rename(columns={'Subject': 'subject'}, inplace=True)
    all_data = pd.merge(demo_data, brain_data, how='right', on = 'subject')
    all_data.drop(columns=['Age'], inplace=True)

    # create a list of all the columns to run a normative model for
    roi_ids = [col for col in all_data.columns if struct_var.upper() in col]

    all_data.rename(columns={'subject': 'participant_id', 'gender':'sex', 'agegroup': 'age'}, inplace=True)

    return all_data, roi_ids