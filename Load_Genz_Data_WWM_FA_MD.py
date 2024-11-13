# This function loads adolescent visit 1 and visit 2 cortical thickness and quality control data. It then only keeps
# the data for the visit of interest and returns the cortical thickness values for all regions, the covariates,
# and a list of all region names
##########

import pandas as pd
from load_raw_data_WM_MPF import load_raw_data_wm_mpf
from load_genz_tract_profile_data import load_genz_tract_profile_data

def load_genz_data_wm_fa_md(struct_var, path, braindatafilename_v1, braindatafilename_v2, demographics_filename):

    visit = 1
    # load data from visit 1
    brain_data_v1_md = load_genz_tract_profile_data('md', visit, path, braindatafilename_v1)
    brain_data_v1_fa = load_genz_tract_profile_data('fa', visit, path, braindatafilename_v1)
    visit = 2
    # load data from visit 2
    brain_data_v2_md = load_genz_tract_profile_data('md', visit, path, braindatafilename_v2)
    brain_data_v2_fa = load_genz_tract_profile_data('fa', visit, path, braindatafilename_v2)

    # put data from both visits in same dataframe
    fa_brain_data = pd.concat([brain_data_v1_fa, brain_data_v2_fa])
    md_brain_data = pd.concat([brain_data_v1_md, brain_data_v2_md])
    # brain_data['Subject'] = brain_data['Subject'].astype('int64')
    # get demographic data
    demo_data = pd.read_csv(f'{path}/{demographics_filename}')
    demo_to_keep = ['subject', 'visit', 'gender', 'agemonths', 'agedays', 'agegroup']
    cols_to_drop = [item for item in demo_data.columns if item not in demo_to_keep]
    demo_data.drop(columns=cols_to_drop, inplace=True)
    demo_data.dropna(inplace=True, ignore_index=True)
    columns_to_convert = demo_data.columns.difference(['subject'])
    demo_data[columns_to_convert] = demo_data[columns_to_convert].astype('int64')

    # merge demo data with brain data
    fa_brain_data.rename(columns={'Subject': 'subject', 'Visit': 'visit'}, inplace=True)
    md_brain_data.rename(columns={'Subject': 'subject', 'Visit': 'visit'}, inplace=True)
    fa_all_data = pd.merge(demo_data, fa_brain_data, how='right', on=['subject', 'visit'])
    fa_all_data.sort_values(by='subject', inplace=True, ignore_index=True)
    md_all_data = pd.merge(demo_data, md_brain_data, how='right', on=['subject', 'visit'])
    md_all_data.sort_values(by='subject', inplace=True, ignore_index=True)

    unique_subjects = fa_all_data['subject'].value_counts()
    unique_subjects = unique_subjects[unique_subjects == 1].index
    subjects_with_one_dataset = fa_all_data[fa_all_data['subject'].isin(unique_subjects)]
    subjects_visit1_data_only = subjects_with_one_dataset[subjects_with_one_dataset['visit'] == 1]
    subjects_visit2_data_only = subjects_with_one_dataset[subjects_with_one_dataset['visit'] == 2]
    subjects_v1_only = subjects_visit1_data_only['subject'].tolist()
    subjects_v2_only = subjects_visit2_data_only['subject'].tolist()

    # create a list of all the columns to run a normative model for
    roi_ids = [col for col in fa_all_data.columns if struct_var.upper() in col]

    fa_all_data.rename(columns={'subject': 'participant_id', 'gender':'sex', 'agegroup': 'age'}, inplace=True)
    md_all_data.rename(columns={'subject': 'participant_id', 'gender': 'sex', 'agegroup': 'age'}, inplace=True)

    all_subjects = fa_all_data['participant_id'].unique().tolist()

    return fa_all_data, md_all_data, roi_ids, all_subjects, subjects_v1_only, subjects_v2_only