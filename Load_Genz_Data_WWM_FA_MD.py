# This function loads adolescent visit 1 and visit 2 cortical thickness and quality control data. It then only keeps
# the data for the visit of interest and returns the cortical thickness values for all regions, the covariates,
# and a list of all region names
##########

import pandas as pd
from load_raw_data_WM_MPF import load_raw_data_wm_mpf
from load_genz_tract_profile_data import load_genz_tract_profile_data

def load_genz_data_wm_fa_md(path, fa_braindatafilename_v1, fa_braindatafilename_v2, md_braindatafilename_v1,
                            md_braindatafilename_v2, mpf_braindatafilename_v1, mpf_braindatafilename_v2, demographics_filename):

    visit = 1
    # load data from visit 1
    brain_data_v1_fa = load_genz_tract_profile_data(visit, path, fa_braindatafilename_v1)
    brain_data_v1_md = load_genz_tract_profile_data(visit, path, md_braindatafilename_v1)
    brain_data_v1_mpf = load_genz_tract_profile_data(visit, path, mpf_braindatafilename_v1)

    visit = 2
    # load data from visit 2
    brain_data_v2_fa = load_genz_tract_profile_data(visit, path, fa_braindatafilename_v2)
    brain_data_v2_md = load_genz_tract_profile_data(visit, path, md_braindatafilename_v2)
    brain_data_v2_mpf = load_genz_tract_profile_data(visit, path, mpf_braindatafilename_v2)

    # put data from both visits in same dataframe
    fa_brain_data = pd.concat([brain_data_v1_fa, brain_data_v2_fa])
    md_brain_data = pd.concat([brain_data_v1_md, brain_data_v2_md])
    mpf_brain_data = pd.concat([brain_data_v1_mpf, brain_data_v2_mpf])

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
    mpf_brain_data.rename(columns={'Subject': 'subject', 'Visit': 'visit'}, inplace=True)

    fa_all_data = pd.merge(demo_data, fa_brain_data, how='right', on=['subject', 'visit'])
    fa_all_data.sort_values(by='subject', inplace=True, ignore_index=True)
    md_all_data = pd.merge(demo_data, md_brain_data, how='right', on=['subject', 'visit'])
    md_all_data.sort_values(by='subject', inplace=True, ignore_index=True)
    mpf_all_data = pd.merge(demo_data, mpf_brain_data, how='right', on=['subject', 'visit'])
    mpf_all_data.sort_values(by='subject', inplace=True, ignore_index=True)

    # Make lists of subjects with data only at timepoint 1, subjects with data at only timepoint 2,
    # and all subjects in dataset for FA and MD data
    unique_subjects = fa_all_data['subject'].value_counts()
    unique_subjects = unique_subjects[unique_subjects == 1].index
    subjects_with_one_dataset = fa_all_data[fa_all_data['subject'].isin(unique_subjects)]
    subjects_visit1_data_only = subjects_with_one_dataset[subjects_with_one_dataset['visit'] == 1]
    subjects_visit2_data_only = subjects_with_one_dataset[subjects_with_one_dataset['visit'] == 2]
    subjects_v1_only = subjects_visit1_data_only['subject'].tolist()
    subjects_v2_only = subjects_visit2_data_only['subject'].tolist()

    # Make lists of subjects with data only at timepoint 1, subjects with data at only timepoint 2,
    # and all subjects in dataset for MPF data
    unique_subjects_mpf = mpf_all_data['subject'].value_counts()
    unique_subjects_mpf = unique_subjects_mpf[unique_subjects_mpf == 1].index
    subjects_with_one_dataset_mpf = mpf_all_data[mpf_all_data['subject'].isin(unique_subjects_mpf)]
    subjects_visit1_data_only_mpf = subjects_with_one_dataset_mpf[subjects_with_one_dataset_mpf['visit'] == 1]
    subjects_visit2_data_only_mpf = subjects_with_one_dataset_mpf[subjects_with_one_dataset_mpf['visit'] == 2]
    mpf_subjects_v1_only = subjects_visit1_data_only_mpf['subject'].tolist()
    mpf_subjects_v2_only = subjects_visit2_data_only_mpf['subject'].tolist()

    # create a list of all the region columns to run a normative model for
    columns_to_exclude = ['subject', 'visit', 'gender', 'agemonths', 'agedays', 'agegroup', 'Age']
    roi_ids = [col for col in fa_all_data.columns if col not in columns_to_exclude]

    tmp_reg_to_keep = ['Left Thalamic Radiation', 'Right Thalamic Radiation', 'Left IFOF', 'Right IFOF', 'Left ILF', 'Right ILF']
    roi_ids = [region for region in roi_ids if any(sub in region for sub in tmp_reg_to_keep)]

    fa_all_data.rename(columns={'subject': 'participant_id', 'gender':'sex', 'agegroup': 'age'}, inplace=True)
    md_all_data.rename(columns={'subject': 'participant_id', 'gender': 'sex', 'agegroup': 'age'}, inplace=True)
    mpf_all_data.rename(columns={'subject': 'participant_id', 'gender': 'sex', 'agegroup': 'age'}, inplace=True)

    all_subjects = fa_all_data['participant_id'].unique().tolist()
    all_subjects_mpf = mpf_all_data['participant_id'].unique().tolist()

    return (fa_all_data, md_all_data, mpf_all_data, roi_ids, all_subjects, all_subjects_mpf, subjects_v1_only,
            subjects_v2_only, mpf_subjects_v1_only, mpf_subjects_v2_only)


def load_genz_data_wm_mpf_v1(struct_var, path, braindatafilename_v1, braindatafilename_v2,
                             demographics_filename):

    visit = 1
    # load data from visit 1
    brain_data_v1 = load_raw_data_wm_mpf(path, braindatafilename_v1)
    visit = 2
    # load data from visit 2
    brain_data_v2 = load_raw_data_wm_mpf(path, braindatafilename_v2)

    # put data from both visits in same dataframe
    brain_data = pd.concat([brain_data_v1, brain_data_v2])

    # get demographic data
    demo_data = pd.read_csv(f'{path}/{demographics_filename}')
    demo_to_keep = ['subject', 'visit', 'gender', 'agemonths', 'agedays', 'agegroup']
    cols_to_drop = [item for item in demo_data.columns if item not in demo_to_keep]
    demo_data.drop(columns=cols_to_drop, inplace=True)
    demo_data.dropna(inplace=True, ignore_index=True)
    columns_to_convert = demo_data.columns.difference(['subject'])
    demo_data[columns_to_convert] = demo_data[columns_to_convert].astype('int64')

    # merge demo data with brain data
    brain_data.rename(columns={'Subject': 'subject', 'Visit': 'visit'}, inplace=True)
    all_data = pd.merge(demo_data, brain_data, how='right', on=['subject', 'visit'])
    all_data.sort_values(by='subject', inplace=True, ignore_index=True)

    # make a list of subjects with data at only timepoint 1, and another for subjects with data only at timepoint 2
    unique_subjects = all_data['subject'].value_counts()
    unique_subjects = unique_subjects[unique_subjects == 1].index
    subjects_with_one_dataset = all_data[all_data['subject'].isin(unique_subjects)]
    subjects_visit1_data_only = subjects_with_one_dataset[subjects_with_one_dataset['visit'] == 1]
    subjects_visit2_data_only = subjects_with_one_dataset[subjects_with_one_dataset['visit'] == 2]
    subjects_v1_only = subjects_visit1_data_only['subject'].tolist()
    subjects_v2_only = subjects_visit2_data_only['subject'].tolist()

    # create a list of all the columns to run a normative model for
    # roi_ids = [col for col in all_data.columns if struct_var.upper() in col]

    all_data.rename(columns={'subject': 'participant_id', 'gender':'sex', 'agegroup': 'age'}, inplace=True)

    # make a list of all unique subject numbers
    all_subjects = all_data['participant_id'].unique().tolist()

    return all_data, all_subjects, subjects_v1_only, subjects_v2_only