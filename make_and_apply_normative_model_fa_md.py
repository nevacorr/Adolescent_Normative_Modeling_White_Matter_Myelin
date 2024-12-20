import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pcntoolkit.normative import estimate, evaluate
from plot_num_subjs import plot_num_subjs
from Load_Genz_Data_WWM_FA_MD import load_genz_data_wm_fa_md, load_genz_data_wm_mpf_v1
from apply_normative_model_time2 import apply_normative_model_time2
from make_model import make_model
from Utility_Functions import makenewdir


def make_and_apply_normative_model_fa_md(struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                              raw_data_dir, working_dir, fa_datafilename_v1, fa_datafilename_v2, md_datafilename_v1,
                              md_datafilename_v2, mpf_datafilename_v1, mpf_datafilename_v2, subjects_to_exclude_v1, subjects_to_exclude_v2,
                              mpf_subjects_to_exclude_v1, mpf_subjects_to_exclude_v2, demographics_filename, n_splits):

    # Load all data
    fa_all_data_both_visits, md_all_data_both_visits, roi_ids, all_subjects, sub_v1_only, sub_v2_only = (
        load_genz_data_wm_fa_md(struct_var, raw_data_dir, fa_datafilename_v1, fa_datafilename_v2, md_datafilename_v1,
        md_datafilename_v2, demographics_filename))

    mpf_all_data_both_visits, all_subjects_mpf, sub_v1_only_mpf, sub_v2_only_mpf = (
        load_genz_data_wm_mpf_v1('mpf', raw_data_dir, mpf_datafilename_v1, mpf_datafilename_v2,
                                             demographics_filename ))

    def process_dataframe(data, visit, subjects_to_exclude):
        df = data.copy()
        df = df[df['visit'] == visit]
        df.reset_index(inplace=True, drop=True)
        df.drop(columns=['visit', 'Age'], inplace=True)
        df = df[~df['participant_id'].isin(subjects_to_exclude)]
        df.loc[df['sex'] == 2, 'sex'] = 0
        return df

    fa_all_data_v1 = process_dataframe(fa_all_data_both_visits, 1, subjects_to_exclude_v1)
    fa_all_data_v2 = process_dataframe(fa_all_data_both_visits, 2, subjects_to_exclude_v2)
    md_all_data_v1 = process_dataframe(md_all_data_both_visits, 1, subjects_to_exclude_v1)
    md_all_data_v2 = process_dataframe(md_all_data_both_visits, 2, subjects_to_exclude_v2)
    mpf_all_data_v1 = process_dataframe(mpf_all_data_both_visits, 1, mpf_subjects_to_exclude_v1)
    mpf_all_data_v2 = process_dataframe(mpf_all_data_both_visits, 2, mpf_subjects_to_exclude_v2)

    # make directories to store files for model creation
    dirpath = os.path.join(working_dir, 'data')
    try:
        shutil.rmtree(dirpath)
        print(f"Directory '{dirpath}' and its contents have been removed.")
    except FileNotFoundError:
        print(f"Directory '{dirpath}' does not exist.")
    makenewdir('{}/data/'.format(working_dir))
    for struct_var_metric in ['fa', 'md', 'mpf']:
        makenewdir('{}/data/{}'.format(working_dir, struct_var_metric))
        makenewdir('{}/data/{}/plots'.format(working_dir, struct_var_metric))

    #make file diretories for model tssting
    dirpath = os.path.join(working_dir, 'predict_files')
    try:
        shutil.rmtree(dirpath)
        print(f"Directory '{dirpath}' and its contents have been removed.")
    except FileNotFoundError:
        print(f"Directory '{dirpath}' does not exist.")
    makenewdir('{}/predict_files/'.format(working_dir))
    for struct_var_metric in ['fa', 'md', 'mpf']:
        makenewdir('{}/predict_files/{}'.format(working_dir, struct_var_metric))
        makenewdir('{}/predict_files/{}/plots'.format(working_dir, struct_var_metric))

    # show bar plots with number of subjects per age group in pre-COVID data
    if show_nsubject_plots:
        plot_num_subjs(fa_all_data_v1, 'Subjects by Age with Pre-COVID FA Data\n'
                                 '(Total N=' + str(fa_all_data_v1.shape[0]) + ')', 'fa', 'pre-covid_allsubj',
                                  working_dir)

    # remove subjects to exclude v1 from sub_v1_only
    sub_v1_only = [val for val in sub_v1_only if val not in subjects_to_exclude_v1]
    # remove subjects to exclude v2 from sub_v2_only
    sub_v2_only = [val for val in sub_v2_only if val not in subjects_to_exclude_v2]
    # remove subjects to exclude from list of all subjects
    all_subjects = fa_all_data_v1['participant_id'].tolist()
    all_subjects.extend(fa_all_data_v2['participant_id'].tolist())
    all_subjects = pd.unique(all_subjects).tolist()
    all_subjects.sort()
    all_subjects_2ts = [sub for sub in all_subjects if (sub not in sub_v1_only and sub not in sub_v2_only)]

    num_subjs_random_add_train = (len(all_subjects)/2) - len(sub_v1_only)
    num_subjs_random_add_test = (len(all_subjects)/2) - len(sub_v2_only)

    fa_df_for_train_test_split = fa_all_data_v1.copy()

    # Create a new column in dataframe that combines age and gender for stratification
    fa_df_for_train_test_split['age_sex'] = fa_df_for_train_test_split['age'].astype(str) + '_' + fa_df_for_train_test_split['sex'].astype(str)

    # Create a dataframe that has only visit 1 data and only subject number, visit, age and sex as columns
    cols_to_keep = ['participant_id', 'visit', 'age', 'sex', 'age_sex']
    cols_to_drop = [col for col in fa_df_for_train_test_split if col not in cols_to_keep]
    fa_df_for_train_test_split.drop(columns=cols_to_drop, inplace=True)
    # keep only the subjects that have data at both time points
    fa_df_for_train_test_split=fa_df_for_train_test_split[fa_df_for_train_test_split['participant_id'].isin(all_subjects_2ts)]

    # Initialize StratifiedShuffleSplit for equal train/test sizes
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.52, random_state=42)

    train_set_list = []
    test_set_list = []
    # Perform the splits
    for i, (train_index, test_index) in enumerate(splitter.split(fa_df_for_train_test_split, fa_df_for_train_test_split['age_sex'])):
        train_set_list_tmp = fa_df_for_train_test_split.iloc[train_index, 0].values.tolist()
        train_set_list_tmp.extend(sub_v1_only)
        test_set_list_tmp = fa_df_for_train_test_split.iloc[test_index, 0].values.tolist()
        test_set_list_tmp.extend(sub_v2_only)
        train_set_list.append(train_set_list_tmp)
        test_set_list.append(test_set_list_tmp)

    train_set_array = np.array(list(train_set_list))
    test_set_array = np.array(list(test_set_list))

    fname_train = '{}/visit1_subjects_train_sets_{}_splits_{}.txt'.format(working_dir, n_splits, struct_var)
    np.save(fname_train, train_set_array)

    fname_test = '{}/visit1_subjects_test_sets_{}_splits_{}.txt'.format(working_dir, n_splits, struct_var)
    np.save(fname_test,test_set_array)

    fa_all_data_v1_orig = fa_all_data_v1
    fa_all_data_v2_orig = fa_all_data_v2

    md_all_data_v1_orig = md_all_data_v1
    md_all_data_v2_orig = md_all_data_v2

    mpf_all_data_v1_orig = mpf_all_data_v1
    mpf_all_data_v2_orig = mpf_all_data_v2

    Z2_all_splits_fa = make_model(fa_all_data_v1_orig, fa_all_data_v2_orig, 'fa', n_splits, train_set_array, test_set_array,
               show_nsubject_plots, working_dir, spline_order, spline_knots, show_plots, roi_ids)

    Z2_all_splits_md = make_model(md_all_data_v1_orig, md_all_data_v2_orig, 'md', n_splits, train_set_array, test_set_array,
               show_nsubject_plots, working_dir, spline_order, spline_knots, show_plots, roi_ids)

    roi_ids_tmp = roi_ids.copy()
    roi_ids_tmp.remove('Left Uncinate FA')
    roi_ids_tmp.remove('Right Uncinate FA')
    Z2_all_splits_mpf = make_model(mpf_all_data_v1_orig, mpf_all_data_v2_orig, 'mpf', n_splits, train_set_array, test_set_array,
               show_nsubject_plots, working_dir, spline_order, spline_knots, show_plots, roi_ids_tmp)

    return roi_ids, Z2_all_splits_fa, Z2_all_splits_md , Z2_all_splits_mpf