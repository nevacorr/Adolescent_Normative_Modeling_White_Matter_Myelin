import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

from numpy.core.defchararray import capitalize
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pcntoolkit.normative import estimate, evaluate
from plot_num_subjs import plot_num_subjs
from Load_Genz_Data_WWM_FA_MD import load_genz_data_wm_fa_md, load_genz_data_wm_mpf_v1
from apply_normative_model_time2 import apply_normative_model_time2
from make_model import make_model
from Utility_Functions import make_nm_directories
from load_data_all import load_data_all


def make_and_apply_normative_model_fa_md(mfseparate, struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                              raw_data_dir, working_dir, fa_datafilename_v1, fa_datafilename_v2, md_datafilename_v1,
                              md_datafilename_v2, mpf_datafilename_v1, mpf_datafilename_v2, subjects_to_exclude_v1, subjects_to_exclude_v2,
                              mpf_subjects_to_exclude_v1, mpf_subjects_to_exclude_v2, demographics_filename, n_splits):

    # Initialize dictionaries for storing Z scores
    Z2_all_splits_fa_dict = {}
    Z2_all_splits_md_dict = {}
    Z2_all_splits_mpf_dict = {}

    # Load data
    (fa_all_data_v1, fa_all_data_v2, md_all_data_v1, md_all_data_v2, mpf_all_data_v1, mpf_all_data_v2, sub_v1_only_orig,
     sub_v2_only_orig, sub_v1_only_mpf_orig, sub_v2_only_mpf_orig, roi_ids) = (
        load_data_all(struct_var, raw_data_dir, fa_datafilename_v1, fa_datafilename_v2, md_datafilename_v1,
                  md_datafilename_v2, mpf_datafilename_v1, mpf_datafilename_v2, demographics_filename,
                  subjects_to_exclude_v1, subjects_to_exclude_v2, mpf_subjects_to_exclude_v1, mpf_subjects_to_exclude_v2))

    # Keep and process only the data for the sexes of interest
    if mfseparate == 0:
        make_nm_directories(working_dir, 'data', 'predict_files')
        sexes = 'all'
        sub_v1_only = sub_v1_only_orig.copy()
        sub_v2_only = sub_v2_only_orig.copy()
        sub_v1_only_mpf = sub_v1_only_mpf_orig.copy()
        sub_v2_only_mpf = sub_v2_only_mpf_orig.copy()
    else:
        sexes = ['females', 'males']
        for sex in sexes:
            make_nm_directories(working_dir, f'data_{sex}', f'predict_files_{sex}')
            if sex == 'females':
                sexflag = 0
                sub_v1_only = [sub for sub in sub_v1_only_orig if sub %2 == 0]
                sub_v2_only = [sub for sub in sub_v1_only_orig if sub % 2 == 0]
                sub_v1_only_mpf = [sub for sub in sub_v1_only_mpf_orig if sub % 2 == 0]
                sub_v2_only_mpf = [sub for sub in sub_v2_only_mpf_orig if sub % 2 == 0]
            elif sex == 'males':
                sexflag = 1
                sub_v1_only = [sub for sub in sub_v1_only_orig if sub % 2 != 0]
                sub_v2_only = [sub for sub in sub_v1_only_orig if sub % 2 != 0]
                sub_v1_only_mpf = [sub for sub in sub_v1_only_mpf_orig if sub % 2 != 0]
                sub_v2_only_mpf = [sub for sub in sub_v2_only_mpf_orig if sub % 2 != 0]

            fa_all_data_v1 = fa_all_data_v1[fa_all_data_v1['sex'] == sexflag].copy()
            fa_all_data_v2 = fa_all_data_v2[fa_all_data_v2['sex'] == sexflag].copy()
            md_all_data_v1 = md_all_data_v1[md_all_data_v1['sex'] == sexflag].copy()
            md_all_data_v2 = md_all_data_v2[md_all_data_v2['sex'] == sexflag].copy()
            mpf_all_data_v1 = mpf_all_data_v1[mpf_all_data_v1['sex'] == sexflag].copy()
            mpf_all_data_v2 = mpf_all_data_v1[mpf_all_data_v2['sex'] == sexflag].copy()

    for sex in sexes:
        # show bar plots with number of subjects per age group in pre-COVID data
        if show_nsubject_plots:
            plot_num_subjs(fa_all_data_v1, f'{capitalize(sex)} Subjects by Age with Pre-COVID FA Data\n'
                                     '(Total N=' + str(fa_all_data_v1.shape[0]) + ')', 'fa', f'pre-covid_{sex} subjects',
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

        # Create a dataframe that has only visit 1 data and only subject number, visit, age and sex as columns
        cols_to_keep = ['participant_id', 'visit', 'age', 'sex', 'age']
        cols_to_drop = [col for col in fa_df_for_train_test_split if col not in cols_to_keep]
        fa_df_for_train_test_split.drop(columns=cols_to_drop, inplace=True)
        # keep only the subjects that have data at both time points
        fa_df_for_train_test_split=fa_df_for_train_test_split[fa_df_for_train_test_split['participant_id'].isin(all_subjects_2ts)]

        # Initialize StratifiedShuffleSplit for equal train/test sizes
        splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.52, random_state=42)

        train_set_list = []
        test_set_list = []
        # Perform the splits
        for i, (train_index, test_index) in enumerate(splitter.split(fa_df_for_train_test_split, fa_df_for_train_test_split['age'])):
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

        Z2_all_splits_fa_dict[sex] = Z2_all_splits_fa
        Z2_all_splits_md_dict[sex] = Z2_all_splits_md
        Z2_all_splits_mpf_dict[sex] = Z2_all_splits_mpf

    return roi_ids, Z2_all_splits_fa_dict, Z2_all_splits_md_dict , Z2_all_splits_mpf_dict