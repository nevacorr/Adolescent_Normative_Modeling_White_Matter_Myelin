import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from make_model import make_model
from Utility_Functions import make_nm_directories
from load_data_all import load_data_all
from evaluate_model_fit_using_validation_set import evaluate_model_fit_using_validation_set
import gc
import sys

def make_and_apply_normative_model_fa_md(struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                              raw_data_dir, working_dir, fa_datafilename_v1, fa_datafilename_v2, md_datafilename_v1,
                              md_datafilename_v2, subjects_to_exclude_v1, subjects_to_exclude_v2, demographics_filename,
                              n_splits, evaluate_model_using_validation_set, sensitivity_analysis):

    # Load data
    (fa_all_data_v1, fa_all_data_v2, md_all_data_v1, md_all_data_v2, sub_v1_only_orig,
     sub_v2_only_orig, roi_ids) = (
        load_data_all(raw_data_dir, fa_datafilename_v1, fa_datafilename_v2, md_datafilename_v1,
                  md_datafilename_v2, demographics_filename, subjects_to_exclude_v1, subjects_to_exclude_v2))

    all_subjects_v1 = fa_all_data_v1['participant_id']
    all_subjects_v2 = fa_all_data_v2['participant_id']
    all_subjects_v1.to_csv('visit1_all_subjects_used_in_analysis.csv', index=False)
    all_subjects_v2.to_csv('visit2_all_subjects_used_in_analysis.csv', index=False)

    make_nm_directories(working_dir, 'data', 'predict_files')

    sub_v1_only = sub_v1_only_orig.copy()
    sub_v2_only = sub_v2_only_orig.copy()

    # remove subjects to exclude v1 from sub_v1_only
    sub_v1_only = [val for val in sub_v1_only if val not in subjects_to_exclude_v1]
    # remove subjects to exclude v2 from sub_v2_only
    sub_v2_only = [val for val in sub_v2_only if val not in subjects_to_exclude_v2]
    # make a list of all subjects with data at both timepoints
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
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.50, random_state=42)

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

    # if evaluate_model_using_validation_set:
    #     evaluate_model_fit_using_validation_set(fa_all_data_v1, 'fa', n_splits, train_set_array, working_dir, spline_order,
    #                                         spline_knots, roi_ids, show_plots)
    #
    #     evaluate_model_fit_using_validation_set(md_all_data_v1, 'md', n_splits, train_set_array, working_dir, spline_order,
    #                                         spline_knots, roi_ids, show_plots)
    #
    #     sys.exit()
    #
    Z2_all_splits_md = make_model(md_all_data_v1, md_all_data_v2, 'md', n_splits, train_set_array, test_set_array,
               show_nsubject_plots, working_dir, spline_order, spline_knots, show_plots, roi_ids, sensitivity_analysis, evaluate_model_using_validation_set)

    Z2_all_splits_md.to_csv(f'{working_dir}/Z2_all_splits_md.csv')

    #
    # gc.collect() # Force garbage collection

    Z2_all_splits_fa = make_model(fa_all_data_v1, fa_all_data_v2, 'fa', n_splits, train_set_array, test_set_array,
               show_nsubject_plots, working_dir, spline_order, spline_knots, show_plots, roi_ids, sensitivity_analysis, evaluate_model_using_validation_set)

    Z2_all_splits_fa.to_csv(f'{working_dir}/Z2_all_splits_fa.csv')

    gc.collect() # Force garbage collection

    sys.exit()

    return Z2_all_splits_fa, Z2_all_splits_md, roi_ids

