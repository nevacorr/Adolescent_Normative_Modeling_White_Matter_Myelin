import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pcntoolkit.normative import estimate, evaluate
from plot_num_subjs import plot_num_subjs
from Utility_Functions import create_design_matrix, plot_data_with_spline
from Utility_Functions import create_dummy_design_matrix
from Utility_Functions import barplot_performance_values, plot_y_v_yhat, makenewdir, movefiles
from Utility_Functions import write_ages_to_file
from Load_Genz_Data_WWM_MPF import load_genz_data_wm_mpf_v1, load_genz_data_wm_mpf_v2
from apply_normative_model_time2 import apply_normative_model_time2

def make_and_apply_normative_model(struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                              data_dir, working_dir, datafilename_v1, datafilename_v2, subjects_to_exclude_v1,
                               subjects_to_exclude_v2, demographics_filename, n_splits):

    # Load all data
    all_data_both_visits, roi_ids, all_subjects, sub_v1_only, sub_v2_only= load_genz_data_wm_mpf_v1(struct_var, data_dir,
                                                            datafilename_v1, datafilename_v2, demographics_filename)

    # Create dataframe with just visit 1 data
    all_data_v1 = all_data_both_visits.copy()
    all_data_v1 = all_data_v1[all_data_v1['visit'] == 1]
    all_data_v1.reset_index(inplace=True, drop=True)
    all_data_v1.drop(columns=['visit', 'Age'],inplace=True)

    # Create dataframe with just visit 2 data
    all_data_v2 = all_data_both_visits.copy()
    all_data_v2 = all_data_v2[all_data_v2['visit'] == 2]
    all_data_v2.reset_index(inplace=True, drop=True)
    all_data_v2.drop(columns=['visit', 'Age'],inplace=True)

    # Remove subjects to exclude for dataframes
    all_data_v1 = all_data_v1[~all_data_v1['participant_id'].isin(subjects_to_exclude_v1)]
    all_data_v2 = all_data_v2[~all_data_v2['participant_id'].isin(subjects_to_exclude_v2)]

    # Replace gender codes 1=male 2=female with binary values (make male=1 and female=0)
    all_data_v1.loc[all_data_v1['sex'] == 2, 'sex'] = 0
    all_data_v2.loc[all_data_v2['sex'] == 2, 'sex'] = 0

    # show bar plots with number of subjects per age group in pre-COVID data
    if show_nsubject_plots:
        plot_num_subjs(all_data_v1, 'Subjects by Age with Pre-COVID Data\n'
                                 '(Total N=' + str(all_data_v1.shape[0]) + ')', struct_var, 'pre-covid_allsubj',
                                  working_dir)

    # remove subjects to exclude v1 from sub_v1_only
    sub_v1_only = [val for val in sub_v1_only if val not in subjects_to_exclude_v1]
    # remove subjects to exclude v2 from sub_v2_only
    sub_v2_only = [val for val in sub_v2_only if val not in subjects_to_exclude_v2]
    # remove subjects to exclude from list of all subjects
    all_subjects = [val for val in all_subjects if (val not in subjects_to_exclude_v1) and (val not in subjects_to_exclude_v2)]
    num_subjects_for_ttsplit = len(all_subjects) - len(sub_v1_only) - len(sub_v2_only)

    subjects_only_one_visit = sub_v1_only + sub_v2_only

    # remove excluded subjects from all_subjects_both_visits_dataframe
    all_data_both_visits = all_data_both_visits[~all_data_both_visits['participant_id'].isin(subjects_to_exclude_v1)]
    all_data_both_visits = all_data_both_visits[~all_data_both_visits['participant_id'].isin(subjects_to_exclude_v2)]

    # remove subjects with only data from one visit from alL_subjects_both_visits dataframe
    all_data_both_visits = all_data_both_visits[~all_data_both_visits['participant_id'].isin(subjects_only_one_visit)]

    # Create a new column in dataframe that combines age and gender for stratification
    all_data_both_visits['age_sex'] = all_data_both_visits['age'].astype(str) + '_' + all_data_both_visits['sex'].astype(str)

    # Create a dataframe that has only visit 1 data and only subject number, visit, age and sex as columns
    cols_to_keep = ['participant_id', 'visit', 'age', 'sex', 'age_sex']
    cols_to_drop = [col for col in all_data_both_visits if col not in cols_to_keep]
    all_data_both_visits.drop(columns=cols_to_drop, inplace=True)
    all_data_all_visits = all_data_both_visits[all_data_both_visits['visit']==1]

    # Initialize StratifiedShuffleSplit for equal train/test sizes
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.64, random_state=42)

    train_set_list = []
    test_set_list = []
    # Perform the splits
    for i, (train_index, test_index) in enumerate(splitter.split(all_data_all_visits, all_data_all_visits['age_sex'])):
        train_set_list_tmp = all_data_all_visits.iloc[train_index, 0].values.tolist()
        train_set_list_tmp.extend(sub_v1_only)
        test_set_list_tmp = all_data_all_visits.iloc[test_index, 0].values.tolist()
        test_set_list_tmp.extend(sub_v2_only)
        train_set_list.append(train_set_list_tmp)
        test_set_list.append(test_set_list_tmp)

    train_set_array = np.array(list(train_set_list))
    test_set_array = np.array(list(test_set_list))

    fname_train = '{}/visit1_subjects_train_sets_{}_splits_{}.txt'.format(working_dir, n_splits, struct_var)
    np.save(fname_train, train_set_array)

    fname_test = '{}/visit1_subjects_test_sets_{}_splits_{}.txt'.format(working_dir, n_splits, struct_var)
    np.save(fname_test,test_set_array)


    Z2_all_splits = pd.DataFrame()
    for split in range(n_splits):
        subjects_train = train_set_array[split, :]
        subjects_test = test_set_array[split, :]
        all_data_v1 = all_data_v1[all_data_v1['participant_id'].isin(subjects_train)]
        all_data_v2 = all_data_v2[all_data_v2['participant_id'].isin(subjects_test)]

        # plot number of subjects of each gender by age who are included in training data set
        if show_nsubject_plots:
            plot_num_subjs(all_data_v1, 'Split ' + str(split) + ' Subjects by Age with Pre-COVID Data used to Train Model\n'
                                     '(Total N=' + str(all_data_v1.shape[0]) + ')', struct_var, 'pre-covid_train',
                                      working_dir)

            # make directories to store files
            makenewdir('{}/data/'.format(working_dir))
            makenewdir('{}/data/{}'.format(working_dir, struct_var))
            makenewdir('{}/data/{}/plots'.format(working_dir, struct_var))
            makenewdir('{}/data/{}/ROI_models'.format(working_dir, struct_var))
            makenewdir('{}/data/{}/covariate_files'.format(working_dir, struct_var))
            makenewdir('{}/data/{}/response_files'.format(working_dir, struct_var))

            # separate the brain features (response variables) and predictors (age) in to separate dataframes
            all_data_features = all_data_v1.loc[:, roi_ids]
            all_data_covariates = all_data_v1[['age', 'agedays', 'sex']]

            # use entire training set to create models
            X_train = all_data_covariates.copy()
            y_train = all_data_features.copy()

            # identify age range in pre-COVID data to be used for modeling
            agemin = X_train['agedays'].min()
            agemax = X_train['agedays'].max()

            write_ages_to_file(working_dir, agemin, agemax, struct_var)

            # save the subject numbers for the training and validation sets to variables
            s_index_train = X_train.index.values
            subjects_train = all_data_v1.loc[s_index_train, 'participant_id'].values

            # drop the age column from the train data set because we want to use agedays as a predictor
            X_train.drop(columns=['age'], inplace=True)

            # change the indices in the train data set because nan values were dropped above
            X_train.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)

            ##########
            # Set up output directories. Save each brain region to its own text file, organized in separate directories,
            # because for each response variable Y (brain region) we fit a separate normative mode
            ##########
            for c in y_train.columns:
                y_train[c].to_csv(f'{working_dir}/resp_tr_' + c + '.txt', header=False, index=False)
                X_train.to_csv(f'{working_dir}/cov_tr.txt', sep='\t', header=False, index=False)
                y_train.to_csv(f'{working_dir}/resp_tr.txt', sep='\t', header=False, index=False)

            for i in roi_ids:
                roidirname = '{}/data/{}/ROI_models/{}'.format(working_dir, struct_var, i)
                makenewdir(roidirname)
                resp_tr_filename = "{}/resp_tr_{}.txt".format(working_dir, i)
                resp_tr_filepath = roidirname + '/resp_tr.txt'
                shutil.copyfile(resp_tr_filename, resp_tr_filepath)
                cov_tr_filepath = roidirname + '/cov_tr.txt'
                shutil.copyfile("{}/cov_tr.txt".format(working_dir), cov_tr_filepath)

            movefiles("{}/resp_*.txt".format(working_dir), "{}/data/{}/response_files/".format(working_dir, struct_var))
            movefiles("{}/cov_t*.txt".format(working_dir), "{}/data/{}/covariate_files/".format(working_dir, struct_var))

            #  this path is where ROI_models folders are located
            data_dir = '{}/data/{}/ROI_models/'.format(working_dir, struct_var)

            # Create Design Matrix and add in spline basis and intercept for validation and training data
            create_design_matrix('train', agemin, agemax, spline_order, spline_knots, roi_ids, data_dir)

            # create dataframe with subject numbers to put the Z scores in.
            subjects_train = subjects_train.reshape(-1, 1)
            Z_score_train_matrix = pd.DataFrame(subjects_train, columns=['subject_id_train'])

            # Estimate the normative model using a for loop to iterate over brain regions. The estimate function uses a few
            # specific arguments that are worth commenting on:
            # ●alg=‘blr’: specifies we should use BLR. See Table1 for other available algorithms
            # ●optimizer=‘powell’:usePowell’s derivative-free optimization method(faster in this case than L-BFGS)
            # ●savemodel=True: do not write out the final estimated model to disk
            # ●saveoutput=False: return the outputs directly rather than writing them to disk
            # ●standardize=False: do not standardize the covariates or response variable

            # Loop through ROIs

            for roi in roi_ids:
                print('Running ROI:', roi)
                roi_dir = os.path.join(data_dir, roi)
                model_dir = os.path.join(data_dir, roi, 'Models')
                os.chdir(roi_dir)

                # configure the covariates to use. Change *_bspline_* to *_int_*
                cov_file_tr = os.path.join(roi_dir, 'cov_bspline_tr.txt')

                # load train response files
                resp_file_tr = os.path.join(roi_dir, 'resp_tr.txt')

                # calculate a model based on the training data and apply to the train dataset. The purpose of
                # running this function is to create and save the model, not to evaluate performance.
                yhat_tr, s2_tr, nm, Z_tr, metrics_tr = estimate(cov_file_tr, resp_file_tr, testresp=resp_file_tr,
                                                                testcov=cov_file_tr, alg='blr', optimizer='powell',
                                                                savemodel=True, saveoutput=False, standardize=False)

                # create dummy design matrices for visualizing model
                dummy_cov_file_path_female, dummy_cov_file_path_male = \
                    create_dummy_design_matrix(struct_var, agemin, agemax, cov_file_tr, spline_order, spline_knots, working_dir)

                # Compute splines and superimpose on data. Show on screen or save to file depending on show_plots value.
                plot_data_with_spline('Training Data', struct_var, cov_file_tr, resp_file_tr, dummy_cov_file_path_female,
                                      dummy_cov_file_path_male, model_dir, roi, show_plots, working_dir)
        mystop=1

        subjects_train = train_set_array[split, :]
        Z_time2 = apply_normative_model_time2(struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                                    working_dir, datafilename_v2, subjects_to_exclude_v2,
                                    demographics_filename, all_data_v2, subjects_test, roi_ids)

        Z_time2['split'] = split

        Z2_all_splits = pd.concat([Z2_all_splits, Z_time2], ignore_index=True)

    Z2_all_splits = Z2_all_splits.groupby(by=['participant_id']).mean().drop(columns=['split'])

    Z2_all_splits.reset_index(drop=True, inplace=True)

    return roi_ids, Z2_all_splits