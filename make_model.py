import pandas as pd
import os
import shutil
from numpy.core.defchararray import capitalize
from pcntoolkit.normative import estimate, evaluate
from plot_num_subjs import plot_num_subjs, plot_num_subjs_one_subject
from Utility_Functions import create_design_matrix, plot_data_with_spline, create_design_matrix_one_gender
from Utility_Functions import create_dummy_design_matrix, plot_data_with_spline_one_gender
from Utility_Functions import barplot_performance_values, plot_y_v_yhat, makenewdir, movefiles
from Utility_Functions import write_ages_to_file
from apply_normative_model_time2 import apply_normative_model_time2

def make_model(all_data_v1_orig, all_data_v2_orig, struct_var_metric, n_splits, train_set_array, test_set_array,
               show_nsubject_plots, working_dir, spline_order, spline_knots, show_plots, roi_ids, sex):

    if sex == 'all':
        dirdata = 'data'
        dirpredict = 'predict_files'
    else:
        dirdata = f'data_{sex}'
        dirpredict = f'predict_files_{sex}'

    # show bar plots with number of subjects per age group in pre-COVID data
    if show_nsubject_plots:
        plot_num_subjs(all_data_v1_orig, f'{capitalize(sex)} Subjects by Age with Pre-COVID {struct_var_metric} Data\n'
                                       '(Total N=' + str(all_data_v1_orig.shape[0]) + ')', struct_var_metric,
                                        f'pre-covid_{sex} subjects',working_dir, dirdata)

    Z2_all_splits = pd.DataFrame()

    for split in range(n_splits):

        subjects_train = train_set_array[split, :]
        subjects_test = test_set_array[split, :]

        all_data_v1 = all_data_v1_orig[all_data_v1_orig['participant_id'].isin(subjects_train)].copy()
        all_data_v2 = all_data_v2_orig[all_data_v2_orig['participant_id'].isin(subjects_test)].copy()
        all_data_v1.reset_index(drop=True, inplace=True)
        all_data_v2.reset_index(drop=True, inplace=True)

        # plot number of subjects of each gender by age who are included in training data set
        if show_nsubject_plots:
            if sex == 'all':
                plot_num_subjs(all_data_v1,
                               'Split ' + str(split) + ' Subjects by Age with Pre-COVID Data used to Train Model\n'
                                                       '(Total N=' + str(all_data_v1.shape[0]) + ')', struct_var_metric,
                               'pre-covid_train', working_dir, dirdata)
            else:
                plot_num_subjs_one_subject(all_data_v1, sex, 'Split ' + str(split) + f'{capitalize(sex)} Subjects by Age with Pre-COVID Data used to Train Model\n'
                                                       '(Total N=' + str(all_data_v1.shape[0]) + ')', struct_var_metric,
                                                        'pre-covid_train', working_dir, dirdata)

        makenewdir('{}/{}/{}/ROI_models'.format(working_dir, dirdata, struct_var_metric))
        makenewdir('{}/{}/{}/covariate_files'.format(working_dir, dirdata, struct_var_metric))
        makenewdir('{}/{}/{}/response_files'.format(working_dir, dirdata, struct_var_metric))

        if struct_var_metric != 'fa':
            roi_ids = [s.replace('FA', struct_var_metric.upper()) for s in roi_ids]

        # separate the brain features (response variables) and predictors (age) in to separate dataframes
        all_data_features = all_data_v1.loc[:, roi_ids]
        all_data_covariates = all_data_v1[['age', 'agedays', 'sex']]

        # use entire training set to create models
        X_train = all_data_covariates.copy()
        y_train = all_data_features.copy()

        # identify age range in pre-COVID data to be used for modeling
        agemin = X_train['agedays'].min()
        agemax = X_train['agedays'].max()

        if struct_var_metric == 'fa':
            write_ages_to_file(working_dir, agemin, agemax, struct_var_metric)

        # drop the age and sex columns from the train data set because we want to use agedays as a predictor
        if sex == 'all':
            X_train.drop(columns=['age'], inplace=True)
        else:
            X_train.drop(columns=['age', 'sex'], inplace=True)

        ##########
        # Set up output directories. Save each brain region to its own text file, organized in separate directories,
        # because for each response variable Y (brain region) we fit a separate normative mode
        ##########

        # Check for nan values in y_train for each region. If nan value exists, remove before writing y_train for that
        # region to file. Also remove the corresponding covariate values for that subject.
        for c in y_train.columns:

            y_train_nan_index = y_train[y_train[c].isna()].index.to_list()

            X_train_copy = X_train.copy()
            y_train_copy = y_train.copy()

            # If there are nan values for this region remove this subject from X_train and y_train for this region only
            if len(y_train_nan_index) == 0:
                X_train_to_file = X_train_copy
                y_train_to_file_region = y_train_copy.loc[:,c]
            else:
                X_train_to_file = X_train_copy.drop(labels=y_train_nan_index).reset_index(drop=True)
                y_train_to_file_region = y_train_copy.loc[:,c].drop(labels=y_train_nan_index).reset_index(drop=True)

            X_train_to_file.to_csv(f'{working_dir}/cov_tr_' + c + '.txt', sep='\t', header=False, index=False)
            y_train_to_file_region.to_csv(f'{working_dir}/resp_tr_' + c + '.txt', header=False, index=False)
            y_train.to_csv(f'{working_dir}/resp_tr.txt', sep='\t', header=False, index=False)

        for i in roi_ids:
            roidirname = '{}/{}/{}/ROI_models/{}'.format(working_dir, dirdata, struct_var_metric, i)
            makenewdir(roidirname)
            cov_tr_filepath = roidirname + '/cov_tr.txt'
            shutil.copyfile("{}/cov_tr_{}.txt".format(working_dir, i), cov_tr_filepath)

            resp_tr_filename = "{}/resp_tr_{}.txt".format(working_dir, i)
            resp_tr_filepath = roidirname + '/resp_tr.txt'
            shutil.copyfile(resp_tr_filename, resp_tr_filepath)

        movefiles("{}/resp_*.txt".format(working_dir), "{}/{}/{}/response_files/".format(working_dir, dirdata, struct_var_metric))
        movefiles("{}/cov_t*.txt".format(working_dir), "{}/{}/{}/covariate_files/".format(working_dir, dirdata, struct_var_metric))

        #  this path is where ROI_models folders are located
        data_dir = '{}/{}/{}/ROI_models/'.format(working_dir, dirdata, struct_var_metric)

        # Create Design Matrix and add in spline basis and intercept for validation and training data
        if sex == 'all':
            create_design_matrix('train', agemin, agemax, spline_order, spline_knots, roi_ids, data_dir)
        else:
            create_design_matrix_one_gender('train', agemin, agemax, spline_order, spline_knots, roi_ids, data_dir)

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
                create_dummy_design_matrix(struct_var_metric, agemin, agemax, cov_file_tr, spline_order, spline_knots,
                                           working_dir)

            if sex == 'all':
                # Compute splines and superimpose on data. Show on screen or save to file depending on show_plots value.
                plot_data_with_spline('Training Data', struct_var_metric, cov_file_tr, resp_file_tr, dummy_cov_file_path_female,
                                      dummy_cov_file_path_male, model_dir, roi, show_plots, working_dir, dirdata)
            elif sex == 'female':
                plot_data_with_spline_one_gender(sex, 'Training Data ', struct_var_metric, cov_file_tr, resp_file_tr,
                                                 dummy_cov_file_path_female,
                                                 model_dir, roi, show_plots, working_dir, dirdata, dirpredict)
            elif sex == 'male':
                plot_data_with_spline_one_gender(sex, 'Training Data ', struct_var_metric, cov_file_tr, resp_file_tr,
                                                 dummy_cov_file_path_male,
                                                 model_dir, roi, show_plots, working_dir, dirdata, dirpredict)

        Z_time2 = apply_normative_model_time2(struct_var_metric, show_plots, show_nsubject_plots, spline_order, spline_knots,
                                    working_dir, all_data_v2, roi_ids, dirdata, dirpredict, sex)

        Z_time2['split'] = split

        Z2_all_splits = pd.concat([Z2_all_splits, Z_time2], ignore_index=True)

    # Z2_all_splits = Z2_all_splits.groupby(by=['participant_id']).mean().drop(columns=['split'])
    # Z2_all_splits = Z2_all_splits.groupby(by=['participant_id']).mean()
    # Z2_all_splits.reset_index(inplace=True)

    return Z2_all_splits