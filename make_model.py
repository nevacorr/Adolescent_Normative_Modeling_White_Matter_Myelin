import numpy as np
import pandas as pd
import os
import shutil
import time
import gc
from numpy.core.defchararray import capitalize
from pcntoolkit.normative import estimate, evaluate
from plot_num_subjs import plot_num_subjs, plot_num_subjs_one_subject
from Utility_Functions import create_design_matrix, plot_data_with_spline, create_design_matrix_one_gender
from Utility_Functions import create_dummy_design_matrix, plot_data_with_spline_one_gender
from Utility_Functions import barplot_performance_values, plot_y_v_yhat, makenewdir, movefiles
from Utility_Functions import write_ages_to_file, write_list_to_file
from apply_normative_model_time2 import apply_normative_model_time2
from evaluate_spline_parameters_cv import evaluate_splines_cv
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def make_model(all_data_v1_orig, all_data_v2_orig, struct_var_metric, n_splits, train_set_array, test_set_array,
               show_nsubject_plots, working_dir, spline_order, spline_knots, show_plots, roi_ids, sensitivity_analysis, evaluate_model_using_validation_set):

    dirdata = 'data'
    dirpredict = 'predict_files'

    # show bar plots with number of subjects per age group in pre-COVID data
    if show_nsubject_plots:
        plot_num_subjs(all_data_v1_orig, f'Subjects by Age with Pre-COVID {struct_var_metric} Data\n'
                                       '(Total N=' + str(all_data_v1_orig.shape[0]) + ')', struct_var_metric,
                                        f'pre-covid subjects',working_dir, dirdata)

    Z2_all_splits = pd.DataFrame()

    total = len(roi_ids)*n_splits  # Get total number of models to be created for this modality
    tcounter = 0  # Initialize counter

    all_Z_score_val = pd.DataFrame()

    all_blr_site_metrics = pd.DataFrame()
    for split in range(n_splits):

        print(f"SPLIT NUMBER = {split+1}/{n_splits}")
        start_time = time.time()  # Record start time

        subjects_train = train_set_array[split, :]
        subjects_test = test_set_array[split, :]

        all_data_v1 = all_data_v1_orig[all_data_v1_orig['participant_id'].isin(subjects_train)].copy()
        all_data_v2 = all_data_v2_orig[all_data_v2_orig['participant_id'].isin(subjects_test)].copy()
        all_data_v1.reset_index(drop=True, inplace=True)
        all_data_v2.reset_index(drop=True, inplace=True)

        if sensitivity_analysis:
            combined_age = pd.concat([all_data_v1["agedays"], all_data_v2["agedays"]])
            agemin = combined_age.min()
            agemax = combined_age.max()

        # plot number of subjects of each gender by age who are included in training data set
        if show_nsubject_plots:
            plot_num_subjs(all_data_v1,
                           'Split ' + str(split) + ' Subjects by Age with Pre-COVID Data used to Train Model\n'
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

        if evaluate_model_using_validation_set:
            X_train, X_val, y_train, y_val = train_test_split(all_data_covariates, all_data_features, stratify=all_data_covariates[['age']], test_size=0.2, random_state=1)
            subjects_val = all_data_v1.loc[X_val.index, 'participant_id'].values
        else:
            # use entire training set to create models
            X_train = all_data_covariates.copy()
            X_val = all_data_covariates.copy()
            y_train = all_data_features.copy()
            y_val = all_data_features.copy()
            subjects_val = all_data_v1.loc[X_val.index, 'participant_id'].values

        if not sensitivity_analysis:
            # identify age range in pre-COVID data to be used for modeling
            agemin = X_train['agedays'].min()
            agemax = X_train['agedays'].max()

        # Run 5 fold cross validation to evaluate best spline parameters
        # if split==0:
        #     cv_dir = f"{working_dir}/spline_cv_split0_{struct_var_metric}"
        #     evaluate_splines_cv(X_train, y_train, roi_ids, agemin, agemax, cv_dir)
        #     os.chdir(working_dir)

        if struct_var_metric == 'fa':
            write_ages_to_file(working_dir, agemin, agemax, struct_var_metric)
        elif struct_var_metric == 'md':
            write_ages_to_file(working_dir, agemin, agemax, struct_var_metric)

        # drop the age column from the train and validation data set because we want to use agedays and sex as predictors
        X_train.drop(columns=['age'], inplace=True)
        X_val.drop(columns=['age'], inplace=True)

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

        y_val_nan_index = {}
        y_val = y_val.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)

        for c in y_val.columns:
            y_val_nan_index[c] = y_val[y_val[c].isna()].index.to_list()
            X_val_copy = X_val.copy()
            y_val_copy = y_val.copy()
            # If there are nan values for this region remove this subject from X_val and y_val for this region only
            if len(y_val_nan_index[c]) == 0:
                X_val_to_file = X_val_copy
                y_val_to_file_region = y_val_copy.loc[:,c]
            else:
                X_val_to_file = X_val_copy.drop(labels=y_val_nan_index[c]).reset_index(drop=True)
                y_val_to_file_region = y_val_copy.loc[:,c].drop(labels=y_val_nan_index[c]).reset_index(drop=True)
            X_val_to_file.to_csv(f'{working_dir}/cov_te_' + c + '.txt', sep='\t', header=False, index=False)
            y_val_to_file_region.to_csv(f'{working_dir}/resp_te_' + c + '.txt', header=False, index=False)
            y_val.to_csv(f'{working_dir}/resp_te.txt', sep='\t', header=False, index=False)

        for i in roi_ids:
            roidirname = '{}/{}/{}/ROI_models/{}'.format(working_dir, dirdata, struct_var_metric, i)
            makenewdir(roidirname)
            cov_tr_filepath = roidirname + '/cov_tr.txt'
            shutil.copyfile("{}/cov_tr_{}.txt".format(working_dir, i), cov_tr_filepath)
            cov_te_filepath = roidirname + '/cov_te.txt'
            shutil.copyfile("{}/cov_te_{}.txt".format(working_dir, i), cov_te_filepath)

            resp_tr_filename = "{}/resp_tr_{}.txt".format(working_dir, i)
            resp_tr_filepath = roidirname + '/resp_tr.txt'
            shutil.copyfile(resp_tr_filename, resp_tr_filepath)
            resp_te_filename = "{}/resp_te_{}.txt".format(working_dir, i)
            resp_te_filepath = roidirname + '/resp_te.txt'
            shutil.copyfile(resp_te_filename, resp_te_filepath)

        movefiles("{}/resp_*.txt".format(working_dir), "{}/{}/{}/response_files/".format(working_dir, dirdata, struct_var_metric))
        movefiles("{}/cov_t*.txt".format(working_dir), "{}/{}/{}/covariate_files/".format(working_dir, dirdata, struct_var_metric))

        #  this path is where ROI_models folders are located
        data_dir = '{}/{}/{}/ROI_models/'.format(working_dir, dirdata, struct_var_metric)

        # Create Design Matrix and add in spline basis and intercept for validation and training data
        create_design_matrix('train', agemin, agemax, spline_order, spline_knots, roi_ids, data_dir)
        create_design_matrix('test', agemin, agemax, spline_order, spline_knots, roi_ids, data_dir)

        # Create pandas dataframes with header names to save performance metrics
        blr_metrics = pd.DataFrame(columns=['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'])
        # blr_site_metrics = pd.DataFrame(
        #     columns=['ROI', 'y_mean', 'y_var', 'yhat_mean', 'yhat_var', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'])

        # create dataframe with subject numbers to put the Z scores in.
        subjects_train = subjects_train.reshape(-1, 1)
        subjects_val = subjects_val.reshape(-1, 1)
        Z_score_train_matrix = pd.DataFrame(subjects_train, columns=['subject_id_train'])
        Z_score_val_matrix = pd.DataFrame(subjects_val, columns=['subject_id_val'])

        # Estimate the normative model using a for loop to iterate over brain regions. The estimate function uses a few
        # specific arguments that are worth commenting on:
        # ●alg=‘blr’: specifies we should use BLR. See Table1 for other available algorithms
        # ●optimizer=‘powell’:usePowell’s derivative-free optimization method(faster in this case than L-BFGS)
        # ●savemodel=True: do not write out the final estimated model to disk
        # ●saveoutput=False: return the outputs directly rather than writing them to disk
        # ●standardize=False: do not standardize the covariates or response variable

        # Loop through ROIs

        roicounter = 0
        for roi in roi_ids:
            print(f"{struct_var_metric} SPLIT NUMBER = {split+1}/{n_splits}")
            print('Running ROI:', roi)
            current_time = time.time()  # Record end time
            elapsed_time = (current_time - start_time) / 60.0  # Calculate elapsed time in minutes
            print(f"Models created for {struct_var_metric}:  {roicounter+1}/{len(roi_ids)} ROIs")
            print(f"Number of times makemodel has been run across all splits = {tcounter+1}/{total}")
            print(f"Elapsed time for split {split+1} for {struct_var_metric} is {elapsed_time:.2f} minutes")
            tcounter += 1  # Increment counter
            roicounter += 1

            print('Running ROI:', roi)
            roi_dir = os.path.join(data_dir, roi)
            model_dir = os.path.join(data_dir, roi, 'Models')
            os.chdir(roi_dir)

            # configure the covariates to use. Change *_bspline_* to *_int_*
            cov_file_tr = os.path.join(roi_dir, 'cov_bspline_tr.txt')
            cov_file_te = os.path.join(roi_dir, 'cov_bspline_te.txt')

            # load train response files
            resp_file_tr = os.path.join(roi_dir, 'resp_tr.txt')
            resp_file_te = os.path.join(roi_dir, 'resp_te.txt')

            try:
                # calculate a model based on the training data and apply to the train dataset. The purpose of
                # running this function is to create and save the model, not to evaluate performance.
                yhat_te, s2_te, nm, Z_te, metrics_te = estimate(cov_file_tr, resp_file_tr, testresp=resp_file_te,
                                                                testcov=cov_file_te, alg='blr', optimizer='powell',
                                                                savemodel=True, saveoutput=False, standardize=False)

                Rho_te=metrics_te['Rho']
                EV_te=metrics_te['EXPV']



                # # create dummy design matrices for visualizing model
                # dummy_cov_file_path_female_tr, dummy_cov_file_path_male_tr = \
                #     create_dummy_design_matrix(struct_var_metric, agemin, agemax, cov_file_tr, spline_order, spline_knots,
                #                                working_dir)
                #
                # dummy_cov_file_path_female_te, dummy_cov_file_path_male_te = \
                #     create_dummy_design_matrix(struct_var_metric, agemin, agemax, cov_file_te, spline_order, spline_knots,
                #                                working_dir)
                #
                # # Compute splines and superimpose on data. Show on screen or save to file depending on show_plots value.
                # plot_data_with_spline('Training Data', struct_var_metric, cov_file_tr, resp_file_tr, dummy_cov_file_path_female_tr,
                #                       dummy_cov_file_path_male_tr, model_dir, roi, show_plots, working_dir, dirdata)
                # plot_data_with_spline('Validation Data', struct_var_metric, cov_file_te, resp_file_te, dummy_cov_file_path_female_te,
                #                       dummy_cov_file_path_male_te, model_dir, roi, show_plots, working_dir, dirdata)

                # Add a row to the blr_metrics dataframe containing ROI, MSLL, EXPV, SMSE, RMSE, and Rho metrics
                blr_metrics.loc[len(blr_metrics)] = [roi, metrics_te['MSLL'][0],
                                                     metrics_te['EXPV'][0], metrics_te['SMSE'][0], metrics_te['RMSE'][0],
                                                     metrics_te['Rho'][0]]
                z_mean = np.mean(Z_te)
                z_std = np.std(Z_te)

                # if metrics_te['EXPV'][0] < 0 and abs(z_mean) < 0.5 and abs(z_std - 1.0) < 0.3:
                #     y_te = np.loadtxt(resp_file_te)
                #
                #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                #
                #     # Left panel: yhat vs y
                #     axes[0].scatter(y_te, yhat_te, alpha=0.6, s=20)
                #     lims = [min(y_te.min(), yhat_te.min()), max(y_te.max(), yhat_te.max())]
                #     axes[0].plot(lims, lims, 'r--', linewidth=1, label='identity')
                #     axes[0].axhline(y=np.mean(yhat_te), color='gray', linestyle=':', linewidth=1, label='mean prediction')
                #     axes[0].set_xlabel('true y')
                #     axes[0].set_ylabel('yhat')
                #     axes[0].set_title(f'yhat vs y\nRho={metrics_te["Rho"][0]:.2f} EXPV={metrics_te["EXPV"][0]:.2f}')
                #     axes[0].legend(fontsize=8)
                #
                #     # Right panel: Z score distribution
                #     axes[1].hist(Z_te, bins=15, edgecolor='black')
                #     axes[1].axvline(x=0, color='r', linestyle='--', linewidth=1, label='mean=0')
                #     axes[1].set_xlabel('Z score')
                #     axes[1].set_ylabel('count')
                #     axes[1].set_title(f'Z distribution\nmean={z_mean:.2f} SD={z_std:.2f}')
                #     axes[1].legend(fontsize=8)
                #
                #     fig.suptitle(f'{roi} split {split + 1}', fontsize=12)
                #     plt.tight_layout()
                #     plt.savefig(f'{working_dir}/data/{struct_var_metric}/yhat_vs_y_{roi}_split{split + 1}.png', dpi=100)
                #     plt.close()

            except:
                yhat_te = np.nan
                s2_te = np.nan
                nm = np.nan
                Z_te = np.full((Z_score_val_matrix.shape[0], 1), np.nan)
                metrics_te = np.nan

            print(f"Z_te shape: {Z_te.shape}")
            print(f"Z_score_val_matrix shape: {Z_score_val_matrix.shape[0]}")
            print(f"nan indices for {roi}: {y_val_nan_index[roi]}")

            if Z_score_val_matrix.shape[0] == Z_te.shape[0]:
                Z_score_val_matrix[roi] = Z_te.squeeze()
            else:
                ind = 0
                for subj in range(Z_score_val_matrix.shape[0]):
                    if subj in y_val_nan_index[roi]:
                        Z_score_val_matrix.loc[subj, roi] = np.nan
                    else:
                        Z_score_val_matrix.loc[subj, roi] = Z_te[ind].squeeze()
                        ind += 1


        blr_metrics['split'] = split + 1

        all_blr_site_metrics = pd.concat([all_blr_site_metrics, blr_metrics], ignore_index=True)

        # Write performance statistics to file
        all_blr_site_metrics.to_csv('{}/data/{}/blr_metrics_{}_{}_splits.txt'.format(working_dir, struct_var_metric, struct_var_metric, n_splits),
                                index=False)

        Z_score_val_matrix['split'] = split + 1
        all_Z_score_val = pd.concat([all_Z_score_val, Z_score_val_matrix], ignore_index=True)
        #
        # Z_time2 = apply_normative_model_time2(struct_var_metric, show_plots, show_nsubject_plots, spline_order,
        #                             spline_knots, working_dir, all_data_v2, roi_ids, dirdata, dirpredict, split,
        #                                       n_splits, sensitivity_analysis)
        #
        # Z_time2['split'] = split
        #
        # Z2_all_splits = pd.concat([Z2_all_splits, Z_time2], ignore_index=True)
        #
        # Z2_all_splits.to_csv(f'{working_dir}/Z_time2_{struct_var_metric}_{n_splits}_splits.csv')
        #
        # write_list_to_file(roi_ids, f'{working_dir}/roi_ids.txt')

        end_time = time.time()  # Record end time
        elapsed_time = (end_time - start_time) / 60.0  # Calculate elapsed time in minutes

        print(f"Elapsed time for split {split+1} for {struct_var_metric} is {elapsed_time:.2f} minutes")

        gc.collect()  # Force garbage collection

    # Z2_all_splits = Z2_all_splits.groupby(by=['participant_id']).mean().drop(columns=['split'])
    # Z2_all_splits = Z2_all_splits.groupby(by=['participant_id']).mean()
    # Z2_all_splits.reset_index(inplace=True)

    all_Z_score_val.to_csv(f'{working_dir}/data/{struct_var_metric}/Z_scores_by_region_validation_{struct_var_metric}_set_all_splits.txt',
                           index=False)

    return Z2_all_splits