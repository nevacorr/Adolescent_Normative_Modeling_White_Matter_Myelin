from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from Utility_Functions import makenewdir, write_ages_to_file, create_design_matrix, movefiles, read_ages_from_file
from Utility_Functions import create_dummy_design_matrix, plot_data_with_spline, make_nm_directories
import os
import shutil
import pandas as pd
from pcntoolkit.normative import estimate
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from normative_edited import predict

def evaluate_model_fit_using_validation_set(all_data_v1_orig, struct_var_metric, n_splits, train_set_array_orig, working_dir, spline_order,
                                            spline_knots, roi_ids, show_plots):
    valdata = 'val_data'
    valpredict = 'val_predict'

    make_nm_directories(working_dir, valdata, valpredict)

    for split in range(n_splits):
        print(f"VALIDATION SPLIT NUMBER = {split+1}/{n_splits}")

        subjects_train = train_set_array_orig[split, :]

        train_orig = all_data_v1_orig[all_data_v1_orig['participant_id'].isin(subjects_train)].copy()
        train_orig.reset_index(drop=True, inplace=True)

        # Initialize StratifiedShuffleSplit for equal train/val sizes
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        train_subset_index, val_subset_index = next(splitter.split(train_orig, train_orig['age']))

        train_subset = train_orig.iloc[train_subset_index]
        val_subset = train_orig.iloc[val_subset_index]

        subjects_train_subset = train_subset['participant_id'].to_numpy()
        subjects_val_subset = val_subset['participant_id'].to_numpy()

        makenewdir('{}/{}/{}/ROI_models'.format(working_dir, valdata, struct_var_metric))
        makenewdir('{}/{}/{}/covariate_files'.format(working_dir, valdata, struct_var_metric))
        makenewdir('{}/{}/{}/response_files'.format(working_dir, valdata, struct_var_metric))

        # separate the brain features (response variables) and predictors (age) in to separate dataframes
        train_subset_features = train_subset.loc[:, roi_ids]
        train_subset_covariates = train_subset[['age', 'agedays', 'sex']]
        val_subset_features = val_subset.loc[:, roi_ids]
        val_subset_covariates = val_subset[['age', 'agedays', 'sex']]

        # use subset of train set
        X_train_subset = train_subset_covariates.copy()
        y_train_subset = train_subset_features.copy()

        # create validation set
        X_val_subset = val_subset_covariates.copy()
        y_val_subset = val_subset_features.copy()

        # identify age range in pre-COVID data to be used for modeling
        agemin = X_train_subset['agedays'].min()
        agemax = X_train_subset['agedays'].max()

        if struct_var_metric == 'fa':
            write_ages_to_file(working_dir, agemin, agemax, struct_var_metric)

        # drop the age column from the train and val data set because we want to use agedays and sex as predictors
        X_train_subset.drop(columns=['age'], inplace=True)
        X_val_subset.drop(columns=['age'], inplace=True)

        ##########
        # Set up output directories. Save each brain region to its own text file, organized in separate directories,
        # because for each response variable Y (brain region) we fit a separate normative mode
        ##########

        # Check for nan values in y_train for each region. If nan value exists, remove before writing y_train for that
        # region to file. Also remove the corresponding covariate values for that subject.
        for c in y_train_subset.columns:

            y_train_nan_index = y_train_subset[y_train_subset[c].isna()].index.to_list()

            X_train_subset_copy = X_train_subset.copy()
            y_train_subset_copy = y_train_subset.copy()

            # If there are nan values for this region remove this subject from X_train and y_train for this region only
            if len(y_train_nan_index) == 0:
                X_train_subset_to_file = X_train_subset_copy
                y_train_subset_to_file_region = y_train_subset_copy.loc[:, c]
            else:
                X_train_subset_to_file = X_train_subset_copy.drop(labels=y_train_nan_index).reset_index(drop=True)
                y_train_subset_to_file_region = y_train_subset_copy.loc[:, c].drop(labels=y_train_nan_index).reset_index(drop=True)

            X_train_subset_to_file.to_csv(f'{working_dir}/cov_tr_' + c + '.txt', sep='\t', header=False, index=False)
            y_train_subset_to_file_region.to_csv(f'{working_dir}/resp_tr_' + c + '.txt', header=False, index=False)
            # y_train_subset.to_csv(f'{working_dir}/resp_tr.txt', sep='\t', header=False, index=False)

        for i in roi_ids:
            roidirname = '{}/{}/{}/ROI_models/{}'.format(working_dir, valdata, struct_var_metric, i)
            makenewdir(roidirname)
            cov_tr_filepath = roidirname + '/cov_tr.txt'
            shutil.copyfile("{}/cov_tr_{}.txt".format(working_dir, i), cov_tr_filepath)

            resp_tr_filename = "{}/resp_tr_{}.txt".format(working_dir, i)
            resp_tr_filepath = roidirname + '/resp_tr.txt'
            shutil.copyfile(resp_tr_filename, resp_tr_filepath)

        movefiles("{}/resp_*.txt".format(working_dir),
                  "{}/{}/{}/response_files/".format(working_dir, valdata, struct_var_metric))
        movefiles("{}/cov_t*.txt".format(working_dir),
                  "{}/{}/{}/covariate_files/".format(working_dir, valdata, struct_var_metric))

        #  this path is where ROI_models folders are located
        val_data_dir = '{}/{}/{}/ROI_models/'.format(working_dir, valdata, struct_var_metric)

        # Create Design Matrix and add in spline basis and intercept for validation and training data
        create_design_matrix('train', agemin, agemax, spline_order, spline_knots, roi_ids, val_data_dir)

        # create dataframe with subject numbers to put the Z scores in.
        subjects_train_subset = subjects_train_subset.reshape(-1, 1)
        Z_score_train_subset_matrix = pd.DataFrame(subjects_train_subset, columns=['subject_id_train'])

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
            print(f"{struct_var_metric} SPLIT NUMBER = {split + 1}/{n_splits}")
            print('Running ROI:', roi)
            print(f"Models created for {struct_var_metric}:  {roicounter + 1}/{len(roi_ids)} ROIs")
            roicounter += 1
            roi_dir = os.path.join(val_data_dir, roi)
            model_dir = os.path.join(val_data_dir, roi, 'Models')
            os.chdir(roi_dir)

            # configure the covariates to use. Change *_bspline_* to *_int_*
            cov_file_tr = os.path.join(roi_dir, 'cov_bspline_tr.txt')

            # load train response files
            resp_file_tr = os.path.join(roi_dir, 'resp_tr.txt')

            try:
                # calculate a model based on the training data and apply to the train dataset. The purpose of
                # running this function is to create and save the model, not to evaluate performance.
                yhat_tr, s2_tr, nm, Z_tr, metrics_tr = estimate(cov_file_tr, resp_file_tr, testresp=resp_file_tr,
                                                                testcov=cov_file_tr, alg='blr', optimizer='powell',
                                                                savemodel=True, saveoutput=False, standardize=False)
            except:
                yhat_tr = np.nan
                s2_tr = np.nan
                nm = np.nan
                Z_tr = np.nan
                metrics_tr = np.nan

            # # create dummy design matrices for visualizing model
            # dummy_cov_file_path_female, dummy_cov_file_path_male = \
            #     create_dummy_design_matrix(struct_var_metric, agemin, agemax, cov_file_tr, spline_order,
            #                                spline_knots,
            #                                working_dir)
            #
            # # Compute splines and superimpose on data. Show on screen or save to file depending on show_plots value.
            # plot_data_with_spline('Training Data', struct_var_metric, cov_file_tr, resp_file_tr,
            #                       dummy_cov_file_path_female,
            #                       dummy_cov_file_path_male, model_dir, roi, show_plots, working_dir, valdata)

        ######################## Apply Normative Model to Validation Data ############################

        makenewdir('{}/{}/{}/ROI_models'.format(working_dir, valpredict, struct_var_metric))
        makenewdir('{}/{}/{}/covariate_files'.format(working_dir, valpredict, struct_var_metric))
        makenewdir('{}/{}/{}/response_files'.format(working_dir, valpredict, struct_var_metric))

        # reset indices
        val_subset.reset_index(inplace=True, drop=True)
        # read agemin and agemax from file
        agemin, agemax = read_ages_from_file(working_dir, struct_var_metric)

        # specify which columns of dataframe to use as covariates
        X_val = val_subset[['agedays', 'sex']]

        # make a matrix of response variables, one for each brain region
        y_val = val_subset.loc[:, roi_ids]

        # specify paths
        training_dir = '{}/{}/{}/ROI_models/'.format(working_dir, valdata, struct_var_metric)
        out_dir = '{}/{}/{}/ROI_models/'.format(working_dir, valpredict, struct_var_metric)
        #  this path is where ROI_models folders are located
        predict_files_dir = '{}/{}/{}/ROI_models/'.format(working_dir, valpredict, struct_var_metric)

        ##########
        # Create output directories for each region and place covariate and response files for that region in  each directory.
        # Check for nan values in y_val for each region. If nan value exists, remove before writing y_val for that
        # region to file. Also remove the corresponding covariate values for that subject.
        ##########
        y_val_nan_index = {}

        total = len(roi_ids) * n_splits  # total number of models to be applied for this modality

        tcounter = 0  # Initialize counter

        for c in y_val.columns:

            y_val_nan_index[c] = y_val[y_val[c].isna()].index.to_list()

            X_val_copy = X_val.copy()
            y_val_copy = y_val.copy()

            # If there are nan values for this region remove the subject from X_train and y_train for this region only
            if len(y_val_nan_index[c]) == 0:
                X_val_to_file = X_val_copy
                y_val_to_file_region = y_val_copy.loc[:, c]
            else:
                X_val_to_file = X_val_copy.drop(labels=y_val_nan_index[c]).reset_index(drop=True)
                y_val_to_file_region = y_val_copy.loc[:, c].drop(labels=y_val_nan_index[c]).reset_index(
                    drop=True)

            X_val_to_file.to_csv(f'{working_dir}/cov_te_' + c + '.txt', sep='\t', header=False, index=False)
            y_val_to_file_region.to_csv(f'{working_dir}/resp_te_' + c + '.txt', header=False, index=False)
            # y_val.to_csv(f'{working_dir}/resp_te.txt', sep='\t', header=False, index=False)

        for i in roi_ids:
            roidirname = '{}/{}/{}/ROI_models/{}'.format(working_dir, valpredict, struct_var_metric, i)
            makenewdir(roidirname)
            resp_te_filename = "{}/resp_te_{}.txt".format(working_dir, i)
            resp_te_filepath = roidirname + '/resp_te.txt'
            shutil.copyfile(resp_te_filename, resp_te_filepath)
            cov_te_filepath = roidirname + '/cov_te.txt'
            shutil.copyfile("{}/cov_te_{}.txt".format(working_dir, i), cov_te_filepath)

        movefiles("{}/resp_*.txt".format(working_dir), "{}/{}/{}/response_files/"
                  .format(working_dir, valpredict, struct_var_metric))
        movefiles("{}/cov_t*.txt".format(working_dir), "{}/{}/{}/covariate_files/"
                  .format(working_dir, valpredict, struct_var_metric))

        # Create dataframe to store Zscores
        Z_time2 = pd.DataFrame()
        Z_time2['participant_id'] = val_subset['participant_id'].copy()
        Z_time2.reset_index(inplace=True, drop=True)

        ####Make Predictions of Brain Structural Measures Post-Covid based on Pre-Covid Normative Model

        # create design matrices for all regions and save files in respective directories
        create_design_matrix('test', agemin, agemax, spline_order, spline_knots, roi_ids, predict_files_dir)

        roicounter = 0

        for roi in roi_ids:
            print(f"SPLIT NUMBER = {split + 1}/{n_splits}")
            print('Running ROI:', roi)
            print(f"Models applied for {struct_var_metric}:  {roicounter + 1}/{len(roi_ids)}")
            print(f"Number of times applymodel has been run for this split = {tcounter + 1}")
            tcounter += 1  # Increment counter
            roicounter += 1

            roi_dir = os.path.join(predict_files_dir, roi)
            model_dir = os.path.join(training_dir, roi, 'Models')
            os.chdir(roi_dir)

            # configure the covariates to use.
            cov_file_te = os.path.join(roi_dir, 'cov_bspline_te.txt')

            # load val response files
            resp_file_te = os.path.join(roi_dir, 'resp_te.txt')

            try:
                # make predictions
                yhat_te, s2_te, Z = predict(cov_file_te, respfile=resp_file_te, alg='blr', model_path=model_dir)

            except:
                yhat_te = np.nan
                s2_te = np.nan
                Z = np.full((X_val.shape[0], 1), np.nan)

            valid_idx = ~y_val[roi].isna()

            y_true = y_val.loc[valid_idx, roi].reset_index(drop=True)

            results_df = pd.DataFrame({
                'y_true': y_true,
                'yhat_te': yhat_te.flatten(),
                'Z_score': Z.flatten()
            })

            # Save to file for this ROI and this split
            results_file = os.path.join(roi_dir, f'predictions_true_yhat_Z_{struct_var_metric}_{roi}_split{split}.csv')
            results_df.to_csv(results_file, index=False)
            print(f"Saved predictions and Z-scores to {results_file}")

            ind = 0
            if Z_time2.shape[0] == Z.shape[0]:
                Z_time2[roi] = Z
            else:
                for subj in range(Z_time2.shape[0]):
                    if subj in y_val_nan_index[roi]:
                        Z_time2.loc[subj, roi] = np.nan
                    else:
                        Z_time2.loc[subj, roi] = Z[ind]
                        ind += 1

            # create dummy design matrices
            # dummy_cov_file_path_female, dummy_cov_file_path_male = \
                # create_dummy_design_matrix(struct_var_metric, agemin, agemax, cov_file_te, spline_order,
                #                            spline_knots,
                #                            working_dir)

            # plot_data_with_spline('Postcovid (Test) Data ', struct_var_metric, cov_file_te, resp_file_te,
            #                       dummy_cov_file_path_female, dummy_cov_file_path_male, model_dir, roi,
            #                       show_plots, working_dir, valdata)


        Z_time2.to_csv('{}/{}/{}/Z_scores_by_region_postcovid_valset_{}_Final.txt'
                       .format(working_dir, valpredict, struct_var_metric,split), index=False)

        plt.show()

        print(f"finished SPLIT NUMBER = {split}/{n_splits}")






