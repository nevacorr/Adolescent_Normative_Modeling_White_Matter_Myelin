#####
# This program implements the bayesian linear regression normative modeling outlined by Rutherford et al.
# NatureProtocols 2022 (https://doi.org/10.1038/s41596-022-00696-5). Here the modeling is applied to
# adolescent MPF data collected at two time points (before and after the COVID lockdowns).
# This program creates models of MPF change in white matter tractsbetween 9 and 17 years of age for our pre-COVID data and
# stores these models to be applied in another script (Apply_Normative_Model_to_Genz_Time2.py).
# Author: Neva M. Corrigan
######

import pandas as pd
import os
from Utility_Functions import plot_age_acceleration
from make_time1_normative_model import make_time1_normative_model
from apply_normative_model_time2 import apply_normative_model_time2
from plot_and_compute_zdistributions import plot_and_compute_zcores_by_gender

struct_var = 'mpf'
show_plots = 0          #set to 1 to show training and test data ymvs yhat and spline fit plots.
show_nsubject_plots = 0 #set to 1 to plot number of subjects used in analysis, for each age and gender
spline_order = 1        # order of spline to use for model
spline_knots = 2        # number of knots in spline to use in model
perform_train_test_split_precovid = 0 #flag indicating whether to split the training set (pre-COVID data) into train and validation data
nbootstrap = 1000         #number of bootstrap to use in calculating confidence intervals for age acceleration separately by sex
data_dir = '/home/toddr/neva/PycharmProjects/data_dir'
visit1_datafile = 'tableGenzVisit1_allTracts_Oct22.csv'
visit2_datafile = 'tableGenzVisit2_allTracts_Oct22.csv'
subjects_to_exclude_time1 = [106, 107, 111, 121, 122, 126, 127, 208, 209, 210, 211, 214, 215, 221, 226, 309, 323, 335, 405, 418, 421, 423, 524]
subjects_to_exclude_time2 = [105, 117, 119, 201, 209, 215, 301, 306, 319, 321, 325, 406, 418, 421, 515, 527]
file_with_demographics = 'Adol_CortThick_data.csv'

run_make_norm_model = 0
run_apply_norm_model = 1
calc_brain_age_acc = 0
calc_CI_age_acc_bootstrap = 0

working_dir = os.getcwd()

if run_make_norm_model:

    Z_time1, roi_ids = make_time1_normative_model(struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                           data_dir, working_dir, visit1_datafile, subjects_to_exclude_time1, file_with_demographics)

    Z_time1.drop(columns=['subject_id_test'], inplace=True)

if run_apply_norm_model:

    Z_time2 = apply_normative_model_time2(struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                           data_dir, working_dir, visit1_datafile, visit2_datafile, subjects_to_exclude_time2, file_with_demographics)

# if calc_brain_age_acc:
#
#     calculate_avg_brain_age_acceleration_make_model(gender, orig_struct_var, show_nsubject_plots, show_plots,
#                                                                spline_order, spline_knots, orig_data_dir, working_dir)
#
#     mean_agediff[gender] = calculate_avg_brain_age_acceleration_apply_model(gender, orig_struct_var, show_nsubject_plots, show_plots,
#                                                            spline_order, spline_knots, orig_data_dir, working_dir, num_permute=0, permute=False, shuffnum=0)
#
# if calc_CI_age_acc_bootstrap:
#
#     ageacc_from_bootstraps[gender] = calculate_avg_brain_age_acceleration_apply_model_bootstrap(gender, orig_struct_var, show_nsubject_plots, show_plots,
#                                                        spline_order, spline_knots, orig_data_dir, working_dir, nbootstrap)
#     # Write age acceleration from bootstrapping to file
#     with open(f"{working_dir}/ageacceleration_dictionary {nbootstrap} bootstraps.txt", 'w') as f:
#         for key, value in ageacc_from_bootstraps.items():
#             f.write('%s:%s\n' % (key, value))

    plot_and_compute_zcores_by_gender(Z_time2, struct_var, roi_ids, working_dir)
