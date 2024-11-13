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
from make_and_apply_normative_model import make_and_apply_normative_model
from apply_normative_model_time2 import apply_normative_model_time2
from plot_and_compute_zdistributions import plot_and_compute_zcores_by_gender

struct_var = 'fa'
n_splits = 1   #Number of train/test splits
show_plots = 0          #set to 1 to show training and test data ymvs yhat and spline fit plots.
show_nsubject_plots = 0 #set to 1 to plot number of subjects used in analysis, for each age and gender
spline_order = 1        # order of spline to use for model
spline_knots = 2        # number of knots in spline to use in model
perform_train_test_split_precovid = 0 #flag indicating whether to split the training set (pre-COVID data) into train and validation data
data_dir = '/home/toddr/neva/PycharmProjects/data_dir'

if struct_var == 'md':
    visit1_datafile = 'genz_tract_profile_data/genzMD_tractProfiles_visit1.csv'
    visit2_datafile = 'genz_tract_profile_data/genzMD_tractProfiles_visit2.csv'
    subjects_to_exclude_time1 = []
    subjects_to_exclude_time2 = []
elif struct_var == 'fa':
    visit1_datafile = 'genz_tract_profile_data/genzFA_tractProfiles_visit1.csv'
    visit2_datafile = 'genz_tract_profile_data/genzFA_tractProfiles_visit2.csv'
    subjects_to_exclude_time1 = []
    subjects_to_exclude_time2 = []
elif struct_var == 'mpf':
    visit1_datafile = 'tableGenzVisit1_allTracts_Oct22.csv'
    visit2_datafile = 'tableGenzVisit2_allTracts_Oct22.csv'
    subjects_to_exclude_time1 = [106, 107, 111, 121, 122, 126, 127, 208, 209, 210, 211, 214, 215, 221, 226, 309, 323, 335, 405, 418, 421, 423, 524]
    subjects_to_exclude_time2 = [105, 117, 119, 201, 209, 215, 301, 306, 319, 321, 325, 406, 418, 421, 515, 527]

file_with_demographics = 'Adol_CortThick_data.csv'

run_make_norm_model = 1
calc_brain_age_acc = 0
calc_CI_age_acc_bootstrap = 0

working_dir = os.getcwd()

if run_make_norm_model:

    roi_ids, Z_time2 = make_and_apply_normative_model(struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                           data_dir, working_dir, visit1_datafile, visit2_datafile, subjects_to_exclude_time1,
                           subjects_to_exclude_time2, file_with_demographics, n_splits)

    plot_and_compute_zcores_by_gender(Z_time2, struct_var, roi_ids, working_dir)

    mystop = 1
