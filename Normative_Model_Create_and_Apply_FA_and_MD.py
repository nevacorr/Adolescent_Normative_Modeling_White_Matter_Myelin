#####
# This program implements the bayesian linear regression normative modeling outlined by Rutherford et al.
# NatureProtocols 2022 (https://doi.org/10.1038/s41596-022-00696-5). Here the modeling is applied to
# adolescent MPF data collected at two time points (before and after the COVID lockdowns).
# This program creates models of FA and MD change in white matter tractsbetween 9 and 17 years of age for our pre-COVID data and
# stores these models to be applied in another script (Apply_Normative_Model_to_Genz_Time2.py).
# Author: Neva M. Corrigan
######

import os
from plot_and_compute_zdistributions import plot_and_compute_zcores_by_gender
from make_and_apply_normative_model_fa_md import make_and_apply_normative_model_fa_md

struct_var = 'fa_and_md'
n_splits = 1   #Number of train/test splits
show_plots = 0          #set to 1 to show training and test data ymvs yhat and spline fit plots.
show_nsubject_plots = 0 #set to 1 to plot number of subjects used in analysis, for each age and gender
spline_order = 1        # order of spline to use for model
spline_knots = 2        # number of knots in spline to use in model
perform_train_test_split_precovid = 0 #flag indicating whether to split the training set (pre-COVID data) into train and validation data
data_dir = '/home/toddr/neva/PycharmProjects/data_dir'

fa_visit1_datafile = 'genz_tract_profile_data/genzFA_tractProfiles_visit1.csv'
fa_visit2_datafile = 'genz_tract_profile_data/genzFA_tractProfiles_visit2.csv'
md_visit1_datafile = 'genz_tract_profile_data/genzMD_tractProfiles_visit1.csv'
md_visit2_datafile = 'genz_tract_profile_data/genzMD_tractProfiles_visit2.csv'
subjects_to_exclude_time1 = []
subjects_to_exclude_time2 = []

file_with_demographics = 'Adol_CortThick_data.csv'

run_make_norm_model = 1

working_dir = os.getcwd()

if run_make_norm_model:

    roi_ids, Z_time2_fa, Z_time2_md = make_and_apply_normative_model_fa_md(struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                           data_dir, working_dir, fa_visit1_datafile, fa_visit2_datafile, md_visit1_datafile, md_visit2_datafile,
                           subjects_to_exclude_time1, subjects_to_exclude_time2, file_with_demographics, n_splits)

    plot_and_compute_zcores_by_gender(Z_time2_fa, 'fa', roi_ids, working_dir)

    mystop=1

    roi_ids = [s.replace('FA', 'MD') for s in roi_ids]
    plot_and_compute_zcores_by_gender(Z_time2_md, 'md', roi_ids, working_dir)

    mystop = 1
