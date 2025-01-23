#####
# This program implements the bayesian linear regression normative modeling outlined by Rutherford et al.
# NatureProtocols 2022 (https://doi.org/10.1038/s41596-022-00696-5). Here the modeling is applied to
# adolescent DTI and MPF data collected at two time points (before and after the COVID lockdowns).
# This program creates models of FA and MD change in white matter tractsbetween 9 and 17 years of age for our pre-COVID data and
# stores these models to be applied in another script (Apply_Normative_Model_to_Genz_Time2.py).
# Author: Neva M. Corrigan
######

import os
import matplotlib.pyplot as plt
from plot_and_compute_zdistributions import plot_and_compute_zcores_by_gender
from make_and_apply_normative_model_fa_md import make_and_apply_normative_model_fa_md
from compute_df_correlations import compute_df_correlations

struct_var = 'fa_and_md_and_mpf'
n_splits = 100   #Number of train/test splits
show_plots = 0          #set to 1 to show training and test data ymvs yhat and spline fit plots.
show_nsubject_plots = 0 #set to 1 to plot number of subjects used in analysis, for each age and gender
spline_order = 1        # order of spline to use for model
spline_knots = 2        # number of knots in spline to use in model
perform_train_test_split_precovid = 0 #flag indicating whether to split the training set (pre-COVID data) into train and validation data
data_dir = '/home/toddr/neva/PycharmProjects/data_dir'
mf_separate = 1  # indicate whether to create separate models for males and females

fa_visit1_datafile = 'genz_tract_profile_data/genzFA_tractProfiles_visit1.csv'
fa_visit2_datafile = 'genz_tract_profile_data/genzFA_tractProfiles_visit2.csv'
md_visit1_datafile = 'genz_tract_profile_data/genzMD_tractProfiles_visit1.csv'
md_visit2_datafile = 'genz_tract_profile_data/genzMD_tractProfiles_visit2.csv'
mpf_visit1_datafile = 'tableGenzVisit1_allTracts_Oct22.csv'
mpf_visit2_datafile = 'tableGenzVisit2_allTracts_Oct22.csv'
subjects_to_exclude_time1 = [525]  # subjects to exclude for FA and MD
subjects_to_exclude_time2 = [525]  # subjects to exclude for FA and MD
mpf_subjects_to_exclude_time1 = []#[106, 107, 111, 121, 122, 126, 127, 208, 209, 210, 211, 214, 215, 221, 226, 309, 323, 335,405, 418, 421, 423, 524]
mpf_subjects_to_exclude_time2 = [] #[105, 117, 119, 201, 209, 215, 301, 306, 319, 321, 325, 406, 418, 421, 515, 527]

file_with_demographics = 'Adol_CortThick_data.csv'

run_make_norm_model = 1

working_dir = os.getcwd()

if run_make_norm_model:

    roi_ids, Z_time2_fa, Z_time2_md, Z_time2_mpf = make_and_apply_normative_model_fa_md(mf_separate, struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                           data_dir, working_dir, fa_visit1_datafile, fa_visit2_datafile, md_visit1_datafile, md_visit2_datafile, mpf_visit1_datafile,
                           mpf_visit2_datafile, subjects_to_exclude_time1, subjects_to_exclude_time2, mpf_subjects_to_exclude_time1,
                           mpf_subjects_to_exclude_time2, file_with_demographics, n_splits)

    compute_df_correlations(Z_time2_fa, Z_time2_md)

    plt.show(block=False)

    tmp = Z_time2_fa.groupby(by=['participant_id'])
    Z_time2_fa = Z_time2_fa.groupby(by=['participant_id']).mean().drop(columns=['split'])
    Z_time2_md = Z_time2_md.groupby(by=['participant_id']).mean().drop(columns=['split'])
    Z_time2_mpf = Z_time2_mpf.groupby(by=['participant_id']).mean().drop(columns=['split'])
    Z_time2_fa.reset_index(inplace=True)
    Z_time2_md.reset_index(inplace=True)
    Z_time2_mpf.reset_index(inplace=True)

    plot_and_compute_zcores_by_gender(Z_time2_fa, 'fa', roi_ids, working_dir, n_splits)
    Z_time2_fa.to_csv(f'{working_dir}/Z_time2_fa_{n_splits}_splits.csv')

    roi_ids_md = roi_ids.copy()
    roi_ids_md = [s.replace('FA', 'MD') for s in roi_ids_md]
    plot_and_compute_zcores_by_gender(Z_time2_md, 'md', roi_ids_md, working_dir, n_splits)
    Z_time2_md.to_csv(f'{working_dir}/Z_time2_md_{n_splits}_splits.csv')

    roi_ids_mpf = roi_ids.copy()
    roi_ids_mpf.remove('Left Uncinate FA')
    roi_ids_mpf.remove('Right Uncinate FA')
    roi_ids_mpf = [s.replace('FA', 'MPF') for s in roi_ids_mpf]
    plot_and_compute_zcores_by_gender(Z_time2_mpf, 'mpf', roi_ids_mpf, working_dir, n_splits)
    Z_time2_mpf.to_csv(f'{working_dir}/Z_time2_mpf_{n_splits}_splits.csv')

    plt.show()

    mystop=1