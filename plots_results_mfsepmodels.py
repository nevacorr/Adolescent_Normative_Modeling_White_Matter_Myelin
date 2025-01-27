
import pandas as pd
import glob
from plot_and_compute_zdistributions import plot_and_compute_zcores_by_gender
from plot_z_scores_MFseparate import plot_and_compute_zcores_by_gender_MFsep
import os
import matplotlib.pyplot as plt

def plot_results(working_dir, mfseparate, Z2_all_splits_fa_dict, Z2_all_splits_md_dict, Z2_all_splits_mpf_dict,
                 n_splits, roi_ids):
    # Remove Z score outputs from results from previous runs
    pattern = os.path.join(working_dir, 'Z_time2_*.csv')
    for file in glob.glob(pattern):
        try:
            os.remove(file)
        except:
            pass

    # Keep and process only the data for the sexes of interest
    if mfseparate == 0:
        sexes = ['all']
    else:
        sexes = ['females', 'males']

    if mfseparate == 0:

        Z_time2_fa = Z2_all_splits_fa_dict['all'].groupby(by=['participant_id']).mean().drop(columns=['split'])
        Z_time2_md = Z2_all_splits_md_dict['all'].groupby(by=['participant_id']).mean().drop(columns=['split'])
        Z_time2_mpf = Z2_all_splits_mpf_dict['all'].groupby(by=['participant_id']).mean().drop(columns=['split'])
        Z_time2_fa.reset_index(inplace=True)
        Z_time2_md.reset_index(inplace=True)
        Z_time2_mpf.reset_index(inplace=True)

        Z_time2_fa.to_csv(f'{working_dir}/Z_time2_fa_allsubjs_{n_splits}_splits.csv')
        Z_time2_md.to_csv(f'{working_dir}/Z_time2_md_allsubjs_{n_splits}_splits.csv')

        plot_and_compute_zcores_by_gender(Z_time2_fa, 'fa', roi_ids, working_dir, n_splits)

        roi_ids_md = roi_ids.copy()
        roi_ids_md = [s.replace('FA', 'MD') for s in roi_ids_md]
        plot_and_compute_zcores_by_gender(Z_time2_md, 'md', roi_ids_md, working_dir, n_splits)

        roi_ids_mpf = roi_ids.copy()
        roi_ids_mpf.remove('Left Uncinate FA')
        roi_ids_mpf.remove('Right Uncinate FA')
        roi_ids_mpf = [s.replace('FA', 'MPF') for s in roi_ids_mpf]
        plot_and_compute_zcores_by_gender(Z_time2_mpf, 'mpf', roi_ids_mpf, working_dir, n_splits)

        plt.show()

    else:

        Z_time2_fa_mandf = {}
        Z_time2_md_mandf = {}
        Z_time2_mpf_mandf = {}

        for sex in ['females', 'males']:

            Z_time2_fa_mandf[sex] = Z2_all_splits_fa_dict[sex].groupby(by=['participant_id']).mean().drop(columns=['split'])
            Z_time2_md_mandf[sex] = Z2_all_splits_md_dict[sex].groupby(by=['participant_id']).mean().drop(columns=['split'])
            Z_time2_mpf_mandf[sex] = Z2_all_splits_mpf_dict[sex].groupby(by=['participant_id']).mean().drop(columns=['split'])
            Z_time2_fa_mandf[sex].reset_index(inplace=True)
            Z_time2_md_mandf[sex].reset_index(inplace=True)
            Z_time2_mpf_mandf[sex].reset_index(inplace=True)

        Z_time2_fa_mandf[sex].to_csv(f'{working_dir}/Z_time2_fa_male_female_separate_models_{n_splits}_splits.csv')
        Z_time2_md_mandf[sex].to_csv(f'{working_dir}/Z_time2_md_{sex}_{n_splits}_splits.csv')
        Z_time2_mpf_mandf[sex].to_csv(f'{working_dir}/Z_time2_mpf_male_and_female_separate_models_{n_splits}_splits.csv')

        plot_and_compute_zcores_by_gender_MFsep('fa', Z_time2_fa_mandf, working_dir, roi_ids)

        roi_ids_md = roi_ids.copy()
        roi_ids_md = [s.replace('FA', 'MD') for s in roi_ids_md]
        plot_and_compute_zcores_by_gender_MFsep('md', Z_time2_md_mandf, working_dir, roi_ids_md)

        roi_ids_mpf = roi_ids.copy()
        roi_ids_mpf.remove('Left Uncinate FA')
        roi_ids_mpf.remove('Right Uncinate FA')
        roi_ids_mpf = [s.replace('FA', 'MPF') for s in roi_ids_mpf]
        plot_and_compute_zcores_by_gender_MFsep('mpf', Z_time2_mpf_mandf, working_dir, roi_ids_mpf)

        plt.show()

        mystop = 1

