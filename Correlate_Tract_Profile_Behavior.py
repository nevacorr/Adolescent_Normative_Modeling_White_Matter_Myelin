
from load_genz_tract_profile_data_no_avg import load_genz_tract_profile_data_noavg
import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt

struct_var = 'FA'
visit = 2
data_dir = '/home/toddr/neva/PycharmProjects/data_dir'

# load cognitive behavior data for both visits
cogdat_v1 = pd.read_csv(
    '/home/toddr/neva/PycharmProjects/Adolescent_Brain_Behavior_Longitudinal_Analysis/'
    'T1_GenZ_Cognitive_Data_for_corr_with_behav.csv')
cogdat_v1.rename({'FlankerStandardUncorrected':
    'FlankerSU', 'DCStandardUncorrected': 'DCSU',
                  'VocabStandardUncorrected': 'VocabSU', 'WMemoryStandardUncorrected':
                      'WMemorySU'}, axis=1, inplace=True)
cogdat_v1['visit'] = 1

cogdat_v2 = pd.read_csv(
    '/home/toddr/neva/PycharmProjects/Adolescent_Brain_Behavior_Longitudinal_Analysis/'
    'T2_GenZ_Cognitive_Data_for_corr_with_behav.csv')
cogdat_v2.rename({'FlankerStandardUncorrected':
    'FlankerSU', 'DCStandardUncorrected': 'DCSU',
                  'VocabStandardUncorrected': 'VocabSU', 'WMemoryStandardUncorrected':
                      'WMemorySU'}, axis=1, inplace=True)
cogdat_v2['visit'] = 2
# remove raw and age corrected scores for cognitive data
remove_cols = [x for x in cogdat_v2.columns if ('Raw' in x) or ('AgeCorrected' in x)]
cogdat_v2.drop(columns=remove_cols, inplace=True)
remove_cols = [x for x in cogdat_v1.columns if ('Raw' in x) or ('AgeCorrected' in x)]
cogdat_v1.drop(columns=remove_cols, inplace=True)

behav_zs = pd.read_csv('/home/toddr/neva/PycharmProjects/AdolNormativeModelingCOVID/'
                       'Z_scores_all_meltzoff_cogn_behav_visit2.csv', usecols=lambda column: column != 'Unnamed: 0')

for visit in [1, 2]:

    fa_datafile = f'genz_tract_profile_data/genzFA_tractProfiles_visit{visit}.csv'
    md_datafile = f'genz_tract_profile_data/genzMD_tractProfiles_visit{visit}.csv'

    data_fa = load_genz_tract_profile_data_noavg(visit, data_dir, fa_datafile)
    data_md = load_genz_tract_profile_data_noavg(visit, data_dir, md_datafile)

    fa_all_data= cogdat_v2.merge(data_fa, on='Subject')
    fa_all_data.drop(columns=['Subject', 'Age', 'AgeGrp', 'visit', 'Age', 'Visit'], inplace=True)
    md_all_data = cogdat_v2.merge(data_md, on='Subject')
    md_all_data.drop(columns=['Subject', 'Age', 'AgeGrp', 'visit', 'Age', 'Visit'], inplace=True)

    fa_all_col_corr = fa_all_data.corr()
    fa_all_col_corr.drop(index=['FlankerSU', 'DCSU', 'VocabSU', 'WMemorySU'], inplace=True)

    md_all_col_corr = md_all_data.corr()
    md_all_col_corr.drop(index=['FlankerSU', 'DCSU', 'VocabSU', 'WMemorySU'], inplace=True)

    fa_finalcorr_df = pd.DataFrame()
    md_finaLcorr_df = pd.DataFrame()

    columns_to_copy = ['FlankerSU', 'DCSU', 'VocabSU', 'WMemorySU']
    columns_to_copy = ['FlankerSU', 'DCSU', 'VocabSU', 'WMemorySU']
    fa_finalcorr_df[columns_to_copy] = fa_all_col_corr[columns_to_copy]
    md_finaLcorr_df[columns_to_copy] = md_all_col_corr[columns_to_copy]

    thresh = 0.30

    fa_finalcorr_df_thresh = fa_finalcorr_df[(fa_finalcorr_df.abs() > thresh).any(axis=1)]
    md_finalcorr_df_thresh = md_finaLcorr_df[(md_finaLcorr_df.abs() > thresh).any(axis=1)]

    fa_finalcorr_df_thresh[fa_finalcorr_df_thresh.abs() <= thresh]  = 0
    md_finalcorr_df_thresh[md_finalcorr_df_thresh.abs() <= thresh]  = 0

    sns.heatmap(fa_finalcorr_df_thresh, cmap='bwr', vmin=-0.5, vmax=0.5)
    plt.title(f'Time {visit} All Subjects FA abs(corr) > {thresh}')
    plt.show(block=False)

    plt.figure()
    sns.heatmap(md_finalcorr_df_thresh, cmap='bwr', vmin=-0.5, vmax=0.5)
    plt.title(f'Time {visit} All Subjects MD abs(corr) > {thresh}')
    plt.show()

mystop=1