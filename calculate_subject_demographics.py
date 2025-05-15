import os
import pandas as pd
import numpy as np

struct_var = 'fa_and_md_and_mpf'
working_dir = os.getcwd()
n_splits = 100
data_dir = '/home/toddr/neva/PycharmProjects/data_dir'
conv_days_to_years = 365.25

file_with_demographics = 'Adol_CortThick_data.csv'
demographics = pd.read_csv(os.path.join(data_dir, file_with_demographics))
demo_data = demographics[['subject', 'visit', 'gender', 'agemonths', 'agedays', 'agegroup']]
v1_demo_data = demo_data[demo_data['visit']==1]
v2_demo_data = demo_data[demo_data['visit']==2]

visit1_subjects_fname = f'{working_dir}/visit1_all_subjects_used_in_analysis.csv'
visit1_subjects = pd.read_csv(visit1_subjects_fname)
visit1_subjects.rename(columns={'participant_id': 'subject'}, inplace=True)

visit2_subjects_fname = f'{working_dir}/visit2_all_subjects_used_in_analysis.csv'
visit2_subjects = pd.read_csv(visit2_subjects_fname)
visit2_subjects.rename(columns={'participant_id': 'subject'}, inplace=True)

visit1_all = visit1_subjects.merge(v1_demo_data, on='subject', how='left')
visit2_all = visit2_subjects.merge(v2_demo_data, on='subject', how='left')

def summarize_subjs(df):
    summary = (
        df.groupby('gender')['agedays']
                  .agg(N='count', mean_days='mean', std_days='std')
                  .assign(
                    mean_years = lambda x: x['mean_days'] / conv_days_to_years,
                    std_years = lambda x: x['std_days'] / conv_days_to_years
                  )
                  .reset_index()
    )
    agegroup_counts = (
        df.groupby(['gender', 'agegroup'])
        .size()
        .unstack(fill_value=0))

    return summary, agegroup_counts

v1_summary, v1_agegroup_counts = summarize_subjs(visit1_all)
v2_summary, v2_agegroup_counts = summarize_subjs(visit2_all)

print('visit 1')
print(v1_summary)
print('\n')
print(v1_agegroup_counts)
print('\n')
print('visit 2')
print(v2_summary)
print('\n')
print(v2_agegroup_counts)

mystop=1