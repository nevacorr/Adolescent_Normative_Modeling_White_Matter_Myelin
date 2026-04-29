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

dses_t1 = pd.read_csv(os.path.join("/home/toddr/neva/PycharmProjects/AdolNormativeModelingCOVID", 'DemographicsAndEFScoresForTables_visit1.csv'))
dses_t2 = pd.read_csv(os.path.join("/home/toddr/neva/PycharmProjects/AdolNormativeModelingCOVID", 'DemographicsAndEFScoresForTables_visit2.csv'))

dses_t1 = dses_t1[['subject', 'SES']]
dses_t2 = dses_t2[['subject', 'SES']]

demo_data = demographics[['subject', 'visit', 'gender', 'agemonths', 'agedays', 'agegroup']]
v1_demo_data = demo_data[demo_data['visit']==1]
v2_demo_data = demo_data[demo_data['visit']==2]

visit1_subjects_fname = f'{working_dir}/visit1_all_subjects_used_in_analysis.csv'
visit1_subjects = pd.read_csv(visit1_subjects_fname)
visit1_subjects.rename(columns={'participant_id': 'subject'}, inplace=True)

visit2_subjects_fname = f'{working_dir}/visit2_all_subjects_used_in_analysis.csv'
visit2_subjects = pd.read_csv(visit2_subjects_fname)
visit2_subjects.rename(columns={'participant_id': 'subject'}, inplace=True)

visit1_all = visit1_subjects.merge(v1_demo_data, on='subject', how='left').merge(dses_t1, on='subject', how='left')
visit2_all = visit2_subjects.merge(v2_demo_data, on='subject', how='left').merge(dses_t2, on='subject', how='left')

def summarize_visit(df):
    # Mean and std age and SES by gender and agegroup
    agegroup_gender_summary = (
        df.groupby(['gender', 'agegroup'])[['agedays', 'SES']]
        .agg(
            N=('agedays', 'count'),
            mean_days=('agedays', 'mean'),
            std_days=('agedays', 'std'),
            mean_SES=('SES', 'mean'),
            std_SES=('SES', 'std')
        )
        .assign(
            mean_years=lambda x: x['mean_days'] / conv_days_to_years,
            std_years=lambda x: x['std_days'] / conv_days_to_years
        )
        .reset_index()
    )
    return agegroup_gender_summary

def print_summaries(agegroup_gender_summary, visit_name="Visit"):

    print(f"\n--- {visit_name} Age Summary by Gender and Age Group ---")
    # Format floats to 2 decimals for readability
    formatted = agegroup_gender_summary.copy()
    formatted['mean_days'] = formatted['mean_days'].round(2)
    formatted['std_days'] = formatted['std_days'].round(2)
    formatted['mean_years'] = formatted['mean_years'].round(2)
    formatted['std_years'] = formatted['std_years'].round(2)

    formatted['mean_SES'] = formatted['mean_SES'].round(1)
    formatted['std_SES'] = formatted['std_SES'].round(1)

    print(formatted.to_string(index=False))

v1_agegroup_gender_summary = summarize_visit(visit1_all)
v2_agegroup_gender_summary = summarize_visit(visit2_all)

print_summaries(v1_agegroup_gender_summary, visit_name="Visit 1")
print_summaries(v2_agegroup_gender_summary, visit_name="Visit 2")

