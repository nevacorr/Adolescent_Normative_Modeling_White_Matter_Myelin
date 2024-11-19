import pandas as pd
import os

data_dir = '/home/toddr/neva/PycharmProjects/data_dir'
fa_visit1_datafile = 'genz_tract_profile_data/genzFA_tractProfiles_visit1.csv'
fa_visit2_datafile = 'genz_tract_profile_data/genzFA_tractProfiles_visit2.csv'

tracts_to_plot = ['Right ILF', 'Right IFOF']
metric = 'FA'

fa_visit1 = pd.read_csv(os.path.join(data_dir, fa_visit1_datafile))
fa_visit2 = pd.read_csv(os.path.join(data_dir, fa_visit2_datafile))

fa_visit1['Subject'] = fa_visit1['Subject'].str.replace('sub-genz', '').astype(int)
fa_visit2['Subject'] = fa_visit2['Subject'].str.replace('sub-genz', '').astype(int)



def add_sex(df):
    df['Sex'] = df.apply(lambda row: 0 if row['Subject'] % 2 == 0 else 1, axis=1)
    sex = df.pop('Sex')
    df.insert(1, 'Sex', sex)
    return df

fa_visit1 = add_sex(fa_visit1)
fa_visit2 = add_sex(fa_visit2)

# plot male vs female trajectories for tract metric for tracts of interest



mystop=1
