import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

data_dir = '/home/toddr/neva/PycharmProjects/data_dir'
visit1_datafile = 'genz_tract_profile_data/genzFA_tractProfiles_visit1.csv'
visit2_datafile = 'genz_tract_profile_data/genzFA_tractProfiles_visit2.csv'

variables_to_keep = ['Subject', 'Right ILF', 'Right IFOF']
metric = 'FA'

visit1 = pd.read_csv(os.path.join(data_dir, visit1_datafile))
visit2 = pd.read_csv(os.path.join(data_dir, visit2_datafile))

visit1['Subject'] = visit1['Subject'].str.replace('sub-genz', '').astype(int)
visit2['Subject'] = visit2['Subject'].str.replace('sub-genz', '').astype(int)

tracts_to_remove = [col for col in visit1.columns if not any(sub in col for sub in variables_to_keep)]
visit1.drop(columns=tracts_to_remove, inplace=True)
visit2.drop(columns=tracts_to_remove, inplace=True)

def add_sex(df):
    df['Sex'] = df.apply(lambda row: 0 if row['Subject'] % 2 == 0 else 1, axis=1)
    sex = df.pop('Sex')
    df.insert(1, 'Sex', sex)
    return df

# Make a separate dataframe for each tract to plot
variables_to_keep.remove('Subject')
tracts_to_plot = variables_to_keep

def plot_tracts_by_sex(visit, tracts_to_plot, visit_num):
    # Make a dictionary of dataframes
    df_dict = {}
    for tract in tracts_to_plot:
        cols_tract = [col for col in visit.columns if tract in col]
        tract_df = pd.DataFrame()
        tract_df['Subject'] = visit['Subject'].copy()
        tract_df = add_sex(tract_df)
        tract_df[cols_tract] = visit[cols_tract].copy()
        df_dict[tract] = tract_df

    for tract in tracts_to_plot:
        cols_tract = [col for col in df_dict[tract] if tract in col]
        plt.figure()
        x = range(21, len(cols_tract)+1-22)

        # Separate data by sex
        data_sex_f = df_dict[tract].loc[df_dict[tract]['Sex'] == 0, cols_tract].T
        data_sex_m = df_dict[tract].loc[df_dict[tract]['Sex'] == 1, cols_tract].T
        data_sex_f_remove_ends = data_sex_f.iloc[21:79,:]
        data_sex_m_remove_ends = data_sex_m.iloc[21:79, :]

        mean_data_sex_f = data_sex_f_remove_ends.mean(axis=1)
        mean_data_sex_m = data_sex_m_remove_ends.mean(axis=1)

        # Plot all rows for both sexes
        plt.plot(x, data_sex_f_remove_ends, color="red", alpha=0.3)
        plt.plot(x, data_sex_m_remove_ends, color="blue", alpha=0.3)
        # Plot mean waveforms
        plt.plot(x, mean_data_sex_f, color='red', linewidth=5, label='Female')
        plt.plot(x, mean_data_sex_m, color='blue', linewidth=5, label='Male')
        # Add labels and legend
        plt.xlabel('Node')
        plt.ylabel(f'{metric} Value')
        plt.title(f'Visit {visit_num} {tract} {metric}')
        plt.legend()
        plt.grid(alpha=0.3)
        # Show the plot
        plt.show(block=False)

plot_tracts_by_sex(visit1, tracts_to_plot, 1)
plot_tracts_by_sex(visit2, tracts_to_plot, 2)
plt.show()

mystop=1

