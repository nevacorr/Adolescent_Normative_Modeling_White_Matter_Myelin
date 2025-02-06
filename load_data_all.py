

from Load_Genz_Data_WWM_FA_MD import load_genz_data_wm_fa_md, load_genz_data_wm_mpf_v1

def load_data_all(raw_data_dir, fa_datafilename_v1, fa_datafilename_v2, md_datafilename_v1,
              md_datafilename_v2, mpf_datafilename_v1, mpf_datafilename_v2, demographics_filename,
              subjects_to_exclude_v1, subjects_to_exclude_v2, mpf_subjects_to_exclude_v1, mpf_subjects_to_exclude_v2):

# Load all data
    (fa_all_data_both_visits, md_all_data_both_visits, mpf_all_data_both_visits,  roi_ids, all_subjects,
    all_subjects_mpf, sub_v1_only, sub_v2_only, sub_v1_only_mpf, sub_v2_only_mpf) = (
        load_genz_data_wm_fa_md(raw_data_dir, fa_datafilename_v1, fa_datafilename_v2, md_datafilename_v1,
        md_datafilename_v2, mpf_datafilename_v1, mpf_datafilename_v2, demographics_filename))

    def process_dataframe(data, visit, subjects_to_exclude):
        df = data.copy()
        df = df[df['visit'] == visit]
        df.reset_index(inplace=True, drop=True)
        df.drop(columns=['visit', 'Age'], inplace=True)
        df = df[~df['participant_id'].isin(subjects_to_exclude)]
        df.loc[df['sex'] == 2, 'sex'] = 0
        return df

    fa_all_data_v1 = process_dataframe(fa_all_data_both_visits, 1, subjects_to_exclude_v1)
    fa_all_data_v2 = process_dataframe(fa_all_data_both_visits, 2, subjects_to_exclude_v2)
    md_all_data_v1 = process_dataframe(md_all_data_both_visits, 1, subjects_to_exclude_v1)
    md_all_data_v2 = process_dataframe(md_all_data_both_visits, 2, subjects_to_exclude_v2)
    mpf_all_data_v1 = process_dataframe(mpf_all_data_both_visits, 1, mpf_subjects_to_exclude_v1)
    mpf_all_data_v2 = process_dataframe(mpf_all_data_both_visits, 2, mpf_subjects_to_exclude_v2)

    return (fa_all_data_v1, fa_all_data_v2, md_all_data_v1, md_all_data_v2, mpf_all_data_v1,
            mpf_all_data_v2, sub_v1_only, sub_v2_only, sub_v1_only_mpf, sub_v2_only_mpf, roi_ids)
