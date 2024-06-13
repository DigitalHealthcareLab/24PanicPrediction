import pandas as pd

foot = pd.read_feather('data/processed/foot_date.feather')
foot['date'] = pd.to_datetime(foot['date'])
id_list = foot['ID'].unique()

foot_delta = pd.DataFrame(columns=['ID', 'date', 'foot', 'foot_max', 'foot_mean', 'foot_hvar_mean', 'foot_delta', 'foot_max_delta',
                                   'foot_mean_delta', 'foot_hvar_mean_delta', 'foot_delta2', 'foot_max_delta2',
                                   'foot_mean_delta2', 'foot_hvar_mean_delta2'])
for id in id_list:
    foot_id = foot.loc[(foot.ID == id)]
    time_per_day = pd.date_range(foot_id.date.min(), foot_id.date.max(), freq='D')
    temp = pd.DataFrame()
    temp['date'] = time_per_day
    foot_id = pd.merge(foot_id, temp, how='right', on='date')
    foot_id.ID = id
    foot_id['foot_delta'] = foot_id['foot'].diff()
    foot_id['foot_delta2'] = foot_id['foot'].diff(periods=2)
    foot_id['foot_max_delta'] = foot_id['foot_max'].diff()
    foot_id['foot_max_delta2'] = foot_id['foot_max'].diff(periods=2)
    foot_id['foot_mean_delta'] = foot_id['foot_mean'].diff()
    foot_id['foot_mean_delta2'] = foot_id['foot_mean'].diff(periods=2)
    foot_id['foot_hvar_mean_delta'] = foot_id['foot_hvar_mean'].diff()
    foot_id['foot_hvar_mean_delta2'] = foot_id['foot_hvar_mean'].diff(periods=2)
    foot_delta = pd.concat([foot_delta, foot_id], axis=0)


foot_delta = foot_delta.dropna(subset= ['foot'], axis=0)
foot_delta = foot_delta.fillna(0)
foot_delta['date'] = foot_delta['date'].dt.strftime('%Y-%m-%d')
foot_delta.reset_index(drop=True, inplace=True)

# %%
foot_delta.to_feather("data/processed/foot_delta.feather")