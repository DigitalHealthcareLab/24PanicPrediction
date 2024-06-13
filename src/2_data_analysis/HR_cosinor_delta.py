import pandas as pd

circadian = pd.read_feather('data/processed/circadian_parameter.feather')
circadian['date'] = pd.to_datetime(circadian['date'])
id_list = circadian['ID'].unique()

circadian_delta = pd.DataFrame(columns=['ID', 'date', 'acr', 'amp', 'mesor','acr_delta', 'acr_delta2', 'amp_delta', 'amp_delta2', 'mesor_delta', 'mesor_delta2'])
for id in id_list:
    circadian_id = circadian.loc[(circadian.ID == id)]
    time_per_day = pd.date_range(circadian_id.date.min(), circadian_id.date.max(), freq='D')
    temp = pd.DataFrame()
    temp['date'] = time_per_day
    circadian_id = pd.merge(circadian_id, temp, how='right', on='date')
    circadian_id.ID = id
    circadian_id['acr_delta'] = circadian_id['acr'].diff()
    circadian_id['acr_delta2'] = circadian_id['acr'].diff(periods=2)
    circadian_id['amp_delta'] = circadian_id['amp'].diff()
    circadian_id['amp_delta2'] = circadian_id['amp'].diff(periods=2)
    circadian_id['mesor_delta'] = circadian_id['mesor'].diff()
    circadian_id['mesor_delta2'] = circadian_id['mesor'].diff(periods=2)
    circadian_delta = pd.concat([circadian_delta, circadian_id], axis=0)


circadian_delta = circadian_delta.dropna(subset= ['acr'], axis=0)
circadian_delta = circadian_delta.fillna(0)
circadian_delta['date'] = circadian_delta['date'].dt.strftime('%Y-%m-%d')
circadian_delta.reset_index(drop=True, inplace=True)

# %%
circadian_delta.to_feather("data/processed/circadian_delta.feather")