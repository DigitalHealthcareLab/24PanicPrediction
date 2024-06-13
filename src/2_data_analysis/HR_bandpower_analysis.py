import numpy as np
import pandas as pd
import os
from utils_for_analysis import bandpower, check_bandpower_value_a, check_bandpower_value_b, check_bandpower_value_c, check_bandpower_value_d
from tqdm import tqdm
from joblib import Parallel, delayed 

#data load
HR= pd.read_feather('data/processed/HR_interpolated.feather')

#data preprocessing
HR['date'] = pd.to_datetime(HR['date'])
HR['HR'] = pd.to_numeric(HR['HR'])
id_list = HR.ID.unique()

#making time-list which fill the gap of time
df_per_min = pd.DataFrame(columns=['ID','HR','date'])
for id in id_list:
    df_id = HR.loc[(HR.ID == id)]
    time_per_min = pd.date_range(df_id.date.min(), df_id.date.max(), freq='min')
    temp = pd.DataFrame()
    temp['date'] = time_per_min
    df_id = pd.merge(df_id,temp, how='right', on='date')
    df_id.ID = id
    df_per_min = pd.concat([df_per_min,df_id],axis=0)

#making dataframe with bandpower value    
id_list = df_per_min['ID'].unique()

def process_date(id, temp_id):
    results = []
    date_list = temp_id['date'].unique()

    for date in date_list:
        temp_date = temp_id.loc[temp_id['date'] == date]
        temp_date.reset_index(inplace=True, drop=True)

        if len(temp_date) > 360:
            bandpower_a = check_bandpower_value_a(temp_date.index, temp_date['HR'])
            bandpower_b = check_bandpower_value_b(temp_date.index, temp_date['HR'])
            bandpower_c = check_bandpower_value_c(temp_date.index, temp_date['HR'])
            bandpower_d = check_bandpower_value_d(temp_date.index, temp_date['HR'])
            results.append([id, date, bandpower_a, bandpower_b, bandpower_c, bandpower_d])

    return results

def function_test():
    results = Parallel(n_jobs=-1)(delayed(process_date)(id, df_per_min.loc[df_per_min['ID'] == id]) for id in tqdm(id_list))
    flat_results = [item for sublist in results for item in sublist]
    bandpower_df = pd.DataFrame(flat_results, columns=['ID', 'date', 'bandpower_a', 'bandpower_b', 'bandpower_c', 'bandpower_d'])
    return bandpower_df

bandpower_df = function_test()

#save data to feather
bandpower_df.to_feather("data/processed/bandpower.feather")