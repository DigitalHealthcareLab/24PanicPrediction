import pandas as pd
import numpy as np
from utils_for_preprocessing import load_raw_file, exclude_duplicated_id, exclude_miband_user_data

#load file
sleep_raw = load_raw_file(path= 'data/raw/Lifestyle - Sleep.xlsx', sheet_name='Sheet1')

#data preprocessing
sleep = exclude_duplicated_id(sleep_raw)
sleep = exclude_miband_user_data(sleep)
sleep.drop(['Measurement_types', 'Units_of_measure', 'Measurement_(_1:_no_value,_0:_unknown_sleep,_1:_wake,_2:_light_sleep,_3:_light_sleep,_4:_deep_sleep)'], axis=1, inplace=True)
sleep.reset_index(drop=True, inplace=True)
sleep['date'] = sleep['Date'].astype(str).str[:10] 
sleep['Date'] = pd.to_datetime(sleep['Date'], format = '%Y-%m-%d %H:%M:%S')
sleep['Bedtime'] = pd.to_datetime(sleep['Bedtime'], format = '%Y-%m-%d %H:%M:%S')
sleep['Wake_up_time'] = pd.to_datetime(sleep['Wake_up_time'], format = '%Y-%m-%d %H:%M:%S')
sleep['sleep_duration'] = sleep['Wake_up_time'] - sleep['Bedtime'] 
sleep['sleep_in'] = sleep['Date'] - sleep['Bedtime'] 
sleep['sleep_out'] = sleep['Date'] - sleep['Wake_up_time'] 
sleep.sleep_in = sleep.sleep_in.dt.total_seconds()/3600
sleep.sleep_out = sleep.sleep_out.dt.total_seconds()/3600
sleep.sleep_duration = sleep.sleep_duration.dt.total_seconds()/3600
sleep.drop(['Date','Bedtime','Wake_up_time'], axis=1, inplace=True)
sleep.columns = ['ID','date','sleep_duration', 'sleep_in', 'sleep_out']

#save to feather
sleep.to_feather("data/processed/sleep.feather")


