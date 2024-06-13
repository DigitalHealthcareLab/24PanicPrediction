import pandas as pd
import numpy as np
from utils_for_preprocessing import load_raw_file, exclude_duplicated_id, exclude_miband_user_data

#load file
sleep_raw = load_raw_file('라이프로그-수면')

#data preprocessing
sleep = exclude_duplicated_id(sleep_raw)
sleep = exclude_miband_user_data(sleep)
sleep.drop(['측정 유형', '측정 단위', '측정값(-1 : 값 없음, 0 : 알수 없는 수면, 1 : 깨어남, 2 : 램 수면, 3 : 얕은 수면, 4 : 깊은 수면)'], axis=1, inplace=True)
sleep.reset_index(drop=True, inplace=True)
sleep['date'] = sleep['날짜'].str[:10] 
sleep['날짜'] = pd.to_datetime(sleep['날짜'], format = '%Y-%m-%d %H:%M:%S')
sleep['취침시간'] = pd.to_datetime(sleep['취침시간'], format = '%Y-%m-%d %H:%M:%S')
sleep['기상시간'] = pd.to_datetime(sleep['기상시간'], format = '%Y-%m-%d %H:%M:%S')
sleep['sleep_duration'] = sleep['기상시간'] - sleep['취침시간'] 
sleep['sleep_in'] = sleep['날짜'] - sleep['취침시간'] 
sleep['sleep_out'] = sleep['날짜'] - sleep['기상시간'] 
sleep.sleep_in = sleep.sleep_in.dt.total_seconds()/3600
sleep.sleep_out = sleep.sleep_out.dt.total_seconds()/3600
sleep.sleep_duration = sleep.sleep_duration.dt.total_seconds()/3600
sleep.drop(['날짜','취침시간','기상시간'], axis=1, inplace=True)
sleep.columns = ['ID','date','sleep_duration', 'sleep_in', 'sleep_out']

#save to feather
sleep.to_feather("data/processed/sleep.feather")


