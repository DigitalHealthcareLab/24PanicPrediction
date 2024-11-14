import pandas as pd
import numpy as np
import datetime
import os
from utils_for_preprocessing import load_raw_file, exclude_duplicated_id

panic = load_raw_file('data/raw/Panic diary.xlsx', sheet_name='Sheet1')

#data preprocessing
panic = exclude_duplicated_id(panic)
panic.drop(['End_time', 'Duration', 'Intensity', 'Comorbidities'], axis=1, inplace=True)
panic['Start_time'] = panic['Start_time'].astype(str) + ':00'
panic.columns = ['ID','date','time']
panic['datetime'] = panic['date'] + ' ' + panic['time']
panic['panic'] = 1 
panic.reset_index(drop=True, inplace=True)

panic['date'] = pd.to_datetime(panic['date'], format = '%Y-%m-%d')
panic.drop_duplicates(['ID', 'date'], keep='first', inplace=True, ignore_index=False)
panic.drop(['datetime', 'time', 'panic'], axis=1, inplace=True)
panic_tomorrow = panic.copy()
panic_tomorrow.date = panic_tomorrow.date - datetime.timedelta(1)
panic_tomorrow['panic'] = 1
panic['panic'] = 2
panic_multi = pd.concat([panic_tomorrow, panic], axis=0)
panic_multi.sort_values(by=['panic'], ascending=True, inplace=True)
panic_multi.reset_index(drop=True, inplace=True)
panic_multi.drop_duplicates(['ID', 'date'], keep='first', inplace=True, ignore_index=False)
panic_multi.reset_index(drop=True, inplace=True)
panic_multi['date'] = panic_multi['date'].dt.strftime('%Y-%m-%d')
panic_multi.to_feather("data/processed/panic.feather")