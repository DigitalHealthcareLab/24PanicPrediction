import pandas as pd
import numpy as np
from utils_for_preprocessing import load_raw_file, exclude_duplicated_id, exclude_15min_interval_data, exclude_miband_user_data, exclude_empty_data, serialize_lifelog_data

#load file
HR_raw = load_raw_file('라이프로그-심박수')

#data preprocessing
HR = exclude_duplicated_id(HR_raw)
HR = exclude_15min_interval_data(HR)
HR = exclude_miband_user_data(HR)
HR = exclude_empty_data(HR, reference_column='측정값(-1 : 값 없음)', keyword = ',')
exclude_total_HR_zero_data = HR[HR['평균 심박수']==0].index
HR.drop(exclude_total_HR_zero_data, axis=0, inplace=True)
HR.drop(['평균 심박수', '측정 단위'], axis=1, inplace=True)
HR.reset_index(drop=True, inplace=True)
HR_melted = serialize_lifelog_data(HR)
HR_melted.rename(columns = {'lifelog_data':'HR'}, inplace=True)
HR_melted.loc[HR_melted['HR'] == '-1', 'HR'] = '0'
HR_melted.fillna('0', inplace=True)
HR_melted['date'] = HR_melted['date'].str[:10] 

#data save to feather
HR_melted.to_feather("data/processed/HR.feather")