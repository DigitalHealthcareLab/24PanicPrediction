import pandas as pd
import numpy as np
from utils_for_preprocessing import load_raw_file, exclude_duplicated_id, exclude_15min_interval_data, exclude_miband_user_data, exclude_empty_data, serialize_lifelog_data


#load file
foot_raw = load_raw_file(path = 'data/raw/Lifestyle - Step count.xlsx', sheet_name='Sheet1')

#data preprocessing
foot = exclude_duplicated_id(foot_raw)
foot = exclude_15min_interval_data(foot)
foot = exclude_miband_user_data(foot)
foot = exclude_empty_data(foot, reference_column='Measure_(_1:_no_value)', keyword = ',')
exclude_total_foot_zero_data = foot[foot['Total_steps']==0].index
foot.drop(exclude_total_foot_zero_data, axis=0, inplace=True)
foot.drop(['Total_steps', 'Measurement_types'], axis=1, inplace=True)
foot.reset_index(drop=True, inplace=True)
foot_melted = serialize_lifelog_data(foot)
foot_melted.rename(columns = {'lifelog_data':'foot'}, inplace=True)
foot_melted.loc[foot_melted['foot'] == '-1', 'foot'] = '0'
foot_melted.fillna('0', inplace=True)
foot_melted['date'] = foot_melted['date'].str[:10] 

#data save to feather
foot_melted.to_feather("data/processed/foot.feather")


