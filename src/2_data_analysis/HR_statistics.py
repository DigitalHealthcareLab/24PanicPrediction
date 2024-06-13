import pandas as pd

#load data
HR= pd.read_feather('data/processed/HR.feather')

#data preprocessing
HR['HR'] = pd.to_numeric(HR['HR'])
HR_nonzero = HR[HR.HR != 0]

#statistical analysis
HR_mean = HR_nonzero.groupby(['ID','date'])['HR'].mean().reset_index()
HR_var = HR_nonzero.groupby(['ID','date'])['HR'].var().reset_index()
HR_min = HR_nonzero.groupby(['ID','date'])['HR'].min().reset_index()
HR_max = HR_nonzero.groupby(['ID','date'])['HR'].max().reset_index()

#calculation of HR_hvar_mean
HR['hour'] = pd.to_datetime(HR['time']).dt.hour 
HR_hvar = HR.groupby(['ID','date','hour'])['HR'].var().reset_index()
HR_hvar_mean = HR_hvar.groupby(['ID','date'])['HR'].mean().reset_index()

#data merge
HR_statistics_merged = pd.merge(left=HR, right=HR_var, how="left", on =['date','ID'], suffixes=['', '_var'])
HR_statistics_merged = pd.merge(left=HR_statistics_merged, right=HR_min, how="left", on =['date','ID'], suffixes=['', '_min'])
HR_statistics_merged = pd.merge(left=HR_statistics_merged, right=HR_max, how="left", on =['date','ID'], suffixes=['', '_max'])
HR_statistics_merged = pd.merge(left=HR_statistics_merged, right=HR_mean, how="left", on =['date','ID'], suffixes=['', '_mean'])
HR_statistics_merged = pd.merge(left=HR_statistics_merged, right=HR_hvar_mean, how="left", on =['date','ID'], suffixes=['', '_hvar_mean'])

#data preprocessing
HR_statistics_merged['datetime'] = HR_statistics_merged['date'] + ' ' + HR_statistics_merged['time']
HR_statistics_merged.drop('hour', axis=1, inplace=True)

#data save to feather
HR_statistics_merged.to_feather("data/processed/hr_stactistics.feather")

#data per date
HR_date = pd.merge(left=HR_var, right=HR_max, how="left", on =['date','ID'], suffixes=['', '_max'])
HR_date = pd.merge(left=HR_date, right=HR_mean, how="left", on =['date','ID'], suffixes=['', '_mean'])
HR_date = pd.merge(left=HR_date, right=HR_hvar_mean, how="left", on =['date','ID'], suffixes=['', '_hvar_mean'])
HR_date.rename(columns = {'HR':'HR_var'}, inplace=True)

#data save to feather
HR_date.to_feather("data/processed/HR_date.feather")