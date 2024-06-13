import pandas as pd

#load data
foot= pd.read_feather('data/processed/foot.feather')

#data preprocessing
foot['foot'] = pd.to_numeric(foot['foot'])
foot_nonzero = foot[foot.foot != 0]

#statistical analysis
foot_mean = foot_nonzero.groupby(['ID','date'])['foot'].mean().reset_index()
foot_var = foot_nonzero.groupby(['ID','date'])['foot'].var().reset_index()
foot_max = foot_nonzero.groupby(['ID','date'])['foot'].max().reset_index()

#calculation of foot_hvar_mean
foot_nonzero['hour'] = pd.to_datetime(foot_nonzero['time']).dt.hour 
foot_hvar = foot_nonzero.groupby(['ID','date','hour'])['foot'].var().reset_index()
foot_hvar= foot_hvar.dropna(axis=0)
foot_hvar_mean = foot_hvar.groupby(['ID','date'])['foot'].mean().reset_index()

#data merge
foot_statistics_merged= pd.merge(left=foot, right=foot_var, how="left", on =['date','ID'], suffixes=['', '_var'])
foot_statistics_merged= pd.merge(left=foot_statistics_merged, right=foot_max, how="left", on =['date','ID'], suffixes=['', '_max'])
foot_statistics_merged= pd.merge(left=foot_statistics_merged, right=foot_mean, how="left", on =['date','ID'], suffixes=['', '_mean'])
foot_statistics_merged= pd.merge(left=foot_statistics_merged, right=foot_hvar_mean, how="left", on =['date','ID'], suffixes=['', '_hvar_mean'])

#data preprocessing
foot_statistics_merged['datetime'] = foot_statistics_merged['date'] + ' ' + foot_statistics_merged['time']

#data save to feather
foot_statistics_merged.to_feather("data/processed/foot_stactistics.feather")

#data per date
foot_date= pd.merge(left=foot_var, right=foot_max, how="left", on =['date','ID'], suffixes=['', '_max'])
foot_date= pd.merge(left=foot_date, right=foot_mean, how="left", on =['date','ID'], suffixes=['', '_mean'])
foot_date= pd.merge(left=foot_date, right=foot_hvar_mean, how="left", on =['date','ID'], suffixes=['', '_hvar_mean'])

#data save to feather
foot_date.to_feather("data/processed/foot_date.feather")