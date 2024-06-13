import pandas as pd
from functools import reduce

#data per ID
questionnaire_byID = pd.read_feather('data/processed/questionnaire_byID.feather')
demographic_data = pd.read_feather('data/processed/demographic_data.feather')
df_ID = pd.merge(left=questionnaire_byID, right=demographic_data, how="right", on =['ID'])

#data per ID/date
alcohol_per_date = pd.read_feather('data/processed/alcohol_per_date.feather')
coffee_date = pd.read_feather('data/processed/coffee_date.feather')
smoking_diet_mens = pd.read_feather('data/processed/smoking_diet_mens.feather')
exercise_date = pd.read_feather('data/processed/exercise_date.feather')
panic = pd.read_feather('data/processed/panic.feather')

alcohol_per_date.drop_duplicates(subset=['ID','date'], keep='first', inplace=True, ignore_index=True)

emotion_diary = pd.read_feather('data/processed/emotion_diary.feather')
questionnaire_bydate = pd.read_feather('data/processed/questionnaire_bydate.feather')

foot_date = pd.read_feather('data/processed/foot_date.feather')
HR_date = pd.read_feather('data/processed/HR_date.feather')
circadian_delta = pd.read_feather('data/processed/circadian_delta.feather')
bandpower = pd.read_feather('data/processed/bandpower.feather')
sleep = pd.read_feather('data/processed/sleep.feather')

#data merge
temp = pd.merge(left=HR_date, right=df_ID, how="left", on =['ID'])
bandpower.date = bandpower.date.dt.strftime('%Y-%m-%d')
data_list = [temp, alcohol_per_date, coffee_date, smoking_diet_mens, exercise_date, panic, emotion_diary, questionnaire_bydate, foot_date, circadian_delta, bandpower, sleep]
for i in data_list:
    print(i.columns)
    print(i.shape)
    i.drop_duplicates(subset=['ID','date'], keep='first', inplace=True, ignore_index=True)
    print(i.shape, '//')

df_date = reduce(lambda x, y : pd.merge(x, y,on=['ID', 'date'], how='left'), data_list)

#data save
df_date.drop(['ht','wt','late_night_snack'], axis=1, inplace=True)
df_date = df_date.dropna(subset=['acr'], axis=0)
df_date.fillna(0, inplace=True)
df_date.drop_duplicates(subset=['ID','date'], keep='first', inplace=True, ignore_index=True)
df_date.reset_index(drop=True, inplace=True)
df_date.rename(columns = {'foot':'foot_var'},inplace=True)
# renaming columns of data
df_date.rename(columns = {'amp':'HR_amplitude', 'mesor':'HR_mesor', 'acr':'HR_acrophase', 'amp_delta':'HR_amplitude_difference','mesor_delta':'HR_mesor_difference', 'acr_delta':'HR_acrophase_difference', 'amp_delta2':'HR_amplitude_difference_2d','mesor_delta2':'HR_mesor_difference_2d', 'acr_delta2':'HR_acrophase_difference_2d', 'positive':'positive_feeling', 'foot_max':'steps_maximum', 'foot_var':'steps_variance', 'foot_mean':'steps_mean', 'bandpower_a':'bandpower(0.001-0.0005Hz)', 'bandpower_b':'bandpower(0.0005-0.0001Hz)', 'bandpower_c':'bandpower(0.0001-0.00005Hz)', 'bandpower_d':'bandpower(0.00005-0.00001Hz)', 'sleep_in':'sleep_onset_time','foot_hvar_mean':'steps_hvar_mean', 'sleep_out':'sleep_out_time', 'suicide_need_in_month':'suicide_need'},inplace=True)

exclude_duplicated_id = "ID != ['SYM2-1-137', 'SYM2-1-334']"
df_date = df_date.query(exclude_duplicated_id)
    
df_date = df_date.drop_duplicates(subset=['ID', 'date'], keep='first')

df_date = df_date[df_date['panic'] != 2]

df_date.reset_index(drop=True, inplace=True)

df_date.to_feather("data/processed/merged_df.feather")
