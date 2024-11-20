import pandas as pd
import numpy as np
from datetime import datetime
from utils_for_preprocessing import load_raw_file, exclude_duplicated_id

#load file
demographic_data= load_raw_file(path='data/raw/Basic research participation information.xlsx', sheet_name='Sheet1')

#data preprocessing
demographic_data = exclude_duplicated_id(demographic_data)
demographic_data.drop(['Date_of_study_consent', 'Date_of_study_consent_withdrawal', 'Experience_of_physical_illness', 'Income', 'Recent_drinking_information', 'Menopausal_status_(M_male)',	'Age_at_first_suicide_attempt',	'Number_of_suicide_attempts_in_past_month', 'Education_level',	'Education_level_(other)', 'Amount_of_smoking', 'Problems_with_drinking_status', 	'Drinking_information', 	
         'Past_12_weeks_drinking_status',	'Current_or_past_problems_with_drinking_status',	'Prescription_drug_dosage_in_the_past_month_(%)', 'Occupation',	'Occupation_(other)',	'Number_of_children',	'Religion',	'Religion_(other)',	
         'Psychiatric_illnesses',	'List_of_psychiatric_illnesses', 'Experience_of_mental_illness', 'Past_12_weeks_smoking_status', 'Self_harm_in_the_past_month_(Y/N)','Prescription_drugs_in_the_past_month'], axis=1, inplace=True)
demographic_data['Date_of_birth'] = pd.to_datetime(demographic_data['Date_of_birth'], format = '%Y%m%d')
demographic_data.reset_index(drop=True, inplace=True)

#check age of participants
for i in range(demographic_data.shape[0]):
    today = datetime.now().date()
    birth = demographic_data['Date_of_birth'][i]
    year = today.year - birth.year
    if today.month < birth.month or (today.month == birth.month and today.day < birth.day):
        year -= 1
    demographic_data['Date_of_birth'][i] = year

#data preprocessing
demographic_data.columns = ["ID", "age", "gender", 'ht', 'wt', 'marriage', 'job','smkHx','drinkHx','suicideHx','suicide_need_in_month','medication_in_month']
demographic_data['marriage'][(demographic_data['marriage'] == 'Y')] = 1
demographic_data['marriage'][(demographic_data['marriage'] != 1)] = 0
demographic_data['job'][(demographic_data['job'] == 'Y')] = 1
demographic_data['job'][(demographic_data['job'] != 1)] = 0
demographic_data['smkHx'][(demographic_data['smkHx'] == 'Y')] = 1
demographic_data['smkHx'][(demographic_data['smkHx'] != 1)] = 0
demographic_data['drinkHx'][(demographic_data['drinkHx'] == 'Y')] = 1
demographic_data['drinkHx'][(demographic_data['drinkHx'] != 1)] = 0
demographic_data['suicideHx'][(demographic_data['suicideHx'] == 'Y')] = 1
demographic_data['suicideHx'][(demographic_data['suicideHx'] != 1)] = 0
demographic_data['gender'][(demographic_data['gender'] == 'M')] = 1
demographic_data['gender'][(demographic_data['gender'] == 'F')] = 0
demographic_data['suicide_need_in_month'][(demographic_data['suicide_need_in_month'] == 'Y')] = 1
demographic_data['suicide_need_in_month'][(demographic_data['suicide_need_in_month'] != 1)] = 0
demographic_data['medication_in_month'][(demographic_data['medication_in_month'] == 'Y')] = 1
demographic_data['medication_in_month'][(demographic_data['medication_in_month'] != 1)] = 0

#data save to feather
demographic_data.to_feather("data/processed/demographic_data.feather")


