import pandas as pd
import numpy as np
from utils_for_preprocessing import load_raw_file, exclude_duplicated_id
import datetime

#load file
smoking_diet_mens = load_raw_file('data/raw/Lifestyle - Smoking, Eating, Menstruation.xlsx', sheet_name='Sheet1')

#data preprocessifng
smoking_diet_mens = exclude_duplicated_id(smoking_diet_mens)
smoking_diet_mens['Amount_smoked'][(smoking_diet_mens['Amount_smoked'] > 0)] = 1
smoking_diet_mens['Amount_smoked'][(smoking_diet_mens['Amount_smoked'] != 1)] = 0
smoking_diet_mens['Menstruation'][(smoking_diet_mens['Menstruation'] == 'Y')] = 1
smoking_diet_mens['Menstruation'][(smoking_diet_mens['Menstruation'] != 1)] = 0
smoking_diet_mens['Midnight_snacks'][(smoking_diet_mens['Midnight_snacks'] == 'Y')] = 1
smoking_diet_mens['Midnight_snacks'][(smoking_diet_mens['Midnight_snacks'] != 1)] = 0
smoking_diet_mens.drop(['Breakfast', 'Lunch', 'Dinner', 'Morning_snack', 'Afternoon_snack'], axis=1, inplace=True)
smoking_diet_mens.columns = ["ID", "date", "smoking", "late_night_snack", "menstruation"]
smoking_diet_mens.reset_index(drop=True, inplace=True)

#data save to feather
smoking_diet_mens.to_feather("data/processed/smoking_diet_mens.feather")

#data for 1wk before menstruation 
menstruation = smoking_diet_mens[['ID', 'date', 'menstruation']]
menstruation['date'] = pd.to_datetime(menstruation['date'])
menstruation['date'] = menstruation['date'] - datetime.timedelta(7)
menstruation_before_1wk = menstruation
menstruation_before_1wk.reset_index(drop=True, inplace=True)

#data save to feather
menstruation_before_1wk.to_feather("data/processed/menstruation_before_1wk.feather")