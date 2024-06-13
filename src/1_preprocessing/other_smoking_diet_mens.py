import pandas as pd
import numpy as np
from utils_for_preprocessing import load_raw_file, exclude_duplicated_id
import datetime

#load file
smoking_diet_mens= load_raw_file('생활패턴-흡연,식사,생리')

#data preprocessifng
smoking_diet_mens = exclude_duplicated_id(smoking_diet_mens)
smoking_diet_mens['흡연량'][(smoking_diet_mens['흡연량'] > 0)] = 1
smoking_diet_mens['흡연량'][(smoking_diet_mens['흡연량'] != 1)] = 0
smoking_diet_mens['생리'][(smoking_diet_mens['생리'] == 'Y')] = 1
smoking_diet_mens['생리'][(smoking_diet_mens['생리'] != 1)] = 0
smoking_diet_mens['야식'][(smoking_diet_mens['야식'] == 'Y')] = 1
smoking_diet_mens['야식'][(smoking_diet_mens['야식'] != 1)] = 0
smoking_diet_mens.drop(['아침식사', '점심식사', '저녁식사', '오전간식', '오후간식'], axis=1, inplace=True)
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