import pandas as pd
import numpy as np
from utils_for_preprocessing import extract_questionnaire_from_raw, extract_emotion_diary_from_raw
from functools import reduce

positive = extract_emotion_diary_from_raw(questionnaire_sheet='정서일지', df_name = 'positive', questionnaire_column = '긍정적기분')
negative = extract_emotion_diary_from_raw(questionnaire_sheet='정서일지', df_name = 'negative', questionnaire_column = '부정적기분')
positive_E = extract_emotion_diary_from_raw(questionnaire_sheet='정서일지', df_name = 'positive_E', questionnaire_column = '긍정적에너지')
negative_E = extract_emotion_diary_from_raw(questionnaire_sheet='정서일지', df_name = 'negative_E', questionnaire_column = '부정적에너지')
anxiety = extract_emotion_diary_from_raw(questionnaire_sheet='정서일지', df_name = 'anxiety', questionnaire_column = '불안')
annoying = extract_emotion_diary_from_raw(questionnaire_sheet='정서일지', df_name = 'annoying', questionnaire_column = '짜증')


data_list = [positive, negative, positive_E, negative_E, anxiety, annoying]
emotion_diary = reduce(lambda x, y : pd.merge(x, y,on=['ID', 'date'], how='outer'), data_list)

emotion_diary.to_feather("data/processed/emotion_diary.feather")