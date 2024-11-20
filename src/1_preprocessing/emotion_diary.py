import pandas as pd
import numpy as np
from utils_for_preprocessing import extract_questionnaire_from_raw, extract_emotion_diary_from_raw
from functools import reduce

#load file
positive = extract_emotion_diary_from_raw(path = 'data/raw/Emotion Diary.xlsx', questionnaire_sheet='Sheet1', df_name = 'positive', questionnaire_column = 'Positive_Mood')
negative = extract_emotion_diary_from_raw(path = 'data/raw/Emotion Diary.xlsx', questionnaire_sheet='Sheet1', df_name = 'negative', questionnaire_column = 'Negative_Mood')
positive_E = extract_emotion_diary_from_raw(path = 'data/raw/Emotion Diary.xlsx', questionnaire_sheet='Sheet1', df_name = 'positive_E', questionnaire_column = 'Positive_Energy')
negative_E = extract_emotion_diary_from_raw(path = 'data/raw/Emotion Diary.xlsx', questionnaire_sheet='Sheet1', df_name = 'negative_E', questionnaire_column = 'Negative_Energy')
anxiety = extract_emotion_diary_from_raw(path = 'data/raw/Emotion Diary.xlsx', questionnaire_sheet='Sheet1', df_name = 'anxiety', questionnaire_column = 'Anxiety')
annoying = extract_emotion_diary_from_raw(path = 'data/raw/Emotion Diary.xlsx', questionnaire_sheet='Sheet1', df_name = 'annoying', questionnaire_column = 'Irritability')

#data preprocessing
data_list = [positive, negative, positive_E, negative_E, anxiety, annoying]
emotion_diary = reduce(lambda x, y : pd.merge(x, y,on=['ID', 'date'], how='outer'), data_list)

#data save to feather
emotion_diary.to_feather("data/processed/emotion_diary.feather")