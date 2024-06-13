import pandas as pd
import numpy as np
from utils_for_preprocessing import extract_questionnaire_from_raw, extract_emotion_diary_from_raw
from functools import reduce

#data loading from raw data
ACQ = extract_questionnaire_from_raw(questionnaire_sheet='9광장공포인지', df_name = 'ACQ', questionnaire_column = '1.광장공포 인지 척도 1번 그룹 점수 합산')
APPQ_1 = extract_questionnaire_from_raw(questionnaire_sheet='11공황-공포', df_name = 'APPQ_1', questionnaire_column = '1.알바니 공포-공황 척도 요인1. 광장공포 점수 합계')
APPQ_2 = extract_questionnaire_from_raw(questionnaire_sheet='11공황-공포', df_name = 'APPQ_2', questionnaire_column = '2.알바니 공포-공황 척도 요인2. 사회공포 점수 합계')
APPQ_3 = extract_questionnaire_from_raw(questionnaire_sheet='11공황-공포', df_name = 'APPQ_3', questionnaire_column = '3.알바니 공포-공황 척도 요인3. 내부감각두려움 점수 합계')
BSQ = extract_questionnaire_from_raw(questionnaire_sheet='10신체감각', df_name = 'BSQ', questionnaire_column = '1.신체감각 척도 1번 그룹 점수 합산')
BFNE = extract_questionnaire_from_raw(questionnaire_sheet='6부정적평가에 대한 두려움', df_name = 'BFNE', questionnaire_column = '1.두려움 척도 1번 그룹 점수 합산')
CES_D = extract_questionnaire_from_raw(questionnaire_sheet='32우울증', df_name = 'CES_D', questionnaire_column = '1.우울 척도 개정판 1번 그룹 점수 합산')

GAD_7 = extract_questionnaire_from_raw(questionnaire_sheet='8범불안 장애', df_name = 'GAD_7', questionnaire_column = '1.범불안 장애 척도 1번 그룹 점수 합산')
KOSSSF = extract_questionnaire_from_raw(questionnaire_sheet='12직무스트레스', df_name = 'KOSSSF', questionnaire_column = '1.직무스트레스 단축형 척도 1번 그룹 점수 합산')
PHQ_9 = extract_questionnaire_from_raw(questionnaire_sheet='1우울증 선별', df_name = 'PHQ_9', questionnaire_column = '1.우울증 척도 1번 그룹 점수 합산')
SADS = extract_questionnaire_from_raw(questionnaire_sheet='7사회적회피 및 불편감', df_name = 'SADS', questionnaire_column = '1.사회적 회피 및 불편감 척도 1번 그룹 점수 합산')
STAI_X1 = extract_questionnaire_from_raw(questionnaire_sheet='4상태 불안', df_name = 'STAI_X1', questionnaire_column = '1.불안 척도 1번 그룹 점수 합산')

#data merge
data_list = [ACQ, APPQ_1, APPQ_2, APPQ_3, BSQ, BFNE, CES_D, GAD_7, KOSSSF, PHQ_9, SADS, STAI_X1 ]
questionnaire_bydate = reduce(lambda x, y : pd.merge(x, y,on=['ID', 'date'], how='outer'), data_list)

#convert type from object to float
col_list = ['ACQ', 'APPQ_1', 'APPQ_2', 'APPQ_3', 'BSQ', 'BFNE', 'CES_D', 'GAD_7', 'KOSSSF', 'PHQ_9', 'SADS', 'STAI_X1']
for i in col_list:
    questionnaire_bydate[i] = pd.to_numeric(questionnaire_bydate[i])

#data save to feather
questionnaire_bydate.to_feather("data/processed/questionnaire_bydate.feather")