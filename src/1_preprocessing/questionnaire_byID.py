import pandas as pd
import numpy as np
from utils_for_preprocessing import extract_questionnaire_byID_from_raw
from functools import reduce

BRIAN = extract_questionnaire_byID_from_raw(questionnaire_sheet='22생물학적 리듬', df_name = 'BRIAN', questionnaire_column = '1.생물학적 리듬 평가 척도 1번 히든 그룹 합산')
CSM = extract_questionnaire_byID_from_raw(questionnaire_sheet='20아침형-저녁형', df_name = 'CSM', questionnaire_column = '1.조합 척도 1~13번 문항 점수 합산')
CTQ_1 = extract_questionnaire_byID_from_raw(questionnaire_sheet='27유년기 외상', df_name = 'CTQ_1', questionnaire_column = '1.유년기 외상 척도 요인1. 정서방임 점수 합산')
CTQ_2 = extract_questionnaire_byID_from_raw(questionnaire_sheet='27유년기 외상', df_name = 'CTQ_2', questionnaire_column = '2.유년기 외상 척도 요인2. 신체학대 점수 합산')
CTQ_3 = extract_questionnaire_byID_from_raw(questionnaire_sheet='27유년기 외상', df_name = 'CTQ_3', questionnaire_column = '3.유년기 외상 척도 요인3. 성학대 점수 합산')
CTQ_4 = extract_questionnaire_byID_from_raw(questionnaire_sheet='27유년기 외상', df_name = 'CTQ_4', questionnaire_column = '4.유년기 외상 척도 요인4. 정서학대 점수 합산')
CTQ_5 = extract_questionnaire_byID_from_raw(questionnaire_sheet='27유년기 외상', df_name = 'CTQ_5', questionnaire_column = '5.유년기 외상 척도 요인5. 신체방임 점수 합산')

KRQ = extract_questionnaire_byID_from_raw(questionnaire_sheet='13회복탄력성', df_name = 'KRQ', questionnaire_column = '1.회복탄력성 척도 1번 그룹 점수 합산')
MDQ = extract_questionnaire_byID_from_raw(questionnaire_sheet='2기분 장애', df_name = 'MDQ', questionnaire_column = '1.기분 장애 척도 1번 그룹 점수 합산')
SPAQ_1 = extract_questionnaire_byID_from_raw(questionnaire_sheet='21계절성 양상', df_name = 'SPAQ_1', questionnaire_column = '1.계절성 양상 척도 2번 그룹 점수 합산')
SPAQ_2 = extract_questionnaire_byID_from_raw(questionnaire_sheet='21계절성 양상', df_name = 'SPAQ_2', questionnaire_column = '2.계절성 양상 척도 3번 문항 점수')
STAI_X2 = extract_questionnaire_byID_from_raw(questionnaire_sheet='5특성 불안', df_name = 'STAI_X2', questionnaire_column = '1.특성 불안 척도 1번 그룹 점수 합산')

data_list = [BRIAN, CSM, CTQ_1, CTQ_2, CTQ_3, CTQ_4, CTQ_5, KRQ, MDQ, SPAQ_1, SPAQ_2, STAI_X2 ]
questionnaire_byID = reduce(lambda x, y : pd.merge(x, y,on=['ID'], how='outer'), data_list)


col_list = ['BRIAN', 'CSM', 'CTQ_1', 'CTQ_2', 'CTQ_3', 'CTQ_4', 'CTQ_5', 'KRQ', 'MDQ', 'SPAQ_1', 'SPAQ_2', 'STAI_X2']
for i in col_list:
    questionnaire_byID[i] = pd.to_numeric(questionnaire_byID[i])
    
questionnaire_byID.to_feather("data/processed/questionnaire_byID.feather")