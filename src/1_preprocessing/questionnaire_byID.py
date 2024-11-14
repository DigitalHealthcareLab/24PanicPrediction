import pandas as pd
import numpy as np
from utils_for_preprocessing import extract_questionnaire_byID_from_raw
from functools import reduce

BRIAN = extract_questionnaire_byID_from_raw(path = 'data/raw/22 Biological rhythms.xlsx', questionnaire_sheet='Sheet1', df_name = 'BRIAN', questionnaire_column = '1.Biological_Rhythms_Rating_Scale_1_Hidden_Group_Summation')
CSM = extract_questionnaire_byID_from_raw(path = 'data/raw/20 Morning-Evening person.xlsx', questionnaire_sheet='Sheet1', df_name = 'CSM', questionnaire_column = '1.Sum_the_scores_of_questions_1_13_on_the_combination_scale')
CTQ_1 = extract_questionnaire_byID_from_raw(path = 'data/raw/27 Childhood Trauma.xlsx', questionnaire_sheet='Sheet1', df_name = 'CTQ_1', questionnaire_column = 'Childhood_Trauma_Scale_Factor_1._Summing_the_Emotional_Neglect_Scores')
CTQ_2 = extract_questionnaire_byID_from_raw(path = 'data/raw/27 Childhood Trauma.xlsx', questionnaire_sheet='Sheet1', df_name = 'CTQ_2', questionnaire_column = '2.Childhood_Trauma_Scale_Factor_2._Total_Physical_Abuse_Scores')
CTQ_3 = extract_questionnaire_byID_from_raw(path = 'data/raw/27 Childhood Trauma.xlsx', questionnaire_sheet='Sheet1', df_name = 'CTQ_3', questionnaire_column = '3.Childhood_Trauma_Scale_Factor_3._Sexual_Abuse_Total_Score')
CTQ_4 = extract_questionnaire_byID_from_raw(path = 'data/raw/27 Childhood Trauma.xlsx', questionnaire_sheet='Sheet1', df_name = 'CTQ_4', questionnaire_column = '4.Childhood_Trauma_Scale_Factor_4._Emotional_Abuse_Scores_Summed')
CTQ_5 = extract_questionnaire_byID_from_raw(path = 'data/raw/27 Childhood Trauma.xlsx', questionnaire_sheet='Sheet1', df_name = 'CTQ_5', questionnaire_column = '5.Childhood_Trauma_Scale_Factor_5._Physical_Neglect_Scores_Summed')

KRQ = extract_questionnaire_byID_from_raw(path = 'data/raw/13 Resilience.xlsx', questionnaire_sheet='Sheet1', df_name = 'KRQ', questionnaire_column = '1.Summing_Resilience_Scale_1_Group_Scores')
MDQ = extract_questionnaire_byID_from_raw(path = 'data/raw/2 Mood disorders.xlsx', questionnaire_sheet='Sheet1', df_name = 'MDQ', questionnaire_column = '1.Mood_Disorders_Scale_1_group_scores_combined\t')
SPAQ_1 = extract_questionnaire_byID_from_raw(path = 'data/raw/21 Seasonality.xlsx', questionnaire_sheet='Sheet1', df_name = 'SPAQ_1', questionnaire_column = '1.Sum_group_scores_for_Seasonality_Aspects_Scale_Question_2')
SPAQ_2 = extract_questionnaire_byID_from_raw(path = 'data/raw/21 Seasonality.xlsx', questionnaire_sheet='Sheet1', df_name = 'SPAQ_2', questionnaire_column = '2.Scores_for_question_3_of_the_Seasonality_Aspect_Scale')
STAI_X2 = extract_questionnaire_byID_from_raw(path = 'data/raw/5 Trait anxiety.xlsx', questionnaire_sheet='Sheet1', df_name = 'STAI_X2', questionnaire_column = '1.Sum_Anxiety_Scale_Group_1_scores')

data_list = [BRIAN, CSM, CTQ_1, CTQ_2, CTQ_3, CTQ_4, CTQ_5, KRQ, MDQ, SPAQ_1, SPAQ_2, STAI_X2 ]
questionnaire_byID = reduce(lambda x, y : pd.merge(x, y,on=['ID'], how='outer'), data_list)


col_list = ['BRIAN', 'CSM', 'CTQ_1', 'CTQ_2', 'CTQ_3', 'CTQ_4', 'CTQ_5', 'KRQ', 'MDQ', 'SPAQ_1', 'SPAQ_2', 'STAI_X2']
for i in col_list:
    questionnaire_byID[i] = pd.to_numeric(questionnaire_byID[i])
    
questionnaire_byID.to_feather("data/processed/questionnaire_byID.feather")