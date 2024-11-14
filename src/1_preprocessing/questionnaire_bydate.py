import pandas as pd
import numpy as np
from utils_for_preprocessing import extract_questionnaire_from_raw, extract_emotion_diary_from_raw
from functools import reduce

#data loading from raw data
ACQ = extract_questionnaire_from_raw(path = 'data/raw/9 Agoraphobia cognition.xlsx', questionnaire_sheet='Sheet1', df_name = 'ACQ', questionnaire_column = '1.Summing_agoraphobia_perception_scale_1_group_scores')
APPQ_1 = extract_questionnaire_from_raw(path = 'data/raw/11 Panic-Fear.xlsx', questionnaire_sheet='Sheet1', df_name = 'APPQ_1', questionnaire_column = 'Albany_Fear_Panic_Scale_Factors_1._Agoraphobia_Score_Total')
APPQ_2 = extract_questionnaire_from_raw(path = 'data/raw/11 Panic-Fear.xlsx', questionnaire_sheet='Sheet1', df_name = 'APPQ_2', questionnaire_column = '2.Albany_Fear_Panic_Scale_Factor_2._sum_of_Social_Fear_scores')
APPQ_3 = extract_questionnaire_from_raw(path = 'data/raw/11 Panic-Fear.xlsx', questionnaire_sheet='Sheet1', df_name = 'APPQ_3', questionnaire_column = '3.Albany_Fear_Panic_Scale_Factor_3._Internalizing_Fear_Score_Sum')
BSQ = extract_questionnaire_from_raw(path = 'data/raw/10 Physical sensations.xlsx', questionnaire_sheet='Sheet1', df_name = 'BSQ', questionnaire_column = '1.Summarize_the_scores_of_the_somatosensory_scale_group_1')
BFNE = extract_questionnaire_from_raw(path = 'data/raw/6 Fear of negative evaluation.xlsx', questionnaire_sheet='Sheet1', df_name = 'BFNE', questionnaire_column = '1.Fear_Scale_1_group_scores_summed_up')
CES_D = extract_questionnaire_from_raw(path = 'data/raw/32 Depression.xlsx', questionnaire_sheet='Sheet1', df_name = 'CES_D', questionnaire_column = '1.Depression_Scale_Revision_1_Group_Scores_Summed')

GAD_7 = extract_questionnaire_from_raw(path = 'data/raw/8 Generalized Anxiety disorder.xlsx', questionnaire_sheet='Sheet1', df_name = 'GAD_7', questionnaire_column = '1.Summing_the_Generalized_Anxiety_Disorder_Scale_1_Group_Scores')
KOSSSF = extract_questionnaire_from_raw(path = 'data/raw/12 Job stress.xlsx', questionnaire_sheet='Sheet1', df_name = 'KOSSSF', questionnaire_column = '1.Summing_group_scores_for_the_Short_Form_of_Job_Stress_Scale_1')
PHQ_9 = extract_questionnaire_from_raw(path = 'data/raw/1 Screening for depression.xlsx', questionnaire_sheet='Sheet1', df_name = 'PHQ_9', questionnaire_column = '1.Group_scores_for_Depression_Scale_1_summed_up\t')
SADS = extract_questionnaire_from_raw(path = 'data/raw/7 Social avoidance and discomfort.xlsx', questionnaire_sheet='Sheet1', df_name = 'SADS', questionnaire_column = '1.Summing_group_scores_for_Social_Avoidance_and_Discomfort_Scale_1')
STAI_X1 = extract_questionnaire_from_raw(path = 'data/raw/4 State anxiety.xlsx', questionnaire_sheet='Sheet1', df_name = 'STAI_X1', questionnaire_column = '1.Sum_Anxiety_Scale_Group_1_scores')

#data merge
data_list = [ACQ, APPQ_1, APPQ_2, APPQ_3, BSQ, BFNE, CES_D, GAD_7, KOSSSF, PHQ_9, SADS, STAI_X1 ]
questionnaire_bydate = reduce(lambda x, y : pd.merge(x, y,on=['ID', 'date'], how='outer'), data_list)

#convert type from object to float
col_list = ['ACQ', 'APPQ_1', 'APPQ_2', 'APPQ_3', 'BSQ', 'BFNE', 'CES_D', 'GAD_7', 'KOSSSF', 'PHQ_9', 'SADS', 'STAI_X1']
for i in col_list:
    questionnaire_bydate[i] = pd.to_numeric(questionnaire_bydate[i])

#data save to feather
questionnaire_bydate.to_feather("data/processed/questionnaire_bydate.feather")