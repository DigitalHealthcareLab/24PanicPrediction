from openpyxl import load_workbook
from pandas import DataFrame
from itertools import islice
import pandas as pd 
import numpy as np 

def load_raw_file(sheet_name):
    raw_file_1 = "data/raw/SYM2_updated_202206.xlsx"
    workbook = load_workbook(raw_file_1, data_only=True)
    worksheet = workbook[sheet_name]
    data = worksheet.values
    cols = next(data)[0:]
    data = list(data)
    idx = [r[0] for r in data]
    data = (islice(r, 0, None) for r in data)
    df_from_raw_1  = DataFrame(data, columns=cols)
    
    raw_file_2 = "data/raw/backup_SYM2-1 (고려병원)_20231219.xlsx"
    workbook = load_workbook(raw_file_2, data_only=True)
    worksheet = workbook[sheet_name]
    data = worksheet.values
    cols = next(data)[0:]
    data = list(data)
    idx = [r[0] for r in data]
    data = (islice(r, 0, None) for r in data)
    df_from_raw_2  = DataFrame(data, columns=cols)
    
    result = pd.concat([df_from_raw_1, df_from_raw_2])
    return result

def exclude_duplicated_id(df):
    exclude_duplicated_id = "비식별키 != ['SYM2-1-101', 'SYM2-1-135', 'SYM2-1-278', 'SYM2-1-292', 'SYM2-1-349', 'SYM2-1-472', 'SYM2-1-474', 'SYM2-1-62','SYM2-1-96', 'SYM2-1-143', 'SYM2-1-275', 'SYM2-1-246', 'SYM2-1-471', 'SYM2-1-448', 'SYM2-1-473']"
    df = df.query(exclude_duplicated_id)
    return df

def exclude_miband_user_data(df):
    exclude_miband_data = df[df['측정 유형'].str.contains('MI-BAND')].index
    df.drop(exclude_miband_data, axis=0, inplace=True)
    exclude_user_data= df[df['측정 유형'].str.contains('User')].index
    df.drop(exclude_user_data, axis=0, inplace=True)
    return df
    
def exclude_15min_interval_data(df):
    exclude_15min_interval_data = df[df['측정 단위'].str.contains('15분 단위')].index
    df.drop(exclude_15min_interval_data, axis=0, inplace=True)
    return df

def exclude_empty_data(df, reference_column, keyword):
    df =df.loc[df[reference_column].str.contains(keyword)]
    return df

def serialize_lifelog_data(df):
    #make time index
    temp_idx = pd.date_range('2018-01-01', periods=1440, freq='1T')
    col_minutes = pd.DataFrame(temp_idx)
    col_minutes.columns = ['time']
    col_minutes.time = col_minutes.time.dt.strftime('%H:%M:%S')
    col_minutes = col_minutes.time.squeeze()
    df_splitted = pd.DataFrame(df['측정값(-1 : 값 없음)'].str.split(',').tolist(),columns=col_minutes)
    df_splitted.reset_index(drop=True, inplace=True)
    df_merged = pd.merge(df,df_splitted,how="left",left_index=True, right_index=True)
    df_merged.drop(['측정 유형','측정값(-1 : 값 없음)'], axis=1, inplace=True)
    df_melted = pd.melt(df_merged, id_vars=['날짜','비식별키'])
    df_melted.columns = ["date", "ID", "time", 'lifelog_data']
    return df_melted

def extract_questionnaire_from_raw(questionnaire_sheet, df_name ,questionnaire_column):
    df = load_raw_file(sheet_name=questionnaire_sheet)
    df = exclude_duplicated_id(df)
    df = df[['비식별키', '설문종료일', '설문완료일', questionnaire_column]]
    df['설문완료일'] = np.where(pd.notnull(df['설문완료일']) == True, df['설문완료일'], df['설문종료일'])
    df = df.loc[(df[questionnaire_column]!='')]
    df.drop(['설문종료일'], axis=1, inplace=True)
    df= df.dropna(axis=0)
    df.columns = ["ID", 'date', df_name]
    df.drop_duplicates(['ID','date'], keep='last', inplace=True, ignore_index=False)
    return df

def extract_emotion_diary_from_raw(questionnaire_sheet, df_name ,questionnaire_column):
    df = load_raw_file(sheet_name=questionnaire_sheet)
    df = exclude_duplicated_id(df)
    df = df[['비식별키', '날짜', questionnaire_column]]
    df = df.loc[(df[questionnaire_column]!='')]
    df= df.dropna(axis=0)
    df.columns = ["ID", 'date', df_name]
    df.drop_duplicates(['ID','date'], keep='last', inplace=True, ignore_index=False)
    return df

def extract_questionnaire_byID_from_raw(questionnaire_sheet, df_name ,questionnaire_column):
    df = load_raw_file(sheet_name=questionnaire_sheet)
    df = exclude_duplicated_id(df)
    df = df[['비식별키', questionnaire_column]]
    df = df.loc[(df[questionnaire_column]!='')]
    df.columns = ["ID", df_name]
    df.drop_duplicates(['ID'], keep='first', inplace=True, ignore_index=False)
    return df