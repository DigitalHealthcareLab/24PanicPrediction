from openpyxl import load_workbook
from pandas import DataFrame
from itertools import islice
import pandas as pd 
import numpy as np 

def load_raw_file(path, sheet_name):
    raw_file_1 = path
    workbook = load_workbook(raw_file_1, data_only=True)
    worksheet = workbook[sheet_name]
    data = worksheet.values
    cols = next(data)[0:]
    data = list(data)
    idx = [r[0] for r in data]
    data = (islice(r, 0, None) for r in data)
    df  = DataFrame(data, columns=cols)
    df.columns = [
        col.replace(' ', '_').replace('-', '_') if col is not None else col 
        for col in df.columns
    ]
    return df

def exclude_duplicated_id(df):
    exclude_duplicated_id = "Non_identifying_keys != ['SYM2-1-101', 'SYM2-1-135', 'SYM2-1-278', 'SYM2-1-292', 'SYM2-1-349', 'SYM2-1-472', 'SYM2-1-474', 'SYM2-1-62','SYM2-1-96', 'SYM2-1-143', 'SYM2-1-275', 'SYM2-1-246', 'SYM2-1-471', 'SYM2-1-448', 'SYM2-1-473']"
    df = df.query(exclude_duplicated_id)
    return df

def exclude_miband_user_data(df):
    exclude_miband_data = df[df['Units_of_measure'].str.contains('MI-BAND')].index
    df.drop(exclude_miband_data, axis=0, inplace=True)
    exclude_user_data= df[df['Units_of_measure'].str.contains('User')].index
    df.drop(exclude_user_data, axis=0, inplace=True)
    return df
    
def exclude_15min_interval_data(df):
    exclude_15min_interval_data = df[df['Units_of_measure'].str.contains('15 minute increments')].index
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
    df_splitted = pd.DataFrame(df['Measure_(_1:_no_value)'].str.split(',').tolist(),columns=col_minutes)
    df_splitted.reset_index(drop=True, inplace=True)
    df_merged = pd.merge(df,df_splitted,how="left",left_index=True, right_index=True)
    df_merged.drop(['Units_of_measure','Measure_(_1:_no_value)'], axis=1, inplace=True)
    df_melted = pd.melt(df_merged, id_vars=['Date','Non_identifying_keys'])
    df_melted.columns = ["date", "ID", "time", 'lifelog_data']
    return df_melted

def extract_questionnaire_from_raw(path, questionnaire_sheet, df_name ,questionnaire_column):
    df = load_raw_file(path, sheet_name=questionnaire_sheet)
    df = exclude_duplicated_id(df)
    df = df[['Non_identifying_keys', 'Survey_end_date', 'Survey_completion_date', questionnaire_column]]
    df['Survey_completion_date'] = np.where(pd.notnull(df['Survey_completion_date']) == True, df['Survey_completion_date'], df['Survey_end_date'])
    df = df.loc[(df[questionnaire_column]!='')]
    df.drop(['Survey_end_date'], axis=1, inplace=True)
    df= df.dropna(axis=0)
    df.columns = ["ID", 'date', df_name]
    df.drop_duplicates(['ID','date'], keep='last', inplace=True, ignore_index=False)
    return df

def extract_emotion_diary_from_raw(path, questionnaire_sheet, df_name ,questionnaire_column):
    df = load_raw_file(path, sheet_name=questionnaire_sheet)
    df = exclude_duplicated_id(df)
    df = df[['Non_identifying_keys', 'Date', questionnaire_column]]
    df = df.loc[(df[questionnaire_column]!='')]
    df= df.dropna(axis=0)
    df.columns = ["ID", 'date', df_name]
    df.drop_duplicates(['ID','date'], keep='last', inplace=True, ignore_index=False)
    return df

def extract_questionnaire_byID_from_raw(path, questionnaire_sheet, df_name ,questionnaire_column):
    df = load_raw_file(path, sheet_name=questionnaire_sheet)
    df = exclude_duplicated_id(df)
    df = df[['Non_identifying_keys', questionnaire_column]]
    df = df.loc[(df[questionnaire_column]!='')]
    df.columns = ["ID", df_name]
    df.drop_duplicates(['ID'], keep='first', inplace=True, ignore_index=False)
    return df