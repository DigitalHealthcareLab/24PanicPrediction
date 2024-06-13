import pandas as pd
from utils_for_analysis import mesor, amplitude, acrophase

HR_interpolated = pd.read_feather('data/processed/HR_interpolated.feather')
HR_interpolated['HR'] = pd.to_numeric(HR_interpolated['HR'])

id_list = HR_interpolated['ID'].unique()
circadian_data = pd.DataFrame(columns=['ID','date','acr','amp','mesor'])
                  
for id in id_list:
    temp_id = HR_interpolated.loc[(HR_interpolated['ID'] == id)]
    temp_id.reset_index(inplace=True)
    temp_id.drop('index', axis=1, inplace=True)
    date_list =temp_id['date'].unique()
    for date in date_list:
        temp_date = temp_id.loc[(temp_id['date'] == date)]
        temp_date.reset_index(inplace=True)
        temp_date.drop('index', axis=1, inplace=True)
        temp_date.reset_index(inplace=True)  
        if temp_date.HR.count() > 360:
            acr = acrophase(temp_date['index'], temp_date['HR'])
            amp = amplitude(temp_date['index'], temp_date['HR'])
            mes = mesor(temp_date['index'], temp_date['HR'])
            circadian_data = circadian_data.append(pd.DataFrame([[id, date, acr, amp, mes]], columns=['ID','date','acr','amp','mesor']), ignore_index=True)
            print(id, date, acr, amp, mes)
        else:
            pass


circadian_data.to_feather("data/processed/circadian_parameter.feather")