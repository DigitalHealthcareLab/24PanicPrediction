import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

HR = pd.read_feather('data/processed/HR.feather')
HR['HR'] = pd.to_numeric(HR['HR'])
id_list = HR['ID'].unique()

def plot_df(x, y, file_name, xlabel='Date', ylabel='Value', dpi=100): 
    fig = plt.figure(figsize=(18,4), dpi=dpi) 
    plt.plot(x, y, color='tab:red') 
    plt.gca().set(title=file_name, xlabel=xlabel, ylabel=ylabel) 
    fig.patch.set_facecolor('white')
    plt.savefig('data/hr/%s.png'% file_name) 
    plt.close(fig)

#def visualize_hr(id):

for id in id_list:
    temp_id = HR.loc[(HR['ID'] == id)]
    temp_id.reset_index(inplace=True)
    temp_id.drop('index', axis=1, inplace=True)
    date_list =temp_id['date'].unique()
    for date in date_list:
        temp_date = temp_id.loc[(temp_id['date'] == date)]
        temp_date.reset_index(inplace=True)
        temp_date.drop('index', axis=1, inplace=True)
        temp_date.reset_index(inplace=True)  
        file_name =  id + ' ' + date
        plot_df(temp_date['index'], temp_date['HR'], file_name)
        print(id, date)
        