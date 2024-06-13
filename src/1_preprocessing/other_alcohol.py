import pandas as pd
import numpy as np
from utils_for_preprocessing import load_raw_file, exclude_duplicated_id

#load file
alcohol_raw= load_raw_file('생활패턴-음주')

#data preprocessing
alcohol = exclude_duplicated_id(alcohol_raw)
alcohol.drop(['음주량',	'단위', '주종'], axis=1, inplace=True)
alcohol.columns = ["ID", "date", "time"]
alcohol['alcohol'] =1
alcohol = alcohol.replace({'time':'저녁'},'21:00:00')
alcohol = alcohol.replace({'time':'오전'},'09:00:00')
alcohol = alcohol.replace({'time':'오후'},'15:00:00')
alcohol.reset_index(drop=True, inplace=True)

#data save to feather
alcohol.to_feather("data/processed/alcohol.feather")

#alcohol data per date
alcohol.drop(['time'], axis=1, inplace=True)
alcohol.to_feather("data/processed/alcohol_per_date.feather")


