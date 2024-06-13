import pandas as pd
import numpy as np
from utils_for_preprocessing import load_raw_file, exclude_duplicated_id

#load file
coffee_raw= load_raw_file('생활패턴-카페인')

#data preprocessing
coffee = exclude_duplicated_id(coffee_raw)
coffee.drop(['종류', '섭취량', '단위'], axis=1, inplace=True)
coffee.columns = ["ID", "date", "time"]
coffee['coffee'] =1
coffee = coffee.replace({'time':'저녁'},'21:00:00')
coffee = coffee.replace({'time':'오전'},'09:00:00')
coffee = coffee.replace({'time':'오후'},'15:00:00')
coffee.reset_index(drop=True, inplace=True)

coffee.to_feather("data/processed/coffee.feather")

coffee.drop(['time'], axis=1, inplace=True)
coffee.to_feather("data/processed/coffee_date.feather")


