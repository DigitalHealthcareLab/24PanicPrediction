import pandas as pd
import numpy as np
from utils_for_preprocessing import load_raw_file, exclude_duplicated_id

#load file
exercise= load_raw_file('생활패턴-운동')

#data preprocessing
exercise = exclude_duplicated_id(exercise)
exercise.drop(['운동 종류', '운동시간(분)'], axis=1, inplace=True)
exercise.columns = ["ID", "date", "time"]
exercise['exercise'] =1
exercise = exercise.replace({'time':'저녁'},'21:00:00')
exercise = exercise.replace({'time':'오전'},'09:00:00')
exercise = exercise.replace({'time':'오후'},'15:00:00')
exercise.reset_index(drop=True, inplace=True)

#data save to feather
exercise.to_feather("data/processed/exercise.feather")

#data per date
exercise.drop(['time'], axis=1, inplace=True)
exercise.to_feather("data/processed/exercise_date.feather")


