import pandas as pd
import numpy as np
from utils_for_preprocessing import load_raw_file, exclude_duplicated_id

#load file
exercise = load_raw_file('data/raw/Lifestyle - Workouts.xlsx', sheet_name='Sheet1')

#data preprocessing
exercise = exclude_duplicated_id(exercise)
exercise.drop(['Workout_types', 'Workout_time_(minutes)'], axis=1, inplace=True)
exercise.columns = ["ID", "date", "time"]
exercise['exercise'] =1
exercise = exercise.replace({'time':'Evening'},'21:00:00')
exercise = exercise.replace({'time':'Morning'},'09:00:00')
exercise = exercise.replace({'time':'Afternoon'},'15:00:00')
exercise.reset_index(drop=True, inplace=True)

#data save to feather
exercise.to_feather("data/processed/exercise.feather")

#data per date
exercise.drop(['time'], axis=1, inplace=True)
exercise.to_feather("data/processed/exercise_date.feather")


