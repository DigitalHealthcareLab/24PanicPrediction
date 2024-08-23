## Project Name
24PanicPrediction

## Prerequisites
- **Python version:** 3.8.0
- **Pandas version:** 1.4.4
- **NumPy version:** 1.24.4
- **SciKit-Learn version:** 1.3.2
- **Matplotlib version:** 3.7.4
- **Shap version:** 0.44.0

## Objective
The objective of this project is to develop a machine learning model to predict impending panic symptoms using digital phenotyping data collected from smartphone applications and wearable devices.

## Project Structure

### 1. Data Preprocessing
- `emotion_diary.py`: Extracts and processes emotion diary data from raw data files ('정서일지' sheet).
- `lifelog_foot.py`: Extracts and processes step count data from raw data files ('라이프로그-걸음수' sheet).
- `lifelog_HR.py`: Extracts and processes heart rate data from raw data files ('라이프로그-심박수' sheet).
- `lifelog_sleep.py`: Extracts and processes sleep data from raw data files ('라이프로그-수면' sheet).
- `other_alcohol.py`: Extracts and processes alcohol consumption data from raw data files ('생활패턴-음주' sheet).
- `other_coffee.py`: Extracts and processes coffee consumption data from raw data files ('생활패턴-카페인' sheet).
- `other_demographics.py`: Extracts and processes demographic information from raw data files ('연구 참여자 기본 정보' sheet).
- `other_exercise.py`: Extracts and processes exercise data from raw data files ('생활패턴-운동' sheet).
- `other_panic.py`: Extracts and processes panic symptom reports from raw data files ('공황일지' sheet).
- `other_smoking_diet_mens.py`: Extracts and processes data related to smoking habits, diet, and menstruation from raw data files ('생활패턴-흡연,식사,생리' sheet).
- `questionnaire_bydate.py`: Organizes and processes questionnaire data by date from various sheets of raw data files.
- `questionnaire_byID.py`: Organizes and processes questionnaire data by participant ID from various sheets of raw data files.
- `utils_for_preprocessing.py`: Contains utility functions for data preprocessing.

### 2. Data Analysis
- `foot_delta.py`: Analyzes changes in step count data, calculating differences between current day and previous day, as well as two days prior.
- `foot_statistics.py`: Calculates statistical measures for step count data, including mean, variance, maximum, and mean of hourly variance.
- `HR_bandpower_analysis.py`: Performs bandpower analysis on heart rate data.
- `HR_cosinor.py` and `HR_cosinor_delta.py`: Performs cosinor analysis on heart rate data to analyze circadian rhythms and changes in these rhythms.
- `HR_interpolation.py`: Interpolates missing heart rate data.
- `HR_statistics.py`: Calculates statistical measures for heart rate data, including mean, variance, maximum, minimum, and mean of hourly variance.
- `HR_visualization.py`: Creates visualizations for heart rate data to aid in analysis and interpretation.
- `utils_for_analysis.py`: Contains utility functions for data analysis.

### 3. Data Merge
- `data_merge.py`: Merges all processed data into a single dataset.

### 4. Prediction Models
- `gradientboost_model.py`: Implements a Gradient Boosting model for panic prediction.
- `randomforest_model.py`: Implements a Random Forest model for panic prediction.
- `xgboost_model.py`: Implements an XGBoost model for panic prediction using the full feature set.
- `xgboost_model_constant_only.py`: XGBoost model using only demographic data.
- `xgboost_model_dailylog_only.py`: XGBoost model using only daily log data.
- `xgboost_model_lifelog_only.py`: XGBoost model using only lifelog data from wearable devices.
- `xgboost_model_psychological_only.py`: XGBoost model using only psychological questionnaire data.
- `xgboost_model_top10_shapley.py`: XGBoost model using only the top 10 features based on Shapley values.
- `utils.py`: Contains utility functions for model training, evaluation, and visualization, including ROC curve plotting and performance metric calculations.

## Usage
To run the full pipeline:

1. Execute the preprocessing scripts in the `1_preprocessing` directory.
2. Run the analysis scripts in the `2_data_analysis` directory.
3. Merge the processed data using `3_data_merge/data_merge.py`.
4. Train and evaluate the prediction models using the scripts in the `4_prediction_model` directory.
