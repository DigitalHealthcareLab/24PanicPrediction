import pandas as pd
import numpy as np
from datetime import datetime
from utils_for_preprocessing import load_raw_file, exclude_duplicated_id

#load file
demographic_data= load_raw_file('연구 참여자 기본 정보')

#data preprocessing
demographic_data = exclude_duplicated_id(demographic_data)
demographic_data.drop(['연구 동의일', '연구 동의 철회일', '육체적 질병 경험',  '수입', '최근 음주 정보', '폐경 여부(M-남자)',	'최초 자살 시도 나이',	'자살 시도 횟수', '교육 수준',	'교육 수준(기타)', '흡연량', '음주로 인한 문제 발생 여부', 	'음주 정보', 	
         '최근 12주 음주 여부',	'현재 혹은 과거 음주로 인한 문제 발생 여부',	'최근 1달 처방약 복약량(%)', '직업',	'직업(기타)',	'자녀수',	'종교',	'종교(기타)',	
         '정신과적 질병 유무',	'정신과적 질병 리스트', '정신적 질병 경험', '최근 12주 흡연 여부', '최근 1달 자해 여부(Y/N)','최근 1달 자살 시도 여부'], axis=1, inplace=True)
demographic_data['생년월일'] = pd.to_datetime(demographic_data['생년월일'], format = '%Y-%m-%d')
demographic_data.reset_index(drop=True, inplace=True)

#check age of participants
for i in range(demographic_data.shape[0]):
    today = datetime.now().date()
    birth = demographic_data['생년월일'][i]
    year = today.year - birth.year
    if today.month < birth.month or (today.month == birth.month and today.day < birth.day):
        year -= 1
    demographic_data['생년월일'][i] = year

#data preprocessing
demographic_data.columns = ["ID", "age", "gender", 'ht', 'wt', 'marriage', 'job','smkHx','drinkHx','suicideHx','suicide_need_in_month','medication_in_month']
demographic_data['marriage'][(demographic_data['marriage'] == 'Y')] = 1
demographic_data['marriage'][(demographic_data['marriage'] != 1)] = 0
demographic_data['job'][(demographic_data['job'] == 'Y')] = 1
demographic_data['job'][(demographic_data['job'] != 1)] = 0
demographic_data['smkHx'][(demographic_data['smkHx'] == 'Y')] = 1
demographic_data['smkHx'][(demographic_data['smkHx'] != 1)] = 0
demographic_data['drinkHx'][(demographic_data['drinkHx'] == 'Y')] = 1
demographic_data['drinkHx'][(demographic_data['drinkHx'] != 1)] = 0
demographic_data['suicideHx'][(demographic_data['suicideHx'] == 'Y')] = 1
demographic_data['suicideHx'][(demographic_data['suicideHx'] != 1)] = 0
demographic_data['gender'][(demographic_data['gender'] == 'M')] = 1
demographic_data['gender'][(demographic_data['gender'] == 'F')] = 0
demographic_data['suicide_need_in_month'][(demographic_data['suicide_need_in_month'] == 'Y')] = 1
demographic_data['suicide_need_in_month'][(demographic_data['suicide_need_in_month'] != 1)] = 0
demographic_data['medication_in_month'][(demographic_data['medication_in_month'] == 'Y')] = 1
demographic_data['medication_in_month'][(demographic_data['medication_in_month'] != 1)] = 0

#data save to feather
demographic_data.to_feather("data/processed/demographic_data.feather")


