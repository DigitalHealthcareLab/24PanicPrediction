import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime
import os
import shap

from sklearn.model_selection import StratifiedKFold
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from functools import partial
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from utils import plot_roc_curve, plot_roc_curve_3class, createDirectory, find_optimal_thresholds, calculate_performance_metrics, calculate_confidence_interval, calculate_performance_metrics_infold, calculate_performance_metrics_totalfold, plot_roc_curve_averagefold, plot_roc_curve_with_ci, save_data

# parameter setting
file_name = os.path.basename(__file__)
save_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")  
target = 'panic'  

# save model_results
createDirectory('data/model_results/' + file_name)
sys.stdout = open('data/model_results/' + file_name + '/' + save_time + '.txt', 'w')

# loading data
df = pd.read_feather('data/processed/merged_df.feather')
df = df.drop_duplicates(subset=['ID', 'date'], keep='first')

def RFC_cv(n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features, X_train, y_train, X_val, y_val):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        min_samples_split=int(min_samples_split),
        max_features=max_features,
        oob_score=True,
        random_state=123456
    )
    model.fit(X_train, y_train)
    predicted = model.predict(X_val)
    score = f1_score(y_val, predicted, average='macro')
    return score

pbounds = {
    'n_estimators': (50, 200),
    'max_depth': (5, 8),
    'min_samples_leaf': (1, 4),
    'min_samples_split': (2, 5),
    'max_features': (0.3, 0.7)
}


df = df.drop_duplicates(subset=['ID', 'date'], keep='first')

# make empty lists
ix_training, ix_test = [], []
predicted_aggr = []
predict_proba_aggr = []
y_test_aggr = []
SHAP_values_per_fold = []
scaled_X_test_aggr = []
shap_values = [[] for _ in range(3)]

accuracy_scores = []
roc_auc_macro_scores = []
roc_auc_classes_0_scores = []
roc_auc_classes_1_scores = []
roc_auc_classes_2_scores = []
prec_macro_scores = []
prec_classes_0_scores = []
prec_classes_1_scores = []
prec_classes_2_scores = []
recall_macro_scores = []
recall_classes_0_scores = []
recall_classes_1_scores = []
recall_classes_2_scores = []
f1_macro_scores = []
f1_classes_0_scores = []
f1_classes_1_scores = []
f1_classes_2_scores = []
y_trues = []
y_scores = []
roc_auc_scores = []
precision_scores = []
recall_scores = [] 
f1_scores = []

# Stratified KFold
str_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y=df[target]
group = df['ID']
n_iter = 0


for fold in str_kf.split(df, y):
    ix_training.append(fold[0]), ix_test.append(fold[1])
    
# model training
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

                       
for i, (train_outer_ix, test_outer_ix) in enumerate(zip(ix_training, ix_test)): #-#-#
    #Verbose
    print('\n------ Fold Number:',i)
    X_train1, X_test = df.iloc[train_outer_ix], df.iloc[test_outer_ix]
    y_train1, y_test = y.iloc[train_outer_ix], y.iloc[test_outer_ix]
    str_kf_2 = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    y_2=X_train1[target]
    group = X_train1['ID']
    ex_training, ex_test = [], []
    for fold in str_kf_2.split(X_train1, y_2):
        ex_training.append(fold[0]), ex_test.append(fold[1])
        
    for i, (train_inner_ix, test_inner_ix) in enumerate(zip(ex_training, ex_test)):
        pass 
    X_train, X_val = X_train1.iloc[train_inner_ix], X_train1.iloc[test_inner_ix]
    y_train, y_val = y_train1.iloc[train_inner_ix], y_train1.iloc[test_inner_ix]
    
    
    X_train = X_train.drop(['ID', 'date', target], axis=1)
    X_train1 = X_train1.drop(['ID', 'date', target], axis=1)
    X_val = X_val.drop(['ID', 'date', target], axis=1)
    X_test = X_test.drop(['ID', 'date', target], axis=1)

    ss = StandardScaler()
    scaled_X_train = ss.fit_transform(X_train)
    scaled_X_train1 = ss.transform(X_train1)
    scaled_X_val = ss.transform(X_val)
    scaled_X_test = ss.transform(X_test)
    
    mms = MinMaxScaler()
    scaled_X_train = mms.fit_transform(scaled_X_train)
    scaled_X_val = mms.transform(scaled_X_val)
    scaled_X_test = mms.transform(scaled_X_test)
    
    # define model
    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train1)
    
    RFC_cv_temp = partial(RFC_cv, X_train=scaled_X_train, y_train=y_train, X_val=scaled_X_val, y_val=y_val)
    rfcBO = BayesianOptimization(f=RFC_cv_temp, pbounds=pbounds, verbose=2, random_state=42)
    rfcBO.maximize(init_points=5, n_iter=10)

    model = RandomForestClassifier(
    n_estimators=int(rfcBO.max['params']['n_estimators']),
    max_depth=int(rfcBO.max['params']['max_depth']),
    min_samples_leaf=int(rfcBO.max['params']['min_samples_leaf']),
    min_samples_split=int(rfcBO.max['params']['min_samples_split']),
    max_features=rfcBO.max['params']['max_features'],
    oob_score=True,
    random_state=123456)

    #run model
    model.fit(scaled_X_train1, y_train1, sample_weight=classes_weights)
    predicted = model.predict(scaled_X_test)
    predict_proba = model.predict_proba(scaled_X_test)
    
    y_trues.append(y_test)
    y_scores.append(predict_proba)
    
    predicted_aggr.extend(predicted)
    predict_proba_aggr.extend(predict_proba)
    y_test_aggr.extend(y_test)
    scaled_X_test_aggr.extend(scaled_X_test)
    
    accuracy, roc_auc, precision, recall, f1 = calculate_performance_metrics_infold(model, y_test, predicted, predict_proba, accuracy_scores, roc_auc_scores, precision_scores, recall_scores, f1_scores)

    explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
    shap_values_fold = explainer.shap_values(scaled_X_test, check_additivity=False)
    for SHAP_values in shap_values_fold:
        SHAP_values_per_fold.append(SHAP_values)

    
plot_roc_curve_with_ci(y_trues, y_scores, file_name, save_time)
save_data(y_trues, y_scores, file_name, save_time)

# 평균 및 신뢰 구간 계산 및 출력
accuracy_scores_mean, accuracy_scores_lowci, accuracy_scores_highci, roc_auc_scores_mean, roc_auc_scores_lowci, roc_auc_scores_highci,  prec_scores_mean, prec_scores_lowci, prec_scores_highci,  recall_scores_mean, recall_scores_lowci, recall_scores_highci, f1_scores_mean, f1_scores_lowci, f1_scores_highci = calculate_performance_metrics_totalfold(accuracy_scores, roc_auc_scores, precision_scores,  recall_scores, f1_scores)
    
   
new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]
fig_name = 'shap_'
df = df.drop(['ID', 'date', target], axis=1)
shap.summary_plot(np.array(SHAP_values_per_fold), df.reindex(new_index), max_display = 100, show=False)
plt.savefig('data/model_results/'+ file_name + '/' + fig_name + save_time +'.png', dpi=300)
plt.close()


fig_name = 'shap_top20_'
shap.summary_plot(np.array(SHAP_values_per_fold), df.reindex(new_index), max_display = 20, show=False)
plt.savefig('data/model_results/'+ file_name + '/' + fig_name + save_time +'.png', dpi=300)
plt.close()


fig_name = 'shap_bar_top20_'
shap.summary_plot(np.array(SHAP_values_per_fold), df.reindex(new_index), max_display = 20, show=False, plot_type="bar")
plt.savefig('data/model_results/'+ file_name + '/' + fig_name + save_time +'.png', dpi=300)
plt.close()