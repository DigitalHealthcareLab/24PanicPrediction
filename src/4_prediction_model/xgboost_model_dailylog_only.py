import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime
import os
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from functools import partial
from xgboost import XGBClassifier
from sklearn.utils import class_weight
from utils import createDirectory, calculate_performance_metrics_infold, calculate_performance_metrics_totalfold, plot_roc_curve_with_ci, save_data

# Initialize empty lists for data aggregation across folds
ix_training, ix_test = [], []
predicted_aggr = []
predict_proba_aggr = []
y_test_aggr = []
SHAP_values_per_fold = []
scaled_X_test_aggr = []
shap_values = [[] for _ in range(3)]

# Initialize lists for performance metrics
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

# Define cross-validation function for XGBoost with given hyperparameters
def XGB_cv(max_depth, learning_rate, n_estimators,  X_train, X_val, y_train, y_val,
           colsample_bytree, min_child_weight):
    model = XGBClassifier(max_depth=int(max_depth),
                          objective='binary:logistic',
                          learning_rate=learning_rate,
                          n_estimators=int(n_estimators),
                          colsample_bytree=colsample_bytree,
                          eval_metric='logloss',
                          early_stopping_rounds=10,  
                          n_jobs=50,
                          min_child_weight = 3,
                          verbosity=0,
                          scale_pos_weight=1)  
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, sample_weight=classes_weights)
    predicted = model.predict(X_val)
    predict_proba = model.predict_proba(X_val)
    score = average_precision_score(y_val, predict_proba[:, 1])
    return score

# Define hyperparameter search bounds for Bayesian Optimization
pbounds = {'max_depth': (6, 9),
          'learning_rate': (0.00005, 0.001),
          'n_estimators': (100, 180),
          'colsample_bytree' :(0.2, 0.4),
          'min_child_weight': (2, 4)
          }

# Set up directory and timestamp for saving results
file_name = os.path.basename(__file__)
save_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")  
target = 'panic'  

# save model_results
createDirectory('data/model_results/' + file_name)
sys.stdout = open('data/model_results/' + file_name + '/' + save_time + '.txt', 'w')

# Load and preprocess dataset
df = pd.read_feather('data/processed/merged_df.feather')
df = df.drop_duplicates(subset=['ID', 'date'], keep='first')

# Define feature groups 
question = ['ACQ', 'APPQ_1', 'APPQ_2', 'APPQ_3', 'BSQ', 'BFNE', 'CES_D', 'GAD_7', 'KOSSSF', 'PHQ_9', 'SADS', 'STAI_X1', 'BRIAN', 'CSM', 'CTQ_1', 'CTQ_2', 'CTQ_3', 'CTQ_4', 'CTQ_5', 'KRQ', 'MDQ', 'SPAQ_1', 'SPAQ_2', 'STAI_X2',]
dailylog = ['alcohol', 'coffee', 'smoking', 'menstruation', 'exercise', 'positive_feeling', 'negative', 'positive_E', 'negative_E', 'anxiety', 'annoying', 'suicide_need', 'medication_in_month']
lifelog = ['HR_var', 'HR_max', 'HR_mean', 'HR_hvar_mean', 'steps_variance', 'steps_maximum', 'steps_mean', 'steps_hvar_mean', 'HR_acrophase', 'HR_amplitude', 'HR_mesor', 'HR_acrophase_difference', 'HR_acrophase_difference_2d',     'HR_amplitude_difference', 'HR_amplitude_difference_2d', 'HR_mesor_difference', 'HR_mesor_difference_2d', 'bandpower(0.001-0.0005Hz)', 'bandpower(0.0005-0.0001Hz)', 'bandpower(0.0001-0.00005Hz)', 'bandpower(0.00005-0.00001Hz)', 'sleep_duration', 'sleep_onset_time', 'sleep_out_time']
constant = [ 'age', 'gender', 'marriage', 'job', 'smkHx', 'drinkHx', 'suicideHx']
top_10 =['CTQ_2', 'CTQ_5', 'anxiety', 'STAI_X2', 'BRIAN', 'CTQ_4', 'age', 'KRQ', 'CTQ_1', 'SPAQ_1']
df = df[['ID', 'date', 'panic']+ dailylog]

# Set up Stratified KFold cross-validation
str_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y=df[target]
group = df['ID']

# Split data into training and testing sets for outer folds
for fold in str_kf.split(df, y):
    ix_training.append(fold[0]), ix_test.append(fold[1])
    
# Outer loop for model training and evaluation across folds
for i, (train_outer_ix, test_outer_ix) in enumerate(zip(ix_training, ix_test)):
    print('\n------ Fold Number:',i)
    X_train1, X_test = df.iloc[train_outer_ix], df.iloc[test_outer_ix]
    y_train1, y_test = y.iloc[train_outer_ix], y.iloc[test_outer_ix]
    str_kf_2 = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    y_2=X_train1[target]
    group = X_train1['ID']
    ex_training, ex_test = [], []

    # Split training data further for inner cross-validation
    for fold in str_kf_2.split(X_train1, y_2):
        ex_training.append(fold[0]), ex_test.append(fold[1])
    
    # Inner cross-validation to define validation set within training data
    for i, (train_inner_ix, test_inner_ix) in enumerate(zip(ex_training, ex_test)):
        pass 
    X_train, X_val = X_train1.iloc[train_inner_ix], X_train1.iloc[test_inner_ix]
    y_train, y_val = y_train1.iloc[train_inner_ix], y_train1.iloc[test_inner_ix]
    
    X_train = X_train.drop(['ID', 'date', target], axis=1)
    X_train1 = X_train1.drop(['ID', 'date', target], axis=1)
    X_val = X_val.drop(['ID', 'date', target], axis=1)
    X_test = X_test.drop(['ID', 'date', target], axis=1)

    # Standardize and normalize the data
    ss = StandardScaler()
    scaled_X_train = ss.fit_transform(X_train)
    scaled_X_train1 = ss.transform(X_train1)
    scaled_X_val = ss.transform(X_val)
    scaled_X_test = ss.transform(X_test)
    mms =  MinMaxScaler()
    scaled_X_train = mms.fit_transform(scaled_X_train)
    scaled_X_val = mms.transform(scaled_X_val)
    scaled_X_test = mms.transform(scaled_X_test)
    
    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    
    # Set up Bayesian Optimization for XGBoost hyperparameters
    XGB_cv_temp = partial(XGB_cv, X_train = scaled_X_train, y_train = y_train, X_val = scaled_X_val, y_val = y_val)
    xgboostBO = BayesianOptimization(f = XGB_cv_temp, pbounds = pbounds, verbose = False, random_state = 42 )
    xgboostBO.maximize(init_points=4, n_iter = 8)

    # Initialize XGBoost model with optimized hyperparameters 
    model = XGBClassifier(
    n_estimators=int(xgboostBO.max['params']['n_estimators']),
    max_depth=int(xgboostBO.max['params']['max_depth']),
    learning_rate=xgboostBO.max['params']['learning_rate'],
    colsample_bytree=xgboostBO.max['params']['colsample_bytree'],
    min_child_weight = int(xgboostBO.max['params']['min_child_weight']),
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=1,
    verbosity=0)  # 'silent' 대신 'verbosity' 사용

    # Train and evaluate the model on validation and test sets
    model.fit(scaled_X_train, y_train, eval_set=[(scaled_X_val,y_val)], sample_weight=classes_weights, verbose=False)
    predicted = model.predict(scaled_X_test)
    predict_proba = model.predict_proba(scaled_X_test)
    
    # Aggregate predictions and metrics
    y_trues.append(y_test)
    y_scores.append(predict_proba)
    predicted_aggr.extend(predicted)
    predict_proba_aggr.extend(predict_proba)
    y_test_aggr.extend(y_test)
    scaled_X_test_aggr.extend(scaled_X_test)
    
    # Calculate performance metrics for each fold
    accuracy, roc_auc, precision, recall, f1 = calculate_performance_metrics_infold(model, y_test, predicted, predict_proba, accuracy_scores, roc_auc_scores, precision_scores, recall_scores, f1_scores)

    explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
    shap_values_fold = explainer.shap_values(scaled_X_test, check_additivity=False)
    for SHAP_values in shap_values_fold:
        SHAP_values_per_fold.append(SHAP_values)

# Plot and save ROC curve with confidence intervals
plot_roc_curve_with_ci(y_trues, y_scores, file_name, save_time)
save_data(y_trues, y_scores, file_name, save_time)

# Calculate average and confidence intervals for performance metrics
accuracy_scores_mean, accuracy_scores_lowci, accuracy_scores_highci, roc_auc_scores_mean, roc_auc_scores_lowci, roc_auc_scores_highci,  prec_scores_mean, prec_scores_lowci, prec_scores_highci,  recall_scores_mean, recall_scores_lowci, recall_scores_highci, f1_scores_mean, f1_scores_lowci, f1_scores_highci = calculate_performance_metrics_totalfold(accuracy_scores, roc_auc_scores, precision_scores,  recall_scores, f1_scores)
    
# Generate index for reordering SHAP values
new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]


# Map prefix labels for different feature groups
prefix_map = {
    'question': '(C) ',
    'dailylog': '(D) ',
    'lifelog': '(L) ',
    'constant': '(S) '
}

# Add appropriate prefixes to each variable name in the dataframe
for col in df.columns:
    if col in question:
        df.rename(columns={col: prefix_map['question'] + col}, inplace=True)
    elif col in dailylog:
        df.rename(columns={col: prefix_map['dailylog'] + col}, inplace=True)
    elif col in lifelog:
        df.rename(columns={col: prefix_map['lifelog'] + col}, inplace=True)
    elif col in constant:
        df.rename(columns={col: prefix_map['constant'] + col}, inplace=True)
        
# Generate and save SHAP summary plots
fig_name = 'shap_'
df = df.drop(['ID', 'date', target], axis=1)
shap.summary_plot(np.array(SHAP_values_per_fold), df.reindex(new_index), max_display = 100, show=False)
plt.savefig('data/model_results/'+ file_name + '/' + fig_name + save_time +'.png', dpi=300)
plt.close()

# Generate and save top 20 features SHAP summary plot
fig_name = 'shap_top20_'
shap.summary_plot(np.array(SHAP_values_per_fold), df.reindex(new_index), max_display = 20, show=False)
plt.savefig('data/model_results/'+ file_name + '/' + fig_name + save_time +'.png', dpi=300)
plt.close()

# Generate and save bar plot for top 20 SHAP features
fig_name = 'shap_bar_top20_'
shap.summary_plot(np.array(SHAP_values_per_fold), df.reindex(new_index), max_display = 20, show=False, plot_type="bar")
plt.savefig('data/model_results/'+ file_name + '/' + fig_name + save_time +'.png', dpi=300)
plt.close()

# Generate and save top 10 features SHAP summary plot with adjusted size and no color bar
fig_name = 'shap_top10_'
shap.summary_plot(np.array(SHAP_values_per_fold), df.reindex(new_index), max_display = 10, show=False, color_bar=False, plot_size=(6.7, 6.5))
plt.savefig('data/model_results/'+ file_name + '/' + fig_name + save_time +'.png', dpi=300)
plt.close()