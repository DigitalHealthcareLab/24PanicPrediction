import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import auc
from scipy import stats
from sklearn.metrics import roc_curve
from scipy import interp
import pickle

def load_data(path):
    # Load data from a file
    with open(path, 'rb') as f:
        y_trues, y_scores = pickle.load(f)
    return y_trues, y_scores



def save_data(y_trues, y_scores, file_name, save_time):
    save_path = f'data/model_results/{file_name}/roc_data_{save_time}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump((y_trues, y_scores), f)
    
    
def plot_roc_curve_with_ci(y_trues, y_scores, file_name, save_time, n_splits=5,):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Compute ROC curve for each fold
    for i in range(n_splits):
        fpr, tpr, _ = roc_curve(y_trues[i], y_scores[i][:,1])
        roc_auc = auc(fpr, tpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)

    # Calculate mean and standard deviation for ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)

    # Compute confidence interval
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    # Plot ROC curve
    plt.plot(mean_fpr, mean_tpr, lw = 2, color="black", label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc))
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='grey', alpha=.7)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="lower right")
    plt.savefig('data/model_results/'+ file_name + '/' + 'roc_curve_with_ci' + save_time +'.png', dpi=300)
    plt.close()
    
def plot_roc_curve_averagefold(y_test_aggr, predict_proba_aggr, roc_auc_scores_mean, title, save_path):
    # Plot average ROC curve across folds
    fpr, tpr, thresholds = roc_curve(y_test_aggr, np.array(predict_proba_aggr)[:, 0], pos_label=0)
    lw = 2
    plt.figure(figsize=(10,8))
    plt.plot(fpr, tpr, color="#1f77b4", lw=lw, linestyle="--",
                    label="ROC curve (area = {0:0.3f})".format(roc_auc_scores_mean))

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
    plt.title(title)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.legend(loc="lower right")

    # Ensure directory exists and save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    
def calculate_performance_metrics_totalfold(accuracy_scores, roc_auc_scores, precision_scores, recall_scores, f1_scores):
    # Calculate mean and 95% confidence intervals for performance metrics
    accuracy_mean, accuracy_lowci, accuracy_highci = calculate_confidence_interval(accuracy_scores)
    roc_auc_mean, roc_auc_lowci, roc_auc_highci = calculate_confidence_interval(roc_auc_scores)
    precision_mean, precision_lowci, precision_highci = calculate_confidence_interval(precision_scores)
    recall_mean, recall_lowci, recall_highci = calculate_confidence_interval(recall_scores)
    f1_mean, f1_lowci, f1_highci = calculate_confidence_interval(f1_scores)
    
    # Print aggregated results with confidence intervals
    print('<Aggregated Results>')
    print(f"Accuracy: {accuracy_mean:.3f} (95% CI {accuracy_lowci:.3f}-{accuracy_highci:.3f})")
    print(f"ROC AUC Score: {roc_auc_mean:.3f} (95% CI {roc_auc_lowci:.3f}-{roc_auc_highci:.3f})")
    print(f"Precision Score: {precision_mean:.3f} (95% CI {precision_lowci:.3f}-{precision_highci:.3f})")
    print(f"Recall Score: {recall_mean:.3f} (95% CI {recall_lowci:.3f}-{recall_highci:.3f})")
    print(f"F1 Score: {f1_mean:.3f} (95% CI {f1_lowci:.3f}-{f1_highci:.3f})")
    print()

    return accuracy_mean, accuracy_lowci, accuracy_highci, roc_auc_mean, roc_auc_lowci, roc_auc_highci, precision_mean, precision_lowci, precision_highci, recall_mean, recall_lowci, recall_highci, f1_mean, f1_lowci, f1_highci


def calculate_performance_metrics_infold(model, y_test, predicted, predict_proba, accuracy_scores, roc_auc_scores, precision_scores, recall_scores, f1_scores):
    # Calculate accuracy, ROC AUC, precision, recall, and F1 score, then store in respective lists
    accuracy = accuracy_score(y_test, predicted)
    accuracy_scores.append(accuracy)

    roc_auc = roc_auc_score(y_test, predict_proba[:, 1]) 
    roc_auc_scores.append(roc_auc)

    precision = precision_score(y_test, predicted)
    precision_scores.append(precision)

    recall = recall_score(y_test, predicted)
    recall_scores.append(recall)

    f1 = f1_score(y_test, predicted)
    f1_scores.append(f1)
    
    # Print performance results for the fold
    print('<', model.__class__.__name__, '-test>')
    print(f'Confusion Matrix: {confusion_matrix(y_test, predicted)}')
    print(f'Mean Accuracy Score: {accuracy:.4}')
    print(f'ROC AUC Score: {roc_auc:.3}')
    print(f'Precision Score: {precision:.3}')
    print(f'Recall Score: {recall:.3}')
    print(f'F1 Score: {f1:.3}')
    print()
    
    return accuracy, roc_auc, precision, recall, f1
    
def calculate_confidence_interval(data, confidence=0.95):
    # Calculate mean and 95% confidence interval
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean - margin, mean + margin


def find_optimal_thresholds(y_true, predict_proba):
    # Calculate the optimal threshold for positive class based on ROC curve and Youden's J statistic
    positive_proba = predict_proba[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true, positive_proba)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def calculate_performance_metrics_with_thresholds(model_name, y_true, predict_proba, optimal_thresholds):
    # Calculate and print performance metrics using optimal thresholds
    predicted_optimal = [np.argmax([predict_proba[i, j] >= optimal_thresholds[j] for j in range(3)]) for i in range(len(predict_proba))]
    calculate_performance_metrics(model_name, y_true, predicted_optimal, predict_proba)

def calculate_performance_metrics(model_name, y_true, predicted, predict_proba):
    # Calculate and print accuracy, ROC AUC, precision, recall, and F1 scores for each class
    y_true_binarized = label_binarize(y_true, classes=[0, 1, 2])
    accuracy = accuracy_score(y_true, predicted)
    roc_auc_macro = roc_auc_score(y_true_binarized, predict_proba, multi_class='ovr', average='macro')
    roc_auc_classes = roc_auc_score(y_true_binarized, predict_proba, multi_class='ovr', average=None)
    precision_macro = precision_score(y_true, predicted, average='macro')
    precision_classes = precision_score(y_true, predicted, average=None)
    recall_macro = recall_score(y_true, predicted, average='macro')
    recall_classes = recall_score(y_true, predicted, average=None)
    f1_macro = f1_score(y_true, predicted, average='macro')
    f1_classes = f1_score(y_true, predicted, average=None)
    
    print(f'{model_name} Performance:')
    print(f'Accuracy Score: {accuracy:.3f}')
    print(f'ROC-AUC Score (Macro): {roc_auc_macro:.3f}')
    print(f'ROC-AUC Score (by Class*): {" / ".join([f"{x:.3f}" for x in roc_auc_classes])}')
    print(f'Precision Score (Macro): {precision_macro:.3f}')
    print(f'Precision Score (by Class*): {" / ".join([f"{x:.3f}" for x in precision_classes])}')
    print(f'Recall Score (Macro): {recall_macro:.3f}')
    print(f'Recall Score (by Class*): {" / ".join([f"{x:.3f}" for x in recall_classes])}')
    print(f'F1 Score (Macro): {f1_macro:.3f}')
    print(f'F1 Score (by Class*): {" / ".join([f"{x:.3f}" for x in f1_classes])}')
    print()

def plot_roc_curve(y_test, y_pred, title, path):
    n_classes = len(np.unique(y_test))
    y_test = label_binarize(y_test, classes=np.arange(n_classes))

  # Compute ROC curve and ROC area for ea|ch class
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], threshold[i] = metrics.roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        optimal_index = np.argmax(tpr[i] - fpr[i])
        optimal_threshold = threshold[i][optimal_index]
        print(f'Optimal threshold for class{i}')
        print(f': {optimal_threshold}')

  # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

  # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

  # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  # Plot all ROC curves
    plt.figure(figsize=(10,8))
    lw = 2

    colors = cycle(["gray", "green", "blue", "yellow", "red",'black','brown','goldenrod','gold',
                    'aqua','violet','darkslategray','mistyrose','darkorange','tan'])
    
    plt.plot(fpr[0], tpr[0], color="#1f77b4", lw=lw, linestyle="--",
                 label="ROC curve of class 'SD' (area = {0:0.3f})".format(roc_auc[0]))
    plt.plot(fpr[1], tpr[1], color="#ff7f0e", lw=lw, linestyle="--",
                 label="ROC curve of class 'DBP' (area = {0:0.3f})".format(roc_auc[1]))

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300)
    
    
def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
        
def preprocess_before_modelling_with_val(df): 
    df.drop(['date'], axis=1, inplace=True)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test_set_by_record(df)
    X_train.drop(['ID'], axis=1, inplace=True)
    X_val.drop(['ID'], axis=1, inplace=True)
    X_test.drop(['ID'], axis=1, inplace=True)
    
    ss = StandardScaler()
    scaled_X_train = ss.fit_transform(X_train)
    scaled_X_val = ss.transform(X_val)
    scaled_X_test = ss.transform(X_test)

    mms = MinMaxScaler()
    scaled_X_train = mms.fit_transform(scaled_X_train)
    scaled_X_val = mms.transform(scaled_X_val)
    scaled_X_test = mms.transform(scaled_X_test)
    return X_train, X_val, X_test, scaled_X_train, scaled_X_val, scaled_X_test, y_train, y_val, y_test

def split_train_val_test_set_by_record(df):
    y=df['panic']
    df.drop(['panic'], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=12345)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, stratify=y_train, random_state=12345)
    print('<train set : label proportion>')
    print(y_train.value_counts())
    print('\n<valid set: label proportion>')
    print(y_val.value_counts())
    print('\n<test set: label proportion>')
    print(y_test.value_counts())
    return X_train, X_val, X_test, y_train, y_val, y_test
  
  
  
def plot_roc_curve_3class(y_test, y_pred, title, path):
    n_classes = len(np.unique(y_test))
    y_test = label_binarize(y_test, classes=np.arange(n_classes))

    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], threshold[i] = metrics.roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(10, 8))
    lw = 2

    class_names = ['SD', 'DBP', 'DP']
    class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=class_colors[i], lw=lw, linestyle="--",
                 label="ROC curve of class {0} (area = {1:0.3f})".format(class_names[i], roc_auc[i]))

    plt.plot(fpr["macro"], tpr["macro"], color="purple", linestyle="-.", linewidth=0.7,
             label="macro-average ROC curve (area = {0:0.3f})".format(roc_auc["macro"]))

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300)
