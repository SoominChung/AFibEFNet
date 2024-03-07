import numpy as np
from sklearn import metrics

def roc_result(total_truth,total_prediction):
    fpr, tpr, thresholds = metrics.roc_curve(np.array(total_truth),np.array(total_prediction), pos_label=1)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]    
    return fpr,tpr,best_thresh,ix

def get_pred_label(total_prediction,threshold):
    total_pred_label = []
    for i in total_prediction:
        if i > threshold:
            total_pred_label.append(1)
        else:
            total_pred_label.append(0)
    return total_pred_label

def calc_metrics(total_truth, total_prediction, total_pred_label, round=3):

    cf_mat = metrics.confusion_matrix(total_truth, total_pred_label, labels=[0, 1])  
    TN = cf_mat[0][0]
    FP = cf_mat[0][1]
    FN = cf_mat[1][0]
    TP = cf_mat[1][1]
    sensitivity = round(TP/(TP+FN), round)
    specificity = round(TN/(FP+TN), round)
    PPV = round(TP/(TP+FP), round)
    NPV = round(TN/(FN+TN), round)
    precision, recall, _ = metrics.precision_recall_curve(total_truth, total_prediction)          
    acc = round(metrics.accuracy_score(total_truth,total_pred_label))
    f1 = round(metrics.f1_score(total_truth,total_pred_label),round)
    auprc = round(metrics.average_precision_score(total_truth,total_prediction),round)

    return sensitivity,specificity,PPV,NPV,precision,recall,acc,f1,auprc
