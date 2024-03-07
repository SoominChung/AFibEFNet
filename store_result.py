
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import torch
import warnings
warnings.filterwarnings(action='ignore')

def save_to_csv(classification_result, title):
    
    df = pd.DataFrame(classification_result).transpose()
    df.to_csv(title+".csv")


def make_calibration_plot(total_train,total_valid,label,path):
    plt.plot(range(len(total_train)), total_train, 'b', range(len(total_valid)), total_valid, 'r') #각 epoch에서의 acc
    blue_patch = mpatches.Patch(color='blue', label=f'Train {label}')
    red_patch = mpatches.Patch(color='red', label=f'Validation {label}')
    plt.legend(handles=[red_patch, blue_patch])
    a =plt.xlabel('Epochs',color='black')
    a =plt.ylabel(f'{label}',color='black')
    plt.savefig(path+f'/[{label}] train valid.png',dpi=200)
    plt.close()       


def make_roc_curve(fpr, tpr, auc,ix,best_thresh,sens,spec, RESULT_FOLD,title,line_str,is_test=False):

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='red', alpha=0.7, linewidth=1.3, label='{} (AUROC={})'.format(line_str, np.round(auc, 3)))
    plt.plot([0, 1], [0, 1], color='lightslategrey', linestyle=':', )
    
    plt.scatter(fpr[ix], tpr[ix], marker='+', s=100, color='r', 
            label='Best threshold = %.3f, \nSensitivity = %.3f, \nSpecificity = %.3f' % (best_thresh, sens, spec))
    plt.title(title)
    plt.xlabel('1 - specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right', edgecolor='white')
    plt.grid(False)
    plt.savefig(RESULT_FOLD+f'/[{title}]ROC curve.jpg',dpi=300)
    plt.close() 

def make_prc_curve(recall, precision, auprc, RESULT_FOLD,title,line_str,is_test=False):

    plt.figure(figsize=(6,6))
    plt.plot(recall, precision, color='red', alpha=0.7, linewidth=1.3, label='{} (AUPRC={})'.format(line_str, np.round(auprc, 3)))

    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower right', edgecolor='white')
    plt.savefig(RESULT_FOLD+f'/[{title}]PRC curve.jpg',dpi=300)
    plt.close()

def make_confusion_matrix(y,output,RESULT_FOLD,title):

    confusion = confusion_matrix(y,output) 
    TN = confusion[0][0]
    FP = confusion[0][1]
    FN = confusion[1][0]
    TP = confusion[1][1]
    sens = round(TP/(TP+FN), 3)
    spec = round(TN/(FP+TN), 3)
    PPV = round(TP/(TP+FP), 3)
    NPV = round(TN/(FN+TN), 3)

    labels = ['class 0', 'class 1']
    fig = plt.figure()
    fig.patch.set_facecolor('xkcd:white')
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion, cmap=plt.cm.Blues)
    cbar = fig.colorbar(cax)
    cbar.ax.yaxis.set_tick_params(labelcolor='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    a =ax.set_xticklabels([''] + labels,color='black')
    a =ax.set_yticklabels([''] + labels,color='black')
    fmt = 'd' 
    thresh = confusion.max()
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, format(confusion[i,j],fmt),
            ha="center",va="center",color="white" if confusion[i,j]>=thresh else "black")
    plt.xlabel('Predicted',color='black')
    plt.ylabel('Expected',color='black')
    plt.savefig(RESULT_FOLD+f'/{title} Confusion Matrix.png',dpi=300)
    plt.close()
    precision = precision_score(y,output)
    recall = recall_score(y,output)
    f1_score_value = f1_score(y,output) 
    f = open(RESULT_FOLD+f'/{title}_Result.txt','w')
    f.write(f'TN {TN} FP {FP} FN {FN} TP {TP}\n')
    f.write(f'precision {precision} recall {recall} f1 {f1_score_value}\n')
    f.write(f'sens {sens} spec {spec} PPV {PPV} NPV {NPV} \n ')
    f.close()


def save_checkpoint(save_path, model, optimizer, valid_loss):

    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    #print(f'Model saved to ==> {save_path}')
    print('model save')


def load_checkpoint(load_path, model, optimizer,device):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    #print(f'Model loaded from <== {load_path}')
    print('model load')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']
