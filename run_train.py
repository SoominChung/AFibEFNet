
import numpy as np
import os
import torch
import torch.optim as optim 
from torch import nn
from sklearn import metrics
import calculate_metrics
import store_result
from utils import save_pickle


def run_train_valid(model,train_loader, valid_loader, epoch, parameter_dict,BEST_dict):
    
    ############################################# Get Parameter
    learning_rate = parameter_dict['learning_rate']
    weight_decay = parameter_dict['weight_decay']
    add_clin_feature = parameter_dict['add_clin_feature']
    device = parameter_dict['device']
    RESULT_PATH = parameter_dict['RESULT_PATH']
    BEST_LOSS_v = BEST_dict['BEST_LOSS_v']
    BEST_ACC_v = BEST_dict['BEST_ACC_v']
    BEST_AUC_v = BEST_dict['BEST_AUC_v']
    BEST_AUPRC_v = BEST_dict['BEST_AUPRC_v']
    BEST_F1_v = BEST_dict['BEST_F1_v']

    ############################################# Preparation
    total_truth,total_loss,total_prediction = [],[],[]
    model.train()
    criterion = nn.BCELoss()
    lr = learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    
    ######################################### Train
    for data in train_loader:

        if add_clin_feature:
            inputs, labels, features = data
            inputs, labels, features = inputs.to(device,dtype=torch.float), labels.to(device,dtype=torch.float), features.to(device,dtype=torch.float) 
        else:
            inputs, labels, _ = data
            inputs, labels = inputs.to(device,dtype=torch.float), labels.to(device,dtype=torch.float) 
        
        optimizer.zero_grad() 

        if add_clin_feature:
            output = model(inputs, features).to(device, dtype=torch.float) 
        else:
            output = model(inputs).to(device, dtype=torch.float) 

        truth = (labels.detach().cpu().numpy())
        output = torch.reshape(output,(output.shape[0],1))
        labels = torch.reshape(labels,(labels.shape[0],1))        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        output = (output.detach().cpu().numpy())
        total_truth.extend(truth)
        total_prediction.extend(output)
        total_loss.append(loss.data.item())

    ######################################### Calculate metrics
    fpr,tpr,best_thresh,_ = calculate_metrics.roc_result(total_truth,total_prediction)
    total_pred_label = calculate_metrics.get_pred_label(total_prediction,best_thresh)

    train_auc = round(metrics.auc(fpr,tpr),3)
    train_loss = round(np.average(total_loss),3)
    _,_,_,_,_,_,train_acc,train_f1,train_auprc = calculate_metrics.calc_metrics(total_truth, total_prediction, total_pred_label)

    one_epoch_train_result = {'acc':train_acc,'auc':train_auc,'loss':train_loss, 'f1':train_f1,'auprc':train_auprc}

    ######################################### Validation
    model.eval()
    criterion = nn.BCELoss()

    total_prediction,total_truth,total_loss,total_positive_predict_proba = [],[],[],[]
    with torch.no_grad():
        for data in valid_loader:

            if add_clin_feature:
                inputs, labels, features = data
                inputs, labels, features = inputs.to(device,dtype=torch.float), labels.to(device,dtype=torch.float), features.to(device,dtype=torch.float) 
            else:
                inputs, labels, _ = data
                inputs, labels = inputs.to(device,dtype=torch.float), labels.to(device,dtype=torch.float) 
            
            optimizer.zero_grad() 

            if add_clin_feature:
                output = model(inputs, features).to(device, dtype=torch.float) 
            else:
                output = model(inputs).to(device, dtype=torch.float) 

            truth = (labels.detach().cpu().numpy())       
            output = torch.reshape(output,(output.shape[0],1))
            labels = torch.reshape(labels,(labels.shape[0],1))
            loss =criterion(output, labels)

            positive_predict_proba = output
            positive_predict_proba = (positive_predict_proba.detach().cpu().numpy())
            total_truth.extend(truth)
            output = (output.detach().cpu().numpy())
            total_prediction.extend(output)
            total_loss.append(loss.data.item())
            total_positive_predict_proba.extend(positive_predict_proba)

        ######################################### Calculate metrics
        fpr,tpr,best_thresh,ix = calculate_metrics.roc_result(total_truth,total_prediction)
        valid_thresholds = {'best_thresh':best_thresh}          
        total_pred_label = calculate_metrics.get_pred_label(total_prediction,best_thresh)

        valid_auc = round(metrics.auc(fpr,tpr),3)
        valid_loss = round(np.average(total_loss),3)
        valid_sens,valid_spec,valid_ppv,valid_npv,valid_precision,valid_recall,valid_acc,valid_f1,valid_auprc = calculate_metrics.calc_metrics(total_truth, total_prediction, total_pred_label)

        if epoch % 20 ==0:
            print(f'valid f1 : {valid_f1}')
            print(f'valid auc : {valid_auc}')
            print(f'valid acc : {valid_acc}')
            print(f'valid auprc : {valid_auprc}')

        one_epoch_valid_result = {'acc':valid_acc,'auc':valid_auc,'loss':valid_loss,'f1':valid_f1,'auprc':valid_auprc,'sens':valid_sens,'spec':valid_spec,
                                  'ppv':valid_ppv,'npv':valid_npv}

        if valid_f1 > BEST_F1_v: 
            BEST_F1_v = valid_f1
            BEST_AUC_v = valid_auc
            BEST_ACC_v = valid_acc
            BEST_LOSS_v = np.round(valid_loss,3)
            BEST_AUPRC_v = valid_auprc
            BEST_sens = valid_sens
            BEST_spec = valid_spec
            BEST_ppv = valid_ppv
            BEST_npv = valid_npv
            BEST_dict= {'BEST_AUC_v':BEST_AUC_v, 'BEST_ACC_v':BEST_ACC_v, 'BEST_LOSS_v':BEST_LOSS_v, 'BEST_F1_v':BEST_F1_v,
                        'BEST_AUPRC_v':BEST_AUPRC_v,'BEST_sens':BEST_sens,'BEST_spec':BEST_spec,'BEST_ppv':BEST_ppv,'BEST_npv':BEST_npv}

            save_pickle(data=valid_thresholds,path=os.path.join(RESULT_PATH,'roc_threshold.pkl')) 
            store_result.make_roc_curve(fpr,tpr,valid_auc,ix,best_thresh,BEST_sens,BEST_spec,RESULT_PATH,title='valid')
            store_result.make_prc_curve(valid_recall,valid_precision,valid_auprc,RESULT_PATH,title='valid')
            store_result.make_confusion_matrix(total_truth,total_pred_label,RESULT_PATH,title='valid')
     
        return one_epoch_train_result,one_epoch_valid_result,BEST_dict



def run_test(model, test_loader, add_clin_feature, device, valid_best_threshold, RESULT_FOLD):
    
    model.eval()
    criterion = nn.BCELoss()

    total_truth,total_loss,total_prediction,total_positive_predict_proba = [],[],[],[]
    with torch.no_grad():
        for data in test_loader:

            if add_clin_feature:
                inputs, labels, features = data
                inputs, labels, features = inputs.to(device,dtype=torch.float), labels.to(device,dtype=torch.float), features.to(device,dtype=torch.float) 
            else:
                inputs, labels, _ = data
                inputs, labels = inputs.to(device,dtype=torch.float), labels.to(device,dtype=torch.float) 
        
            if add_clin_feature:
                output = model(inputs, features).to(device, dtype=torch.float) 
            else:
                output = model(inputs).to(device, dtype=torch.float) 
     
            truth = (labels.detach().cpu().numpy())
            output = torch.reshape(output,(output.shape[0],1))
            labels = torch.reshape(labels,(labels.shape[0],1))            
            loss =criterion(output, labels)

            positive_predict_proba = output
            positive_predict_proba = (positive_predict_proba.detach().cpu().numpy())
            total_truth.extend(truth)
            output = (output.detach().cpu().numpy())
            total_prediction.extend(output)
            total_loss.append(loss.data.item())
            total_positive_predict_proba.extend(positive_predict_proba)
        
        fpr, tpr, thresholds = metrics.roc_curve(np.array(total_truth),np.array(total_positive_predict_proba), pos_label=1)
        best_thresh = valid_best_threshold
        ix = (np.abs(thresholds-best_thresh)).argmin()
        total_pred_label = calculate_metrics.get_pred_label(total_prediction,best_thresh)
        
        test_auc = round(metrics.auc(fpr,tpr),3)
        test_loss = round(np.average(total_loss),3)
        test_sens,test_spec,test_ppv,test_npv,test_precision,test_recall,test_acc,test_f1,test_auprc = calculate_metrics.calc_metrics(total_truth, total_prediction, total_pred_label)

    FINAL_RESULT = {'FINAL_AUC':test_auc,'FINAL_ACC':test_acc,'FINAL_LOSS':test_loss,
                    'FINAL_F1_SCORE':test_f1,'FINAL_SENS':test_sens,'FINAL_SPEC':test_spec,
                    'FINAL_PPV':test_ppv,'FINAL_NPV':test_npv,'FINAL_AUPRC':test_auprc}
    store_result.make_confusion_matrix(total_truth,total_pred_label,RESULT_FOLD,title='test')
    store_result.make_roc_curve(fpr,tpr,test_auc,ix,best_thresh,test_sens,test_spec,RESULT_FOLD,title='test')
    store_result.make_prc_curve(test_recall,test_precision,test_auprc,RESULT_FOLD,title='test')

    return total_pred_label, total_truth,total_positive_predict_proba,FINAL_RESULT

