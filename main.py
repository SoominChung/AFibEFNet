import argparse
import itertools 
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import model as models
import run_train
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path','-p',type=str, default='/home/AFEF/result',
                        help='result folder path')
    parser.add_argument('--data_path','-dat',type=str, default='/home/AFEF/data',
                        help='data folder path')
    parser.add_argument('--device_num', '-d',type=int, default=7,
                        help='GPU device number')
    parser.add_argument('--epochs',type=int, default=40)
    parser.add_argument('--add_clin_feature','-f',type=bool, action='store_true',
                        help='whether clinical features are included')
    parser.add_argument('--what_model','-m',type=str, default='AFibEFNet',
                        help='model name')
    parser.add_argument('--ef_threshold', '-ef',type=int, default=50)
    args = parser.parse_args()

    device_num = args.device_num 
    device = torch.device(f'cuda:{device_num}')
    
    result_path = os.path.join(args.result_path,f'EF{str(args.ef_threshold)}',str(args.add_clin_feature),args.what_model)

    ###################### Parameter tuning with 5-fold validation
    params = {
        'drop_rate': [0.3,0.4,0.5],
        'learning_rate' : [1e-03,1e-04,1e-05],
        'weight_decay' : [1e-04,1e-05,1e-06],
        'batch_size' : [8,16,32,64]
    }
    params_combination = itertools.product(params['drop_rate'],params['learning_rate'],params['weight_decay'],params['batch_size'])
    params_combination_list = list(params_combination)
    print(f'Parameter Tuning : total {len(params_combination_list)}')
    param_idx = 0
    for drop_rate, learning_rate, weight_decay,batch_size in params_combination_list:
        param_idx +=1
        print(f'========================================={param_idx}th /{len(list(params_combination_list))}')
            
        parameter_option  = f'{args.ef_threshold}ef_{args.add_clin_feature}demo_{args.what_model}_{drop_rate}dr_{learning_rate}lr_{weight_decay}weight_decay_{batch_size}batch_size_{args.epochs}epoch'
        PARAM_PATH = os.path.join(result_path,parameter_option)      

        fold_f1,fold_sens,fold_spec,fold_PPV,fold_NPV,fold_auc,fold_auprc,fold_acc,fold_loss = [],[],[],[],[],[],[],[],[]
        for fold in range(0,5): 
            print('#########Fold {}'.format(fold + 1))
            K_FOLD_PATH = os.path.join(PARAM_PATH,f'{fold+1}')    
            RESULT_PATH = K_FOLD_PATH
            os.makedirs(RESULT_PATH, exist_ok=True) 
            
            ################################################## Get Data
            x_train = np.load(os.path.join(args.data_path,str(fold+1),'train','ecg.npy'))
            y_train = np.load(os.path.join(args.data_path,str(fold+1),'train',f'label{args.ef_threshold}.npy'))
            x_val = np.load(os.path.join(args.data_path,str(fold+1),'valid','ecg.npy'))
            y_val = np.load(os.path.join(args.data_path,str(fold+1),'valid',f'label{args.ef_threshold}.npy'))
            if args.add_clin_feature:
                feature_train = np.load(os.path.join(args.data_path,str(fold+1),'train',f'normalization_feature{args.add_clin_feature}.npy'))
                feature_val = np.load(os.path.join(args.data_path,str(fold+1),'valid',f'normalization_feature{args.add_clin_feature}.npy'))
            if args.what_model =='lstm':
                x_train = np.transpose(x_train, (0, 2, 1))
                x_val = np.transpose(x_val, (0, 2, 1))

            train_ecgs = torch.Tensor(x_train)
            train_targets = torch.Tensor(y_train)
            if args.add_clin_feature:  
                train_features = torch.Tensor(feature_train)
            else:
                train_features = None

            val_ecgs = torch.Tensor(x_val)
            val_targets = torch.Tensor(y_val)
            if args.add_clin_feature:  
                val_features = torch.Tensor(feature_val)
            else:
                val_features = None

            train_data = ECGDataset(train_ecgs, train_targets, train_features,args.add_clin_feature)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
            val_data = ECGDataset(val_ecgs, val_targets, val_features,args.add_clin_feature)
            valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=False)

            ################################################## Choose Model
            if args.what_model == 'AFibEFNet':
                _model = models.AFibEFNet.AFibEFNet(add_clin_feature=int(args.add_clin_feature),drop_rate=drop_rate)                                                                        
            elif args.what_model == 'resnet50':
                _model = models.ResNet.ResNet(layers=[3, 4, 6, 3],what_block='bottleneck',feature_num=args.add_clin_feature)
            elif args.what_model == 'efficientNet_b5':
                _model = models.EfficientNet.efficientnet_b5(feature_num=args.add_clin_feature)
            elif args.what_model == 'lstm':
                _model = models.LSTM.LSTMModel(input_dim=8, hidden_dim=128, layer_dim=1, output_dim=1,feature_num=args.add_clin_feature) 
            model = _model.to(device)
            weights_init(model,'xavier')

            ################################################## Preparing for training
            parameter_dict = {'what_model':args.what_model,'weight_decay':weight_decay,
                            'learning_rate':learning_rate,
                                'add_clin_feature':args.add_clin_feature,'device':device,'RESULT_PATH':RESULT_PATH}        
            total_loss_t,total_loss_v,total_auc_t,total_auc_v= [],[],[],[]
            BEST_AUC_v,BEST_ACC_v,BEST_F1_v,BEST_AUPRC_v,BEST_LOSS_v = 0,0,0,0,1e+12
            BEST_dict = {'BEST_AUC_v':BEST_AUC_v, 'BEST_ACC_v':BEST_ACC_v, 'BEST_LOSS_v':BEST_LOSS_v,'BEST_F1_v':BEST_F1_v,'BEST_AUPRC_v':BEST_AUPRC_v}
            early_stopping = EarlyStopping(path=os.path.join(RESULT_PATH,'checkpoint.pt'))
            
            ################################################## Train
            print('*** START EPOCH***')
            for epoch in range(args.epochs):
                print(f'# Epoch " {epoch+1}/{args.epochs}')
                one_epoch_train_result,one_epoch_valid_result,BEST_dict=run_train.run_train_valid(model,train_loader, valid_loader,
                                                                                              epoch,parameter_dict,BEST_dict)

                train_auc = one_epoch_train_result['auc']
                train_loss= one_epoch_train_result['loss']

                valid_acc = one_epoch_valid_result['acc']
                valid_auc = one_epoch_valid_result['auc']
                valid_loss= one_epoch_valid_result['loss']
                valid_f1=   one_epoch_valid_result['f1']
                valid_auprc=one_epoch_valid_result['auprc']
                valid_sens = one_epoch_valid_result['sens']
                valid_spec = one_epoch_valid_result['spec']
                valid_ppv = one_epoch_valid_result['ppv']
                valid_npv = one_epoch_valid_result['npv']
                
                total_loss_t.append(train_loss)
                total_loss_v.append(valid_loss)
                total_auc_t.append(train_auc)
                total_auc_v.append(valid_auc)

                early_stopping(valid_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break      
        
            run_train.make_calibration_plot(total_loss_t,total_loss_v,label='Loss',path=RESULT_PATH)
            run_train.make_calibration_plot(total_auc_t,total_auc_v,label='AUC',path=RESULT_PATH)

            fold_acc.append(valid_acc)
            fold_loss.append(valid_loss)
            fold_f1.append(valid_f1)
            fold_auc.append(valid_auc)
            fold_sens.append(valid_sens)
            fold_spec.append(valid_spec)
            fold_PPV.append(valid_ppv)
            fold_NPV.append(valid_npv)
            fold_auprc.append(valid_auprc)

        print(f'********************** total-fold valid result')
        fold_average_path1 = '5-fold-average'
        fold_average_path = os.path.join(PARAM_PATH,fold_average_path1)      
        os.makedirs(fold_average_path, exist_ok=True) 
        def round_val(val,round_n=3):
            return round(np.average(val),round_n),round(np.std(val),round_n)
                
        fold_acc_average,fold_acc_std = round_val(fold_acc)
        fold_loss_average,fold_loss_std = round_val(fold_loss)
        fold_f1_average,fold_f1_std = round_val(fold_f1)
        fold_auc_average,fold_auc_std = round_val(fold_auc)
        fold_sens_average,fold_sens_std = round_val(fold_sens)
        fold_spec_average,fold_spec_std = round_val(fold_spec)
        fold_PPV_average,fold_PPV_std = round_val(fold_PPV)
        fold_NPV_average,fold_NPV_std = round_val(fold_NPV)
        fold_auprc_average,fold_auprc_std = round_val(fold_auprc)
        fold_total_result_dict = {
                    'fold_acc_average' : fold_acc_average,  'fold_acc_std':fold_acc_std,            
                    'fold_loss_average' : fold_loss_average,  'fold_loss_std':fold_loss_std,            
                    'fold_f1_average' : fold_f1_average,  'fold_f1_std':fold_f1_std,            
                    'fold_auc_average' : fold_auc_average,  'fold_auc_std':fold_auc_std,            
                    'fold_sens_average' : fold_sens_average,  'fold_sens_std':fold_sens_std,        
                    'fold_spec_average' : fold_spec_average,  'fold_spec_std':fold_spec_std,        
                    'fold_PPV_average' : fold_PPV_average,  'fold_PPV_std':fold_PPV_std,            
                    'fold_NPV_average' : fold_NPV_average,  'fold_NPV_std':fold_NPV_std,            
                    'fold_auprc_average' : fold_auprc_average,  'fold_auprc_std':fold_auprc_std
                    }
        save_pickle(data=fold_total_result_dict, path = os.path.join(fold_average_path,'5_fold_total_result_dict.pkl'))         
        


if __name__ == "__main__":
    main()