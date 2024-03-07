import argparse
import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import model as models
import EF.code.to_github.run_train as run_train
import store_result
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path','-p',type=str, default='/home/AFEF/result',
                        help='result folder path')
    parser.add_argument('--data_path','-dat',type=str, default='/home/AFEF/data',
                        help='data folder path')
    parser.add_argument('--device_num', '-d',type=int, default=7,
                        help='GPU device number')
    parser.add_argument('--add_clin_feature','-f',type=bool, action='store_true',
                        help='whether clinical features are included')
    parser.add_argument('--what_model','-m',type=str, default='AFibEFNet',
                        help='model name')
    parser.add_argument('--ef_threshold', '-ef',type=int, default=50)
    args    = parser.parse_args()

    device_num = args.device_num 
    device = torch.device(f'cuda:{device_num}')
    
    result_path = os.path.join(args.result_path,f'EF{str(args.ef_threshold)}',str(args.add_clin_feature),args.what_model)
    
    ################################################## Parameters you want to use
    drop_rate = 0.3
    learning_rate = 1e-03
    weight_decay = 1e-05
    batch_size = 32
    epochs = 40
    
    ################################################## Get Data
    x_test = np.load(os.path.join(args.data_path,'after_2021','ecg.npy'))
    y_test = np.load(os.path.join(args.data_path,'after_2021',f'label{args.ef_threshold}.npy'))
    if args.add_clin_feature:
        feature_test = np.load(os.path.join(args.data_path,'after_2021',f'normalization_feature{args.add_clin_feature}.npy'))
    if args.what_model =='lstm':
        x_test = np.transpose(x_test, (0, 2, 1))
    test_ecgs = torch.Tensor(x_test)
    test_targets = torch.Tensor(y_test)
    if args.add_clin_feature:
        test_features = torch.Tensor(feature_test)       
    else:
        test_features = None
    test_data = ECGDataset(test_ecgs, test_targets, test_features,args.add_clin_feature)  
    test_loader = DataLoader(test_data, batch_size=test_ecgs.shape[0], shuffle=False, drop_last=False)    

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

    ################################################## Preparing for test
    parameter_option  = f'{args.ef_threshold}ef_{args.add_clin_feature}demo_{args.what_model}_{drop_rate}dr_{learning_rate}lr_{weight_decay}weight_decay_{batch_size}batch_size_{epochs}epoch'
    PARAM_PATH = os.path.join(result_path,parameter_option)  
    final_model_path = os.path.join(PARAM_PATH,'model.pt')
    best_model = (_model).to(device)
    optimizer = optim.Adam(best_model.parameters(), lr=float(learning_rate))
    store_result.load_checkpoint(final_model_path, best_model, optimizer,device=device)
    valid_thresholds = open_pickle(os.path.join(PARAM_PATH,'roc_threshold.pkl'))
    valid_best_threshold = valid_thresholds['best_thresh']
    RESULT_PATH = os.path.join(PARAM_PATH,'final_test_result')
    os.makedirs(RESULT_PATH,exist_ok=True) 
    
    ################################################## Test
    pred_test, truth_test, positive_predict_proba_test,test_result_dict= run_train.run_test(best_model,test_loader,add_clin_feature=args.add_clin_feature,valid_best_threshold=valid_best_threshold,
                                                                                            device=device,RESULT_FOLD=RESULT_PATH)


    save_pickle(data=pred_test,path=RESULT_PATH+'/pred_test.pkl')
    save_pickle(data=truth_test,path=RESULT_PATH+'/truth_test.pkl')
    save_pickle(data=positive_predict_proba_test,path=RESULT_PATH+'/positive_predict_proba_test.pkl')
    save_pickle(data=test_result_dict,path=RESULT_PATH+'/test_result.pkl')
    

if __name__ == "__main__":
    main()