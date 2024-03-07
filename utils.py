import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def open_pickle(path):
  try:
    with open(path,'rb') as f:
      data = pickle.load(f)
  except:
    print(path)
  return data

def save_pickle(path, data):
  with open(path,'wb') as f:
    pickle.dump(data,f)
    

def weights_init(m, what_init='xavier'):
    if what_init == 'xavier':
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
    if what_init == 'kaiming':
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight)        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)       
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.path.split('/')[-1] == '5':
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

        
class ECGDataset(Dataset):
    def __init__(self, x, y, features,add_clin_feature=False, transform=None):
        self.x = x
        self.y = y
        self.features = features
        self.add_clin_feature = add_clin_feature
        self.transform = transform
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)

        if self.add_clin_feature:
            feat = self.features[idx]
            return x,y,feat
        else:
            feat = None
            return x, y, feat
    