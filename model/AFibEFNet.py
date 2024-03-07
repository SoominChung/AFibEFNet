import torch
import torch.nn as nn
import tools as T # pip install tools-jsyoo61

# %%
class ReduceBlock1(nn.Module): # stride size =1
    def __init__(self, in_channel=1, out_channel=64, kernel_size=5,drop_rate=0.5):
        super().__init__()
        layers=[
        nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Dropout(drop_rate)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ReduceBlock2(nn.Module): # stride size =2
    def __init__(self, in_channel=1, out_channel=64, kernel_size=11,drop_rate=0.5):
        super().__init__()
        layers=[
        nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Dropout(drop_rate)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        layers=[
        nn.Conv1d(in_channel, in_channel, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm1d(in_channel),
        nn.ReLU(),
        nn.Conv1d(in_channel, in_channel, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm1d(in_channel),
        ]
        self.layers = nn.Sequential(*layers)#

    def forward(self, x):
        return self.layers(x)

class CNN_Res(nn.Module):
    def __init__(self, in_channel=1, n_repeat=2):
        super().__init__()
        blocks=[ResBlock(in_channel=in_channel) for i in range(n_repeat)]

        self.blocks = nn.Sequential(*blocks)
        self.activation = nn.ReLU()

    def forward(self, x):
        for block in self.blocks:
            x = self.activation(block(x)+x)

        return x

class AFibEFNet(nn.Module):

    def __init__(self, SET_ADVANCED_TUNING,n_classes=1, drop_rate=0.5,kernel_size=7):
        self.SET_ADVANCED_TUNING = SET_ADVANCED_TUNING
        super(AFibEFNet,self).__init__()
        self.reduceblock2_1 = nn.Sequential(
            nn.Conv1d(8, 64, kernel_size=11, stride=2, padding=11//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(drop_rate)
        )
        self.reduceblock2_2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=7//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(drop_rate)
        )
        self.reduceblock2_3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(drop_rate)
        )
        self.reduceblock1_1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(drop_rate)            
        )
        self.reduceblock1_2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(drop_rate)            
        )
        self.reduceblock1_3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(drop_rate)            
        )
        self.reduceblock1_4 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(drop_rate)            
        )                        
        self.CNN_Resblock1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm1d(64)
        )
        self.CNN_Resblock2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm1d(128)
        )
        self.CNN_Resblock3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm1d(256)
        )    
        self.CNN_Resblock4 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm1d(512)
        )             
        self.CNN_Resblock5 = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm1d(1024)
        )     
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.hidden1= nn.Linear(4096,512)
        self.bn1 = nn.BatchNorm1d(512)
        self.hidden2 = nn.Linear(512,256)
        self.bn2 = nn.BatchNorm1d(256)
        self.hidden3= nn.Linear(256,64)
        self.bn3 = nn.BatchNorm1d(64)
        self.hidden4 = nn.Linear(64+self.SET_ADVANCED_TUNING,16)
        self.bn4 = nn.BatchNorm1d(16)
        self.hidden5 = nn.Linear(16,1)
        
        if self.SET_ADVANCED_TUNING !=0:
            self.fc_feature = nn.Linear(self.SET_ADVANCED_TUNING,self.SET_ADVANCED_TUNING)

    def forward(self, x, demographic=[]):

        out = self.reduceblock2_1(x)
        out = self.reduceblock2_2(out)
        out = self.reduceblock2_3(out)
        blocks1 = [self.CNN_Resblock1 for _ in range(2)]
        for block in blocks1:
            out = self.relu(block(out)+out)
        out = self.reduceblock1_1(out)  
        blocks2 = [self.CNN_Resblock2 for _ in range(2)]
        for block in blocks2:
            out = self.relu(block(out)+out)
        out = self.reduceblock1_2(out)  
        blocks3 = [self.CNN_Resblock3 for _ in range(2)]
        for block in blocks3:
            out = self.relu(block(out)+out)
        out = self.reduceblock1_3(out)  
        blocks4 = [self.CNN_Resblock4 for _ in range(2)]
        for block in blocks4:
            out = self.relu(block(out)+out)
        out = self.reduceblock1_4(out)      
        blocks5 = [self.CNN_Resblock5 for _ in range(2)]
        for block in blocks5:
            out = self.relu(block(out)+out)
        out = out.view(out.size(0), -1)  
        out = self.leakyrelu(self.bn1(self.hidden1(out)))
        out = self.leakyrelu(self.bn2(self.hidden2(out)))
        out = self.leakyrelu(self.bn3(self.hidden3(out)))
        
        if self.SET_ADVANCED_TUNING !=0:
            demographic = self.fc_feature(demographic)
            out = torch.cat((out, demographic), dim=1)       

        out = self.leakyrelu(self.bn4(self.hidden4(out)))
        out = self.leakyrelu(self.hidden5(out))
        out = torch.sigmoid(out)
            
        return out


