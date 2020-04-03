import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(DNN, self).__init__()
        
        self.fully = nn.Sequential(
                nn.Linear(in_dim, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(True),
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(True),
                nn.Dropout(0.4),
                
                nn.Linear(4096,2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(True),
                nn.Linear(2048,2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(True),
                nn.Dropout(0.35),  
                
                nn.Linear(2048,1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Dropout(0.3),                  
                
                nn.Linear(1024,512),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Dropout(0.25),  
                
                nn.Linear(512,n_class)             
                )
        
    def forward(self,x):   
        x = x.view(-1,48*48)
        out = self.fully(x)   
        return out
        