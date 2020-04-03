import sys
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

### Parameters ###
RAWDATA_PATH = './dataset/train.csv'
OBSERVE_PATH = './dataset/test.csv'
SUBMISSION_PATH = './dataset/sample.csv'
#RAWDATA_PATH = sys.argv[1]
#OBSERVE_PATH = sys.argv[2]
#SUBMISSION_PATH = sys.argv[3]
MODE_PATH = './model'
HIS_PATH = './history'
PIC_PATH = './picture'
##################

def readfile(path):
    print("Reading File...")
    x_train = []
    x_label = []
    val_data = []
    val_label = []

    raw_train = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    for i in range(len(raw_train)):
        tmp = np.array(raw_train[i, 1].split(' ')).reshape(1, 48, 48)
        if (i % 10 == 0):
            val_data.append(tmp)
            val_label.append(raw_train[i][0])
        else:
            x_train.append(tmp)
            x_train.append(np.flip(tmp, axis=2))    # simple example of data augmentation
            x_label.append(raw_train[i][0])
            x_label.append(raw_train[i][0])

    x_train = np.array(x_train, dtype=float) / 255.0
    val_data = np.array(val_data, dtype=float) / 255.0
    x_label = np.array(x_label, dtype=int)
    val_label = np.array(val_label, dtype=int)
    x_train = torch.FloatTensor(x_train)
    val_data = torch.FloatTensor(val_data)
    x_label = torch.LongTensor(x_label)
    val_label = torch.LongTensor(val_label)

    return x_train, x_label, val_data, val_label

x_train, x_label, val_data, val_label = readfile(RAWDATA_PATH)    # 'train.csv'

train_set = TensorDataset(x_train, x_label)
val_set = TensorDataset(val_data, val_label)

batch_size = 256
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),  # [64, 24, 24]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [64, 12, 12]

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [128, 6, 6]

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0)       # [256, 3, 3]
        )

        self.fc = nn.Sequential(
            nn.Linear(256*3*3, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(512, 7)
        )

        self.cnn.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        out = self.cnn(x)
#        print("The output size",out.size())
        out = out.view(out.size()[0], -1) # Flatten
#        print("After Flatten",out.size())
        return self.fc(out)

model = Classifier().cuda()
# print(model)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_acc = 0.0

num_epoch = 16
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        progress = ('#' * int(float(i)/len(train_loader)*40)).ljust(40)
        print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, num_epoch, \
                (time.time() - epoch_start_time), progress), end='\r', flush=True)
    model.eval()
    for i, data in enumerate(val_loader):
        val_pred = model(data[0].cuda())
        batch_loss = loss(val_pred, data[1].cuda())

        val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        val_loss += batch_loss.item()

        progress = ('#' * int(float(i)/len(val_loader)*40)).ljust(40)
        print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, num_epoch, \
                (time.time() - epoch_start_time), progress), end='\r', flush=True)

    val_acc = val_acc/val_set.__len__()
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss, val_acc, val_loss))

    if (val_acc > best_acc):
        torch.save(model.state_dict(), MODE_PATH+'/model.pth')
        best_acc = val_acc
        print ('Model Saved!')

def readtestfile(path):
    print("Reading File...")
    x_test = []
    x_label = []

    raw_test = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    for i in range(len(raw_test)):
        tmp = np.array(raw_test[i, 1].split(' ')).reshape(1, 48, 48)
        x_test.append(tmp)
        x_label.append(raw_test[i][0])

    x_test = np.array(x_test, dtype=float) / 255.0
    x_label = np.array(x_label, dtype=int)
    x_test = torch.FloatTensor(x_test)
    x_label = torch.LongTensor(x_label)

    return x_test, x_label

x_test, x_label = readtestfile(OBSERVE_PATH)    # 'test.csv'
test_set = TensorDataset(x_test, x_label)
test_loader = DataLoader(test_set)
print('1')

submission = [['id', 'label']]
for i, data in enumerate(test_loader):
        
        test_pred = model(data[0].cuda())
        pred = np.argmax(test_pred.cpu().data.numpy(), axis=1)[0]
        submission.append([i, pred])
with open(SUBMISSION_PATH, 'w') as submissionFile:
    writer = csv.writer(submissionFile)
    writer.writerows(submission)
    
print('Writing Complete!')