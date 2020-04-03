import sys
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import resnet_v2
from torch.utils.data import TensorDataset, DataLoader

### Parameters ###
RAWDATA_PATH = './dataset/train.csv'
OBSERVE_PATH = './dataset/test.csv'
SUBMISSION_PATH = './dataset/sample.csv'
#RAWDATA_PATH = sys.argv[1]
#OBSERVE_PATH = sys.argv[2]
#SUBMISSION_PATH = sys.argv[3]
MODEL_PATH = './model'
HIS_PATH = './history'
PIC_PATH = './picture'
CLASSES = ("Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral")
##################

### Golbal Parameters ###
LEARNING_RATE = 0.001
LR_DECAY = 0.8
REG = 2 # L2-
WEIGHT_DECAY = 1e-04
PATIENCE = 16
#########################

class Data():
    def __init__(self):
        self.X = []
        self.y = []
    def Read(self,path):
        raw = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
        for i in range(len(raw)):
            tmp = np.array(raw[i, 1].split(' ')).reshape(1, 48, 48) # (RGB-channel, height, width)
            self.X.append(tmp)
            self.y.append(np.array(raw[i, 0]))
        self.X = torch.FloatTensor(np.array(self.X, dtype=float))
        self.y = torch.LongTensor(np.array(self.y, dtype=int)) 

def Normalization(imgs,method='Divide',_max='None',_min='None',trans=False):
    if trans is False:
        _max = torch.max(imgs, dim = 0) # (Number of Images, RGB, height, width)
        _min = torch.min(imgs, dim = 0)
    if method == 'Divide':
        imgs /= 255.0
        return imgs
    elif method == 'Rescaling':
        imgs = (imgs - _min) / (_max - _min)
    if trans is False:
        return imgs, _max, _min
    else:
        return imgs
    
def Standardization(imgs,_mean='None',_std='None',trans=False):
    if trans is False:
        _mean = torch.mean(imgs, dim = 0)
        _std = torch.std(imgs, dim = 0)          
    imgs = (imgs - _mean) / _std
    if trans is False:
        return imgs, _mean, _std
    else:
        return imgs

def Split_train_val(x,y,train_rate=0.8):
    if len(x) != len(y):
        return print("no match size!")
    perm = torch.randperm(len(y)) # Shuffle
    x, y = x[perm], y[perm]
    split_pos = int(np.round(len(y)*train_rate))
    return x[:split_pos], x[split_pos:], y[:split_pos], y[split_pos:]

#def Encoding(y,n_class,method='One-Hot'):
#    size = len(y)
#    if method == 'One-Hot':
#        encoding = torch.LongTensor(size,n_class)
#        encoding.zero_()
#        encoding.scatter_(1,y.view(-1,1),1)
#    return encoding
  
def Fit(train_set,val_set,model,loss,optimizer,batch_size,epochs):
    
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)   
    
    loss_history, acc_history = [], []
    best_acc = 0.0
    
    patience = 0
    
    if WEIGHT_DECAY > 0:
        import Regular
        reg_loss = Regular.Regularization(model, WEIGHT_DECAY, p=REG)
        print('Regularization...')
    else:
        reg_loss = 0.
        print('No Regularization...')
    
    print('Image Augment...')
    from keras.preprocessing.image import ImageDataGenerator 
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=[0.8, 1.2],
        shear_range=0.2,
        horizontal_flip=True)
    
    for epoch in range(epochs):
        if patience == PATIENCE:
            print("Early Stopping...")
            break
        
        epoch_start_time = time.time()
        
        adjust_learning_rate(optimizer, epoch)
        
        train_acc, train_loss = 0.0, 0.0
        val_acc, val_loss = 0.0, 0.0
    
        model.train() # Switch to train mode
        for i, batch in enumerate(datagen.flow(train_set[:][0].view(len(train_set[:][0]),48,48,1),\
                                               train_set[:][1], batch_size=batch_size)):
            
            if i == int(train_set.__len__()/batch_size)+1: # one epoch
                break
            else:
                i += 1
                
            # type transform
            batch = (torch.FloatTensor(batch[0]), torch.LongTensor(batch[1]))
            batch = (batch[0].view(len(batch[0]),1,48,48), batch[1])
            
            # compute output
            train_pred = model(batch[0].cuda())
            batch_loss = loss(train_pred, batch[1].cuda()) + reg_loss(model)
            
            # compute gradient and do step
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            train_pred = train_pred.float()
            batch_loss = batch_loss.float()
            
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == batch[1].numpy())
            train_loss += batch_loss.item()
                 
        model.eval() # Switch to evaluate mode
        for i, data in enumerate(val_loader):
            # compute output
            with torch.no_grad():
                val_pred = model(data[0].cuda())
                batch_loss = loss(val_pred, data[1].cuda())
    
            val_pred = val_pred.float()
            batch_loss = batch_loss.float()
    
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()
            
        train_acc = train_acc/train_set.__len__()
        val_acc = val_acc/val_set.__len__()
        
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.3f Loss: %3.3f | Val Acc: %3.3f loss: %3.3f' % \
                (epoch + 1, epochs, time.time()-epoch_start_time, \
                 train_acc, train_loss, val_acc, val_loss))
        
        loss_history.append((train_loss,val_loss))
        acc_history.append((train_acc,val_acc))
        
        # Early Stopping
        if (val_acc > best_acc):
            torch.save(model.state_dict(), MODEL_PATH+'/model.pth')
            best_acc = val_acc
            patience = 0
            print ('Model Saved!') 
        else:
            patience += 1
            
    # Save the last model
    torch.save(model.state_dict(), MODEL_PATH+'/last_model.pth')
    print ('Last Model Saved!')
        
    return np.asarray(loss_history), np.asarray(acc_history)

def Evaluate(test_set,classifier,save_model):
    test_loader = DataLoader(test_set, num_workers=8)
    acc = 0.0
    y_pred = []
    classifier.eval()
    for i, data in enumerate(test_loader):
            test_pred = classifier(data[0].cuda())
            batch_pred = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            acc+=np.sum(batch_pred == data[1].numpy())
            y_pred.append(batch_pred)
    print("Test Accuracy: %.3f" %(acc/test_set.__len__()) )
    Plot_Confusion_Matrix(list(test_set[:][1].numpy()), y_pred)
    if save_model == False:     
        plt.savefig(PIC_PATH+'/_confusion.png')
    else:
        plt.savefig(PIC_PATH+'/'+save_model+'_confusion.png')
    plt.show()
    plt.close()
    
def Plot_History(history,save_model):
    plt.clf()
    loss_t, loss_v = history[:,0], history[:,1]
    plt.plot(loss_t,'b')
    plt.plot(loss_v,'r')
    if "loss" in save_model:
        plt.legend(['loss', 'val_loss'], loc="upper left")
        plt.ylabel("loss")
    elif "acc" in save_model:
        plt.legend(['acc', 'val_acc'], loc="upper left")
        plt.ylabel("acc")        
    plt.xlabel("epoch")
    plt.title("Training Process")
    if save_model == False:     
        plt.savefig(PIC_PATH+'/_history.png')
    else:
        plt.savefig(PIC_PATH+'/'+save_model+'_history.png')
    plt.show()
    plt.close()

def Plot_Confusion_Matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    import itertools
    conf_matrix = confusion_matrix(y_true, y_pred)
    title='Normalized Confusion Matrix'
    cm = conf_matrix.astype('float')/conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)
    
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i,j]), horizontalalignment="center", 
                 color="white" if cm[i,j] > thresh else "black")    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return 

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 32 epochs """
    lr = LEARNING_RATE * (LR_DECAY ** (epoch // 32))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
def Resnet_():
    print("Reading Raw File...")
    raw = Data()
    raw.Read(RAWDATA_PATH)
    X_train, X_test, y_train, y_test = Split_train_val(raw.X, raw.y, train_rate=0.8)
    
    print("Normalization...")
    X_train = Normalization(X_train)
    
    print("Standardization...")
    X_train, _mean, _std = Standardization(X_train)
    
    ## Split to training, validation  
    X_train, X_val, y_train, y_val = Split_train_val(X_train, y_train, train_rate=0.8)
    train_set, val_set = TensorDataset(X_train, y_train), TensorDataset(X_val, y_val)
    
    print("Building Model...")
    model = resnet_v2.resnet18().cuda()
    loss = nn.CrossEntropyLoss() # The criterion combines nn.LogSoftmax()
    loss = loss.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)    
    
    ## Training infomation
    print('Total parameters:',sum(p.numel() for p in model.parameters()),\
          ', Trainable parameters:comp',sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Training set size:',len(X_train),', Validation set size:',len(X_val))
    
    print("Fitting Model...")
    loss_history, acc_history = Fit(train_set,val_set,model,loss,optimizer,batch_size=256,epochs=128)
    
    Plot_History(loss_history,'Resnet50_loss')
    Plot_History(acc_history,'Resnet50_acc')
    
    print("Loading Model...")
    model = resnet_v2.resnet18().cuda()
    model.load_state_dict(torch.load(MODEL_PATH+'/model.pth'))
    
    print("Evaluate and Plot Confusion Matrix...")
    print("Normalization...")
    X_test = Normalization(X_test,'Divide', 'None', 'None', True)
    
    print("Standardization...")
    X_test = Standardization(X_test, _mean, _std, True)
    test_set = TensorDataset(X_test, y_test)
    
    Evaluate(test_set,model,'Resnet50')
    
    print("Reading Observe File...")
    ob = Data()
    ob.Read(OBSERVE_PATH)
    
    print("Normalization...")
    ob.X = Normalization(ob.X,'Divide', 'None', 'None', True)
    
    print("Standardization...")
    ob.X = Standardization(ob.X, _mean, _std, True) 
    
    ob_set = TensorDataset(ob.X, ob.y) ## test.y is ID not label.
    ob_loader = DataLoader(ob_set, num_workers=8)
    
    print("Submission...")
    submission = [['id', 'label']]
    for i, data in enumerate(ob_loader):
        test_pred = model(data[0].cuda())
        pred = np.argmax(test_pred.cpu().data.numpy(), axis=1)[0]
        submission.append([i, pred])
    with open(SUBMISSION_PATH, 'w') as submissionFile:
        writer = csv.writer(submissionFile)
        writer.writerows(submission)
    
    print('Writing Complete!') 

if __name__ == '__main__':    
    Resnet_()