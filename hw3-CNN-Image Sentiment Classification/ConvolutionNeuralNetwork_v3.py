import sys
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import resnet_v2
import torchvision.transforms as transforms
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
WORKERS = 0
ENSEMBLE_NUM = 8
LEARNING_RATE = 0.002
AUG_SIZE = 8 # one picture will be seen 8 times.
EPOCHS = 80
BATCH_SIZE = 256
PER_EPOCHS_TO_DECAY_LR = 32
LR_DECAY = 0.8
REG = 0 # L2-Norm
WEIGHT_DECAY = 1e-09
PATIENCE = 64
PRINT_FREQ = 256
#########################

def Resnet_():
    global best_prec1
    print("Building Model...")
    model = resnet_v2.resnet18().cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  

    cudnn.benchmark = True

    print("Reading Raw File...")
    raw = Data()
    raw.Read(RAWDATA_PATH)
    X_train, X_test, y_train, y_test = Split_train_val(raw.X, raw.y, train_rate=0.8)
    
    ## Split to training, validation  
    X_train, X_val, y_train, y_val = Split_train_val(X_train, y_train, train_rate=0.8)


    ## Image Augment
#    normalize = transforms.Normalize(mean=[0.485, 0.456],
#                                     std=[0.229, 0.224])
    train_aug = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(48),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
#                normalize,
                ])
    test_aug = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(32),
                transforms.CenterCrop(48),
                transforms.ToTensor(),
#                normalize,
                ])  
    
    for i, img in enumerate(X_train):
        X_train[i] = train_aug(img)
        
    for i, img in enumerate(X_val):
        X_val[i] = test_aug(img)
        
    ## Loader
    train_set, val_set = TensorDataset(X_train, y_train), TensorDataset(X_val, y_val)
    test_set = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)
   
    ## Fitting
    best_prec1 = 0.
    for epoch in range(0, EPOCHS):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            torch.save(model.state_dict(), MODEL_PATH+'/model.pth')

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
  
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to train mode
    model.train()
    
    if WEIGHT_DECAY > 0:
        import Regular
        reg_loss = Regular.Regularization(model, WEIGHT_DECAY, p=REG)
        print('Regularization...')
    else:
        reg_loss = 0.
        print('No Regularization...')
        
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var.cuda())
        loss = criterion(output, target_var.cuda()) + reg_loss(model)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var.cuda())
        loss = criterion(output, target_var.cuda())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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
    


if __name__ == '__main__':    
    Resnet_()