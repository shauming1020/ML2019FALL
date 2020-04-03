import numpy as np
import sys
import csv
from numpy.linalg import inv

### Parameters ###
RAWDATA_PATH = './dataset/train.csv'
OBSERVE_PATH = './dataset/test.csv'
SUBMISSION_PATH = './dataset/sample_submission.csv'
RAW_FEATURE_PATH = './dataset/X_train' # after proprecessing with raw data
RAW_LABEL_PATH = './dataset/Y_train'
OB_FEATURE_PATH = './dataset/X_test'
#RAWDATA_PATH = sys.argv[1]
#OBSERVE_PATH = sys.argv[2]
#SUBMISSION_PATH = sys.argv[3]
MODE_PATH = './model'
HIS_PATH = './history'
PIC_PATH = './picture'
######################

class Generative_Model():
    def __init__(self):
        self.data = {}  
    
    # row : number of data as 'n', col : number of feature as 'm'
    def read(self,name,path):
        with open(path,newline = '') as csvfile:
            rows = np.array(list(csv.reader(csvfile))[1:] ,dtype = float)  
            if name == 'X_train':
                self.mean = np.mean(rows,axis = 0).reshape(1,-1)
                self.std = np.std(rows,axis = 0).reshape(1,-1)
                self.theta = np.ones((rows.shape[1] + 1,1),dtype = float) 
                # Normalize
                for i in range(rows.shape[0]):
                    rows[i,:] = (rows[i,:] - self.mean) / self.std  
            elif name == 'X_test': 
                # Normalize
                for i in range(rows.shape[0]):
                    rows[i,:] = (rows[i,:] - self.mean) / self.std 
            self.data[name] = rows  

    def get_u(self,x):
        return np.mean(x,axis = 0)
    def get_cov(self,x,u):
        n, m = x.shape[0], x.shape[1]
        Z = np.zeros((m,m))
        for i in range(n):
            z = x[i] - u
            Z += np.dot(z[:,None],z[None,:]) # z.T dot z
        return Z/n
    
    def get_parameter(self):
        class_0_id, class_1_id = [],[]
        for i in range(self.data['Y_train'].shape[0]):
            if self.data['Y_train'][i][0] == 0:
                class_0_id.append(i)
            else:
                class_1_id.append(i)
                
        class_0, class_1 = self.data['X_train'][class_0_id], self.data['X_train'][class_1_id]
        u_0,u_1 = self.get_u(class_0), self.get_u(class_1)
        cov_0,cov_1 = self.get_cov(class_0,u_0), self.get_cov(class_1,u_1)
        
        n_class_0, n_class_1 = class_0.shape[0], class_1.shape[0]
        cov = (cov_0 * n_class_0 + cov_1 * n_class_1) / (n_class_0 + n_class_1) 
        
        self.w = np.transpose(((u_0 - u_1)).dot(inv(cov)) )
        self.b = (- 0.5)* (u_0).dot(inv(cov)).dot(u_0)\
                + 0.5 * (u_1).dot(inv(cov)).dot(u_1)\
                + np.log(float(n_class_0) / n_class_1)

    def func(self,x):
        probability = np.empty([x.shape[0],1],dtype=float)
        for i in range(x.shape[0]):
            z = x[i,:].dot(self.w) + self.b
            z *= (-1)
            probability[i][0] = 1 / (1 + np.exp(z))
        return np.clip(probability, 1e-8, 1-(1e-8))

    def classify(self,x):
        pred = np.ones([x.shape[0],1],dtype=int)
        for i in range(x.shape[0]):
            if x[i] > 0.5:
                pred[i] = 0; # class-0
        return pred

    def write_file(self,path):
        result = self.func(self.data['X_test'])
        pred = self.classify(result)
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile) 
            writer.writerow(['id','label']) 
            for i in range(pred.shape[0]):
                writer.writerow([i+1,pred[i][0]])

dm = Generative_Model()
dm.read('X_train','./dataset/X_train')
dm.read('Y_train','./dataset/Y_train')
dm.read('X_test','./dataset/X_test')

dm.get_parameter()
dm.write_file(SUBMISSION_PATH)
