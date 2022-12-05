# -*- coding: utf-8 -*-

"""
Created on Wed Jun  8 18:23:55 2022

@author: Mayubin

xshell use utf-8 

这个程序是一个TBF的实例程序的torch部分，

第一步：需要输入训练数据，验证数据，测试数据，
	1.计算一个y所需要的x1，x2...xm,这个是m = feature_count .

第二步：网络架构设计：设置有几层，每层有几个节点，哪类激活函数
	1.每层有几个节点在point_count,feature,n = point_count 设置；
	2.有几层这个在nn.sequential{}中设置，然后在后面的o = {激活函数的个数} 
	3.在nn.sequential 中激活函数和线性层交替排列，现代深度学习一般relu为激活函数	      也可以使sigmod、tanh等，这个在torch最好设为一样的，在KBF目前只能设置成一样	  的。

第三步：设置优化方法，主要是对优化方法，npoch,batch_size,等进行修改，以达到最好的优化效果，得到最优参数W,b,c,d

第四步：查看训练的结果，两张图，方差，均方根误差等决定需不需要再次重复上面步骤。

第五步：在训练效果很好，得到正确参数的情况下，传输模型参数先输出为txt文件，
	然后用fortran读取到KBF中。
	1.检查m,n,o,function_kind,shuchucanshu.txt
	2.检查w_input.txt w_dense.txt w_output.txt
	3.检查b_input.txt b_dense.txt b_output.txt 

第六步：将*.txt 传输到torch_bridge_fortran 文件夹中
之后的操作留给fortran程序来解决。


这个示例程序，用ERA5的资料进行深度学习训练边界层高度模型
主要可以修改的部分是因子选择，训练方法，绘图等等

"""

#time 
import datetime




# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os
import pandas as pd
import xarray as xr
import math

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#for earthlab no GPU
plt.switch_backend('agg')



#==================================================================================
#数据准备

path = 'h_small.nc'                                 #路径和文件
#用xarray输入数据
data=xr.open_dataset(path)
#输入数据
lon = data.longitude.data
lat = data.latitude.data
time  = data.time.data 

t2m =  data.t2m.data        
bld =  data.bld.data                
zust=    data.zust.data       
gwd =     data.gwd.data        
sst =     data.sst.data       
skt =     data.skt.data        
slhf =     data.slhf.data      
ssr=    data.ssr.data        
st =     data.str.data        
sp =     data.sp.data        
ssh =     data.sshf.data

pblh =     data.blh.data

#将数据转为一维数据，-1 表示默认待定
t2m1 =  t2m.reshape(-1,1)
bld1 =  bld.reshape(-1,1)
zust1=  zust.reshape(-1,1)       
gwd1 =  gwd.reshape(-1,1) 
sst1 =  sst.reshape(-1,1)   
skt1 =  skt.reshape(-1,1)     
slhf1=  slhf.reshape(-1,1)
ssr1=   ssr.reshape(-1,1)     
st1 =   st.reshape(-1,1) 
sp1 =   sp.reshape(-1,1)   
ssh1=   ssh.reshape(-1,1)

pblh1=  pblh.reshape(-1,1) 
#将一维列数据拼接成矩阵,hstack 水平拼接，vstack垂直拼接：
data_for_dp = np.hstack((t2m1,bld1,zust1,gwd1,sst1,skt1,slhf1,ssr1,st1,sp1,ssh1,pblh1))
#第一个索引表示行，第二表示列，选取训练数据量

#训练数据
data_train   =   data_for_dp[0:3000,:]
#测试数据，只给因子，不给y,：-1表示到倒数第二列
data_test  = data_for_dp[3001:5000,:-1]

#转化为pandas的数据结构：表，为了后面使用to_save函数
data_train = pd.DataFrame(data_train)
data_test =pd.DataFrame(data_test)
#保存数据到csv
data_test.to_csv('test.csv',index=False, header=None)
data_train.to_csv('train.csv',index=False, header=None)

# feature = [0,1,2,3,4,5,6,7,8,9,10]
# feature_for_train=[0,1,2,3,4,5,6,7,8,9,10,11]

#对因子选择
feature = [2,6,10]
#训练时需要加上y
feature_for_train=[2,6,10,11]
#统计因子个数
feature_count = np.size(feature)
#每层的节点数
point_count = 64 #每层的节点数


#%%

tr_path = 'train.csv'  # 训练数据的路径
tt_path = 'test.csv'   # 测试数据的路径



myseed = 42069  # 设置随机数种子
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# 一些实用函数


#获得计算设备：是CPU还是GPU,是否支持显卡加速计算
def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

#绘制学习曲线函数
def plot_learning_curve(loss_record, title=''):
    '''绘制DNN学习曲线(训练和验证的误差函数)'''
    total_steps = len(loss_record['train'])

    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]

    figure(figsize=(6, 4))

    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')

    plt.ylim(0.0, 2000.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()

    plt.savefig(r'learncurve.jpg')
#    plt.show()#在超算上不能够直接显示
 
#未来需要实现的功能：每次训练之后的数据，模型，代码，图片等都备份到一个特殊命名文件夹中


#数据加载和前处理
class myDataset(Dataset):
    ''' 数据集的加载和预处理'''
    def __init__(self,
                 path,
                 mode='train',
                 target_only=False):
        self.mode = mode

        # 将数据读到numpy数组中
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[:])[:, :].astype(float)

#代码这个地方存在BUG,导致因子是前三个，后面改一下        
#给feats 赋值根据target_only
        if not target_only:
            feats = list(range(feature_count))
        else:
           feats = list(range(feature_count))

#根据test或者是train或者是dev分配数据
        if mode == 'test':
            # 测试数据 
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # 训练数据
            target = data[:, -1]
            data = data[:, feats]
            
            # 将训练数据分为训练集和验证集，这里是每十个里面选一个作为验证集
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            
            # 将数据转化为PyTorch的数据格式 tensor
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])


         #对数据正则化(可选，对参数优化有帮助) = 距平/标准差
 #       self.data[:, :] =  (self.data[:, :] - self.data[:, :].mean(dim=0, keepdim=True))   / self.data[:, :].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('结束读取Finished reading the {} set of Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)
 

def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = myDataset(path, mode=mode, target_only=target_only)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader


# # **Deep Neural Network**
# 
# `NeuralNet` is an `nn.Module` designed for regression.
# The DNN consists of 2 fully-connected layers with ReLU activation.
# This module also included a function `cal_loss` for calculating loss.
# 
class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
#	point_count=4 #每层的节点数
        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim,point_count),
	    nn.ReLU(),
            nn.Linear(point_count,point_count),
	    nn.ReLU(),
            nn.Linear(point_count, 1)
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L2 regularization here
        return self.criterion(pred, target)


# # **Train/Dev/Test**

# ## **Training**

# In[7]:


def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 100000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
#        print(epoch)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


# ## **Validation**

def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss


# ## **Testing**

def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds



#%%
#关键的地方了
# # **Setup Hyper-parameters**
# 
# `config` contains hyper-parameters for training and the path to save your model.

device = get_device()                 # get the current available device ('cpu' or 'cuda')
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
target_only = False                 # TODO: Using 40 states & 2 tested_positive features

# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 3000,                # maximum number of epochs
    'batch_size': 300,               # mini-batch size for dataloader
#    'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
    'optimizer': 'Adam',
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
#        'lr': 0.001,                 # learning rate of SGD
#        'momentum': 0.09              # momentum for SGD
    },
    'early_stop': 100,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth'  # your model will be saved here
}


# # **Load data and model**

tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)

model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device

# # **Start Training!**

model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

plot_learning_curve(model_loss_record, title='deep model')

del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)

def plot_pred(dv_set, model, device, lim=600, preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.savefig(r'preds.jpg')
#    plt.show()


plot_pred(dv_set, model, device)  # Show prediction on the validation set


# # **Testing**
# The predictions of your model on testing set will be stored at `pred.csv`.

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
save_pred(preds, 'pred.csv')         # save prediction file to pred.csv


# # **Hints**
# 
# ## **Medium Baseline**
# * Feature selection: 40 states + 2 `tested_positive` (`TODO` in dataset)
# 

# * Feature selection (what other features are useful?)
# * DNN architecture (layers? dimension? activation function?)
# * Training (mini-batch? optimizer? learning rate?)
# * L2 regularization

# calculate parameters number sum  计算模型参数总数。 
total = sum(p.numel() for p in model.parameters())
print("Total params: %.2f" % (total))



#%%
weight00 = model.net[0].weight.data.cpu().numpy()
weight02 = model.net[2].weight.data.cpu().numpy()
bias00 = model.net[0].bias.data.cpu().numpy()
bias02 = model.net[2].bias.data.cpu().numpy()

#为了直接在cesm中手动加入，早期方法
w00_fortran = weight00.T.reshape(-1,1)
#np.savetxt("weight00.txt", w00_fortran,fmt='%f',delimiter=' ')

w02_fortran = weight02.T.reshape(-1,1)
#np.savetxt("weight02.txt", w02_fortran,fmt='%f',delimiter=' ')

b00_fortran = bias00.reshape(-1,1)
#np.savetxt("bias00.txt", b00_fortran,fmt='%f',delimiter=' ')

b02_fortran = bias02
#np.savetxt("bias02.txt", b02_fortran,fmt='%f',delimiter=' ')

#将pth 的模型参数输出出来


#必须输出

#有几个因子  m = ?
m = len(feature)
#有几层      o = 2?
o = 2          #   目前o 需要手动设定，为relu 的个数
#我们说的有几层指的是有几个激活函数，就有几层

#每层有几个节点 n = ?
n = point_count

#输入层的权重和偏差
w_input =  model.net[0].weight.data.cpu().numpy()
b_input =  model.net[0].bias.data.cpu().numpy()

print( model.net[0].weight.data.cpu().numpy())
np.savetxt('w_input.txt',w_input,fmt='%f')
np.savetxt('b_input.txt',b_input,fmt='%f')

#中间层的权重和偏差
os.system('rm w_dense.txt ')
os.system('rm b_dense.txt')

for i in range(2,2*o,2):# range(x,y) 不包含y 
	print(model.net[i].weight.data.cpu().numpy())

	w_dense = model.net[i].weight.data.cpu().numpy()
	b_dense = model.net[i].bias.data.cpu().numpy()

	with open("w_dense.txt","ab") as f:#追加写入模式
		np.savetxt(f,w_dense,fmt = '%f')
	with open("b_dense.txt","ab") as g:
		np.savetxt(g,b_dense,fmt = '%f')


#输出层的权重和误差

print( model.net[2*o].weight.data.cpu().numpy())

w_output =  model.net[2*o].weight.data.cpu().numpy()
b_output =  model.net[2*o].bias.data.cpu().numpy()

np.savetxt('w_output.txt',w_output,fmt='%f')
np.savetxt('b_output.txt',b_output,fmt='%f')


#向文件写入维度参数
#
note = open ('shuchucanshu.txt','w')
current_time = datetime.datetime.now()
note.write(str(current_time))
note.write('\n存放深度学习模型数据，用于输入到TBF中')

note.write('\nm\n')
note.write(str(m))

note.write('\nn\n')
note.write(str(n))

note.write('\no\n')
note.write(str(o))

note.close

#将TXT传到TBF文件夹中：
os.system('cp *.txt $deeplearn/torch_bridge_fortran/') 
#一些有用的代码
#查看tt_set 里面的x
for x in tt_set:
	print(x)

