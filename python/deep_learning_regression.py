#!/public/software/apps/anaconda3/5.3.0/bin/python
# -*- coding: utf-8 -*-

"""
Created on Wed Jun  8 18:23:55 2022

@author: 马钰斌

xshell use utf-8 

这个程序是一个TBF的实例程序的torch部分，

第一步：需要输入训练数据，验证数据，测试数据，
	1.计算一个y所需要的x1，x2...xm,这个是m = feature_count .

第二步：网络架构设计：设置有几层，每层有几个节点，哪类激活函数
	1.每层有几个节点在point_count,feature,n = point_count 设置；
	2.有几层这个在nn.sequential{}中设置，然后在后面的o = {激活函数的个数} 
	3.在nn.sequential 中激活函数和线性层交替排列，现代深度学习一般relu为激活函数	      也可以使sigmod、tanh等，这个在torch最好设为一样的，在KBF目前只能设置成一样的。

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
time=  data.time.data 

t2m =  data.t2m.data        
bld =  data.bld.data                
zust=  data.zust.data       
gwd =  data.gwd.data        
sst =  data.sst.data       
skt =  data.skt.data        
slhf=  data.slhf.data      
ssr =  data.ssr.data        
st  =  data.str.data        
sp  =  data.sp.data        
ssh =  data.sshf.data

pblh=  data.blh.data

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
#feature = [2,6,10]
feature = [0,2,6,10]
#训练时需要加上y
#feature_for_train=[2,6,10,11]
feature_for_train=[0,2,6,10,11]
#统计因子个数
feature_count = np.size(feature)
#每层的节点数
point_count = 64 #每层的节点数


#%%=======================================================================

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
    ''' 获得设备 ( 如果 GPU 可用, 则用 GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

#函数：绘制学习曲线
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


#函数：绘制验证集和预测值的对比图，验证模型训练效果
def plot_pred(dv_set, model, device, lim=600, preds=None, targets=None):
    ''' 绘制DNN模型输出和验证集 '''
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

#给feats 赋值根据target_only
        if not target_only:
            feats = list((feature))
        else:
           feats = list((feature))

#根据test或者是train或者是dev分配数据
        if mode == 'test':
            # 测试数据 
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # 训练数据,-表示倒数第一个
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

	#统计数据个数
        self.dim = self.data.shape[1]

        print('结束读取 {} 集， ({} 样本被输入, 每个 有 {} 变量)'
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
        # 返回数据尺寸
        return len(self.data)
 

def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    '''建立一个数据集，然后将它放在dataloader中 '''
    dataset = myDataset(path, mode=mode, target_only=target_only)  # 建立数据集
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # 建立dataloader
    return dataloader


# # 深度神经网络
# 
# NeuralNet 被设计用来深度学习回归方程
# 这个DNN含有多个全连接层和多个ReLU激活函数
# 这个网络还定义了一个误差函数来计算误差
# 
class NeuralNet(nn.Module):
    ''' 一个简单的全连接深度神经网络（DNN） '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        # 在这里定义神经网络
	#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        self.net = nn.Sequential(

            nn.Linear(input_dim  ,point_count),

	    nn.ReLU(),

            nn.Linear(point_count,point_count),

	    nn.ReLU(),

            nn.Linear(point_count,point_count),

	    nn.ReLU(),

            nn.Linear(point_count,point_count),

	    nn.ReLU(),

            nn.Linear(point_count,point_count),

	    nn.ReLU(),

            nn.Linear(point_count, 1)

        )

        # 均方根误差
        self.criterion = nn.MSELoss(reduction='mean')
	#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' 计算损失函数 '''
        # 这里可以选择正则化
        return self.criterion(pred, target)

#=================================================================================================================

# ## **训练**


def train(tr_set, dv_set, model, config, device):
    ''' DNN 训练 '''

    n_epochs = config['n_epochs']  # 最大训练轮数

    # 设置优化
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 100000.       #能够被保存模型的 最小的误差
    loss_record = {'train': [], 'dev': []}      # 为了保存训练的误差函数

    early_stop_cnt = 0#初始化提前终止步数
    epoch = 0#初始化轮数
    while epoch < n_epochs:
        model.train()                           # 设置模型到训练模式
        for x, y in tr_set:                     # 通过dataloader迭代
            optimizer.zero_grad()               # 设置梯度为0
            x, y = x.to(device), y.to(device)   # 移动数据到设备中 (cpu/cuda)
            pred = model(x)                     # 计算输出forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # 计算代价函数（误差函数）
            mse_loss.backward()                 # 计算梯度 (反向传播算法)
            optimizer.step()                    # 更新模型参数
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # 结束每一轮训练后，使用验证集测试模型
        dev_mse = dev(dv_set, model, device)
#        print(epoch)
        if dev_mse < min_mse:
            # 如果模型优化了，则保存模型
            min_mse = dev_mse
            print('保存模型(epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # 保存模型到特定的路径
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            #结束训练如果模型很久不更新参数了 ，需要在超参数中设置提前终止选项和轮数"config['early_stop']"
            break

    print('结束训练！经历了 {} 轮！'.format(epoch))
    return min_mse, loss_record

#-----------------------------------------------------------------------------------------------------------------
# ## **验证**

def dev(dv_set, model, device):
    model.eval()                                # 设置模型到评估模式
    total_loss = 0
    for x, y in dv_set:                         # 通过dataloader 迭代
        x, y = x.to(device), y.to(device)       # 移动数据到‘设备’ (cpu/cuda)
        with torch.no_grad():                   # 关闭梯度计算
            pred = model(x)                     # 计算输出forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # 计算代价函数
        total_loss += mse_loss.detach().cpu().item() * len(x)  # 计算累计误差
    total_loss = total_loss / len(dv_set.dataset)              # 计算平均误差

    return total_loss

#-----------------------------------------------------------------------------------------------------------------
# ## **测试**

def test(tt_set, model, device):
    model.eval()                                # 设置模型到评估模式
    preds = []
    for x in tt_set:                            # 通过dataloader 迭代
        x = x.to(device)                        # 移动数据到‘设备’ (cpu/cuda)
        with torch.no_grad():                   # 关闭梯度计算
            pred = model(x)                     # 计算输出forward pass (compute output)
            preds.append(pred.detach().cpu())   # 收集预测值
    preds = torch.cat(preds, dim=0).numpy()     # 收集所有的预测值并转化为一个numpy array 数组
    return preds

#================================================================================================================

#%%
# 
# # **设置超参数**
# 
# 设置训练的超参数和存放模型的路径

device = get_device()                 # 获得可用设备 ('cpu' 或者 'cuda')
os.makedirs('models', exist_ok=True)  # 训练的模型将会放在当前目录下的models中 ./models/
target_only = False                 #可以在选择因子时使用

# 改变优化超参数去改进模型训练
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#根据数据量改变优化参数

config = {
    'n_epochs': 3000,                # 最大epochs数
    'batch_size': 300,               # mini-batch 尺寸，batch包含的数据量
#    'optimizer': 'SGD',              # 优化算法选择 optimization algorithm (optimizer in torch.optim)
    'optimizer': 'Adam',  
    'optim_hparas': {                #优化算法的超参数(取决于选取的优化算法)
#        'lr': 0.001,                 # SGD(随机梯度下降算法)的学习率
#        'momentum': 0.09              # SGD的动能
    },
    'early_stop': 100,               # 提前终止的最大步数 (自从上次模型参数更新的轮数，如果这么多步仍然没有更新则提前终止优化)1起到正则化作用，2减少计算资源消耗
    'save_path': 'models/model.pth'  # 模型保存路径
}

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# # **加载数据和模型**

tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)

dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)

tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)

model = NeuralNet(tr_set.dataset.dim).to(device)  #实例化(创建)模型并移动到设备


# # **开始训练!**

model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)    #  训练过程

plot_learning_curve(model_loss_record, title='deep model')  #绘制学习曲线 

del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # 加载最好的模型
model.load_state_dict(ckpt)	#加载模型参数



plot_pred(dv_set, model, device)  # 绘制验证集和预测值对比 


# # **测试**
# 利用测试集和模型计算的预测值将会被保存在pred.csv 中

def save_pred(preds, file):
    ''' 保存预测值到文件中 '''
    print('保存结果到 {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


preds = test(tt_set, model, device)  #使用测试集和训练好的深度模型 预测 y

save_pred(preds, 'pred.csv')         # 保存预测数据到 pred.csv 文件


##========================================================================
#模型参数传递torch->TBF->Fortran


#  计算模型参数总数。 
total = sum(p.numel() for p in model.parameters())
print("总参数个数: %.2f" % (total))

#将pth 的模型参数输出出来

#有几个因子  m = ?
m = len(feature)

#有几层  o = ?
o = ( len(list(model.net)) -1 )//2 #自动获取层数，减去输出层然后线性层和激活函数层的和整除以2

#每层有几个节点 n = ?
n = point_count

#输入层的权重和偏差
w_input =  model.net[0].weight.data.cpu().numpy()
b_input =  model.net[0].bias.data.cpu().numpy()

#print( model.net[0].weight.data.cpu().numpy())
np.savetxt('w_input.txt',w_input,fmt='%f')
np.savetxt('b_input.txt',b_input,fmt='%f')

#中间层的权重和偏差
os.system('rm w_dense.txt ')
os.system('rm b_dense.txt')

for i in range(2,2*o,2):# range(x,y) 不包含y 
#	print(model.net[i].weight.data.cpu().numpy())

	w_dense = model.net[i].weight.data.cpu().numpy()
	b_dense = model.net[i].bias.data.cpu().numpy()

	with open("w_dense.txt","ab") as f:#追加写入模式
		np.savetxt(f,w_dense,fmt = '%f')
	with open("b_dense.txt","ab") as g:
		np.savetxt(g,b_dense,fmt = '%f')


#输出层的权重和误差

#print( model.net[2*o].weight.data.cpu().numpy())

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
os.system('cp *.txt /data/chengxl/pblh_deeplearning/torch_bridge_fortran/') 
#一些有用的代码
#查看tt_set 里面的x

#for x in tt_set:
#	print(x)

