#-*- coding:utf-8 -*-
#'''
# Created on 2020/9/10  10:32
#
# @Author: Jun Wang
#'''

import os
import time
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
from numpy.random import choice
import pandas as pd
import matplotlib.pyplot as plt
import PIL

from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision
from sklearn.model_selection import train_test_split

np.random.seed(42)
DATA_DIR = '~/'
# train_dir = os.path.join(DATA_DIR, 'train')
# test_dir  = os.path.join(DATA_DIR, 'test')
train_dir = '~/data/train/'
test_dir = '~/data/test/'
"""
def train_validation_split(df, val_fraction=0.1):
    val_ids  = np.random.choice(df.id, size=int(len(df) * val_fraction))
    val_df   = df.query('id     in @val_ids')
    train_df = df.query('id not in @val_ids')
    return train_df, val_df


train_label_df, val_label_df = train_validation_split(pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv')),
                                                      val_fraction=0.1)
"""
# DATA_DIR = '/home/dl/zy_Histopathologic_CanDet/input/'


labels = pd.read_csv('/home/ubuntu/junwang/paper/Boosted_EffNet/code/data/train_labels.csv')

train_label_df, val_label_df = train_test_split(labels, stratify=labels.label, test_size=0.1, random_state=123)


# os.environ['CUDA_VISIBLE_DEVICES'] = "2, 3"


def SPC(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).astype(np.float32)
    
    TN = cm[1, 1]
    FP = cm[0, 1]
    return TN / (TN + FP)


def function_timer(function):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        duration = time.time() - start
        
        hours = int(duration // 60 ** 2)
        minutes = int((duration % 60 ** 2) // 60)
        seconds = int(duration % 60)
        print(f'execution-time of function "{function.__name__}": {hours}h {minutes}m {seconds}s')
        
        return result
    
    return wrapper


class HistoPatches(Dataset):
    
    def __init__(self,
                 image_dir: str,
                 label_df=None,
                 transform=transforms.ToTensor(),
                 sample_n=None,
                 in_memory=False):
        """
        @ image_dir:   path to directory with images
        @ label_df:    df with image id (str) and label (0/1) - only for labeled test-set
        @ transforms:  image transformation; by default no transformation
        @ sample_n:    if not None, only use that many observations
        """
        self.image_dir = image_dir
        self.label_df = label_df
        self.transform = transform
        self.in_memory = in_memory
        
        if label_df is not None:
            if sample_n:
                self.label_df = self.label_df.sample(n=sample_n)
            ids = set(self.label_df.id)
            self.img_files = [f for f in os.listdir(image_dir) if f.split('.')[0] in ids]
        else:
            if sample_n is not None:
                print('subsampling is currently only implemented when a label-dataframe is provided.')
                return
            self.img_files = os.listdir(image_dir)
        
        if in_memory:
            self.id2image = self._load_images()
        
        print(f'Initialized datatset with {len(self.img_files)} images.\n')
    
    @function_timer
    def _load_images(self):
        print('loading images in memory...')
        id2image = {}
        
        for file_name in self.img_files:
            img = PIL.Image.open(os.path.join(self.image_dir, file_name))
            X = self.transform(img)
            id_ = file_name.split('.')[0]
            id2image[id_] = X
        
        return id2image
    
    def __getitem__(self, idx):
        file_name = self.img_files[idx]
        id_ = file_name.split('.')[0]
        
        if self.in_memory:
            X = self.id2image[id_]
        else:
            img = PIL.Image.open(os.path.join(self.image_dir, file_name))
            X = self.transform(img)
        
        if self.label_df is not None:
            y = float(self.label_df.query('id == @id_').label)
            return X, y
        else:
            return X, id_
    
    def __len__(self):
        return len(self.img_files)


memory = False
batchsize = 256
image_trans = transforms.Compose([  # transforms.CenterCrop(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.70017236, 0.5436771, 0.6961061],
                         std=[0.22246036, 0.26757348, 0.19798167])
])

# pad + RandomCrop, RCC
train_trans = transforms.Compose([
    transforms.Pad(8),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(size=96),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.70017236, 0.5436771, 0.6961061],
                         std=[0.22246036, 0.26757348, 0.19798167])
])

train = HistoPatches(train_dir,
                     train_label_df,
                     transform=train_trans,
                     # sample_n=1000,
                     in_memory=memory)

val = HistoPatches(train_dir,
                   val_label_df,
                   transform=image_trans,
                   # sample_n=100,
                   in_memory=memory)

train_loader = DataLoader(train, batch_size=batchsize, shuffle=True, num_workers=8)
val_loader = DataLoader(val, batch_size=batchsize * 4, shuffle=False, num_workers=4)



# net = models.densenet121(pretrained=False)
# model = torch.load('/home/dl/zy/zy_Histopathologic_CanDet/models/densenet121-a639ec97.pth')
#
# model = {k.replace('.1.', '1.'): v for k, v in model.items()}
# model = {k.replace('.2.', '2.'): v for k, v in model.items()}

# net.load_state_dict(model)
from torchvision.models.densenet import _densenet

model = 'efficientnet-b3'

from efficientnet_pytorch import EfficientNet


def load_network(model: nn.Module, path):
    state = torch.load(str(path))
    model.load_state_dict(state)
    return model


class SEModule(nn.Module):
    
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.model = EfficientNet.from_name(model)
        # RDS
        self.model._conv_stem = nn.Conv2d(self.model._conv_stem.in_channels, self.model._conv_stem.out_channels,
                                          kernel_size=3, stride=1, bias=False, padding=1) # RDS
        
        self.model._fc = nn.Linear(600, 1)
        self.model._dropout = nn.Dropout(0.3)
        
        self.semodule1 = SEModule(32, 8)
        self.semodule2 = SEModule(48, 8)
        self.semodule3 = SEModule(136, 8)
        self.semodule4 = SEModule(384, 16)
    
    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        
        # Stem
        x = self.model._swish(self.model._bn0(self.model._conv_stem(inputs)))
        
        output = []
        # Blocks
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            # print(len(output), x.shape)
            output.append(x)
        
        # Head
        x = self.model._swish(self.model._bn1(self.model._conv_head(x)))
        output.append(x)
	# SE and FF
        x = torch.cat([F.adaptive_avg_pool2d(self.semodule1(output[4]), 1),
               F.adaptive_avg_pool2d(self.semodule2(output[7]), 1),
               F.adaptive_avg_pool2d(self.semodule3(output[17]), 1),
               F.adaptive_avg_pool2d(self.semodule4(output[25]), 1)], 1)
       
        return x
    
    def forward(self, x):
        # See note [TorchScript super()]
        bs = x.size(0)
        # Convolution layers
        x = self.extract_features(x)
        
        # Pooling and final linear layer
        # x = self.model._avg_pooling(x)
        x = x.view(bs, -1)
        x = self.model._dropout(x)
        x = self.model._fc(x)
        return x


net = mynet()


@function_timer
def train_model(net, train, validation, optimizer, device, max_epoch=100, verbose=False):
    """
    This function returns nothing. The parametes of @net are updated in-place
    and the error statistics are written to a global variable. This allows to
    stop the training at any point and still have the results.

    @ net: a defined model - can also be pretrained
    @ train, test: DataLoaders of training- and test-set
    @ max_epoch: stop training after this number of epochs
    """
    global error_df  # to track error log even when training aborted
    error_df = pd.DataFrame(
        columns=['train_bce', 'train_acc', 'train_auc', 'train_SEN', 'train_SPE', 'train_F1 score', 'val_bce',
                 'val_acc', 'val_auc', 'val_SEN', 'val_SPE', 'val_F1 score'])
    
    criterion = nn.BCEWithLogitsLoss()
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [15, 23])
    
    net.to(device)
    
    print(
        'epoch\tLR\ttr-BCE\ttr-Acc\ttr-AUC\ttr-SEN\ttr-SPE\ttr-F1-score\t\tval-BCE\tval-Acc\tval-AUC\tval-SEN\tval-SPE\tval-F1-score')
    for epoch in tqdm(range(max_epoch)):
        net.train()
        training_bce = training_acc = training_auc = training_SEN = training_SPE = training_f1 = 0
        # print('qingkaishinidebiaoyan')
        for X, y in train:
            # print(X.shape, y.shape)
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            # prediction and error:
            out = net(X).squeeze()
            
            labels = y.detach().cpu().numpy()
            probabilities = torch.sigmoid(out).detach().cpu().numpy()
            predictions = probabilities.round()
            loss = criterion(out.type(torch.DoubleTensor).cuda(), y)
            
            training_bce += loss.item()
            training_acc += np.mean(labels == predictions) * 100
            training_auc += roc_auc_score(y_true=labels, y_score=probabilities)
            training_SEN += recall_score(y_true=labels, y_pred=predictions)
            training_SPE += SPC(y_true=labels, y_pred=predictions)
            training_f1 += f1_score(y_true=labels, y_pred=predictions)
            
            # update parameters:
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():  # no backpropagation necessary
            net.eval()
            validation_bce = validation_acc = validation_auc = validation_SEN = validation_SPE = validation_f1 = 0
            
            for X, y in validation:
                X, y = X.to(device), y.to(device)
                
                # prediction and error:
                out = net(X).squeeze()
                
                labels = y.detach().cpu().numpy()
                probabilities = torch.sigmoid(out).detach().cpu().numpy()
                predictions = probabilities.round()
                
                validation_bce += criterion(out.type(torch.DoubleTensor).cuda(), y).item()
                validation_acc += np.mean(labels == predictions) * 100
                validation_auc += roc_auc_score(y_true=labels, y_score=probabilities)
                validation_SEN += recall_score(y_true=labels, y_pred=predictions)
                validation_SPE += SPC(y_true=labels, y_pred=predictions)
                validation_f1 += f1_score(y_true=labels, y_pred=predictions)
        
        # convert to batch loss:
        training_bce /= len(train)
        training_acc /= len(train)
        training_auc /= len(train)
        training_SEN /= len(train)
        training_SPE /= len(train)
        training_f1 /= len(train)
        
        validation_bce /= len(validation)
        validation_acc /= len(validation)
        validation_auc /= len(validation)
        validation_SEN /= len(validation)
        validation_SPE /= len(validation)
        validation_f1 /= len(validation)
        scheduler.step()

        torch.save(net.state_dict(), 'checkpoint/'+model +'_'+str(epoch)+ '_net.pt')
        # torch.save(net.state_dict(), f'epoch{epoch}.pt')
        error_stats = [training_bce, training_acc, training_auc, training_SEN, training_SPE, training_f1,
                       validation_bce, validation_acc, validation_auc, validation_SEN, validation_SPE, validation_f1]
        error_df = error_df.append(pd.Series(error_stats, index=error_df.columns), ignore_index=True)
        print(
            '{}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t\t{:.4f}\t{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
                .format(epoch, optimizer.param_groups[0]['lr'], *error_stats))


optimizer = torch.optim.Adam(net.parameters(), lr=0.003)  # 5e-6)
net = nn.DataParallel(net, device_ids=[0, 1, 2, 3]) # multi-gpu

train_model(net,
            train_loader,
            val_loader,
            optimizer,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            max_epoch=30,
            verbose=False)
model = model + '-ALL' # save model
torch.save(net.state_dict(), 'checkpoint/'+model + '_net.pth')

error_df.to_csv('./' + model + '.csv')
