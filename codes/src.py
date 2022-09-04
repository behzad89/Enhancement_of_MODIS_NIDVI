# Name: src.py
# Description: The tools to to train and validate the model
# Author: Behzad Valipour Sh. <behzad.valipour@outlook.com>
# Date: 04.09.2022

'''
lines (17 sloc)  1.05 KB

MIT License

Copyright (c) 2022 Behzad Valipour Sh.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn, optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

import rasterio as rio
from rasterio.plot import reshape_as_image

import os,sys


class LoadImageData(Dataset):
    def __init__(self,datasetPath,transform=None):
        # data loading
        self.path = datasetPath
        self.dataset = os.listdir(datasetPath)
        self.n_samples = len(self.dataset)
        
        self.transform = transform
    def __len__(self):
        return self.n_samples

    def __getitem__(self,idx):
        image = rio.open(f'{self.path}/{self.dataset[idx]}').read()
        self.feat = reshape_as_image(image[:4,...])
        self.label = np.expand_dims(image[4,...],0)
        
        if self.transform:
            transformedX = self.transform(image = self.feat, mask= self.label)
            self.feat = transformedX["image"]
            self.label = transformedX["mask"]
        
        return self.feat,self.label[:,8:-8,8:-8]
    
class NDVIModelNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cnn_layers = nn.Sequential(
            
            # input patches: 4*128*128
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=9, stride=1, padding='valid'), # 64*120*120
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding='valid'), # 32*120*120
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding='valid'), # 1*116*116
        )
        
    def forward(self,inPut):
        X = self.cnn_layers(inPut)
        return X
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding='valid'):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.active = nn.ReLU(inplace=True)

    def forward(self,X):
        X = self.conv(X)
        return self.active(X)
    
class NDVIModelUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1_d1 = ConvBlock(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding='valid')
        self.conv_2_d1 = ConvBlock(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding='valid')
        
        self.conv_1_d2 = ConvBlock(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding='valid')
        self.conv_2_d2 = ConvBlock(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding='valid')
        
        self.conv_1_U1 = ConvBlock(in_channels=128, out_channels=64, kernel_size=2, stride=1, padding='valid')
        self.conv_2_U1 = ConvBlock(in_channels=64, out_channels=32, kernel_size=2, stride=1, padding='valid')
        
        self.conv_1_U2 = ConvBlock(in_channels=64, out_channels=32, kernel_size=2, stride=1, padding='valid')
        self.conv_2_U2 = ConvBlock(in_channels=32, out_channels=16, kernel_size=2, stride=1, padding='valid')
        
        self.LastConvBlock = nn.Conv2d(16, 1, kernel_size=2, stride=1, padding='valid')
        
        self.MaxPool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2.0)
        
    def forward(self,X):
        C1 = self.conv_1_d1(X)
        C2 = self.conv_2_d1(C1)
        P1 = self.MaxPool(C2)
        C3 = self.conv_1_d2(P1)
        C4 = self.conv_2_d2(C3)
        P2 = self.MaxPool(C4)
        
        U1 = self.upsample(P2)
        concat1 = torch.concat([U1,C4], axis=1)
        C5 = self.conv_1_U1(concat1)
        C6 = self.conv_2_U1(C5)
        U2 = self.upsample(C6)
        concat2 = torch.concat([U2[:,:,1:,1:],C2[:,:,5:-5,5:-5]], axis=1)
        C7 = self.conv_1_U2(concat2)
        C8 = self.conv_2_U2(C7)
        
        Y = self.LastConvBlock(C8)
        return Y
    
    
    
class NDVIModel(pl.LightningModule):
    def __init__(self,learning_rate = 0.001):
        super(NDVIModel,self).__init__()
        
        
        self.model = NDVIModelUNet()
        self.learning_rate = learning_rate
        self.loss = nn.MSELoss()
        self.save_hyperparameters()
        
    def forward(self,InPut):
        return self.model(InPut)
    
    def training_step(self, batch, batch_idx):
        X,y = batch
        
        pred = self(X)
        loss = self.loss(pred,y)
        self.log('Train_Loss', loss, on_epoch=True, on_step=True,prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        X,y = batch
        
        pred = self(X)
        loss = self.loss(pred,y)
        self.log('validation_Loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
        
        
    def test_step(self, batch, batch_idx):
        X,y = batch
        pred = self(X)
        loss = self.loss(pred,y)
        self.log('Test_Loss', loss, on_epoch=True, on_step=True)
        return loss
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(),lr=self.learning_rate)
        return optimizer