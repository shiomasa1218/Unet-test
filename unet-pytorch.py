# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

num_classes = 2
img_height,img_width = 236,236
out_height,out_width = 52,52
GPU = False
torch.manual_seed(0)


# -

def crop_layer(layer,size):
    _,_,h,w = layer.size()
    _,_,_h,_w = size
    ph = int((h-_h)/2)
    pw = int((w-_w)/2)
    return layer[:,:,ph:ph+_h,pw:pw+_w]


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        
        base = 64
        
        self.enc1 = torch.nn.Sequential()
        for i in range(2):
            f = 3 if i == 0 else base
            self.enc1.add_module("enc1_{}".format(i+1), torch.nn.Conv2d(f, base, kernel_size=3, padding=0, stride=1))
            self.enc1.add_module("enc1_relu_{}".format(i+1), torch.nn.ReLU())
            self.enc1.add_module("enc1_bn_{}".format(i+1),torch.nn.BatchNorm2d(base))
            
        self.enc2 = torch.nn.Sequential()
        for i in range(2):
            f = base if i == 0 else base*2
            self.enc2.add_module("enc2_{}".format(i+1), torch.nn.Conv2d(f, base*2, kernel_size=3, padding=0, stride=1))
            self.enc2.add_module("enc2_relu_{}".format(i+1), torch.nn.ReLU())
            self.enc2.add_module("enc2_bn_{}".format(i+1), torch.nn.BatchNorm2d(base*2))
            
        self.enc3 = torch.nn.Sequential()
        for i in range(2):
            f = base*2 if i == 0 else base*2*2
            self.enc3.add_module("enc3_{}".format(i+1), torch.nn.Conv2d(f, base*2*2, kernel_size=3, padding=0, stride=1))
            self.enc3.add_module("enc3_relu_{}".format(i+1), torch.nn.ReLU())
            self.enc3.add_module("enc3_bn_{}".format(i+1), torch.nn.BatchNorm2d(base*2*2))
        
        self.enc4 = torch.nn.Sequential()
        for i in range(2):
            f = base*2*2 if i == 0 else base*2*2*2
            self.enc4.add_module("enc4_{}".format(i+1), torch.nn.Conv2d(f, base*2*2*2, kernel_size=3, padding=0, stride=1))
            self.enc4.add_module("enc4_relu_{}".format(i+1), torch.nn.ReLU())
            self.enc4.add_module("enc4_bn_{}".format(i+1), torch.nn.BatchNorm2d(base*2*2*2))

        self.enc5 = torch.nn.Sequential()
        for i in range(2):
            f = base*2*2*2 if i == 0 else base*2*2*2*2
            self.enc5.add_module("enc5_{}".format(i+1), torch.nn.Conv2d(f, base*2*2*2*2, kernel_size=3, padding=0, stride=1))
            self.enc5.add_module("enc5_relu_{}".format(i+1), torch.nn.ReLU())
            self.enc5.add_module("enc5_bn_{}".format(i+1), torch.nn.BatchNorm2d(base*2*2*2*2))
            
        self.tconv4 = torch.nn.ConvTranspose2d(base*2*2*2*2, base*2*2*2, kernel_size=2, stride=2)
        self.tconv4_bn = torch.nn.BatchNorm2d(base*2*2*2)
        
        self.dec4 = torch.nn.Sequential()
        for i in range(2):
            f = base*2*2*2*2 if i == 0 else base*2*2*2
            self.dec4.add_module("dec4_{}".format(i+1), torch.nn.Conv2d(f,base*2*2*2, kernel_size=3, padding=0, stride=1))
            self.dec4.add_module("dec4_relu_{}".format(i+1), torch.nn.ReLU())
            self.dec4.add_module("dec4_bn_{}".format(i+1), torch.nn.BatchNorm2d(base*2*2*2))
        
        self.tconv3 = torch.nn.ConvTranspose2d(base*2*2*2, base*2*2, kernel_size=2, stride=2)
        self.tconv3_bn = torch.nn.BatchNorm2d(base*2*2)
        
        self.dec3 = torch.nn.Sequential()
        for i in range(2):
            f = base*2*2*2 if i == 0 else base*2*2
            self.dec3.add_module("dec3_{}".format(i+1), torch.nn.Conv2d(f, base*2*2, kernel_size=3, padding=0, stride=1))
            self.dec3.add_module("dec3_relu_{}".format(i+1), torch.nn.ReLU())
            self.dec3.add_module("dec3_bn_{}".format(i+1), torch.nn.BatchNorm2d(base*2*2))
            
        self.tconv2 = torch.nn.ConvTranspose2d(base*2*2, base*2, kernel_size=3, stride=2)
        self.tconv2_bn = torch.nn.BatchNorm2d(base*2)
        
        self.dec2 = torch.nn.Sequential()
        for i in range(2):
            f = base*2*2 if i == 0 else base*2
            self.dec2.add_module("dec2_{}".format(i+1), torch.nn.Conv2d(f, base*2, kernel_size=3, padding=0, stride=1))
            self.dec2.add_module("dec2_relu_{}".format(i+1), torch.nn.ReLU())
            self.dec2.add_module("dec2_bn_{}".format(i+1), torch.nn.BatchNorm2d(base*2))
            
        self.tconv1 = torch.nn.ConvTranspose2d(base*2, base, kernel_size=3, stride=2)
        self.tconv1_bn = torch.nn.BatchNorm2d(base)
        
        self.dec1 = torch.nn.Sequential()
        for i in range(2):
            f = base*2 if i == 0 else base
            self.dec1.add_module("dec1_{}".format(i+1), torch.nn.Conv2d(f, base, kernel_size=3, padding=0, stride=1))
            self.dec1.add_module("dec1_relu_{}".format(i+1), torch.nn.ReLU())
            self.dec1.add_module("dec1_bn_{}".format(i+1), troch.nn.BatchNorm2d(base))
            
        self.out = torch.nn.Conv2d(base, num_classes+1, kernel_size=1, padding=0, stride=1)
        
        
    def forward(self, x):
        
        x_enc1 = self.enc1(x)
        x = F.max_pool2d(x_enc1, 2, stride=2, padding=0)
        
        x_enc2 = self.enc2(x)
        x = F.max_pool2d(x_enc2, 2, stride=2, padding=0)
        
        x_enc3 = self.enc3(x)
        x = F.max_pool2d(x_enc3, 2, stride=2, padding=0)
        
        x_enc4 = self.enc4(x)
        x = F.max_pool2d(x_enc4, 2, stride=2, padding=0)
        
        x = self.enc5(x)
        
        x = self.tconv4_bn(self.tconv4(x))
        _x = crop_layer(x_enc4, x.size())
        x = torch.cat((_x,x), dim=1)
        x = self.dec4(x)
        
        x = self.tconv3_bn(self.tconv3(x))
        _x = crop_layer(x_enc3, x.size())
        x = torch.cat((_x,x), dim=1)
        x = self.dec3(x)
        
        x = self.tconv2_bn(self.tconv2(x))
        _x = crop_layer(x_enc2, x.size())
        x = torch.cat((_x,x), dim=1)
        x = self.dec2(x)
        
        x = self.tconv1_bn(self.tconv1(x))
        _x = crop_layer(x_enc1, x.size())
        x = torch.cat((_x,x), dim=1)
        x = self.dec1(x)
        
        x = self.out(x)

CLS = {
    'akahara':[0,0,128],
    'madara':[0,128,0]
}


def data_load(path, hf=False, vf=False):
    xs = []
    ts = []
    paths = []
    
    for dir_path in glob(path+'/*'):
        for path in glob(dir_path+'/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width,img_height)).astype(np.float32)
            x /= 255.
            x = x[..., ::-1]
            xs.append(x)
            
            gt_path
