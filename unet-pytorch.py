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
            
            gt_path = path.replace("images","seg_images").replace(".jpg",".png")
            gt = cv2.imread(gt_path)
            gt = resize(gt,(out_width,out_height), interpolation=cv2.INTER_NEAREST)
            
            t = np.zeros((out_height, out_width), dtype=np.int)
            
            for i,(_,vs) in enumerate(CLS.items()):
                ind = (gt[...,0] == vs[0])*(gt[...,1] == vs[1])*(gt[...,2] == vs[2])
                t[ind] = i+1
                
                ts.append(t)
                paths.append(path)
                
                if hf:
                    xs.append(x[:,::-1])
                    ts.append(t[:,::-1])
                    paths.append(path)
                    
                if vf:
                    xs.append(x[::-1])
                    ts.append(t[::-1])
                    paths.append(path)
                    
                if hf and vf:
                    xs.append(x[::-1, ::-1])
                    ts.append(t[::-1, ::-1])
                    paths.append(path)
    
    xs = np.array(xs)
    ts = np.array(ts)
    xs = xs.transpose(0,3,1,2)
    return xs, ts, paths


def train():
    device = torch.device("cuda" if GPU else "cpu")
    
    model = UNet().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    
    xs, ts, paths = data_load('Dataset/train/images/', hf=True, vf=True)
    
    mb = 4
    mbi = 0
    train_N = len(xs)
    train_ind = np.arange(train_N)
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(100):
        if mbi+mb > train_N:
            mb_ind = train_N[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb - (train_N - mbi))]))
            mbi = mb - (train_N - mbi)
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb
            
        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)
        t = torch.tensor(ts[mb_ind], dtype=torch.float).to(device)
        
        # zero_grad: initialization
        opt.zero_grad()
        y = model(x)
        
        # permute: like np.transpose
        # contiguous : values remapping on contiguous memory region
        y = y.permute(0,2,3,1).contiguous()
        y = y.view(-1, num_classes+1)
        t = t.view(-1)
        
        y = F.log_softmax(y,dim=1)
        loss = torch.nn.CrossentropyLoss()(y,t)
        loss.backward()
        opt.step()
        
        pred = y.argmax(dim=1, keepdim=True)
        acc = pred.eq(t.view_as(pred)).sum().item()/ mb / img_height / img_width
        
        print("iter >>", i+1, ",loss >>", loss.item(), ",acc >>",acc)
        
    torch.save(model.state_dict(),"cnn.pt")


def test():
    device = torch.device("cuda" if GPU else "cpu")
    model = UNet().to(device)
    model.eval()
    model.load_state_dict(torch.load("cnn.pt"))
    
    xs, ts, paths = data_load("Dataset/test/images/")
    
    with torch.no_grad():
        for i in range(len(path)):
            x = xs[i]
            t = ts[i]
            path = paths[i]
            
            x = np.expend_dims(x, axis=0)
            x = torch.tensor(x,dtype=torch.float).to(device)
            
            pred = model(x)
            
            pred = pred.permute(0,2,3,1).reshape(-1, num_classes+1)
            pred = F.softmax(pred,dim=1)
            pred = pred.reshape(-1,out_height, out_width, num_classes+1)
            pred = pred.detach().cpu().numpy()[0]
            pred = pred.argmax(axis=-1)
            
            out = np.zeros((out_height, out_width, 3),dtype=np.uint8)
            for i, (_,vs) in enumerate(CLS.items()):
                out[pred == (i+1)] = vs
            
            print("in {}".format(path))
            
            plt.subplot(1,2,1)
            plt.imshow(x.detach().cpu().numpy()[0].tarnspose(1,2,0))
            plt.subplot(1,2,2)
            plt.imshow(out[...,::-1])
            plt.show()


def arg_parse():
    parser = argparse.ArgumentParser(description="CNN implemented with keras")
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--test", dest="test", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()

    if args.train:
        train()
    if args.test:
        test()

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")


