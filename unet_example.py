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
import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.transforms as transforms
from torch.autograd import Function, Variable

from tqdm import tqdm
#import pydensecrf.densecrf as dcrf
import random
from PIL import Image
import matplotlib.pyplot as plt
# %matplotlib inline
# -

args = {}
args['dir_img'] = 'data/train/'
args['dir_mask'] = 'data/train_mask/'
args['dir_img_test'] = 'data/test/'
args['dir_checkpoint'] = 'checkpoint/'
args['val_percent'] = 0.05
args['scale'] = 0.5
args['n'] = 2
args['batch_size'] = 2
args['epoch'] = 5
args['threshold'] = 0.5


# +
def to_cropped(args, ids, dir, suffix):
    
    for id, pos in ids:
        img = Image.open(dir + id + suffix)

        w = img.size[0]
        h = img.size[1]
        newW = int(w * args['scale'])
        newH = int(h * args['scale'])

        img = img.resize((newW, newH))    
        img = img.crop((0, 0, newW, newH))
        img = np.array(img, dtype=np.float32)
        
        h = img.shape[0]
        if pos == 0:
            img = img[:, :h]
        else:
            img = img[:, -h:]
        
        yield img

        
def get_img_mask(args, ids):
    img = to_cropped(args, ids, args['dir_img'], '.jpg')
    img = map(lambda x: np.transpose(x, axes=[2, 0, 1]), img)
    img = map(lambda x: x/255, img)

    mask = to_cropped(args, ids, args['dir_mask'], '_mask.gif')

    return zip(img, mask)


def batch(iterable, batch_size):
    
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


# +
ids_all = [f[:-4] for f in os.listdir(args['dir_img'])]
ids_all = [(id, i) for i in range(args['n']) for id in ids_all]
random.shuffle(ids_all)
n = int(len(ids_all) * args['val_percent'])
ids = {'train': ids_all[:-n], 'val': ids_all[-n:]}

len_train = len(ids['train'])
len_val = len(ids['val'])
# -


