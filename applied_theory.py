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

# ### image recognization

import numpy as np
import pdb
np.random.seed(0)
xs = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
ts = np.array([[0],[1],[1],[0]],dtype=np.float32)


def sigmoid(x):
    return 1/(1+np.exp(-1*x))


# +
class FullyConnectedLayer():
    def __init__(self, in_n, out_n, use_bias=True, activation=None):
        self.w = np.random.normal(0,1,[in_n,out_n])
        if use_bias:
            self.b = np.random.normal(0,1,[out_n])
        else:
            self.b = None
        if activation is not None:
            self.activation = activation
        else:
            self.activation = None
    
    def set_lr(self, lr=0.1):
        self.lr = lr
        
    def forward(self, feature_in):
        self.f_in = feature_in
#         pdb.set_trace()
        tmp = np.dot(feature_in,self.w)
        
        if self.b is not None:
            tmp += self.b
        if self.activation is not None:
            tmp = self.activation(tmp)
        
        self.ys = tmp
        return tmp
        
    def backward(self, w_pro, grad_pro):
#         pdb.set_trace()
        grad_u = np.dot(grad_pro,w_pro.T)
        if self.activation is sigmoid:
            grad_u *= (self.ys * (1-self.ys))
        grad_w = np.dot(self.f_in.T, grad_u)
        self.w -= self.lr*grad_w
        
        if self.b is not None:
            grad_b = np.dot(np.ones([grad_u.shape[0]]), grad_u)
            self.b -= self.lr*grad_b
        
        return grad_u
    
class Model():
    def __init__(self, *args, lr=0.1):
        self.layers = args
        for l in self.layers:
            l.set_lr(lr=lr)
    
    def forward(self,x):
        for l in self.layers:
            x = l.forward(x)
        self.ys = x
        return x
        
    def backward(self,t):
        En = -(t-self.ys)*self.ys*(1-self.ys)
        grad_pro = En
        ## referensed answer  
        w_pro = np.eye(En.shape[-1])
        ##
        for l in self.layers[::-1]:
            grad_pro = l.backward(w_pro=w_pro,grad_pro=grad_pro)
            w_pro = l.w


# -

model = Model(FullyConnectedLayer(in_n=2, out_n=64,activation= sigmoid),
              FullyConnectedLayer(in_n=64, out_n=32,activation= sigmoid),
              FullyConnectedLayer(in_n=32, out_n=1,activation= sigmoid))
