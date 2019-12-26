# -*- coding: utf-8 -*-
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

# ### perceptron AND

import numpy as np
xs = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
ts = np.array([[-1],[-1],[-1],[1]],dtype=np.float32)

np.random.seed(0)
w = np.random.normal(0.,1,(3))
print("w >> ",w)

input_signal = np.hstack((xs,[[1],[1],[1],[1]])) 

input_signal

y = [sum(d) for d in w*input_signal]

y

# ### perceptron learning

import numpy as np
xs = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
ts = np.array([[-1],[-1],[-1],[1]],dtype=np.float32)
np.random.seed(0)
w = np.random.normal(0.,1,(3))
print("w >> ",w)
input_signal = np.hstack((xs,[[1],[1],[1],[1]])) 

lr = 0.1

rw = w
while True:
    y = [[sum(d)] for d in rw*input_signal]
    round_y = [ [1] if i[0] > 0 else [-1] for i in y]
    flg = round_y == ts
    if all(flg):
        break
    for i,n in enumerate(flg):
        if n[0] == False:
            En = ts[i][0]*input_signal[i]
            rw = rw + lr*En 
            print(rw)

for i,n in enumerate(xs):
    print("input :",n,"output : ",y[i][0])

# ### perceptron sigmoid

import numpy as np
xs = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
ts = np.array([0,0,0,1],dtype=np.float32)
np.random.seed(0)
w = np.random.normal(0.,1,(3))
print("w >> ",w)
input_signal = np.hstack((xs,[[1],[1],[1],[1]])) 
lr = 0.1


def sigmoid(x):
    return 1/(1+np.exp(-1*x))


# + jupyter={"outputs_hidden": true}
rw = w
for i in range(5000):
    ys = sigmoid(np.dot(input_signal,rw))
    
    En = -(ts-ys)*ys*(1-ys)

    grad = np.dot(input_signal.T,En)

    rw -= lr*grad 
    print(rw)
    
# -

for i in range(4):
    ys = sigmoid(np.dot(input_signal[i],w))
    print("input :",xs[i],"output : ",ys)

# ### perceptron bias
# biasを別表記にする

import numpy as np
xs = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
ts = np.array([0,0,0,1],dtype=np.float32)
np.random.seed(0)
wb = np.random.normal(0.,1,(3))
w = wb[0:-1]
b = wb[-1]
print("w:",w,"b:",b)
lr = 0.1


def sigmoid(x):
    return 1/(1+np.exp(-1*x))


# + jupyter={"outputs_hidden": true}
rw = w
rb = b
for i in range(5000):
    ys = sigmoid(np.dot(xs,rw)+rb)
    
    En = -(ts-ys)*ys*(1-ys)

    grad = np.dot(xs.T,En)
    grad_b = np.dot(np.ones([En.shape[0]]),En)
    rw -= lr*grad
    rb -= lr*grad_b
    print(rw,rb)
# -

for i in range(4):
    ys = sigmoid(np.dot(xs[i],rw)+rb)
    print("input :",xs[i],"output : ",ys)

# ### perceptron OR

import numpy as np
xs = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
ts = np.array([0,1,1,1],dtype=np.float32)
np.random.seed(0)
wb = np.random.normal(0.,1,(3))
w = wb[0:-1]
b = wb[-1]
print("w:",w,"b:",b)
lr = 0.1


def sigmoid(x):
    return 1/(1+np.exp(-1*x))


rw = w
rb = b
for i in range(5000):
    ys = sigmoid(np.dot(xs,rw)+rb)
    
    En = -(ts-ys)*ys*(1-ys)

    grad = np.dot(xs.T,En)
    grad_b = np.dot(np.ones([En.shape[0]]),En)
    rw -= lr*grad
    rb -= lr*grad_b
#     print(rw,rb)

for i in range(4):
    ys = sigmoid(np.dot(xs[i],rw)+rb)
    print("input :",xs[i],"output : ",ys)

# ### perceptron NOT

import numpy as np
xs = np.array([[0],[1]],dtype=np.float32)
ts = np.array([1,0],dtype=np.float32)
np.random.seed(0)
wb = np.random.normal(0.,1,(2))
w = [wb[0]]
b = [wb[1]]
print("w:",w,"b:",b)
lr = 0.1


def sigmoid(x):
    return 1/(1+np.exp(-1*x))


# + jupyter={"outputs_hidden": true}
rw = w
rb = b
for i in range(5000):
    ys = sigmoid(np.dot(xs,rw)+rb)
    
    En = -(ts-ys)*ys*(1-ys)

    grad = np.dot(xs.T,En)
    grad_b = np.dot(np.ones([En.shape[0]]),En)
    rw -= lr*grad
    rb -= lr*grad_b
    print(rw,rb)
# -

for i in range(2):
    ys = sigmoid(np.dot(xs[i],rw)+rb)
    print("input :",xs[i],"output : ",ys)

# ### 2dense-perceptron feedforward

import numpy as np
xs = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
ts = np.array([[0],[1],[1],[0]],dtype=np.float32)
np.random.seed(0)
wb = np.random.normal(0.,1,(9))
w1 = wb[0:4].reshape(2,2)
b1 = wb[4:6]
w2 = wb[6:8].reshape(2,1)
b2 = [wb[8]]
print("w1:",w1,"\nb1:",b1,"\nw2:",w2,"\nb2:",b2)
lr = 0.1


def sigmoid(x):
    return 1/(1+np.exp(-1*x))


# +
rw1 = w1
rb1 = b1
rw2 = w2
rb2 = b2

for i in range(5000):
    ys1 = sigmoid(np.dot(xs,rw1)+rb1)
    ys2 = sigmoid(np.dot(ys1,rw2)+rb2)
    
    En = -(ts-ys2)*ys2*(1-ys2)
#     print(En)
#     print(ys1)
#     print(ys2)
    grad_w2 = np.dot(ys1.T,En)
    grad_b2 = np.dot(np.ones([En.shape[0]]),En)
    rw2 -= lr*grad_w2
    rb2 -= lr*grad_b2
    
    
    grad_u1 = np.dot(En,rw2.T)*ys1*(1-ys1)
    grad_w1 = np.dot(xs.T,grad_u1)
    grad_b1 = np.dot(np.ones([grad_u1.shape[0]]),grad_u1)
    rw1 -= lr*grad_w1
    rb1 -= lr*grad_b1
#     print(rw1,rb1,rw2,rb2)
# -

for i in range(4):
    ys1 = sigmoid(np.dot(xs[i],rw1)+rb1)
    ys2 = sigmoid(np.dot(ys1,rw2)+rb2)
    print("input :",xs[i],"output : ",ys2)

# ### multi-perceptron 

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

for i in range(5000):
    model.forward(xs)
    model.backward(ts)

for i in range(4):
    out = model.forward(xs[i])
    print("input :",xs[i],"output : ",out)


