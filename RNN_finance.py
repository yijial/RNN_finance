#-----------------------------------------------------------------------
#     VERSION HISTORY OF RNN_finance.py
#-----------------------------------------------------------------------

# Version: v0.1.0
# Date: 05/03/2016
# Author: Yijia Liu
# Comment: Initial version. 

# Version: v1.0.1
# Date: 05/10/2016
# Author: Yijia Liu
# Comment: Automatic decreasing step size, momentum-dependent current direction and 
#            visualization added. Implementation modified.                        . 

# Version: v1.1.1
# Date: 05/17/2016
# Author: Yijia Liu
# Comment: auto increase/decrease step size, added one hidden layer
#           modefied activation function and improved result-visualization

#-----------------------------------------------------------------------
#     BEGINNING OF THE CODE
#-----------------------------------------------------------------------

import numpy as np
import theano
import theano.tensor as TT
import pandas as pd
import os
import pylab as pl
import matplotlib.pyplot as plt
import math
# import lasagne
# from collections import OrderedDict
from sklearn.cross_validation import train_test_split
from matplotlib.backends.backend_pdf import PdfPages


# set path to the working directory
os.chdir("/Users/yijialiu/Documents/WXML_NeuralNets/RNN_finance")
#-----------------------------------------------------------------------
#     LOADING DATA
#-----------------------------------------------------------------------
# read in returns as data frame
data = pd.read_csv('SimpleReturn.csv')
# usd_index = pd.read_csv('USD_index_DAY.csv')
# convert into matrix and drop the column of date
rt = data.as_matrix()[:,range(1, data.shape[1])]

#-----------------------------------------------------------------------
#     PARAMETER SETTING
#-----------------------------------------------------------------------
# number of hidden units
n = 100
# number of output units
nout = 1
# momentum of current direction
momentum = 0.7
# number of iterations
num_epochs = 200
# initial stepeize
stepsize = 10.0

# input (where first dimension is time)
u = TT.matrix()
# target (where first dimension is time)
t = TT.matrix()
# initial hidden state of the RNN
h0 = TT.vector()
# learning rate
lr = TT.scalar()
# recurrent weights as a theano.shared variable
W1 = TT.matrix()
W2 = TT.matrix()
Whh = TT.matrix()
# input to hidden layer weights
W_in = TT.matrix()
# hidden to output layer weights
W_out = TT.matrix()
GW1 = TT.matrix()
GW2 = TT.matrix()
GWhh = TT.matrix()
GW_in = TT.matrix()
GW_out = TT.matrix()

# training prediction
y = TT.matrix()
# testing prediction
yt = TT.matrix()

#-----------------------------------------------------------------------
#    FUNCTION DEFINING
#-----------------------------------------------------------------------
# recurrent function (using tanh activation function) and linear output
# activation function
def step(u_t, h1_tm1, h2_tm1, W1, W2, Whh, W_in, W_out):
    h1_t = TT.nnet.relu(TT.dot(u_t, W_in) + TT.dot(h1_tm1, W1))
    h2_t = TT.nnet.relu(TT.dot(h1_t, Whh) + TT.dot(h2_tm1, W2))
    y_t = TT.dot(h2_t, W_out)
    return h1_t, h2_t, y_t

[h1, h2, y], _ = theano.scan(step,
                        sequences=u,
                        outputs_info=[h0, h0, None],
                        non_sequences=[W1, W2, Whh, W_in, W_out])
                        
# error between output and target
lam = 15.0
error = (((y - t)**2).sum() + lam * ((GW1**2).sum() + (GW2**2).sum() + (GWhh**2).sum() + (GW_in**2).sum() +
        (GW_out**2).sum())) / y.shape[0]

# gradients on the weights using BPTT
gW1, gW2, gWhh, gW_in, gW_out = TT.grad(error, [W1, W2, Whh, W_in, W_out])

# training function, that computes the error and updates the weights using
# SGD.
fn = theano.function(inputs=[h0, u, t, W1, W2, Whh, W_in, W_out, GW1, GW2, GWhh, GW_in, GW_out],
                      outputs = [error,y])
                    #   updates = OrderedDict([(W, W - lr * gW),
                    #          (W_in, W_in - lr * gW_in),
                    #          (W_out, W_out - lr * gW_out)]))
    
# computes gradient respect to given weights
gradient = theano.function(inputs = [h0,u,t,W1, W2, Whh,W_in,W_out, GW1, GW2, GWhh, GW_in, GW_out],
                            outputs = [gW1, gW2, gWhh,gW_in,gW_out])

#-----------------------------------------------------------------------
#     TRAINING IMPLEMENTATION
#-----------------------------------------------------------------------
# initialzie plotting panel
pp = PdfPages('stockPredictions.pdf')
pt = PdfPages('stockPredictions_test.pdf')

for i in range(5,6):
    # input
    X = rt[range(0,(len(rt)-1)), :][:, [i,6,7,8,9,10,11,12,13]]
    # number of input units
    nin = X.shape[1]
    # output
    y = np.transpose(np.mat(rt[range(1,len(rt)), i]))
    # data split
    X_train = X[range(int(len(X)*0.98)), :]
    X_test = X[range(int(len(X)*0.98)+1, len(X)), :]
    y_train = y[range(int(len(X)*0.98)), :]
    y_test = y[range(int(len(X)*0.98)+1, len(X)), :]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, 
    #     test_size=0.5, random_state=42)
    
    X_train = np.asarray(X_train,dtype='float64')
    y_train = np.asarray(y_train,dtype='float64')
    # for i in range(1):
    #     X_train[:,i] = X_train[:,i] / np.mean(X_train[:,i])
    #     y_train[:,i] = y_train[:,i] / np.mean(y_train[:,i])
    #y_train = y_train/np.sqrt((y_train**2).sum())
    X_test = np.asarray(X_test,dtype='float64')
    y_test = np.asarray(y_test,dtype='float64')
    
    stepsize = 10
    # print header
    print("    Epoch      |    Train err  |   Stepsize  |   Grad Norm ")

    # initialize training weight
    W1 = np.random.uniform(size=(n, n), low=-0.1, high=0.1)
    W2 = np.random.uniform(size=(n, n), low=-0.1, high=0.1)
    Whh = np.random.uniform(size=(n, n), low=-0.1, high=0.1)
    W_in = np.random.uniform(size=(nin, n), low=-0.1, high=0.1)
    W_out = np.random.uniform(size=(n, nout), low=-0.1, high=0.1)
    W1_temp = W1
    W2_temp = W2
    Whh_temp = Whh
    W_in_temp = W_in
    W_out_temp = W_out
    GW1 = np.zeros((n,n))
    GW2 = np.zeros((n,n))
    GWhh = np.zeros((n,n))
    GW_in = np.zeros((nin,n))
    GW_out = np.zeros((n,nout))
    # GW1, GW2, GWhh, GW_in, GW_out = gradient(np.zeros(n), X_train, y_train, W1, W2, Whh, W_in, W_out,
    #                                     GW1, GW2, GWhh, GW_in, GW_out)
      
    # initialize training error  
    pref , _ = fn(np.zeros(n),X_train,y_train,W1, W2, Whh,W_in,W_out,GW1, GW2, GWhh, GW_in, GW_out)
    train_pref = pref
    
    # initialize current direction
    cr_w1 = np.zeros((n,n))
    cr_w2 = np.zeros((n,n))
    cr_whh = np.zeros((n,n))
    cr_w_in = np.zeros((nin,n))
    cr_w_out = np.zeros((n,nout))
    cr_w1_temp = cr_w1
    cr_w2_temp = cr_w2
    cr_whh_temp = cr_whh
    cr_w_in_temp = cr_w_in
    cr_w_out_temp = cr_w_out

    for epoch in range(num_epochs):
        # computes gradient of weights from last step
        GW1, GW2, GWhh, GW_in, GW_out = gradient(np.zeros(n), X_train, y_train, W1, W2, Whh, W_in, W_out,
                                        GW1, GW2, GWhh, GW_in, GW_out)
        # computes new direction
        cr_w1_temp = momentum * cr_w1 + (1- momentum) * GW1
        cr_w2_temp = momentum * cr_w2 + (1- momentum) * GW2
        cr_whh_temp = momentum * cr_whh + (1- momentum) * GWhh
        cr_w_in_temp = momentum * cr_w_in + (1- momentum) * GW_in
        cr_w_out_temp = momentum * cr_w_out + (1- momentum) * GW_out
        # computes new training weights
        W1_temp = W1 - stepsize * cr_w1_temp
        W2_temp = W2 - stepsize * cr_w2_temp
        Whh_temp = Whh - stepsize * cr_whh_temp
        W_in_temp = W_in - stepsize * cr_w_in_temp
        W_out_temp = W_out - stepsize * cr_w_out_temp
        # calculate training error and prediction
        train_pref, y_train_predict = fn(np.zeros(n),X_train,y_train,W1_temp,W2_temp,Whh_temp,W_in_temp,W_out_temp,
                                        GW1, GW2, GWhh, GW_in, GW_out)
        # if(train_pref > 1e100):
        #     break
        pred_imp = - stepsize * ((GW1**2).sum() + (GW2**2).sum() + (GWhh**2).sum() + (GW_in**2).sum() + (GW_out**2).sum()) / 2
        act_imp = train_pref - pref
        ratio = act_imp / pred_imp
        if(ratio < 0.0):
            print("back trace")
            stepsize = stepsize * 0.7
        else:
            cr_w1 = np.copy(cr_w1_temp)
            cr_w2 = np.copy(cr_w2_temp)
            cr_whh = np.copy(cr_whh_temp)
            cr_w_in = np.copy(cr_w_in_temp)
            cr_w_out = np.copy(cr_w_out_temp)
            W1 = np.copy(W1_temp)
            W2 = np.copy(W2_temp)
            Whh = np.copy(Whh_temp)
            W_in = np.copy(W_in_temp)
            W_out = np.copy(W_out_temp)
            pref = train_pref
            if(ratio > 0.8):
                print("nice direction")
                stepsize = stepsize * 2
    
        # calculates testing error    
        test_pref,y_test_predict  = fn(np.zeros(n),X_test,y_test,W1,W2,Whh,W_in,W_out,GW1, GW2, GWhh, GW_in, GW_out)
    
        # print result for desired peroids
        if np.mod(epoch,1) == 0:
            print("{0:15}|{1:15}|{2:15}|{3:15}|{4:15}".format(epoch, train_pref, test_pref,
                    stepsize, (i+1)))
    prd_train = range(len(X_train))
    prd_test = range(len(X_test))
    pl.plot(prd_train,y_train[:,0],prd_train,y_train_predict[:,0])
    pl.savefig("vpacx_train.eps")
    pp.savefig()
    pl.clf()
    pl.plot(prd_test,y_test[:,0],prd_test,y_test_predict[:,0])
    pl.savefig("vpacx_test.eps")
    pt.savefig()
    pl.clf()
pp.close()
pt.close()



#-----------------------------------------------------------------------
#     VISUALIZING TRAINING RESULTS
#-----------------------------------------------------------------------


pl.clf()
pl.plot(prd_test,y_test[:,0],prd_test,y_test_predict[:,0])
pp.savefig()
pp.close()

pp.savefig("single_pred.eps")

pp = PdfPages('stockPredictions.pdf')
for i in range (X_train.shape[1]):
    pl.plot(prd,y_train[:,i],prd,y_train_predict[:,i])
    pp.savefig()
    pl.clf()
pp.close()

f, axarr = plt.subplots(2, 3,sharex='col', sharey='row')
m = 0
for i in range(2):
    for j in range(3):
        axarr[i, j].plot(prd,y_train[:,m],prd,y_train_predict[:,m])
        axarr[i, j].set_title("stock"+ str(m) + "_prediction")
        m = m+1
plt.show()

