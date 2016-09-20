#coding=utf8
import lasagne
import numpy as np
import theano
import theano.tensor as T

def get_dataset():
    x=np.random.random((10000,10))
    y=np.zeros((10000))
    for idx,i in enumerate(x):
        if 0.3<i[0]<0.7:
            y[idx]=1
        elif i[0]>=0.7:
            y[idx]=2
    return x,y


x=T.vector('x')
y=T.matrix('y')
x1=T.vector('x1')
x2=T.matrix('all')

# x_shared=theano.shared(x)
dimention=10
batch_size=16

l_in = lasagne.layers.InputLayer(shape=(None, 1,dimention))
l_theta = lasagne.layers.DenseLayer(l_in,3,W=lasagne.init.Normal(std=0.1))
l_mu=lasagne.layers.NonlinearityLayer(l_theta,nonlinearity=lasagne.nonlinearities.softmax)

# probas = lasagne.layers.helper.get_output(l_mu, {l_in: x_shared})
xx,yy=get_dataset()
yy=np.int32(yy)
pass
