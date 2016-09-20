#coding=utf8
import lasagne
import numpy as np
import theano
import theano.tensor as T

x=T.vector('x')
y=T.matrix('y')
x1=T.vector('x1')
x2=T.matrix('all')
dimention=10
batch_size=16

l_in = lasagne.layers.InputLayer(shape=(None, 1,dimention))
l_theta = lasagne.layers.DenseLayer(l_in,3,W=lasagne.init.Normal(std=0.1))
l_mu=lasagne.layers.NonlinearityLayer(l_theta,nonlinearity=lasagne.nonlinearities.softmax)

