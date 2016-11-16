#coding=utf8
import gzip
import pickle
import numpy as np
import random

import theano
from theano import tensor as T

import lasagne
from lasagne.nonlinearities import rectify, softmax, very_leaky_rectify
from lasagne.updates import nesterov_momentum
from lasagne.layers import InputLayer, MaxPool2DLayer, Conv2DLayer, DenseLayer, DropoutLayer, helper
from sklearn.preprocessing import LabelBinarizer,label_binarize
import image

BATCHSIZE = 32
PIXELS = 30
imageSize = PIXELS * PIXELS
num_features = imageSize
size=1000

# set up functions needed to train the network
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def lasagne_model():
    l_in = InputLayer(shape=(None, 1, 30, 30))

    l_in =lasagne.layers.NonlinearityLayer(l_in,lasagne.nonlinearities.tanh)
    l_conv1 = Conv2DLayer(l_in, num_filters = 128, filter_size=(3,3), nonlinearity=rectify)
    l_conv1b = Conv2DLayer(l_conv1, num_filters = 128, filter_size=(3,3), nonlinearity=rectify)
    l_pool1 = MaxPool2DLayer(l_conv1b, pool_size=(2,2))
    # l_dropout1 = DropoutLayer(l_pool1, p=0.2)

    l_conv2 = Conv2DLayer(l_pool1, num_filters = 256, filter_size=(3,3), nonlinearity=rectify)
    l_conv2b = Conv2DLayer(l_conv2, num_filters = 256, filter_size=(3,3), nonlinearity=rectify)
    l_pool2 = MaxPool2DLayer(l_conv2b, pool_size=(2,2))
    # l_dropout2 = DropoutLayer(l_pool2, p=0.2)

    l_hidden3 = DenseLayer(l_pool2, num_units = 1024, nonlinearity=rectify)
    # l_dropout3 = DropoutLayer(l_hidden3, p=0.5)

    l_hidden4 = DenseLayer(l_hidden3, num_units = 1024, nonlinearity=rectify)
    # l_dropout4 = DropoutLayer(l_hidden4, p=0.5)

    l_out = DenseLayer(l_hidden4, num_units=169, nonlinearity=softmax)

    return l_out

def main():
    # load the training and validation data sets
    labels=int(0.7*image.all_count)
    data=image.ddd
    train_X=np.zeros([size,1,PIXELS,PIXELS])
    train_y=np.zeros([size,labels])
    for i in range(size):
        label=random.sample(range(labels),1)[0]
        train_X[i,0]=0.01*random.sample(data[label],1)[0]
        train_y[i]=label_binarize([label],range(labels))[0]

    X = T.tensor4()
    Y = T.matrix()

    # set up theano functions to generate output by feeding data through network
    output_layer = lasagne_model()
    output_train = lasagne.layers.get_output(output_layer, X)
    output_valid = lasagne.layers.get_output(output_layer, X, deterministic=True)

    # set up the loss that we aim to minimize
    loss_train = T.mean(T.nnet.categorical_crossentropy(output_train, Y))
    loss_valid = T.mean(T.nnet.categorical_crossentropy(output_valid, Y))

    # prediction functions for classifications
    pred = T.argmax(output_train, axis=1)
    pred_valid = T.argmax(output_valid, axis=1)

    # get parameters from network and set up sgd with nesterov momentum to update parameters
    params = lasagne.layers.get_all_params(output_layer)
    updates = nesterov_momentum(loss_train, params, learning_rate=0.03, momentum=0.9)
    # updates =lasagne.updates.adagrad(loss_train, params, learning_rate=0.003)

    # set up training and prediction functions
    train = theano.function(inputs=[X, Y], outputs=[loss_train,pred_valid], updates=updates, allow_input_downcast=True)
    valid = theano.function(inputs=[X, Y], outputs=loss_valid, allow_input_downcast=True)
    predict_valid = theano.function(inputs=[X], outputs=pred_valid, allow_input_downcast=True)

    # loop over training functions for however many iterations, print information while training
    train_eval = []
    valid_eval = []
    valid_acc = []

    for i in range(450):
        batch_total_number = len(train_X) / BATCHSIZE
        for idx_batch in range (batch_total_number):
            xx_batch = np.float32(train_X[idx_batch * BATCHSIZE:(idx_batch + 1) * BATCHSIZE])
            yy_batch = np.float32(train_y[idx_batch * BATCHSIZE:(idx_batch + 1) * BATCHSIZE])
               
            train_loss ,pred = train(xx_batch,yy_batch)
            print i,idx_batch,'| Tloss:', train_loss,'| Count:',np.count_nonzero(np.int32(pred ==np.argmax(yy_batch,axis=1)))
            print pred
            print np.argmax(yy_batch,axis=1)

        acc=0
        for j in range(batch_total_number):
            x_batch = np.float32(train_X[idx_batch * BATCHSIZE:(idx_batch + 1) * BATCHSIZE])
            y_batch = np.float32(train_y[idx_batch * BATCHSIZE:(idx_batch + 1) * BATCHSIZE])
            pred = predict_valid(x_batch)
            acc += np.count_nonzero(np.int32(pred ==np.argmax(y_batch,axis=1)))
        acc=float(acc)/(BATCHSIZE*batch_total_number)
        print 'iter:', i,idx_batch, '| Tloss:', train_loss,'|Acc:',acc


    # save weights
    all_params = helper.get_all_param_values(output_layer)
    f = gzip.open('params/weights.pklz', 'wb')
    pickle.dump(all_params, f)
    f.close()

if __name__ == '__main__':
    main()
