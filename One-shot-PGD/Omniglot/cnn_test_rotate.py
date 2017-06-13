#coding=utf8
import gzip
import pickle
import numpy as np
import random
import time

import theano
from theano import tensor as T

import lasagne
from lasagne.nonlinearities import rectify, softmax, very_leaky_rectify
from lasagne.updates import nesterov_momentum
from lasagne.layers import InputLayer, MaxPool2DLayer, Conv2DLayer, DenseLayer, DropoutLayer, helper
from sklearn.preprocessing import LabelBinarizer,label_binarize
import image_all_rotate as image
from tqdm import tqdm

BATCHSIZE = 128
PIXELS = 20
imageSize = PIXELS * PIXELS
num_features = imageSize
num_labels=4*964
num_images=964*4*15#共77120个图，算上了旋转的
# num_images=964 #共77120个图，算上了旋转的
h_dimension=300
valid_size=1280*5
size=num_images-valid_size

# set up functions needed to train the network
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def lasagne_model():
    l_in = InputLayer(shape=(None, 1, 20, 20))

    # l_in =lasagne.layers.NonlinearityLayer(l_in,lasagne.nonlinearities.tanh)
    l_conv1 = Conv2DLayer(l_in, num_filters = 128, filter_size=(3,3), nonlinearity=rectify)
    l_conv1b = Conv2DLayer(l_conv1, num_filters = 128, filter_size=(3,3), nonlinearity=rectify)
    l_pool1 = MaxPool2DLayer(l_conv1b, pool_size=(2,2))
    # l_dropout1 = DropoutLayer(l_pool1, p=0.2)

    l_conv2 = Conv2DLayer(l_pool1, num_filters = 256, filter_size=(3,3), nonlinearity=rectify)
    l_conv2b = Conv2DLayer(l_conv2, num_filters = 256, filter_size=(3,3), nonlinearity=rectify)
    l_pool2 = MaxPool2DLayer(l_conv2b, pool_size=(2,2))
    # l_dropout2 = DropoutLayer(l_pool2, p=0.2)

    l_hidden3 = DenseLayer(l_pool2, num_units = h_dimension, nonlinearity=rectify)
    # l_dropout3 = DropoutLayer(l_hidden3, p=0.5)

    l_hidden4 = DenseLayer(l_hidden3, num_units = h_dimension, nonlinearity=rectify)
    # l_dropout4 = DropoutLayer(l_hidden4, p=0.5)

    l_out = DenseLayer(l_hidden4, num_units=num_labels, nonlinearity=softmax)

    return l_out

def main():
    # load the training and validation data sets
    # labels=int(0.7*image.all_count)
    data=image.ddd
    labels=data.keys()
    train_X=np.zeros([num_images,1,PIXELS,PIXELS])
    train_y=np.zeros([num_images,len(labels)])
    i=0
    for label in (data.keys()):
        for im in data[label]:
            train_X[i,0]=im
            train_y[i]=label_binarize([label],labels)[0]
            i+=1
            if i>=num_images:
                break
            if i%500==0:
                print 'idx of images:',i
        if i>=num_images:
            break
    zipp=zip(train_X,train_y)
    random.shuffle(zipp)
    xx=np.array([one[0] for one in zipp])
    yy=np.array([one[1] for one in zipp])
    del train_X,train_y
    train_X=xx[:size]
    train_y=yy[:size]
    valid_X=xx[size:]
    valid_y=yy[size:]
    del xx,yy
    print 'Shuffle finish. Begin to build model.'

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
    train = theano.function(inputs=[X, Y], outputs=[loss_train,pred], updates=updates, allow_input_downcast=True)
    valid = theano.function(inputs=[X, Y], outputs=[loss_valid], allow_input_downcast=True)
    predict_valid = theano.function(inputs=[X], outputs=[pred_valid], allow_input_downcast=True)

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
            count=np.count_nonzero(np.int32(pred ==np.argmax(yy_batch,axis=1)))
            print i,idx_batch,'| Tloss:', train_loss,'| Count:',count,'| Acc:',float(count)/(BATCHSIZE)
            print pred
            print np.argmax(yy_batch,axis=1)
            print "time:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            if 0 and idx_batch%1==0:
                acc=0
                valid_batch_number=len(valid_X)/BATCHSIZE
                for j in tqdm(range(valid_batch_number)):
                    x_batch = np.float32(valid_X[idx_batch * BATCHSIZE:(idx_batch + 1) * BATCHSIZE])
                    y_batch = np.float32(valid_y[idx_batch * BATCHSIZE:(idx_batch + 1) * BATCHSIZE])
                    valid_loss = valid(x_batch,y_batch)[0]
                    pred = predict_valid(x_batch)[0]
                    acc += np.count_nonzero(np.int32(pred ==np.argmax(y_batch,axis=1)))
                acc=float(acc)/(valid_batch_number*BATCHSIZE)
                print 'iter:', i,idx_batch, '| Vloss:',valid_loss,'|Acc:',acc


        # save weights
        if i%5:
            all_params = helper.get_all_param_values(output_layer)
            f = gzip.open('params/weights_cnn_only_rotate_{}.pklz'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), 'wb')
            pickle.dump(all_params, f)
            f.close()

if __name__ == '__main__':
    main()
