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
num_triples=10000
num_batches=1000#一个epoch里需要循环多少个不同的batch
# num_images=964 #共77120个图，算上了旋转的
h_dimension=300
valid_size=1280*5
size=num_images-valid_size
alpha=0.05


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


    return l_out,l_hidden4

def main():
    # load the training and validation data sets
    # labels=int(0.7*image.all_count)
    data=image.ddd
    labels=data.keys()
    X = T.tensor4()

    # set up theano functions to generate output by feeding data through network
    output_layer_softmax , output_layer_triplet= lasagne_model()
    output_train = lasagne.layers.get_output(output_layer_triplet, X)
    output_shape = output_train.shape
    output_train = lasagne.layers.reshape(output_train,(output_train[0]/3,3,output_shape[-1]))
    output_0= lasagne.layers.helper.get_output(lasagne.layers.SliceLayer(output_train,0,1))
    output_1= lasagne.layers.helper.get_output(lasagne.layers.SliceLayer(output_train,1,1))
    output_2= lasagne.layers.helper.get_output(lasagne.layers.SliceLayer(output_train,2,1))

    # set up the loss that we aim to minimize
    dis_pos=T.sqrt((T.sum(T.square(T.sub(output_0,output_1)),1)))
    dis_neg=T.sqrt((T.sum(T.square(T.sub(output_0,output_2)),1)))
    dis=(dis_pos-dis_neg+alpha)
    loss_train = T.mean((dis)*(dis>0))

    # prediction functions for classifications
    pred = T.argmax(output_train, axis=1)

    # get parameters from network and set up sgd with nesterov momentum to update parameters
    params = lasagne.layers.get_all_params(output_layer_triplet)
    updates = nesterov_momentum(loss_train, params, learning_rate=0.03, momentum=0.9)
    # updates =lasagne.updates.adagrad(loss_train, params, learning_rate=0.003)

    # set up training and prediction functions
    train = theano.function(inputs=[X, Y], outputs=[loss_train,pred], updates=updates, allow_input_downcast=True)

    # loop over training functions for however many iterations, print information while training
    train_eval = []
    valid_eval = []
    valid_acc = []

    for i in range(450):
        for idx_batch in range (num_batches):
            train_X=np.zeros([BATCHSIZE,3,PIXELS,PIXELS])
            i=0
            for iii in range(BATCHSIZE):
                label=random.choice(labels)
                im_aim=random.choice(data[label])
                im_pos=random.choice(data[label])
                while True:
                    label_neg=random.choice(labels)
                    if label!=label_neg:
                        im_neg=random.choice(data[label_neg])
                        break
                train_X[iii,0]=im_aim
                train_X[iii,1]=im_pos
                train_X[iii,2]=im_neg


            xx_batch = np.float32(train_X[idx_batch * BATCHSIZE:(idx_batch + 1) * BATCHSIZE])
            # yy_batch = np.float32(train_y[idx_batch * BATCHSIZE:(idx_batch + 1) * BATCHSIZE])

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
