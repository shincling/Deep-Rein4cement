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
import image_all_1200_rotate as image
from tqdm import tqdm

BATCHSIZE = 128
PIXELS = 20
imageSize = PIXELS * PIXELS
num_features = imageSize
num_labels=4*964
num_images=964*4*15#共77120个图，算上了旋转的

num_labels=4*1200
num_images=1200*4*20#共77120个图，算上了旋转的

num_triples=10000
num_batches=1000#一个epoch里需要循环多少个不同的batch
# num_images=964 #共77120个图，算上了旋转的
h_dimension=300
valid_size=1280*5
size=num_images-valid_size
alpha=5
load_params='params/params_rotate_95.pklz'
load_params='params/weights_1200_perfect.pklz'
# load_params=0


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
    output_train = lasagne.layers.ReshapeLayer(output_layer_triplet,(-1,3,[1]))
    output_0= lasagne.layers.helper.get_output(lasagne.layers.SliceLayer(output_train,0,1),X)
    output_1= lasagne.layers.helper.get_output(lasagne.layers.SliceLayer(output_train,1,1),X)
    output_2= lasagne.layers.helper.get_output(lasagne.layers.SliceLayer(output_train,2,1),X)
    output= lasagne.layers.helper.get_output(output_layer_softmax,X)

    # set up the loss that we aim to minimize
    eps=1e-10
    dis_pos=T.sqrt(T.sum(T.square(T.sub(output_0,output_1)),1)+eps)
    dis_neg=T.sqrt(T.sum(T.square(T.sub(output_0,output_2)),1)+eps)
    dis=(dis_pos-dis_neg+alpha)
    # dis=(dis_pos-dis_neg)
    loss_train = T.mean((dis)*(dis>0))
    # loss_train = T.sum(T.nnet.relu(dis))
    # loss_train = T.mean(dis)

    # prediction functions for classifications
    pred = T.argmax(output, axis=1)

    # get parameters from network and set up sgd with nesterov momentum to update parameters
    params = lasagne.layers.get_all_params(output_layer_triplet)
    grad=T.grad(loss_train,params)

    # updates = nesterov_momentum(loss_train, params, learning_rate=0.03, momentum=0.9)
    updates =lasagne.updates.rmsprop(loss_train, params, learning_rate=0.003)
    # updates =lasagne.updates.get_or_compute_grads(loss_train, params)

    # set up training and prediction functions
    train = theano.function(inputs=[X], outputs=[loss_train,pred,dis,dis_pos,dis_neg], updates=updates, allow_input_downcast=True)

    if load_params:
        pre_params=pickle.load(gzip.open(load_params))
        lasagne.layers.set_all_param_values(output_layer_softmax,pre_params)
        print 'load Success.'

    for i in range(4500):
        aver_loss=0
        for idx_batch in range (num_batches):
            train_X=np.zeros([BATCHSIZE,3,PIXELS,PIXELS])
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

            train_X=train_X.reshape(BATCHSIZE*3,1,PIXELS,PIXELS)
            xx_batch = np.float32(train_X)
            # print xx_batch.shape
            # yy_batch = np.float32(train_y[idx_batch * BATCHSIZE:(idx_batch + 1) * BATCHSIZE])

            train_loss ,pred ,dis ,dis1,dis2= train(xx_batch)
            aver_loss+=train_loss
            # count=np.count_nonzero(np.int32(pred ==np.argmax(yy_batch,axis=1)))
            if idx_batch%15==0:
                print i,idx_batch,'| Tloss:', train_loss,pred,'\ndis_pos:{}\ndis_neg:{}\ndis:{}'.format(dis1[:20],dis2[:20],dis[:20])
                # print pred
                # print np.argmax(yy_batch,axis=1)
                print "time:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())



        # save weights
        if i%4==0:
            aver_loss=aver_loss/num_batches
            all_params = helper.get_all_param_values(output_layer_softmax)
            f = gzip.open('params/cnn_triplet/weights_{}_cnn_rotate_{}_triplet_{}.pklz'.format(aver_loss,alpha,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), 'wb')
            pickle.dump(all_params, f)
            f.close()

if __name__ == '__main__':
    main()
