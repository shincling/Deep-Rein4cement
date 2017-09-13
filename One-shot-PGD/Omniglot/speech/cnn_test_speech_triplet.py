#coding=utf8
import gzip
import pickle
import numpy as np
import random
import time
import os

import theano
from theano import tensor as T

import lasagne
from lasagne.nonlinearities import rectify, softmax, very_leaky_rectify
from lasagne.updates import nesterov_momentum
from lasagne.layers import InputLayer, MaxPool2DLayer, Conv2DLayer, DenseLayer, DropoutLayer, helper,batch_norm
from sklearn.preprocessing import LabelBinarizer,label_binarize
from tqdm import tqdm

train_load_path='/home/shijing/data/fisher/train_pix40/' #4768个类别
train_load_path='dataset/fisher/train_part/' #
train_load_path='/media/sw/Elements/fisher_dataset/train_pix40/' #
# test_load_path='/home/shijing/data/fisher/test_pix40/' #665个类别
train_files=os.listdir(train_load_path)
# test_files=os.listdir(test_load_path)
# print train_files
# raise EOFError
BATCHSIZE = 32
one_sample=np.load(train_load_path+train_files[0])[0]
PIXELS = one_sample.shape[0]
speechSize = one_sample.shape
num_labels_train=len(train_files)
num_sample_list=[int(file.split('_')[-1].split('.')[0]) for file in train_files]
print 'max num:{},min num:{}'.format(max(num_sample_list),min(num_sample_list))
num_speechs=np.array(num_sample_list).sum()
print 'all labels for CNN:',num_labels_train,'all samples:',num_speechs
h_dimension=300


num_triples=10000
num_batches=500#一个epoch里需要循环多少个不同的batch
# num_images=964 #共77120个图，算上了旋转的
h_dimension=300
alpha=0.5

load_params='params/params_rotate_95.pklz'
load_params='params/weights_1200_perfect.pklz'
load_params='params/cnn_rare/validbest_2_0.949375_2017-05-21 03:09:28.pklz'
load_params='params/cnn_28/validbest_pix28_2_0.90109375_2017-05-29 07:26:04.pklz'
load_params='params/cnn_28/weights_0.123253030458_cnn_rotate_5_triplet_2017-05-29 20:31:01.pklz'
load_params='params/cnn_28/weights_0.0527420222348_last2_cnn_rotate_1_triplet_2017-07-20 21:44:50.pklz'
load_params='params/cnn_28/validbest_pix28_batchnorm_0_0.96578125_2017-07-24 11:07:25.pklz'
load_params='params/cnn_28/weights_0.00839128953114_batchnorm_last4_cnn_rotate_5_triplet_2017-07-24 18:26:28.pklz'
load_params='params/cnn_28/weights_0.00309908878308_batchnorm_lastall_cnn_rotate_0.5_triplet_2017-07-31 01:24:14.pklz'
load_params='speech_params/validbest_cnn_fisher_0_idxbatch15000_0.46875_2017-08-17 14:43:10.pklz' # 1 shot: 99.0 2000多次
load_params='speech_params/validbest_cnn_fisher_0_idxbatch65000_0.75_2017-08-18 12:29:41.pklz' # 77.5
load_params='speech_params/validbest_cnn_fisher_0_validacc0.695228494624_2017-08-21 15:11:12.pklz' #
print load_params
# load_params=0

# set up functions needed to train the network
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def lasagne_model():
    l_in = InputLayer(shape=(None, 1,PIXELS , PIXELS))

    # l_in =lasagne.layers.NonlinearityLayer(l_in,lasagne.nonlinearities.tanh)
    l_conv1 = Conv2DLayer(l_in, num_filters = 128, filter_size=(3,3), nonlinearity=rectify)
    l_conv1b = Conv2DLayer(l_conv1, num_filters = 128, filter_size=(3,3), nonlinearity=rectify)
    l_conv1b =batch_norm(l_conv1b)
    l_pool1 = MaxPool2DLayer(l_conv1b, pool_size=(2,2))
    # l_dropout1 = DropoutLayer(l_pool1, p=0.2)

    l_conv2 = Conv2DLayer(l_pool1, num_filters = 256, filter_size=(3,3), nonlinearity=rectify)
    l_conv2b = Conv2DLayer(l_conv2, num_filters = 256, filter_size=(3,3), nonlinearity=rectify)
    l_conv2b =batch_norm(l_conv2b)
    l_pool2 = MaxPool2DLayer(l_conv2b, pool_size=(2,2))
    # l_dropout2 = DropoutLayer(l_pool2, p=0.2)

    l_hidden3 = DenseLayer(l_pool2, num_units = h_dimension, nonlinearity=rectify)
    l_hidden3 =batch_norm(l_hidden3)
    # l_dropout3 = DropoutLayer(l_hidden3, p=0.5)

    l_hidden4 = DenseLayer(l_hidden3, num_units = h_dimension, nonlinearity=rectify)
    # l_dropout4 = DropoutLayer(l_hidden4, p=0.5)

    l_out = DenseLayer(l_hidden4, num_units=num_labels_train, nonlinearity=softmax)


    return l_out,l_hidden4

def main():
    # load the training and validation data sets
    # labels=int(0.7*image.all_count)
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
    params = lasagne.layers.get_all_params(output_layer_triplet,trainable=True)
    #params = params[-4:]#TODO: !!!!!!!!!!!!!!!!!!!
    grad=T.grad(loss_train,params)

    # updates = nesterov_momentum(loss_train, params, learning_rate=0.03, momentum=0.9)
    updates =lasagne.updates.rmsprop(loss_train, params, learning_rate=0.0002)
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
                label=random.choice(train_files)
                num_slots=random.randint(1,5)
                im_aim_list=random.sample(np.load(train_load_path+label),num_slots)
                tmp_sum=0
                for iidx,shot in enumerate(im_aim_list):
                    tmp_sum+=shot
                im_aim=tmp_sum/float(num_slots)
                # im_aim=tmp_sum

                im_aim_list=random.sample(np.load(train_load_path+label),num_slots)
                tmp_sum=0
                for iidx,shot in enumerate(im_aim_list):
                    tmp_sum+=shot
                im_pos=tmp_sum/float(num_slots)
                # im_pos=tmp_sum

                while True:
                    label_neg=random.choice(train_files)
                    if label!=label_neg:
                        im_neg_list=random.sample(np.load(train_load_path+label_neg),num_slots)
                        tmp_sum=0
                        for iidx,shot in enumerate(im_neg_list):
                            tmp_sum+=shot
                        im_neg=tmp_sum/float(num_slots)
                        # im_neg=tmp_sum
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
            if idx_batch%3==0:
                print i,idx_batch,'| Tloss:', train_loss,pred,'\ndis_pos:{}\ndis_neg:{}\ndis:{}'.format(dis1[:20],dis2[:20],dis[:20])
                # print pred
                # print np.argmax(yy_batch,axis=1)
                print "time:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())



        # save weights
        if i%1==0:
            aver_loss=aver_loss/num_batches
            all_params = helper.get_all_param_values(output_layer_softmax)
            f = gzip.open('speech_params/speech_{}_batchnorm_12345aver_{}_triplet_{}.pklz'.format(aver_loss,alpha,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), 'wb')
            pickle.dump(all_params, f)
            f.close()

if __name__ == '__main__':
    main()
