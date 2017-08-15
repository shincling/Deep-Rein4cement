#coding=utf8
import gzip
import pickle
import numpy as np
import random
import time

import theano
from theano import tensor as T

import lasagne
from lasagne.nonlinearities import rectify, softmax, very_leaky_rectify,tanh
from lasagne.updates import nesterov_momentum
from lasagne.layers import InputLayer, MaxPool2DLayer, Conv2DLayer, DenseLayer, DropoutLayer, helper , batch_norm#batchnorm层噢
from sklearn.preprocessing import LabelBinarizer,label_binarize
# import speech_features_forDirs as speech
from tqdm import tqdm
import os

train_load_path='/home/shijing/data/fisher/train_pix40/' #4768个类别
train_load_path='dataset/fisher/train_part/' #
# test_load_path='/home/shijing/data/fisher/test_pix40/' #665个类别
train_files=os.listdir(train_load_path)
# test_files=os.listdir(test_load_path)
print train_files
# raise EOFError
BATCHSIZE = 32
one_sample=np.load(train_load_path+train_files[0])[0]
PIXELS = one_sample.shape[0]
speechSize = one_sample.shape
num_labels_train=len(train_files)
num_sample_list=[int(file.split('_')[-1].split('.')[0]) for file in train_files]
num_speechs=np.array(num_sample_list).sum()
print 'all labels for CNN:',num_labels_train,'all samples:',num_speechs
h_dimension=300
valid_size=num_speechs/10
size=num_speechs-valid_size
global valid_best
valid_best=0.95
print valid_best


def get_data(train_files,total_num,size,valid_size):
    total_list=[]
    for file in train_files:
        num_file=int(file.split('_')[-1].split('.')[0])
        list_file=[file[:-3]+str(num) for num in range(num_file)] # 02483_b_1048.1 02483_b_1048.2 这种
        total_list.extend(list_file)
    print 'total_list length:',len(total_list)
    assert total_num==len(total_list)
    random.shuffle(total_list)
    # print total_list[500:900]

    train_list=total_list[:size]
    valid_list=total_list[size:]

    return train_list,valid_list

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def lasagne_model():
    l_in = InputLayer(shape=(None, 1)+speechSize)

    l_conv1 = Conv2DLayer(l_in, num_filters = 128, filter_size=(3,3), nonlinearity=rectify)
    l_conv1b = Conv2DLayer(l_conv1, num_filters = 128, filter_size=(3,3), nonlinearity=rectify)
    l_conv1b = batch_norm(l_conv1b)
    l_pool1 = MaxPool2DLayer(l_conv1b, pool_size=(2,2))
    # l_dropout1 = DropoutLayer(l_pool1, p=0.2)

    l_conv2 = Conv2DLayer(l_pool1, num_filters = 256, filter_size=(3,3), nonlinearity=rectify)
    l_conv2b = Conv2DLayer(l_conv2, num_filters = 256, filter_size=(3,3), nonlinearity=rectify)
    l_conv2b = batch_norm(l_conv2b)
    l_pool2 = MaxPool2DLayer(l_conv2b, pool_size=(2,2))
    # l_dropout2 = DropoutLayer(l_pool2, p=0.2)

    l_hidden3 = DenseLayer(l_pool2, num_units = h_dimension, nonlinearity=tanh)
    l_hidden3 = batch_norm(l_hidden3)
    # l_dropout3 = DropoutLayer(l_hidden3, p=0.3)

    l_hidden4 = DenseLayer(l_hidden3, num_units = h_dimension, nonlinearity=tanh)
    l_hidden4 = DropoutLayer(l_hidden4, p=0.5)

    l_out = DenseLayer(l_hidden4, num_units=num_labels_train, nonlinearity=softmax)

    return l_out

def main():
    # load the training and validation data sets
    # labels=int(0.7*speech.all_count)
    # global valid_best
    # data=speech.data_dict
    # labels=data.keys()
    # # assert (PIXELS,PIXELS)==speechSize
    # train_X=np.zeros([num_speechs,1,speechSize[0],speechSize[1]])
    # train_y=np.zeros([num_speechs,len(labels)])
    # i=0
    # for label in (data.keys()):
    #     for im in data[label]:
    #         train_X[i,0]=im
    #         train_y[i]=label_binarize([label],labels)[0]
    #         i+=1
    #         if i>=num_speechs:
    #             break
    #         if i%500==0:
    #             print 'idx of speechs:',i
    #     if i>=num_speechs:
    #         break
    # zipp=zip(train_X,train_y)
    # random.shuffle(zipp)
    # xx=np.array([one[0] for one in zipp])
    # yy=np.array([one[1] for one in zipp])
    # del train_X,train_y
    # train_X=xx[:size]
    # train_y=yy[:size]
    # valid_X=xx[size:]
    # valid_y=yy[size:]
    # del xx,yy
    # print 'Shuffle finish. Begin to build model.'

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
    params = lasagne.layers.get_all_params(output_layer,trainable=True)
    updates = nesterov_momentum(loss_train, params, learning_rate=0.003, momentum=0.9)
    # updates =lasagne.updates.sgd(loss_train, params, learning_rate=0.01)

    # set up training and prediction functions
    train = theano.function(inputs=[X, Y], outputs=[loss_train,pred], updates=updates, allow_input_downcast=True)
    valid = theano.function(inputs=[X, Y], outputs=[loss_valid,pred_valid], allow_input_downcast=True)
    # predict_valid = theano.function(inputs=[X], outputs=[pred_valid], allow_input_downcast=True)

    # loop over training functions for however many iterations, print information while training
    train_eval = []
    valid_eval = []
    valid_acc = []

    for i in range(450):
        train_list,valid_list=get_data(train_files,num_speechs,size,valid_size)
        batch_total_number = size / BATCHSIZE
        for idx_batch in range (batch_total_number):
            batch_list = train_list[idx_batch * BATCHSIZE:(idx_batch + 1) * BATCHSIZE]
            xx_batch=np.zeros((BATCHSIZE,1,speechSize[0],speechSize[1]))
            yy_batch=np.zeros((BATCHSIZE,num_labels_train))
            for ind,one_name in enumerate(batch_list):
                aim_file_name,aim_idx=one_name.split('.')
                aim_file_name+='.npy'
                aim_file_name_full=train_load_path+aim_file_name
                xx_batch[ind,0]=np.load(aim_file_name_full)[int(aim_idx)]
                yy_batch[ind]=label_binarize([train_files.index(aim_file_name)],range(num_labels_train))[0]

            train_loss ,pred = train(xx_batch,yy_batch)
            count=np.count_nonzero(np.int32(pred ==np.argmax(yy_batch,axis=1)))
            print i,idx_batch,'| Tloss:', train_loss,'| Count:',count,'| Acc:',float(count)/(BATCHSIZE)
            print pred
            print np.argmax(yy_batch,axis=1)
            print "time:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # raise EOFError

            if 1 and idx_batch%15==0:
                acc=0
                valid_batch_number=len(valid_X)/BATCHSIZE
                for j in tqdm(range(valid_batch_number)):
                    x_batch = np.float32(valid_X[j* BATCHSIZE:(j+ 1) * BATCHSIZE])
                    y_batch = np.float32(valid_y[j* BATCHSIZE:(j+ 1) * BATCHSIZE])
                    print len(x_batch),len(y_batch)
                    valid_loss,pred = valid(x_batch,y_batch)
                    # pred = predict_valid(x_batch)[0]
                    acc += np.count_nonzero(np.int32(pred ==np.argmax(y_batch,axis=1)))
                acc=float(acc)/(valid_batch_number*BATCHSIZE)
                print 'iter:', i,idx_batch, '| Vloss:',valid_loss,'|Acc:',acc
                if acc>valid_best:
                    print 'new valid_best:',valid_best,'-->',acc
                    valid_best=acc
                    all_params = helper.get_all_param_values(output_layer)
                    f = gzip.open('speech_params/validbest_cnn_allbatchnorm_{}_{}_{}.pklz'.format(i,valid_best,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), 'wb')
                    pickle.dump(all_params, f)
                    f.close()

        # save weights
        if i%5:
            all_params = helper.get_all_param_values(output_layer)
            f = gzip.open('speech_params/validbest_cnn_allbatchnorm_{}_{}_{}.pklz'.format(i,acc,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), 'wb')
            pickle.dump(all_params, f)
            f.close()

if __name__ == '__main__':
    main()
