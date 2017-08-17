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

def shuffle_label(y,counts):
    for i in y:
        uni_labes=list(set(list(i)))
        # random_labels=random.sample(range(5),counts)
        random_labels=random.sample(range(total_labels_per_seq),counts)
        for idx,j in enumerate(i):
            for ind in range(counts):
                if j==uni_labes[ind]:
                    i[idx]=random_labels[ind]
                    break
    return y

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

def get_sequence_speechs(data,path_length,total_label,size,total_roads=10000,path=''):
    final_x=np.zeros((total_roads,path_length,size[0],size[1]))
    final_y=np.zeros((total_roads,path_length))
    labels=data
    # print labels
    for one_sample in tqdm(range(total_roads)):
        label_list=random.sample(labels,total_label)
        # print 'label list:',label_list
        one_shoot_label=label_list[-1]
        final_x[one_sample,-1]=random.sample(np.load(path+one_shoot_label),1)[0]
        final_y[one_sample,-1]=data.index(one_shoot_label)
        for i in range(path_length-1):
            label=label_list[i]
            final_y[one_sample,i]=data.index(label)
            tmp_sum=0
            tmp_sample=random.sample(np.load(path+label),number_shots_total)
            for iidx,shot in enumerate(tmp_sample):
                tmp_sum+=shot
            tmp_sum/=float(number_shots_total)
            final_x[one_sample,i,:]=tmp_sum
            del tmp_sum,tmp_sample
    return final_x,final_y




train_load_path='/home/shijing/data/fisher/train_pix40/' #4768个类别
train_load_path='dataset/fisher/train_part/' #
# test_load_path='/home/shijing/data/fisher/test_pix40/' #665个类别
train_load_path='/media/sw/Elements/fisher_dataset/train_pix40/' #
test_load_path='/media/sw/Elements/fisher_dataset/test_pix40/' #

train_files=os.listdir(train_load_path)
test_files=os.listdir(test_load_path)
# print train_files
# raise EOFError
BATCHSIZE = 32
one_sample=np.load(train_load_path+train_files[0])[0]
PIXELS = one_sample.shape[0]
speechSize = one_sample.shape
num_labels_train=len(train_files)
num_labels_test=len(test_files)
num_sample_list=[int(file.split('_')[-1].split('.')[0]) for file in train_files]
num_speechs=np.array(num_sample_list).sum()
print 'all labels for CNN:',num_labels_train,'all samples:',num_speechs
h_dimension=300

total_labels_per_seq=5
path_length=total_labels_per_seq+1
total_roads=1000
total_roads_test=total_roads
total_roads_test=1000
number_shots_total=1#这个量用来约束到底是几shot
#注意用这个的时候，主程序的n_classes改成5（非必须，但更好） 和 path_length要改成6（必须）

load_data=1
# load_data=0
if load_data:
    print 'Beginto load dataset.'
    tt=time.time()
    x_train,y_train,x_test,y_test,y_train_shuffle,y_test_shuffle=pickle.load(gzip.open('dataset/speech_1of5_5way1shot_train1000test1000.pklz'))
    print 'load dataset finished , cost :',time.time()-tt
    del tt
else:
    x_train,y_train=get_sequence_speechs(train_files,path_length,total_labels_per_seq,speechSize,total_roads=total_roads,path=train_load_path)
    x_test,y_test=get_sequence_speechs(test_files,path_length,total_labels_per_seq,speechSize,total_roads=total_roads_test,path=test_load_path)
    y_train_shuffle=shuffle_label(y_train.copy(),total_labels_per_seq)
    y_test_shuffle=shuffle_label(y_test.copy(),total_labels_per_seq)
    if 1:
        f = gzip.open('dataset/speech_1of5_{}way{}shot_train{}test{}.pklz'.format(total_labels_per_seq,number_shots_total,total_roads,total_roads_test), 'wb')
        pickle.dump([x_train,y_train,x_test,y_test,y_train_shuffle,y_test_shuffle], f)
        f.close()

print y_test[:10]
print y_test_shuffle[:10]
print 'Data Finished.',number_shots_total
