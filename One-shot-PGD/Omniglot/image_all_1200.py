#-*- coding: UTF-8 -*-
import os
import numpy as np
import random
import scipy.misc
from scipy.misc import imread,imresize,imsave,imshow
# from scipy.ndimage import rotate,shift

def get_labels(dir):
    labels_list=[]
    for one_file in dir:
        label=one_file[:4]
        if label not in labels_list:
            labels_list.append(label)
    return labels_list

def get_sequence_images(data,labels,path_length,total_label,size,total_roads=10000):
    final_x=np.zeros((total_roads,path_length,size[0],size[1]))
    final_y=np.zeros((total_roads,path_length))
    labels=data.keys()
    for one_sample in range(total_roads):
        label_list=random.sample(labels,total_label)
        one_shoot_label=label_list[-1]
        insert_idx=np.random.randint(0,path_length-1)
        final_x[one_sample,insert_idx]=random.sample(data[one_shoot_label],1)[0]
        final_y[one_sample,insert_idx]=int(one_shoot_label)
        final_x[one_sample,-1]=random.sample(data[one_shoot_label],1)[0]
        final_y[one_sample,-1]=int(one_shoot_label)
        for i in range(path_length-1):
            if i!=insert_idx:
                label=np.random.choice(label_list[:-1])
                final_y[one_sample,i]=int(label)
                final_x[one_sample,i,:]=random.sample(data[label],1)[0]
    return final_x,final_y

def shuffle_label(y,counts):
    for i in y:
        uni_labes=list(set(list(i)))
        random_labels=random.sample(range(5),counts)
        for idx,j in enumerate(i):
            for ind in range(counts):
                if j==uni_labes[ind]:
                    i[idx]=random_labels[ind]
                    break
    return y

def build(path,pathdir,files,files_eval,labels,labels_eval,all_count,size):
    train_labels=labels
    test_labels=labels_eval
    assert len(train_labels)+len(test_labels)==all_count

    train_dates={label:[] for label in train_labels}
    test_dates={label:[] for label in test_labels}
    for file in (files+files_eval):
        label=file[-11:-7]
        if label in train_labels:
            train_dates[label].append(0.001*(255-np.float32(imresize(imread(file,1),size))))
        else:
            test_dates[label].append(0.001*(255-np.float32(imresize(imread(file,1),size))))

    train_rank_dates={}
    for i in range(len(train_dates)):
        train_rank_dates[i]=train_dates[train_dates.keys()[i]]
    if cnn_only:
        return train_rank_dates
    x_train,y_train=get_sequence_images(train_rank_dates,train_labels,path_length,total_labels_per_seq,size,total_roads)

    # x_train,y_train=get_sequence_images(train_dates,train_labels,path_length,total_labels_per_seq,size,total_roads)
    x_test,y_test=get_sequence_images(test_dates,test_labels,path_length,total_labels_per_seq,size,total_roads)
    return x_train,y_train,x_test,y_test

path='python/backall_1200'
pathdir=os.listdir(path)
files=[path+'/'+ff for ff in pathdir]
labels=get_labels(pathdir)

path_eval='python/backall_423'
pathdir_eval=os.listdir(path_eval)
files_eval=[path_eval+'/'+ff for ff in pathdir_eval]
labels_eval=get_labels(pathdir_eval)

all_count=len(labels)+len(labels_eval)
print "train:{},test:{}".format(len(labels),len(labels_eval))
# ratio=0.7
size=(20,20)
size=(28,28)
total_labels_per_seq=5
path_length=11
total_roads=500
cnn_only=0
label_fixed=1
if cnn_only:
    pass
    # ddd=build(path,pathdir,files,labels,all_count,ratio,size)
else:
    x_train,y_train,x_test,y_test=build(path,pathdir,files,files_eval,labels,labels_eval,all_count,size)
    del files,files_eval
    y_train_shuffle=shuffle_label(y_train.copy(),total_labels_per_seq)
    y_test_shuffle=shuffle_label(y_test.copy(),total_labels_per_seq)
print y_train[0]
print y_train_shuffle[0]
print 'size:',x_train[10][0].shape,'Data Finished.'
# if __name__=="__main__":
#     build(path,pathdir,files,labels,all_count,ratio,size)
