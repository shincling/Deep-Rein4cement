#-*- coding: UTF-8 -*-
import os
import numpy as np
import random
import scipy.misc
from scipy.misc import imread,imresize
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
        # del data
        print final_x[0]
        print final_y
    return final_x,final_y

def build(path,pathdir,files,labels,all_count,ratio,size):

    train_labels=random.sample(labels,int(ratio*len(labels)))
    for i in train_labels:
        labels.remove(i)
    test_labels=labels
    assert len(train_labels)+len(test_labels)==all_count

    train_dates={label:[] for label in train_labels}
    test_dates={label:[] for label in test_labels}
    for file in files:
        label=file[-11:-7]
        if label in train_labels:
            train_dates[label].append(np.float32(imresize(imread(file),size)))
        else:
            test_dates[label].append(np.float32(imresize(imread(file),size)))

    x_train,y_train=get_sequence_images(train_dates,train_labels,11,3,size,10)
    pass
def cc():
    pass

bbb=111

if __name__=="__main__":
    cc()
    path='python/backall'
    pathdir=os.listdir(path)
    files=[path+'/'+ff for ff in pathdir]
    labels=get_labels(pathdir)
    all_count=len(labels)
    ratio=0.7
    size=(20,20)
    build(path,pathdir,files,labels,all_count,ratio,size)
