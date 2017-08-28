#coding=utf8
import gzip
import pickle
import numpy as np
import random
import time
import scipy.io as sio

from sklearn.preprocessing import LabelBinarizer,label_binarize
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
        final_x[one_sample,-1]=random.sample(global_data[one_shoot_label],1)[0]
        final_y[one_sample,-1]=data.index(one_shoot_label)
        for i in range(path_length-1):
            label=label_list[i]
            final_y[one_sample,i]=data.index(label)
            tmp_sum=0
            tmp_sample=random.sample(global_data[label],number_shots_total)
            for iidx,shot in enumerate(tmp_sample):
                tmp_sum+=shot
            tmp_sum/=float(number_shots_total)
            final_x[one_sample,i,:]=tmp_sum
            del tmp_sum,tmp_sample
    return final_x,final_y



aa=sio.loadmat('/home/sw/Shin/Codes/MSR Identity Toolkit v1.0/code/testIvs128_100_665_41.mat')
print type(aa)
data=aa['testIVs']
del aa
data=data.reshape(100,-1).transpose().reshape(665,-1,100,1) #665*41*100的聚脏嗯个
global_data=data
del data

h_dimension=100
speechSize=(h_dimension,1)

total_labels_per_seq=5
path_length=total_labels_per_seq+1
total_roads=1000
total_roads_test=total_roads
total_roads_test=1000
number_shots_total=1#这个量用来约束到底是几shot
#注意用这个的时候，主程序的n_classes改成5（非必须，但更好） 和 path_length要改成6（必须）

load_data=0
# load_data=0
if load_data:
    print 'Beginto load dataset.'
    tt=time.time()
    x_train,y_train,x_test,y_test,y_train_shuffle,y_test_shuffle=pickle.load(gzip.open('dataset/speech_1of5_5way1shot_train1000test1000.pklz'))
    print 'load dataset finished , cost :',time.time()-tt
    del tt
else:
    x_train,y_train=get_sequence_speechs(range(665),path_length,total_labels_per_seq,speechSize,total_roads=total_roads,path=None)
    x_test,y_test=get_sequence_speechs(range(665),path_length,total_labels_per_seq,speechSize,total_roads=total_roads_test,path=None)
    y_train_shuffle=shuffle_label(y_train.copy(),total_labels_per_seq)
    y_test_shuffle=shuffle_label(y_test.copy(),total_labels_per_seq)
    if 0:
        f = gzip.open('dataset/speech_1of5iVectors_{}way{}shot_train{}test{}.pklz'.format(total_labels_per_seq,number_shots_total,total_roads,total_roads_test), 'wb')
        pickle.dump([x_train,y_train,x_test,y_test,y_train_shuffle,y_test_shuffle], f)
        f.close()

print y_test[:10]
print y_test_shuffle[:10]
print 'Data Finished.',number_shots_total
