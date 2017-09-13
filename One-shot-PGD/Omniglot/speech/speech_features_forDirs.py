#coding=utf8
import sys
from python_speech_features import logfbank
from tqdm import tqdm
import random
import scipy.io.wavfile as wav
import numpy as np
import os
import time
import gzip
import pickle

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

def get_sequence_speechs(data,path_length,total_label,size,total_roads=10000):
    final_x=np.zeros((total_roads,path_length,size[0],size[1]))
    final_y=np.zeros((total_roads,path_length))
    labels=data.keys()
    print labels
    for one_sample in range(total_roads):
        label_list=random.sample(labels,total_label)
        # print 'label list:',label_list
        one_shoot_label=label_list[-1]
        final_x[one_sample,-1]=random.sample(data[one_shoot_label],1)[0]
        final_y[one_sample,-1]=int(one_shoot_label)
        for i in range(path_length-1):
            label=label_list[i]
            final_y[one_sample,i]=int(label)
            tmp_sum=0
            tmp_sample=random.sample(data[label],number_shots_total)
            for iidx,shot in enumerate(tmp_sample):
                tmp_sum+=shot
            tmp_sum/=float(number_shots_total)
            final_x[one_sample,i,:]=tmp_sum
            del tmp_sum,tmp_sample
    return final_x,final_y


def get_labels(dir):
    labels_list=[]
    for one_file in dir:
        label=one_file[:3]
        if label not in labels_list:
            labels_list.append(label)
    return labels_list

def split_speech(features,length=None):#如果不给第二个参数，就默认等同于跟特征个数一样的
    list=[]
    if not length:
        length=features.shape[1]
    number=features.shape[0]/length
    for i in range(number):
        split=features[i*length:(i+1)*length]
        list.append(split)
    return list

def get_features(path):
    wav_name=path[path.rfind('/')+1:]
    # print 'wav_name:',wav_name
    label=wav_name[:3]

    # output_dir=sys.argv[2]
    output_dir='/home/sw/Shin/Codes/Deep-Rein4cement/One-shot-PGD/Omniglot/speech/dataset'
    output_dir=output_dir if output_dir[-1]=='/' else output_dir+'/'
    # print 'output_dir:',output_dir

    (rate,sig)=wav.read(path)
    logfbank0015_feat=logfbank(sig,rate,winstep=0.015,nfilt=40)
    feat_list=split_speech(logfbank0015_feat,100)
    return label,feat_list

load_data=None
# load_data='dataset/spk1-15_dict_fbank40.pklz'
# load_data='dataset/spkall_dict_fbank40.pklz'

total_labels_per_seq=5
path_length=total_labels_per_seq+1
total_roads=200
total_roads_test=200
cnn_only=0
test_split=1#意思是要不要从spkall里面分出来部分作为one-shot的test样本
label_fixed=1
number_shots_total=1#这个量用来约束到底是几shot

if load_data:
    print 'load data:',load_data
    timell=time.time()
    data_dict=pickle.load(gzip.open(load_data))
    print 'load cost time:',time.time()-timell
else:
    path='/home/sw/Shin/数据集/多说话人语音数据 - WSJ0/spk1-15'
    # path='/home/sw/Shin/数据集/多说话人语音数据 - WSJ0/spk_all_wav'
    pathdir=os.listdir(path)
    files=[path+'/'+ff for ff in pathdir]
    labels=get_labels(pathdir)
    print 'total labels:',len(labels)
    total_labels=range(len(labels))

    data_dict={label:[] for label in total_labels}
    for file in tqdm(files):
        label,feat_list=get_features(file)
        data_dict[labels.index(label)].extend(feat_list)
    del files

    f = gzip.open('dataset/spkall_dict_fbank40.pklz', 'wb')
    pickle.dump(data_dict, f)
    f.close()

print 'data finished.'

if test_split:
    print 'Test split from spkall start............'
    total_labels_num=len(data_dict)
    test_labels_num=total_labels_num/10 if total_labels_num/10>5 else 5
    print 'total_labes_num:',total_labels_num,'test_labes_num:',test_labels_num
    test_labels=data_dict.keys()[(-1*test_labels_num):]
    data_dict_test={ll:data_dict[ll] for ll in test_labels}

speechSize = data_dict[0][0].shape
if not cnn_only: # 不只是得到cnn的训练dict数据
    x_train,y_train=get_sequence_speechs(data_dict,path_length,total_labels_per_seq,speechSize,total_roads=total_roads)
    x_test,y_test=get_sequence_speechs(data_dict_test,path_length,total_labels_per_seq,speechSize,total_roads=total_roads_test)
    y_train_shuffle=shuffle_label(y_train.copy(),total_labels_per_seq)
    y_test_shuffle=shuffle_label(y_test.copy(),total_labels_per_seq)
    print y_train[:10]
    print y_train_shuffle[:10]


