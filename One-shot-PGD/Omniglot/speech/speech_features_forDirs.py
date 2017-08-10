#coding=utf8
import sys
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np
import os

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
    print 'wav_name:',wav_name
    label=wav_name[:3]

    # output_dir=sys.argv[2]
    output_dir='/home/sw/Shin/Codes/Deep-Rein4cement/One-shot-PGD/Omniglot/speech/dataset'
    output_dir=output_dir if output_dir[-1]=='/' else output_dir+'/'
    print 'output_dir:',output_dir

    (rate,sig)=wav.read(path)
    logfbank0015_feat=logfbank(sig,rate,winstep=0.015,nfilt=40)
    feat_list=split_speech(logfbank0015_feat)
    return label,feat_list


# path=sys.argv[1]
path='/home/sw/Shin/数据集/多说话人语音数据 - WSJ0/spk1-15'
pathdir=os.listdir(path)
files=[path+'/'+ff for ff in pathdir]
labels=get_labels(pathdir)
total_labels=range(len(labels))

data_dict={label:[] for label in total_labels}
for file in files:
    label,feat_list=get_features(file)
    data_dict[labels.index(label)].extend(feat_list)





# np.save(output_dir+wav_name+'.0015',logfbank0015_feat)
pass
