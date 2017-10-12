#coding=utf8
import scipy.io as sio
import os
import numpy as np
import random
from tqdm import tqdm
'''这个文件的目的施把之前存成np的很多个人的语音片段导成matlab格式的东西给后面的ivector程序，
它的还是那个一步是data_fisher1.py，需要的东西施train_pix40/310的文件夹里的东西。
需要注意的一点是，max min的个数影响num_channel的设置。num_channel暂时应该至少小于等于min，是一个sample的指标。
整个程序大概１５分钟左右的样子。
'''
def file2mat(label,num_slots,train_or_test):

    if train_or_test:
        ccc=np.load(train_load_path + label)
        ll=len(ccc)
        if ccc.shape[1]!=310:
            print label,'is not 310,but ',ccc.shape
            raise EOFError
        if ll<num_slots:
            print 'Not enough samples for ',label,':',ll
            return None
        aim_dir=train_save_path+label[:-4]+'/'
        try:
            os.mkdir(aim_dir)
        except OSError:
            pass
        cc=random.sample(ccc, num_slots)
        for idx in range(num_slots):
            name=label[:-4]+'_'+str(idx)
            sio.savemat(aim_dir+name,{'tmp_array':cc[idx]})
    else:
        ccc=np.load(test_load_path + label)
        ll=len(ccc)
        if ccc.shape[1]!=310:
            print label,'is not 310,but ',ccc.shape
            raise EOFError
        if ll<num_slots:
            print 'Not enough samples for ',label,':',ll
            return None
        aim_dir=test_save_path+label[:-4]+'/'
        try:
            os.mkdir(aim_dir)
        except OSError:
            pass
        cc=random.sample(np.load(test_load_path + label), num_slots)
        for idx in range(num_slots):
            name=label[:-4]+'_'+str(idx)
            sio.savemat(aim_dir+name,{'tmp_array':cc[idx]})
    return cc

# train_load_path='/home/shijing/data/fisher/train_pix40/' #4768个类别
# train_load_path='dataset/fisher/train_part/' #
# train_load_path='/media/sw/Elements/fisher_dataset/train_pix40/' #
train_load_path='/media/sw/Elements/fisher_dataset/train_pix310/' #
# test_load_path='/home/shijing/data/fisher/test_pix40/' #665个类别
# test_load_path='/media/sw/Elements/fisher_dataset/test_pix40/' #
test_load_path='/media/sw/Elements/fisher_dataset/test_pix310/' #
train_files=os.listdir(train_load_path)
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

num_channels=41# how many pieces of speech to load
num_channels=10# how many pieces of speech to load,这个东西应该与上面的min num有一定的关系

train_save_path='/media/sw/Elements/fisher_dataset/train_matlab_310/' #
for file in tqdm(train_files):
    file2mat(file,num_channels,1)
del train_files

test_save_path='/media/sw/Elements/fisher_dataset/test_matlab_310/' #
test_files=os.listdir(test_load_path)
for file in tqdm(test_files):
    file2mat(file,num_channels,0)
