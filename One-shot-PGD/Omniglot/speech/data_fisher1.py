# coding=utf-8
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from ark import read_utt_data
import pickle
import gzip
import time

def split_speech(features,length=None):#如果不给第二个参数，就默认等同于跟特征个数一样的
    list=[]
    if not length:
        length=features.shape[1]
    number=features.shape[0]/length
    for i in range(number):
        split=features[i*length:(i+1)*length]
        list.append(split)
    return list

def read_features(train_dict,train_of_test):
    # train_fea={}
    counter=0
    for train in tqdm(train_dict):
        ark_path_list=[]
        for file_idx,num_range in enumerate(fisher_list):
            if num_range[0]<=int(train[:5])<=num_range[1]:
                file_path='/home/sw/Shin/fisher/fisher_16k/pitchlog/raw_fbank_pitch_fisher_16k.'+str(file_idx+1)
                file_cont=open(file_path+'.scp')
                file_cont_lines=file_cont.readlines()
                for line in file_cont_lines:
                    line=line.strip()
                    if train in line:
                        ark_path_list.append(file_path+'.ark:'+line[(line.rindex(':')+1):])
        # print train
        # print ark_path_list,'\n'
        if len(ark_path_list)<5:
            # print 'not enough length.',train
            continue
        # train_fea[train]=np.array([read_utt_data(ark_path) for ark_path in ark_path_list])
        tmp_list=[]
        for ark_path in ark_path_list:
            one_fea=read_utt_data(ark_path)
            tmp_list.extend(split_speech(one_fea))
        # train_fea[train]=tmp_list

        if len(tmp_list )<5:
            # print 'not enough length.',train
            continue


        # train_fea[train]=0
        if train_of_test:
            # np.save('dataset/fisher/train/'+train+str(len(ark_path_list)),train_fea[train])
            np.save('/media/sw/Elements/fisher_dataset/train_pix40/'+train+'_'+str(len(tmp_list)),tmp_list)
        else:
            # np.save('dataset/fisher/test/'+train+str(len(ark_path_list)),train_fea[train])
            np.save('/media/sw/Elements/fisher_dataset/test_pix40/'+train+'_'+str(len(tmp_list)),tmp_list)
        counter+=1

    print 'length:',counter
    return None
    return train_fea

text_detail='/home/sw/Shin/fisher/doc/fe_03_p1_calldata.tbl'
idx_line=0
ff=open(text_detail)
train_maledict=OrderedDict()
train_femaledict=OrderedDict()
train_total_num_perSex=2500
test_maledict=OrderedDict()
test_femaledict=OrderedDict()
test_total_num_perSex=350
total_maledict=OrderedDict()
total_femaledict=OrderedDict()
total_num_perSex=test_total_num_perSex+train_total_num_perSex
total_maledict=OrderedDict()
br_ex=0
sum_ex=0
while True:
    if len(total_maledict)>=total_num_perSex and len(total_femaledict)>=total_num_perSex:
        break
    cont=ff.readline()
    if cont=='':
        break
    # print cont
    cont_list=cont.split(',')
    story_id=cont_list[0]
    a_id=cont_list[5]
    a_sex=cont_list[6][0]
    b_id=cont_list[10]
    b_sex=cont_list[11][0]
    if a_sex=='f':
        if len(total_femaledict)<total_num_perSex and a_id not in total_femaledict.values():
            total_femaledict[story_id+'_'+'a']=a_id
    elif a_sex=='m':
        if len(total_maledict)<total_num_perSex and a_id not in total_maledict.values():
            total_maledict[story_id+'_'+'a']=a_id

    if b_sex=='f':
        if len(total_femaledict)<total_num_perSex and b_id not in total_femaledict.values():
            total_femaledict[story_id+'_'+'b']=b_id
    elif b_sex=='m':
        if len(total_maledict)<total_num_perSex and b_id not in total_maledict.values():
            total_maledict[story_id+'_'+'b']=b_id

    br=len(total_maledict)-len(total_femaledict)
    sum=len(total_maledict)+len(total_femaledict)
    print story_id,len(total_maledict),len(total_femaledict)
    if br!=br_ex:
        print 'hhh'
    if sum!=sum_ex+2:
        print 'Repeated speaker!'
    br_ex=br
    sum_ex=sum

    # print total_maledict
    # print total_femaledict
    # idx_line+=1
    # if idx_line>10:
    #     break

print 'Finished Total dict.'
train_female_label=total_femaledict.keys()[:train_total_num_perSex]
train_male_label=total_maledict.keys()[:train_total_num_perSex]
test_female_label=total_femaledict.keys()[-1*test_total_num_perSex:]
test_male_label=total_maledict.keys()[-1*test_total_num_perSex:]
train_femaledict={label:total_femaledict[label] for label in train_female_label}
train_maledict={label:total_maledict[label] for label in train_male_label}
test_femaledict={label:total_femaledict[label] for label in test_female_label}
test_maledict={label:total_maledict[label] for label in test_male_label}
del total_femaledict,total_maledict
train_femaledict.update(train_maledict)
train_dict=train_femaledict
test_femaledict.update(test_maledict)
test_dict=test_femaledict
print train_dict
print test_dict
print 'Finished train/test dict.'


fisher_list=[(1,391),(391,930),(930,1389),(1389,1888),(1889,2362),(2362,2894),(2894,3352),
             (3352,3837),(3837,4319),(4319,4808),(4808,5321),(5321,5850)]

tt=time.time()
test_fea=read_features(test_dict,0)
# f = gzip.open('dataset/fisher/test_{}.pklz'.format(2*test_total_num_perSex), 'wb')
# pickle.dump(test_fea, f)
# f.close()
# del test_fea
print 'Test pickle cost time :',time.time()-tt

tt=time.time()
train_fea=read_features(train_dict,1)
# f = gzip.open('dataset/fisher/train_{}.pklz'.format(2*train_total_num_perSex), 'wb')
# pickle.dump(train_fea, f)
# f.close()
# del train_fea
print 'Train pickle cost time :',time.time()-tt










