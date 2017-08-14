# coding=utf-8
import numpy as np
from collections import OrderedDict

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
del train_femaledict,train_maledict
print 'Finished train/test dict.'



