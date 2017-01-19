import os
import shutil
path='backall_eval/'
cc=os.listdir(path)
for i in cc:
    if int(i[:4])<1201:
        shutil.copy(path+i,'backall_1200')
    else:
        shutil.copy(path+i,'backall_423')


