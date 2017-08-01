#coding=utf8
import sys
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np

path=sys.argv[1]
wav_name=path[path.rfind('/')+1:]
print 'wav_name:',wav_name
output_dir=sys.argv[2]
output_dir=output_dir if output_dir[-1]=='/' else output_dir+'/'
print 'output_dir:',output_dir
(rate,sig)=wav.read(path)
logfbank_feat=logfbank(sig,rate)
logfbank40_feat=logfbank(sig,rate,winstep=0.01,nfilt=40)
np.save(output_dir+wav_name+'.26',logfbank_feat)
np.save(output_dir+wav_name+'.40',logfbank40_feat)

