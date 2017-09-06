from visdom import Visdom
import numpy as np
import math
import os.path
import getpass
from sys import platform as _platform
from six.moves import urllib
import re

viz = Visdom()
f=open('log_purerl_1to0_5class11').read()
print f[:100]
# ttt0,ttt1,ttt2,ttt3,ttt4,ttt5=[],[],[],[],[],[]
ttt=[]
ttt.append([float(i) for i in re.findall('ttt0 is.* (.*)',f)])
ttt.append([float(i) for i in re.findall('acc:(.*?)\t',f)])
ttt.append([float(i) for i in re.findall('ttt2 is.* (.*)',f)])
ttt.append([float(i) for i in re.findall('ttt3 is.* (.*)',f)])
ttt.append([float(i) for i in re.findall('ttt4 is.* (.*)',f)])
ttt.append([float(i) for i in re.findall('ttt5 is.* (.*)',f)])
times=range(0,306,3)
times=range(0,60,3)

tt=[]
for i in ttt:
    tt.append(i[:20])
#
ttt=tt
# textwindow = viz.text('Hello World!')

win = viz.line(
    X=np.array(times),
    Y=np.array(ttt[0]),
    opts=dict(
        markersize=10,
        markers=1,
        legend=['0-shot accuracy']
    )
)
viz.updateTrace(
    X=np.array(times),
    Y=np.array(ttt[1]),
    win=win,
    name='1-shot accuracy',
    opts=dict(
        markersize=10,
        markers=1,
        markersymbol='square',
    )

)
viz.updateTrace(
    X=np.array(times),
    Y=np.array(ttt[2]),
    win=win,
    name='2-shot accuracy'
)
viz.updateTrace(
    X=np.array(times),
    Y=np.array(ttt[3]),
    win=win,
    name='3-shot accuracy'
)
viz.updateTrace(
    X=np.array(times),
    Y=np.array(ttt[4]),
    win=win,
    name='4-shot accuracy'
)
