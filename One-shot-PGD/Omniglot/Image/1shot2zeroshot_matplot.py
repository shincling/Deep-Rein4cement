# from visdom import Visdom
import numpy as np
import math
# import os.path
# import getpass
# from sys import platform as _platform
# from six.moves import urllib
import re
import matplotlib.pyplot as plt

# viz = Visdom()
f=open('log_purerl_1to0_5class11').read()
print f[:100]
ttt0,ttt1,ttt2,ttt3,ttt4,ttt5=[],[],[],[],[],[]
ttt=[]
ttt.append([float(i) for i in re.findall('ttt0 is.* (.*)',f)])
ttt.append([float(i) for i in re.findall('acc:(.*?)\t',f)])
ttt.append([float(i) for i in re.findall('ttt2 is.* (.*)',f)])
ttt.append([float(i) for i in re.findall('ttt3 is.* (.*)',f)])
ttt.append([float(i) for i in re.findall('ttt4 is.* (.*)',f)])
ttt.append([float(i) for i in re.findall('ttt5 is.* (.*)',f)])
times=range(0,306,3)
# times=range(0,60,3)

tt=[]
times=times[:50]
for i in ttt:
    tt.append(i[:50])

ttt=tt
# textwindow = viz.text('Hello World!')


plt.plot(times, ttt[0], c='b', marker='o')
plt.plot(times, ttt[1], c='r', marker='D')
plt.plot(times, ttt[2], c='#EE7600', marker=5)
plt.plot(times, ttt[3], c='#458B00', marker=7)
plt.plot(times, ttt[4], c='#551A8B', marker=4)
# plt.plot(times, ttt[5], c='c', marker='*')

# plt.plot(times, ttt[0], c='m', marker='o')
# plt.plot(times, ttt[1], c='g', marker='o')
# plt.plot(times, ttt[2], c='r', marker='o')
# plt.plot(times, ttt[3], c='b', marker='o')
# plt.plot(times, ttt[4], c='y', marker='o')
# plt.plot(times, ttt[5], c='c', marker='o')

plt.xlabel('Episode',fontsize='18')
plt.ylabel('Percent Correct',fontsize='18')
plt.legend((u'0-shot accuracy', u'1-shot accuracy', u'2-shot accuracy', u'3-shot accuracy', u'4-shot accuracy'),
           loc=4)
plt.savefig('one-shot.png')
plt.show()













# --------------------------------------------------
# win = viz.line(
#     X=np.array(times),
#     Y=np.array(ttt[0]),
#     opts=dict(
#         markersize=10,
#         markers=1,
#         legend=['0-shot accuracy'],
#         ylabel='Percent Correct',
#         xlabel='Episode',
#         xtick=1,
#         ytick=1,
#         # xtype='log'
#     )
# )
# viz.updateTrace(
#     X=np.array(times),
#     Y=np.array(ttt[1]),
#     win=win,
#     name='1-shot accuracy',
# )
# viz.updateTrace(
#     X=np.array(times),
#     Y=np.array(ttt[2]),
#     win=win,
#     name='2-shot accuracy'
# )
# viz.updateTrace(
#     X=np.array(times),
#     Y=np.array(ttt[3]),
#     win=win,
#     name='3-shot accuracy'
# )
# viz.updateTrace(
#     X=np.array(times),
#     Y=np.array(ttt[4]),
#     win=win,
#     name='4-shot accuracy'
# )
