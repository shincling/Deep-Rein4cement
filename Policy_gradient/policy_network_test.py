#coding=utf8
import lasagne
import numpy as np
import theano
import theano.tensor as T
from sklearn.preprocessing import LabelBinarizer,label_binarize

def get_dataset(dimention):
    x=np.random.random((10000,dimention))
    y=np.zeros((10000))
    for idx,i in enumerate(x):
        if 0.3<i[0]<0.7:
            y[idx]=1
        elif i[0]>=0.7:
            y[idx]=2
    return x,y

dimention=10
xx,yy=get_dataset(dimention)
yy=np.int32(yy)
x=T.matrix('x')
y=T.imatrix('y')
x1=T.vector('x1')
x2=T.matrix('all')

n_classes=3
batch_size=16
n_epoch=500
path_lenth=10
n_paths=1000

x_shared=theano.shared(np.zeros((batch_size,dimention),dtype=theano.config.floatX),borrow=True)
y_shared=theano.shared(np.zeros((batch_size,1),dtype=np.int32),borrow=True)

l_in = lasagne.layers.InputLayer(shape=(None, 1,dimention))
l_theta = lasagne.layers.DenseLayer(l_in,3,W=lasagne.init.Normal(std=1))
l_mu=lasagne.layers.NonlinearityLayer(l_theta,nonlinearity=lasagne.nonlinearities.softmax)

probas = lasagne.layers.helper.get_output(l_mu, {l_in: x_shared})
givens = {
    x: x_shared,
}
output_model = theano.function([],probas, givens=givens,on_unused_input='ignore',allow_input_downcast=True)

if 1:
    x_range=T.tensor3()
    x_action=T.imatrix()
    x_range_shared=theano.shared(np.zeros((batch_size,path_lenth,dimention),dtype=theano.config.floatX),borrow=True)
    x_range_action=theano.shared(np.zeros((batch_size,path_lenth),dtype=np.int32),borrow=True)

    l_range_in = lasagne.layers.InputLayer(shape=(batch_size,path_lenth,dimention))
    l_range_theta = lasagne.layers.ReshapeLayer(l_range_in,[batch_size*path_lenth,dimention])
    l_range_remu = lasagne.layers.DenseLayer(l_range_theta,n_classes,W=l_theta.W,nonlinearity=lasagne.nonlinearities.softmax)
    l_range_mu = lasagne.layers.ReshapeLayer(l_range_remu,[batch_size,path_lenth,n_classes])
    probas_range = lasagne.layers.helper.get_output(l_range_mu, {l_range_in: x_range_shared})
    givens = {
        x_range: x_range_shared,
        x_action: x_range_action
    }

    output_model_range = theano.function([],probas_range,givens=givens,on_unused_input='ignore',allow_input_downcast=True)
    x_range_batch=np.random.rand(batch_size,path_lenth,dimention)
    x_range_action_batch=np.int32(np.random.randint(0,batch_size,size=[batch_size,path_lenth]))
    x_range_shared.set_value(x_range_batch)
    x_range_action.set_value(x_range_action_batch)
    pred=output_model_range()
    print pred.shape

'''

idx_batch=np.random.randint(0,500)
x_batch=xx[idx_batch*batch_size:(idx_batch+1)*batch_size]
target=yy[idx_batch*batch_size:(idx_batch+1)*batch_size]
y_batch=label_binarize(target,range(n_classes))
x_shared.set_value(x_batch)
y_shared.set_value(np.int32(y_batch))
cost,probs,pred=output_model()
print cost,pred
print probs

def sample_one_path(state,prob):
    n_action=len(prob)
    action=np.random.choice([0,1,2],p=prob)
    if state[0]<0.5:
        if action ==0:
            reward=3
            state[0]+=0.1
        elif action ==1:
            reward=2
            # state[1]+=0.1
        else:
            reward=1
    else:
        if action ==0:
            reward=0
            state[0]+=0.1
        elif action ==1:
            reward=2
            # state[1]+=0.1
        else:
            reward=1
    return action,state,reward

states=x_batch
for repeat_time in range(n_paths):
    total_state=np.zeros([batch_size,path_lenth,dimention])
    total_reward=np.zeros([batch_size,path_lenth])
    total_action=np.zeros([batch_size,path_lenth])
    for t in range(path_lenth):#进行了10步
        for idx,prob in enumerate(probs):#对于batch里的每一个样本
            action,state,reward=sample_one_path(states[idx],prob)
            #action是这一步采取的动作，state是进入的新的状态，prob四
            total_state[idx,t]=state
            total_action[idx,t]=action
            total_reward[idx,t]=reward
        x_shared.set_value(total_state[:,t])
        probs=output_model()[1]
    pass
'''

'''
for epoch in range(n_epoch):
    batch_total_number=len(xx)/batch_size
    cost,error_cout=0,0
    for idx_batch in range(batch_total_number):
        x_batch=xx[idx_batch*batch_size:(idx_batch+1)*batch_size]
        target=yy[idx_batch*batch_size:(idx_batch+1)*batch_size]
        y_batch=label_binarize(target,range(n_classes))
        x_shared.set_value(x_batch)
        y_shared.set_value(np.int32(y_batch))
        cost,pred=train_model()
        cost+=cost

        count=np.count_nonzero(pred-target)
        error_cout+=count
    print cost/batch_size
    print 'accuracy:',1-float(error_cout)/(batch_total_number*batch_size)
'''
