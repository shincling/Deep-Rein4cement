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
n_epoch=530
path_lenth=10
n_paths=1000

x_shared=theano.shared(np.zeros((batch_size,dimention),dtype=theano.config.floatX),borrow=True)
y_shared=theano.shared(np.zeros((batch_size,1),dtype=np.int32),borrow=True)

l_in = lasagne.layers.InputLayer(shape=(None, 1,dimention))
# l_in1=lasagne.layers.DenseLayer(l_in,30,W=lasagne.init.Normal(std=1),nonlinearity=lasagne.nonlinearities.softmax)
l_theta = lasagne.layers.DenseLayer(l_in,3,W=lasagne.init.Normal(std=1))
l_mu=lasagne.layers.NonlinearityLayer(l_theta,nonlinearity=lasagne.nonlinearities.softmax)

probas = lasagne.layers.helper.get_output(l_mu, {l_in: x_shared})
pred = T.argmax(probas, axis=1)
cost = T.nnet.categorical_crossentropy(probas, y).sum()
params = lasagne.layers.helper.get_all_params(l_mu, trainable=True)
grads = T.grad(cost, params)
updates = lasagne.updates.sgd(grads, params, learning_rate=0.05)


givens = {
    x: x_shared,
    y: y_shared,
    }

train_model = theano.function([], [cost,pred], givens=givens,updates=updates,on_unused_input='ignore',allow_input_downcast=True)
output_model = theano.function([], [cost,probas,pred], givens=givens,on_unused_input='ignore',allow_input_downcast=True)

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
