# coding=utf8
import time
import lasagne
import numpy as np
import theano
import theano.tensor as T
from sklearn.preprocessing import LabelBinarizer, label_binarize


def get_dataset(dimention):
    x = 0.1 * np.random.random((10000, dimention))
    y = np.zeros((10000))
    for idx, i in enumerate(x):
        if 0.3 < i[0] < 0.7:
            y[idx] = 1
        elif i[0] >= 0.7:
            y[idx] = 2
    return x, y


def action_to_vector(x, n_classes):
    result = np.zeros([x.shape[0], x.shape[1], n_classes])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] == 0:
                result[i, j] = np.array([1.0, 0, 0])
            elif x[i, j] == 1:
                result[i, j] = np.array([0, 1.0, 0])
            elif x[i, j] == 2:
                result[i, j] = np.array([0, 0, 1.0])
    return result


def reward_count(total_reward, length, discout=0.99):
    '''
    for line in range(total_reward.shape[0]):
        for step in range(length):
            result[line]+=total_reward[line,step]*(discout**step)
    '''
    discout_list = np.zeros(length)
    for idx in range(discout_list.shape[0]):
        step = discout ** idx
        discout_list[idx] = step
    result = np.dot(total_reward, discout_list)
    return result


dimention = 10
xx, yy = get_dataset(dimention)
yy = np.int32(yy)
x = T.matrix('x')
y = T.imatrix('y')
x1 = T.vector('x1')
x2 = T.matrix('all')

n_classes = 3
batch_size = 100
n_epoch = 500
path_lenth = 10
n_paths = 1000
max_norm = 40
lr = 0.02
std = 0.1
discout=0.8
print 'batch_size:{},n_paths:{},std:{},lr:{},discount:{}'.format(batch_size, n_paths, std, lr,discout)

x_shared = theano.shared(np.zeros((batch_size, dimention), dtype=theano.config.floatX), borrow=True)
y_shared = theano.shared(np.zeros((batch_size, 1), dtype=np.int32), borrow=True)

l_in = lasagne.layers.InputLayer(shape=(None, dimention))
l_in1 = lasagne.layers.DenseLayer(l_in, dimention, W=lasagne.init.Normal(std=std),
                                  nonlinearity=lasagne.nonlinearities.sigmoid)
l_theta = lasagne.layers.DenseLayer(l_in1, 3, W=lasagne.init.Normal(std=std),nonlinearity=lasagne.nonlinearities.softmax)
l_mu = l_theta
# l_mu = lasagne.layers.NonlinearityLayer(l_theta, nonlinearity=lasagne.nonlinearities.softmax)

probas = lasagne.layers.helper.get_output(l_mu, {l_in: x_shared})
givens = {
    x: x_shared,
}
output_model = theano.function([], probas, givens=givens, on_unused_input='ignore', allow_input_downcast=True)

if 1:
    x_range = T.tensor3()
    x_action = T.tensor3()
    x_reward = T.vector()
    x_range_shared = theano.shared(np.zeros((batch_size, path_lenth, dimention), dtype=theano.config.floatX),
                                   borrow=True)
    x_range_action = theano.shared(np.zeros((batch_size, path_lenth, n_classes), dtype=theano.config.floatX),
                                   borrow=True)
    x_range_reward = theano.shared(np.zeros(batch_size, dtype=theano.config.floatX), borrow=True)

    l_range_in = lasagne.layers.InputLayer(shape=(batch_size, path_lenth, dimention))
    l_range_theta = lasagne.layers.ReshapeLayer(l_range_in, [batch_size * path_lenth, dimention])
    l_range_in1 = lasagne.layers.DenseLayer(l_range_theta, dimention, W=l_in1.W,
                                            nonlinearity=lasagne.nonlinearities.sigmoid)
    l_range_remu = lasagne.layers.DenseLayer(l_range_in1, n_classes, W=l_theta.W,
                                             nonlinearity=lasagne.nonlinearities.softmax)
    # l_range_remu = lasagne.layers.DenseLayer(l_range_theta,n_classes,W=l_theta.W,nonlinearity=lasagne.nonlinearities.softmax)
    l_range_mu = lasagne.layers.ReshapeLayer(l_range_remu, [batch_size, path_lenth, n_classes])
    probas_range = lasagne.layers.helper.get_output(l_range_mu, {l_range_in: x_range_shared})
    params = lasagne.layers.helper.get_all_params(l_range_mu, trainable=True)
    givens = {
        x_range: x_range_shared,
        x_action: x_range_action,
        x_reward: x_range_reward
    }
    cost = -T.mean(T.sum(T.sum(T.log(probas_range) * x_action, axis=2), axis=1) * x_reward)
    grads = T.grad(cost, params)
    scaled_grads = lasagne.updates.total_norm_constraint(grads, max_norm)
    updates = lasagne.updates.rmsprop(grads, params, learning_rate=lr)

    output_model_range = theano.function([], [probas_range, cost], givens=givens, updates=updates,
                                         on_unused_input='ignore', allow_input_downcast=True)
    # x_range_batch=np.random.rand(batch_size,path_lenth,dimention)
    # x_range_action_batch=np.int32(np.random.randint(0,batch_size,size=[batch_size,path_lenth]))
    # x_range_shared.set_value(x_range_batch)
    # x_range_action.set_value(x_range_action_batch)
    # pred=output_model_range()[0]
    # print pred.shape


# idx_batch=np.random.randint(0,500)
# x_batch=xx[idx_batch*batch_size:(idx_batch+1)*batch_size]
# target=yy[idx_batch*batch_size:(idx_batch+1)*batch_size]
# y_batch=label_binarize(target,range(n_classes))
# x_shared.set_value(x_batch)
# y_shared.set_value(np.int32(y_batch))
# probs=output_model()
# print probs

def sample_one_path(state, prob):
    n_action = len(prob)
    action = np.random.choice([0, 1, 2], p=prob)
    if state[0] < 0.5:
        if action == 0:
            reward = 5
            state[0] += 0.1
        elif action == 1:
            reward = 2
            # state[1]+=0.1
        else:
            reward = 1
    else:
        if action == 0:
            reward = 0
            state[0] += 0.1
        elif action == 1:
            reward = 2
            # state[1]+=0.1
        else:
            reward = 1
    return action, state, reward


for epoch in range(n_epoch):
    begin_time = time.time()
    batch_total_number = len(xx) / batch_size
    np.random.shuffle(xx)
    for idx_batch in range(batch_total_number):
        x_batch = xx[idx_batch * batch_size:(idx_batch + 1) * batch_size]
        # 初始化两个循环的参数，state和概率
        x_shared.set_value(x_batch)
        probs_sample_0 = output_model()

        tmp_cost, tmp_result, tmp_reward = 0, 0, 0
        for repeat_time in range(n_paths):  # 每一个batch都要经过多次重复采样获取不同的道路
            # 但是第一步的初始化状态都是一样的
            total_state = np.zeros([batch_size, path_lenth, dimention])
            total_reward = np.zeros([batch_size, path_lenth])
            total_action = np.zeros([batch_size, path_lenth])
            total_probs = np.zeros([batch_size, path_lenth, n_classes])
            total_state[:, 0, :] = x_batch
            total_probs[:, 0, :] = probs_sample_0
            for t in range(path_lenth):  # 进行了10步
                for idx, prob in enumerate(total_probs[:, t, :]):  # 对于batch里的每一个样本
                    now_state = total_state[idx, t].copy()
                    action, new_state, reward = sample_one_path(now_state, prob)
                    # action是这一步采取的动作，state是进入的新的状态，prob四
                    total_action[idx, t] = action
                    total_reward[idx, t] = reward
                    # 更新state和概率,下一个循环使用
                    if t != path_lenth - 1:
                        total_state[idx, t + 1] = new_state
                    del new_state, now_state

                if t != path_lenth - 1:
                    x_shared.set_value(total_state[:, t + 1])
                    total_probs[:, t + 1, :] = output_model()

            x_range_shared.set_value(total_state)
            x_range_action.set_value(action_to_vector(total_action, n_classes))
            x_range_reward.set_value(reward_count(total_reward, length=path_lenth, discout=discout))
            aver_reward = np.mean(np.sum(np.float32(total_reward), axis=1))
            espect_reward = np.mean(np.float32(reward_count(total_reward,path_lenth,discout=discout)), axis=0)
            _, cost = output_model_range()
            tmp_cost += cost
            tmp_result += aver_reward
            tmp_reward += espect_reward
            # print 'cost:{},average_reward:{}'.format(cost,aver_reward)
            # print _[0]
            # print '\n\n\n'
        print 'cost:{},average_reward:{},espect_reward:{}'.format(tmp_cost / n_paths, tmp_result / n_paths, tmp_reward/ n_paths)
        if tmp_result / n_paths > 10:
            print total_state[0]
            print total_action[0]
            print _[0]
            print total_reward[0]
            print '\n\n'

    print 'epoch:{},time:{}'.format(epoch, time.time() - begin_time)

'''
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
    test
'''
