#coding=utf8
import random
import argparse
import time
import pickle
import lasagne
import numpy as np
import theano
import theano.tensor as T
from sklearn.preprocessing import LabelBinarizer,label_binarize
import image

def action_to_vector_real(x, n_classes): #x是bs*path_length
    result = np.zeros([x.shape[0], x.shape[1], n_classes])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result[i,j]=label_binarize([int(x[i,j])],range(n_classes))[0]
    return np.int32(result)

def action_to_vector(x, n_classes): #x是bs*path_length
    p=0.7 #p是标签正常的概率
    result = np.zeros([x.shape[0], x.shape[1], n_classes])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if np.random.rand()<p and j!=x.shape[1]:
                result[i,j]=label_binarize([int(x[i,j])],range(n_classes))[0]
    return np.int32(result)

def reward_count(total_reward, length, discout=0.99):
    discout_list = np.zeros(length)
    for idx in range(discout_list.shape[0]):
        step = discout ** idx
        discout_list[idx] = step
    result = np.dot(total_reward, discout_list)
    return result

def get_dataset(dimention,path_length=10,total_labels=10,total_label=3,total_roads=10000):
    order_list=[[],[],[],[],[],[],[],[],[],[]]
    if not test_mode:
        x=np.random.random((10000,dimention))
        y=np.zeros((10000))
        for i in (x):
            y=int(i[0]/0.1)
            order_list[y].append(i)
    else:
        for i in range(10):
            zz=np.zeros((10))
            zz[i]=1
            order_list[i]=[zz]

    # total_roads=1000
    # total_lables=10
    # path_length=10
    total_label=3
    final_x=np.zeros((total_roads,path_length,dimention))
    final_y=np.zeros((total_roads,path_length))
    for one_sample in range(total_roads):
        label_list=random.sample(range(total_labels),total_label)
        one_shoot_label=label_list[-1]
        insert_idx=np.random.randint(0,path_length-1)
        final_x[one_sample,insert_idx]=random.sample(order_list[one_shoot_label],1)[0]
        final_y[one_sample,insert_idx]=one_shoot_label
        final_x[one_sample,-1]=random.sample(order_list[one_shoot_label],1)[0]
        final_y[one_sample,-1]=one_shoot_label
        for i in range(path_length-1):
            if i!=insert_idx:
                label=np.random.choice(label_list[:-1])
                final_y[one_sample,i]=label
                final_x[one_sample,i,:]=random.sample(order_list[label],1)[0]
    return final_x,np.int32(final_y)

class BatchedDotLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, **kwargs):
        super(BatchedDotLayer, self).__init__(incomings, **kwargs)
        if len(incomings) != 2:
            raise NotImplementedError

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][2])

    def get_output_for(self, inputs, **kwargs):
        return T.batched_dot(inputs[0], inputs[1])

class InnerProductLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, nonlinearity=None, **kwargs):
        super(InnerProductLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = nonlinearity
        if len(incomings) != 2:
            raise NotImplementedError

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:2]

    def get_output_for(self, inputs, **kwargs):
        M = inputs[0]
        u = inputs[1]
        output = T.batched_dot(M, u)
        if self.nonlinearity is not None:
            output = self.nonlinearity(output)
        return output

class finalChoiceLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, W_question, W_choice,W_out, nonlinearity=lasagne.nonlinearities.tanh,**kwargs):
        super(finalChoiceLayer, self).__init__(incomings, **kwargs) #？？？不知道这个super到底做什么的，会引入input_layers和input_shapes这些属性
        if len(incomings) != 2:
            raise NotImplementedError
        # if mask_input is not None:
        #     incomings.append(mask_input)
        batch_size, max_sentlen ,embedding_size = self.input_shapes[0]
        self.batch_size,self.max_sentlen,self.embedding_size=batch_size,max_sentlen,embedding_size
        self.W_h=self.add_param(W_choice,(embedding_size,embedding_size), name='Pointer_layer_W_h')
        self.W_q=self.add_param(W_question,(embedding_size,embedding_size), name='Pointer_layer_W_q')
        self.W_o=self.add_param(W_out,(embedding_size,), name='Pointer_layer_W_o')
        self.nonlinearity=nonlinearity
        # zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        # self.set_zero = theano.function([zero_vec_tensor], updates=[(x, T.set_subtensor(x[0, :], zero_vec_tensor)) for x in [self.A,self.C]])

    def get_output_shape_for(self, input_shapes):
        return (self.batch_size,self.max_sentlen)
    def get_output_for(self, inputs, **kwargs):
        #input[0]:(BS,max_senlen,emb_size),input[1]:(BS,1,emb_size),input[2]:(BS,max_sentlen)
        activation0=(T.dot(inputs[0],self.W_h))
        activation1=T.dot(inputs[1],self.W_q).reshape([self.batch_size,self.embedding_size]).dimshuffle(0,'x',1)
        activation=self.nonlinearity(activation0+activation1)#.dimshuffle(0,'x',2)#.repeat(self.max_sentlen,axis=1)
        final=T.dot(activation,self.W_o) #(BS,max_sentlen)
        # if inputs[2] is not None:
        #     final=inputs[2]*final-(1-inputs[2])*1000000
        alpha=lasagne.nonlinearities.softmax(final) #(BS,max_sentlen)
        # final=T.batched_dot(alpha,inputs[0])#(BS,max_sentlen)*(BS,max_sentlen,emb_size)--(BS,emb_size)
        return alpha

class ChoiceLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, W_question, W_choice,W_out, nonlinearity=lasagne.nonlinearities.tanh,**kwargs):
        super(ChoiceLayer, self).__init__(incomings, **kwargs) #？？？不知道这个super到底做什么的，会引入input_layers和input_shapes这些属性
        if len(incomings) != 2:
            raise NotImplementedError
        # if mask_input is not None:
        #     incomings.append(mask_input)
        batch_size, max_sentlen ,embedding_size = self.input_shapes[0]
        self.batch_size,self.max_sentlen,self.embedding_size=batch_size,max_sentlen,embedding_size
        self.W_h=self.add_param(W_choice,(embedding_size,1), name='Pointer_layer_W_h')
        self.b_h=self.add_param(W_choice,(1,1), name='Pointer_layer_W_b')
        self.W_q=self.add_param(W_question,(embedding_size,1), name='Pointer_layer_W_q')
        self.W_o=self.add_param(W_out,(embedding_size,embedding_size), name='Pointer_layer_W_o')
        self.nonlinearity=nonlinearity
        # zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        # self.set_zero = theano.function([zero_vec_tensor], updates=[(x, T.set_subtensor(x[0, :], zero_vec_tensor)) for x in [self.A,self.C]])

    def get_output_shape_for(self, input_shapes):
        return (self.batch_size,self.max_sentlen)
    def get_output_for(self, inputs, **kwargs):
        #input[0]:(BS,max_senlen,emb_size),input[1]:(BS,1,emb_size),input[2]:(BS,max_sentlen)
        activation0=(T.dot(inputs[0],self.W_h)).reshape([self.batch_size,self.max_sentlen])+self.b_h.repeat(self.batch_size,0).repeat(self.max_sentlen,1)
        activation1=T.dot(inputs[1],self.W_q).reshape([self.batch_size]).dimshuffle(0,'x')
        # activation2=T.batched_dot(inputs[0],inputs[1].reshape([self.batch_size,self.embedding_size,1])).reshape([self.batch_size,self.max_sentlen])
        activation2=T.batched_dot(T.dot(inputs[0],self.W_o),inputs[1].reshape([self.batch_size,self.embedding_size,1])).reshape([self.batch_size,self.max_sentlen])
        activation=(self.nonlinearity(activation0)+self.nonlinearity(activation1)+activation2).reshape([self.batch_size,self.max_sentlen])#.dimshuffle(0,'x',2)#.repeat(self.max_sentlen,axis=1)
        # final=T.dot(activation,self.W_o) #(BS,max_sentlen)
        # if inputs[2] is not None:
        #     final=inputs[2]*final-(1-inputs[2])*1000000
        alpha=lasagne.nonlinearities.softmax(activation) #(BS,max_sentlen)
        # final=T.batched_dot(alpha,inputs[0])#(BS,max_sentlen)*(BS,max_sentlen,emb_size)--(BS,emb_size)
        return alpha

class Model:
    def __init__(self,x_dimension=(20,20),h_dimension=30,n_classes=10,batch_size=32,n_epoch=50,path_length=10,
                 n_paths=1000,max_norm=50,lr=0.05,discount=0.99,update_method='sgd',std=0.5,save_path=str(np.random.randint(100,999)),**kwargs):
        self.h_dim=h_dimension
        self.n_classes=n_classes
        self.n_slot=n_classes
        self.batch_size=batch_size
        self.n_epoch=n_epoch
        self.path_length=path_length
        self.n_paths=n_paths
        self.max_norm=max_norm
        self.lr=lr
        self.std=std
        self.discount=discount
        self.x_dim=x_dimension
        self.save_path=save_path

        if update_method=='sgd':
            self.update_method=lasagne.updates.sgd
        elif update_method=='adagrad':
            self.update_method=lasagne.updates.adagrad
        elif update_method=='adadelta':
            self.update_method=lasagne.updates.adadelta
        elif update_method=='rmsprop':
            self.update_method=lasagne.updates.rmsprop
        self.build()

    def build(self):
        x_range=T.tensor4()
        x_label=T.tensor3()
        x_action=T.tensor3()
        x_reward=T.vector()
        x_memory=T.tensor4()

        self.x_range_shared=theano.shared(np.zeros((self.batch_size,self.path_length,self.x_dim[0],self.x_dim[1]),dtype=theano.config.floatX),borrow=True)
        self.x_range_label=theano.shared(np.zeros((self.batch_size,self.path_length,self.n_classes),dtype=theano.config.floatX),borrow=True)
        self.x_range_action=theano.shared(np.zeros((self.batch_size,self.path_length,self.n_classes),dtype=theano.config.floatX),borrow=True)
        self.x_range_reward=theano.shared(np.zeros(self.batch_size,dtype=theano.config.floatX),borrow=True)
        self.x_range_memory=theano.shared(np.zeros((self.batch_size,self.path_length,self.n_classes,self.h_dim),dtype=theano.config.floatX),borrow=True)

        '''前期的框架模型，主要是得到x到h的映射，以及memory的构建'''
        D1, D2, D3 = lasagne.init.Normal(std=self.std), lasagne.init.Normal(std=self.std), lasagne.init.Normal(std=self.std)
        l_range_in = lasagne.layers.InputLayer(shape=(self.batch_size,self.path_length,self.x_dim[0],self.x_dim[1]))
        l_range_flatten = lasagne.layers.ReshapeLayer(l_range_in,[self.batch_size*self.path_length,1,self.x_dim[0],self.x_dim[1]])
        l_range_flatten = lasagne.layers.NonlinearityLayer(l_range_flatten,nonlinearity=lasagne.nonlinearities.sigmoid)
        # l_range_conv1=lasagne.layers.Conv2DLayer(l_range_flatten,num_filters=128,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify)
        # l_range_conv1=lasagne.layers.Conv2DLayer(l_range_conv1,num_filters=128,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify)
        # l_pool1=lasagne.layers.MaxPool2DLayer(l_range_conv1,pool_size=(2,2))
        # l_dropout1=lasagne.layers.DropoutLayer(l_pool1,p=0.2)
        # l_range_conv2=lasagne.layers.Conv2DLayer(l_pool1,num_filters=128,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify)
        # l_range_conv2=lasagne.layers.Conv2DLayer(l_range_conv2,num_filters=128,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify)
        # l_pool2=lasagne.layers.MaxPool2DLayer(l_range_conv2,pool_size=(2,2))
        # l_dropout2=lasagne.layers.DropoutLayer(l_pool2,p=0.2)

        # l_range_dense2 = lasagne.layers.DenseLayer(l_dropout1,500,W=D1,nonlinearity=lasagne.nonlinearities.tanh) #[bs*path_length,dimension]
        # l_dropout3=lasagne.layers.DropoutLayer(l_range_dense2,p=0.5)
        # l_range_dense2 = lasagne.layers.DenseLayer(l_dropout1,self.h_dim,W=D2,nonlinearity=lasagne.nonlinearities.tanh) #[bs*path_length,dimension]
        l_range_dense2_origin=lasagne.layers.ReshapeLayer(l_range_flatten,[self.batch_size,self.path_length,self.h_dim])
        l_range_label = lasagne.layers.InputLayer(shape=(self.batch_size,self.path_length,self.n_classes))
        if if_cont==1:
            l_range_dense2_origin=lasagne.layers.ConcatLayer((l_range_dense2_origin,l_range_label),axis=2)
            self.h_dim+=self.n_classes
        l_range_hidden=lasagne.layers.ReshapeLayer(l_range_dense2_origin,[self.batch_size*self.path_length,1,self.h_dim])

        # if test_mode:
        #     l_range_dense2,l_range_dense2_origin=l_range_in,l_range_in
        # l_range_mu = lasagne.layers.DenseLayer(l_range_hidden,self.n_slot,W=D2,nonlinearity=lasagne.nonlinearities.softmax)

        '''Policy Gradient Methods的模型，主要是从Memory状态得到action的概率'''
        l_range_memory_in = lasagne.layers.InputLayer(shape=(self.batch_size,self.path_length,self.n_classes,self.h_dim))
        l_range_memory = lasagne.layers.ReshapeLayer(l_range_memory_in,[self.batch_size*self.path_length,self.n_classes,self.h_dim])
        if 0:
            l_range_status=lasagne.layers.ConcatLayer((l_range_memory,l_range_hidden),axis=1) #[bs*pl,(n_class+1),dim]
            l_range_mu=lasagne.layers.DenseLayer(l_range_status,self.n_classes,W=D3,nonlinearity=lasagne.nonlinearities.sigmoid)
            l_range_mu=lasagne.layers.DenseLayer(l_range_mu,self.n_classes,W=D3,nonlinearity=lasagne.nonlinearities.softmax)
            l_range_mu = lasagne.layers.ReshapeLayer(l_range_mu,[self.batch_size,self.path_length,self.n_classes])
        if 1:
            l_range_hidden=lasagne.layers.ReshapeLayer(l_range_dense2_origin,[self.batch_size*self.path_length,1,self.h_dim])
            l_range_status=ChoiceLayer((l_range_memory,l_range_hidden),D3,D3,D3,nonlinearity=lasagne.nonlinearities.tanh) #[bs*pl,(n_class+1),dim]
            l_range_mu = lasagne.layers.ReshapeLayer(l_range_status,[self.batch_size,self.path_length,self.n_classes])

        '''模型的总体参数和更新策略等'''
        hidden = lasagne.layers.helper.get_output(l_range_dense2_origin, {l_range_in: x_range,l_range_label:x_label})
        probas_range = lasagne.layers.helper.get_output(l_range_mu, {l_range_in: x_range,l_range_memory_in:x_memory,l_range_label:x_label})
        params=lasagne.layers.helper.get_all_params(l_range_mu,trainable=True)
        givens = {
            x_range: self.x_range_shared,
            x_label:self.x_range_label,
            x_action: self.x_range_action,
            x_reward: self.x_range_reward,
            x_memory: self.x_range_memory
        }
        cost=-T.mean(T.sum(T.sum(T.log(probas_range)*x_action,axis=2),axis=1)*x_reward)
        grads=T.grad(cost,params)
        scaled_grads = lasagne.updates.total_norm_constraint(grads, self.max_norm)
        updates = self.update_method(scaled_grads, params, learning_rate=self.lr)

        self.output_model_range = theano.function([],[probas_range,cost,hidden],givens=givens,on_unused_input='ignore',allow_input_downcast=True)
        self.output_model_range_updates = theano.function([],[probas_range,cost,hidden],updates=updates,givens=givens,on_unused_input='ignore',allow_input_downcast=True)
        self.output_hidden = theano.function([x_range,x_label],[hidden[:,0]],on_unused_input='ignore',allow_input_downcast=True)

        if hid==1:
            # l_label = lasagne.layers.InputLayer(shape=(self.batch_size,lll))
            # xx_label=T.matrix()
            # l_range_probas=lasagne.layers.SliceLayer(l_range_dense2_origin,0,axis=1)
            # l_range_probas=lasagne.layers.ReshapeLayer(l_range_probas,[self.batch_size,self.h_dim])
            l_range_dense2_origin=lasagne.layers.ReshapeLayer(l_range_dense2,[self.batch_size*self.path_length,self.h_dim])
            l_range_probas=lasagne.layers.DenseLayer(l_range_dense2_origin,lll,W=D3,nonlinearity=lasagne.nonlinearities.softmax)
            ppp=lasagne.layers.helper.get_output(l_range_probas,{l_range_in:x_range,l_range_label:x_label})
            hidden_params=lasagne.layers.helper.get_all_params(l_range_probas,trainable=True)
            hidden_cost = T.nnet.categorical_crossentropy(ppp, x_label.reshape([-1,lll])).sum()
            pred = T.argmax(ppp, axis=1)
            hidden_grads=T.grad(hidden_cost,hidden_params)
            hidden_updates =lasagne.updates.adagrad(hidden_grads, hidden_params, learning_rate=0.01)
            self.hid = theano.function([x_range,x_label],[hidden_cost,pred,ppp],updates=hidden_updates,on_unused_input='ignore',allow_input_downcast=True)
            # self.hid = theano.function([x_range,x_label],[x_label[:,0]],on_unused_input='ignore',allow_input_downcast=True)

        self.network=l_range_mu

    def sample_one_path(self,now_state,now_memory_label, prob,this_label,idx_path_length):
        new_memory_state=np.zeros([self.n_classes,self.h_dim])
        new_memory_label=np.zeros([self.n_classes,self.path_length])
        reward=0.0

        now_memory_label_count=np.int32(now_memory_label!=-1).sum(axis=1)#计算当前memory每个slot各有多少个数据存储
        action=np.random.choice(range(self.n_classes),p=prob)
        index_to_insert=int(np.argwhere(now_memory_label[action]==-1)[0])
        #定义reward
        if now_memory_label[action][0]==-1:#证明是新开的一个slot
            reward=-0
        else: #证明action是一个已经有label的slot位置
            label_count_dict={}
            for jj in (now_memory_label[action]):
                if jj==-1:
                    break
                if jj in label_count_dict.keys():
                    label_count_dict[jj]+=1
                else:
                    label_count_dict[jj]=1
            max_num=max(label_count_dict.values())
            if 1:
                if this_label in label_count_dict and len(label_count_dict)==1:
                    if idx_path_length==self.path_length-1:
                        reward=10000
                        print '100~!\n'
                    else:
                        reward=3
                else:
                    reward=-0
        #更新状态，加权平均，更新label记录
        now_state[action]=now_state[-1]+(now_state[action]*now_memory_label_count[action])
        now_state[action]=now_state[action]/(now_memory_label_count[action]+1)
        new_memory_state=now_state[:-1]
        now_memory_label[action][index_to_insert]=this_label
        new_memory_label=now_memory_label


        return action,new_memory_state,reward,new_memory_label

    def train(self):
        batch_size=self.batch_size
        n_paths=self.n_paths
        path_length=self.path_length
        x_dim=self.x_dim
        h_dim=self.h_dim
        n_classes=self.n_classes
        save_path=self.save_path
        # xx, yy = get_dataset(x_dim,path_length,n_classes) #xx是[sample,path_length,dimension]，yy是[sample.path_length]
        x_train,y_train,x_test,y_test=image.x_train,image.y_train,image.x_test,image.y_test
        yy_train,yy_test=image.y_train_shuffle,image.y_test_shuffle
        xx,yy=x_train,yy_train
        if 0:
            load_params=pickle.load(open('params/params_001_0_44_2016-11-11 14:58:31'))
            lasagne.layers.set_all_param_values(self.network,load_params)
            print 'load succeed!'
        for epoch in range(self.n_epoch):
            begin_time = time.time()
            batch_total_number = len(xx) / batch_size
            '''每一轮的shuffle策略'''
            zipp=zip(xx,yy,y_train)
            random.shuffle(zipp)
            xx=np.array([one[0] for one in zipp])
            yy=np.array([one[1] for one in zipp])
            y_train=np.array([one[2] for one in zipp])

            for repeat_time in range(n_paths):  # 每一个batch都要经过多次重复采样获取不同的道路
                if self.lr<0.01:
                    self.lr*=1.05
                    print "now lr is:",self.lr
                for idx_batch in range(batch_total_number):#对于每一个batch
                # 初始化两个循环的参数，state和概率
                    xx_batch = xx[idx_batch * batch_size:(idx_batch + 1) * batch_size]
                    yy_batch = yy[idx_batch * batch_size:(idx_batch + 1) * batch_size]
                    yy_batch_vector=action_to_vector_real(yy_batch,self.n_classes)

                    # global hid
                    # y_batch = np.int32(y_train)[idx_batch * batch_size:(idx_batch + 1) * batch_size]
                    # if hid==1:
                    #     # self.x_range_shared.set_value(xx_batch)
                    #     # self.x_range_label.set_value(action_to_vector(y_batch,len(image.)))
                    #     ccc,pred,ppp=self.hid(xx_batch,action_to_vector_real(y_batch,lll))
                    #     print 'The hidden classification is :',ccc
                    #     errors=np.count_nonzero(np.int32(pred==y_batch.flatten()))
                    #     print pred[0:100]
                    #     print y_batch.flatten()[0:100]
                    #     print 'right rate:',float(errors)/len(y_batch.flatten())
                    #     if ccc>200:
                    #         continue
                    #     else:
                    #         hid=0

                    xx_batch_0=xx_batch[:,0,:].reshape([xx_batch.shape[0],1,xx_batch.shape[-2],xx_batch.shape[-1]])
                    xx_batch_0_repeat=xx_batch_0.repeat(path_length,axis=1)
                    memory_0=np.zeros((batch_size,self.n_classes,h_dim))
                    memory_0_repeat=np.zeros((batch_size,path_length,self.n_classes,h_dim))

                    tmp_cost, tmp_result, tmp_reward = 0, 0, 0
                    # 但是第一步的初始化状态都是一样的
                    self.x_range_shared.set_value(xx_batch_0_repeat)
                    self.x_range_label.set_value(yy_batch_vector)
                    self.x_range_memory.set_value(memory_0_repeat)
                    probbb = self.output_model_range()[0]
                    probs_sample_0 = probbb[:, 0]

                    total_state = np.zeros([batch_size, path_length, n_classes+1,h_dim])
                    total_memory_label=np.zeros([batch_size,path_length,n_classes,path_length],dtype=np.int32)-1 #取作-1，标志着还没有存放过样本
                    total_reward = np.zeros([batch_size, path_length])
                    total_action = np.zeros([batch_size, path_length])
                    total_probs = np.zeros([batch_size, path_length, n_classes])

                    total_state[:, 0, -1] =self.output_hidden(xx_batch_0_repeat,yy_batch_vector)[0]
                    total_state[:, 0, :-1] = memory_0
                    '''state是[bs,class+1,x_dim]尺度的东西，第二个维度前class个是memory状态,最后一个是当前输入的隐层表达'''
                    total_probs[:, 0, :] = probs_sample_0

                    for t in range(path_length):  # 进行了path_length步
                        for idx, prob in enumerate(total_probs[:, t, :]):  # 对于batch里的每一个样本
                            now_state = total_state[idx, t].copy()
                            now_memory_label=total_memory_label[idx,t].copy()
                            action, new_memory_state, reward, new_memory_label = self.sample_one_path(now_state,now_memory_label, prob,yy_batch[idx,t],t)
                            # action, new_state, reward = sample_one_path_plus(now_state, prob)
                            # action是这一步采取的动作，state是进入的新的状态，prob四
                            total_action[idx, t] = action
                            total_reward[idx, t] = reward
                            # 更新state和概率,下一个循环使用
                            if t != path_length - 1:
                                # TODO:这里需要模型中添加一个只从x得到隐层表示的小函数
                                # new_h=self.output_hidden(xx_batch[idx,t+1])[-1]
                                # new_state=np.concatenate((new_memory_state,new_h),axis=1)
                                total_state[idx, t + 1,:-1] = new_memory_state
                                total_memory_label[idx,t+1]=new_memory_label
                            del new_memory_state, now_state,new_memory_label

                        if t != path_length - 1:
                            total_state[:, t + 1,-1,:] =self.output_hidden(xx_batch[:,t+1].reshape(batch_size,1,x_dim[0],x_dim[1]).repeat(path_length,axis=1),yy_batch_vector[:,t+1].reshape(batch_size,1,n_classes).repeat(path_length,axis=1))[0]
                            self.x_range_shared.set_value(xx_batch[:,t+1].reshape(batch_size,1,x_dim[0],x_dim[1]).repeat(path_length,axis=1))
                            self.x_range_label.set_value(yy_batch_vector[:,t+1].reshape(batch_size,1,n_classes).repeat(path_length,axis=1))
                            self.x_range_memory.set_value(total_state[:,t+1,:-1,:].reshape(batch_size,1,n_classes,h_dim).repeat(path_length,axis=1))
                            total_probs[:, t + 1, :] = self.output_model_range()[0][:,0]

                    self.x_range_shared.set_value(xx_batch)
                    self.x_range_label.set_value(yy_batch_vector)
                    self.x_range_memory.set_value(total_state[:,:,:-1])
                    self.x_range_action.set_value(action_to_vector(total_action, n_classes))
                    self.x_range_reward.set_value(reward_count(total_reward, length=path_length, discout=self.discount))
                    aver_reward = np.mean(np.sum(np.float32(total_reward), axis=1))
                    espect_reward = np.mean(np.float32(reward_count(total_reward,path_length,discout=self.discount)), axis=0)
                    _, cost = self.output_model_range_updates()[:-1]
                    tmp_cost += cost
                    tmp_result += aver_reward
                    tmp_reward += espect_reward
                    print 'cost:{},average_reward:{}'.format(cost,aver_reward)
                    print total_state[0][-1]
                    print total_action[0]
                    print total_memory_label[0][-1]
                    print total_reward[0]
                    print _[0]
                    # print total_probs[0]
                    # print _[0]
                    print '\n\n\n'

                prev_weights = lasagne.layers.helper.get_all_param_values(self.network)
                pickle.dump(prev_weights,open('params/params_{}_{}_{}_{}'.format(save_path,epoch,repeat_time,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),'wb'))

                if tmp_result / n_paths  > 0:
                    print total_state[0][-1]
                    print total_action[0]
                    print total_memory_label[0][-1]
                    print total_reward[0]
                    print _[0]
                    # print total_probs[0]
                    print '\n\n'

            print 'cost:{},average_reward:{},espect_reward:{},save_folder:{}'.format(tmp_cost /path_length, tmp_result /path_length, tmp_reward/path_length, save_path)
            print 'epoch:{},time:{}'.format(epoch, time.time() - begin_time)

test_mode=0
if_cont=0
global hid
hid=0
lll=170
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, default=1, help='Task#')
    if test_mode:
        parser.add_argument('--x_dimension', type=int, default=10, help='Dimension#')
    else:
        parser.add_argument('--x_dimension', type=tuple, default=(10,10), help='Dimension#')
    parser.add_argument('--h_dimension', type=int, default=100, help='Dimension#')
    parser.add_argument('--n_classes', type=int, default=10, help='Task#')
    parser.add_argument('--batch_size', type=int, default=160, help='Task#')
    parser.add_argument('--n_epoch', type=int, default=100, help='Task#')
    parser.add_argument('--path_length', type=int, default=11, help='Task#')
    parser.add_argument('--n_paths', type=int, default=100, help='Task#')
    parser.add_argument('--max_norm', type=float, default=5, help='Task#')
    parser.add_argument('--lr', type=float, default=0.00005, help='Task#')
    parser.add_argument('--discount', type=float, default=0.99, help='Task#')
    parser.add_argument('--std', type=float, default=0.1, help='Task#')
    parser.add_argument('--update_method', type=str, default='rmsprop', help='Task#')
    parser.add_argument('--save_path', type=str, default='104', help='Task#')
    args=parser.parse_args()
    print '*' * 80
    print 'args:', args
    print '*' * 80
    model=Model(**args.__dict__)
    model.train()
