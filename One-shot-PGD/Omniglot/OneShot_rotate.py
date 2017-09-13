#coding=utf8
import random
import argparse
import time
import pickle
import gzip
import lasagne
import numpy as np
import theano
import theano.tensor as T
from sklearn.preprocessing import LabelBinarizer,label_binarize
import image_all_1200 as image_all
from tqdm import tqdm

def action_to_vector_real(x, n_classes): #x是bs*path_length
    result = np.zeros([x.shape[0], x.shape[1], n_classes])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result[i,j]=label_binarize([int(x[i,j])],range(n_classes))[0]
    return np.int32(result)

def action_to_vector(x, n_classes,p=0): #x是bs*path_length
    # p=0 #p是标签正常的概率
    result = np.zeros([x.shape[0], x.shape[1], n_classes])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if np.random.rand()<p and j!=x.shape[1]-1:
                result[i,j]=label_binarize([int(x[i,j])],range(n_classes))[0]
    return np.int32(result)

def reward_count(total_reward, length, discout=0.99):
    discout_list = np.zeros(length)
    for idx in range(discout_list.shape[0]):
        step = discout ** idx
        discout_list[idx] = step
    result = np.dot(total_reward, discout_list)
    global same_batch
    if same_batch:
        inn=total_reward.shape[0]/same_batch#每个样本重复inn次
        for line in range(same_batch):
            patch=np.mean(result[inn*line:(line+1)*inn])
            result[inn*line:(line+1)*inn]-=patch
    return result

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
        # self.W_h=self.add_param(W_choice,(embedding_size,1), name='Pointer_layer_W_h')
        # self.b_h=self.add_param(W_choice,(1,1), name='Pointer_layer_W_b')
        # self.W_q=self.add_param(W_question,(embedding_size,1), name='Pointer_layer_W_q')
        # self.W_o=self.add_param(W_out,(embedding_size,embedding_size), name='Pointer_layer_W_o')
        # self.ratio=self.add_param(W_choice,(1,1), name='Pointer_layer_W_b')
        self.nonlinearity=nonlinearity
        # zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        # self.set_zero = theano.function([zero_vec_tensor], updates=[(x, T.set_subtensor(x[0, :], zero_vec_tensor)) for x in [self.A,self.C]])

    def get_output_shape_for(self, input_shapes):
        return (self.batch_size,self.max_sentlen)
    def get_output_for(self, inputs, **kwargs):
        #input[0]:(BS,max_senlen,emb_size),input[1]:(BS,1,emb_size),input[2]:(BS,max_sentlen)
        # activation0=(T.dot(inputs[0],self.W_h)).reshape([self.batch_size,self.max_sentlen])+self.b_h.repeat(self.batch_size,0).repeat(self.max_sentlen,1)
        # activation1=T.dot(inputs[1],self.W_q).reshape([self.batch_size]).dimshuffle(0,'x')
        # activation2=T.batched_dot(T.dot(inputs[0],self.W_o),inputs[1].reshape([self.batch_size,self.embedding_size,1])).reshape([self.batch_size,self.max_sentlen])

        #正常的点积的方法
        # activation2=T.batched_dot(inputs[0],inputs[1].reshape([self.batch_size,self.embedding_size,1])).reshape([self.batch_size,self.max_sentlen])

        #正常的点积的方法
        # activation2=T.batched_dot(inputs[0],inputs[1].reshape([self.batch_size,self.embedding_size,1])).reshape([self.batch_size,self.max_sentlen])
        # norm1=T.sqrt(T.sum(T.square(inputs[0]),axis=2))+1e-15
        # norm2=T.sqrt(T.sum(T.square(inputs[1]),axis=2))
        # activation2=activation2/(norm1+norm2)

        # 采用欧式距离的相反数的评价的话：
        activation2=-T.sqrt(T.sum(T.square(T.sub(inputs[0],inputs[1].repeat(self.max_sentlen,1))),axis=2)+1e-15)



        # norm2=T.sqrt(T.sum(T.mul(inputs[0],inputs[0]),axis=2))+1e-15
        # activation2=activation2/norm2
        # activation=(self.nonlinearity(activation0)+self.nonlinearity(activation1)+activation2).reshape([self.batch_size,self.max_sentlen])#.dimshuffle(0,'x',2)#.repeat(self.max_sentlen,axis=1)
        # activation2=(activation2).reshape([self.batch_size,self.max_sentlen])#.dimshuffle(0,'x',2)#.repeat(self.max_sentlen,axis=1)
        # final=T.dot(activation,self.W_o) #(BS,max_sentlen)
        # activation3=T.batched_dot(inputs[0],inputs[1].reshape([self.batch_size,self.embedding_size,1])).reshape([self.batch_size,self.max_sentlen])
        # if inputs[2] is not None:
        #     final=inputs[2]*final-(1-inputs[2])*1000000
        alpha=lasagne.nonlinearities.softmax(activation2) #(BS,max_sentlen)
        return alpha

class ContChoiceLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, W_question, W_choice,W_out,h_dim,n_classes, nonlinearity=lasagne.nonlinearities.tanh,**kwargs):
        super(ContChoiceLayer, self).__init__(incomings, **kwargs) #？？？不知道这个super到底做什么的，会引入input_layers和input_shapes这些属性
        if len(incomings) != 2:
            raise NotImplementedError
        batch_size, max_sentlen ,embedding_size = self.input_shapes[0]
        embedding_size=h_dim
        self.batch_size,self.max_sentlen,self.embedding_size=batch_size,max_sentlen,h_dim
        # self.W_h=self.add_param(W_choice,(embedding_size,1), name='Pointer_layer_W_h')
        # self.b_h=self.add_param(W_choice,(1,1), name='Pointer_layer_W_b')
        # self.W_q=self.add_param(W_question,(embedding_size,1), name='Pointer_layer_W_q')
        # self.W_o=self.add_param(W_out,(embedding_size,embedding_size), name='Pointer_layer_W_o')
        # self.W_h_label=self.add_param(W_choice,(n_classes,1), name='Pointer_layer_W_h_label')
        # self.b_h_label=self.add_param(W_choice,(1,1), name='Pointer_layer_W_b_label')
        # self.W_q_label=self.add_param(W_question,(n_classes,1), name='Pointer_layer_W_q_label')
        self.W_o_label=self.add_param(W_out,(n_classes,n_classes), name='Pointer_layer_W_o')
        self.nonlinearity=nonlinearity
        self.h_dim=h_dim
        self.n_classes=n_classes

    def get_output_shape_for(self, input_shapes):
        return (self.batch_size,self.max_sentlen)
    def get_output_for(self, inputs, **kwargs):
        '''input[0]是memory[bs*path_length,n_classes,h_dim]
           input[1]是hidden[bs*path_length,1,h_dim]'''
        '''内容部分的计算'''
        # activation0=(T.dot(inputs[0][:,:,:self.h_dim],self.W_h)).reshape([self.batch_size,self.max_sentlen])+self.b_h.repeat(self.batch_size,0).repeat(self.max_sentlen,1)
        # activation1=T.dot(inputs[1][:,:,:self.h_dim],self.W_q).reshape([self.batch_size]).dimshuffle(0,'x')
        activation2=T.batched_dot(inputs[0][:,:,:self.h_dim],inputs[1][:,:,:self.h_dim].reshape([self.batch_size,self.embedding_size,1])).reshape([self.batch_size,self.max_sentlen])
        # activation2=T.batched_dot(T.dot(inputs[0][:,:,:self.h_dim],self.W_o),inputs[1][:,:,:self.h_dim].reshape([self.batch_size,self.embedding_size,1])).reshape([self.batch_size,self.max_sentlen])
        norm2=T.sqrt(T.sum(T.mul(inputs[0][:,:,:self.h_dim],inputs[0][:,:,:self.h_dim]),axis=2))+0.0000001
        # activation2=activation2/norm2
        activation=(activation2).reshape([self.batch_size,self.max_sentlen])#.dimshuffle(0,'x',2)#.repeat(self.max_sentlen,axis=1)
        alpha=lasagne.nonlinearities.softmax(activation) #(BS,max_sentlen)

        '''标签部分的计算'''
        # activation0=(T.dot(inputs[0][:,:,self.h_dim:],self.W_h_label)).reshape([self.batch_size,self.max_sentlen])+self.b_h_label.repeat(self.batch_size,0).repeat(self.max_sentlen,1)
        # activation1=T.dot(inputs[1][:,:,self.h_dim:],self.W_q_label).reshape([self.batch_size]).dimshuffle(0,'x')
        activation2=T.batched_dot(T.dot(inputs[0][:,:,self.h_dim:],self.W_o_label),inputs[1][:,:,self.h_dim:].reshape([self.batch_size,self.n_classes,1])).reshape([self.batch_size,self.max_sentlen])
        activation=(activation2).reshape([self.batch_size,self.max_sentlen])#.dimshuffle(0,'x',2)#.repeat(self.max_sentlen,axis=1)
        beta=lasagne.nonlinearities.softmax(activation) #(BS,max_sentlen)

        alpha=lasagne.nonlinearities.softmax(alpha+5*beta)

        return alpha
class Model:
    def __init__(self,x_dimension=(20,20),h_dimension=30,tmp_h_dim=300,n_classes=10,batch_size=32,n_epoch=50,path_length=10,
                 n_paths=1000,max_norm=50,lr=0.05,discount=0.99,update_method='sgd',std=0.5,save_path=str(np.random.randint(100,999)),**kwargs):
        self.h_dim=h_dimension
        self.tmp_h_dim=tmp_h_dim
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
        D1, D2, D3 = lasagne.init.Normal(std=self.std,mean=0), lasagne.init.Normal(std=self.std,mean=0), lasagne.init.Normal(std=self.std,mean=0)
        # D1, D2, D3 = lasagne.init.Uniform(-1,1), lasagne.init.Uniform(-1,1), lasagne.init.Uniform(-1,1)

        l_range_in = lasagne.layers.InputLayer(shape=(self.batch_size,self.path_length,self.x_dim[0],self.x_dim[1]))
        l_range_flatten = lasagne.layers.ReshapeLayer(l_range_in,[self.batch_size*self.path_length,1,self.x_dim[0],self.x_dim[1]])
        # l_range_flatten = lasagne.layers.NonlinearityLayer(l_range_flatten,nonlinearity=lasagne.nonlinearities.tanh)
        l_range_conv1=lasagne.layers.Conv2DLayer(l_range_flatten,num_filters=128,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify)
        l_range_conv1=lasagne.layers.Conv2DLayer(l_range_conv1,num_filters=128,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify)
        l_pool1=lasagne.layers.MaxPool2DLayer(l_range_conv1,pool_size=(2,2))
        l_dropout1=lasagne.layers.DropoutLayer(l_pool1,p=0.2)
        l_range_conv2=lasagne.layers.Conv2DLayer(l_pool1,num_filters=256,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify)
        l_range_conv2=lasagne.layers.Conv2DLayer(l_range_conv2,num_filters=256,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify)
        l_pool2=lasagne.layers.MaxPool2DLayer(l_range_conv2,pool_size=(2,2))
        l_dropout2=lasagne.layers.DropoutLayer(l_pool2,p=0.2)

        # l_range_dense2 = lasagne.layers.DenseLayer(l_pool2,300,W=D1,nonlinearity=lasagne.nonlinearities.tanh) #[bs*path_length,dimension]
        # # l_dropout3=lasagne.layers.DropoutLayer(l_range_dense2,p=0.5)
        # # l_range_dense2 = lasagne.layers.DenseLayer(l_range_flatten,1024,W=D1,nonlinearity=None) #[bs*path_length,dimension]
        # l_range_dense2 = lasagne.layers.DenseLayer(l_range_dense2,self.h_dim,W=D2,nonlinearity=lasagne.nonlinearities.tanh) #[bs*path_length,dimension]
        # l_range_dense2 = lasagne.layers.DenseLayer(l_range_dense2,self.h_dim,W=D2,nonlinearity=lasagne.nonlinearities.tanh) #[bs*path_length,dimension]
        # l_range_dense2_origin=lasagne.layers.ReshapeLayer(l_range_dense2,[self.batch_size,self.path_length,self.h_dim])
        # l_range_dense2_origin1=lasagne.layers.DenseLayer(l_range_dense2,lll,W=D3,b=None,nonlinearity=lasagne.nonlinearities.softmax)
        # l_range_dense2_origin2=lasagne.layers.ReshapeLayer(l_range_dense2_origin1,[self.batch_size,self.path_length,self.h_dim])
        # l_range_label = lasagne.layers.InputLayer(shape=(self.batch_size,self.path_length,self.n_classes))
        #
        tmp_h_dim=self.tmp_h_dim
        l_range_dense2 = lasagne.layers.DenseLayer(l_pool2,tmp_h_dim,W=D1,nonlinearity=lasagne.nonlinearities.rectify) #[bs*path_length,dimension]
        # l_dropout3=lasagne.layers.DropoutLayer(l_range_dense2,p=0.5)
        # l_range_dense2 = lasagne.layers.DenseLayer(l_range_flatten,1024,W=D1,nonlinearity=None) #[bs*path_length,dimension]
        # l_range_dense2 = lasagne.layers.DenseLayer(l_range_dense2,self.h_dim,W=D2,nonlinearity=lasagne.nonlinearities.tanh) #[bs*path_length,dimension]
        # l_range_dense2 = lasagne.layers.DenseLayer(l_range_dense2,self.h_dim,W=D2,nonlinearity=lasagne.nonlinearities.tanh) #[bs*path_length,dimension]
        # l_range_dense2_origin=lasagne.layers.ReshapeLayer(l_range_dense2,[self.batch_size,self.path_length,self.h_dim])
        # TODO:这里的层是不是有点多了？
        l_range_dense2 = lasagne.layers.DenseLayer(l_range_dense2,tmp_h_dim,W=D2,nonlinearity=lasagne.nonlinearities.rectify) #[bs*path_length,dimension]
        # l_range_dense2 = lasagne.layers.DenseLayer(l_range_dense2,tmp_h_dim,W=D2,nonlinearity=lasagne.nonlinearities.rectify) #[bs*path_length,dimension]
        l_range_dense2_origin=lasagne.layers.ReshapeLayer(l_range_dense2,[self.batch_size,self.path_length,tmp_h_dim])

        l_range_dense2_origin1=lasagne.layers.DenseLayer(l_range_dense2,lll,W=D3,nonlinearity=lasagne.nonlinearities.softmax)
        # l_range_dense2_origin2=lasagne.layers.ReshapeLayer(l_range_dense2_origin1,[self.batch_size,self.path_length,self.h_dim])
        l_range_label = lasagne.layers.InputLayer(shape=(self.batch_size,self.path_length,self.n_classes))


        if hid==1:
            # TODO:这里改成一起学的，不用slice的最好
            global slice_label
            if slice_label:
                l_label = lasagne.layers.InputLayer(shape=(self.batch_size,lll))
                xx_label=T.matrix()
                l_range_probas=lasagne.layers.SliceLayer(l_range_dense2_origin,0,axis=1)#原来是[bs,path_legth,h_dimension]
                l_range_probas=lasagne.layers.ReshapeLayer(l_range_probas,[self.batch_size,tmp_h_dim])
                # l_range_dense2_origin=lasagne.layers.ReshapeLayer(l_range_dense2,[self.batch_size*self.path_length,self.h_dim])
                l_range_probas=lasagne.layers.DenseLayer(l_range_probas,lll,W=l_range_dense2_origin1.W,b=l_range_dense2_origin1.b,nonlinearity=lasagne.nonlinearities.softmax)
                ppp=lasagne.layers.helper.get_output(l_range_probas,{l_range_in:x_range,l_label:xx_label})
                hidden_params=lasagne.layers.helper.get_all_params(l_range_probas,trainable=True)
                hidden_cost = T.mean(T.nnet.categorical_crossentropy(ppp, xx_label))
                pred = T.argmax(ppp, axis=1)
                hidden_grads=T.grad(hidden_cost,hidden_params)
                hidden_updates =lasagne.updates.nesterov_momentum(hidden_cost, hidden_params, learning_rate=0.01)
                hidden_updates =lasagne.updates.sgd(hidden_cost, hidden_params, learning_rate=0.005)
                self.hid = theano.function([x_range,xx_label],[hidden_cost,pred,ppp],updates=hidden_updates,on_unused_input='ignore',allow_input_downcast=True)
                self.hid_out= theano.function([x_range],[pred],on_unused_input='ignore',allow_input_downcast=True)
                self.nnn=l_range_probas
            else:
                l_label = lasagne.layers.InputLayer(shape=(self.batch_size,lll))
                xx_label=T.matrix()
                # l_range_probas=lasagne.layers.SliceLayer(l_range_dense2_origin,0,axis=1)#原来是[bs,path_legth,h_dimension]
                l_range_probas=lasagne.layers.ReshapeLayer(l_range_dense2_origin,[self.batch_size*self.path_length,tmp_h_dim])
                l_range_probas=lasagne.layers.DenseLayer(l_range_probas,lll,W=l_range_dense2_origin1.W,b=l_range_dense2_origin1.b,nonlinearity=lasagne.nonlinearities.softmax)
                ppp=lasagne.layers.helper.get_output(l_range_probas,{l_range_in:x_range,l_label:xx_label})
                hidden_params=lasagne.layers.helper.get_all_params(l_range_probas,trainable=True)
                hidden_cost = T.mean(T.nnet.categorical_crossentropy(ppp, xx_label))
                pred = T.argmax(ppp, axis=1)
                # hidden_grads=T.grad(hidden_cost,hidden_params)
                hidden_updates =lasagne.updates.nesterov_momentum(hidden_cost, hidden_params, learning_rate=0.1)
                self.hid = theano.function([x_range,xx_label],[hidden_cost,pred,ppp],updates=hidden_updates,on_unused_input='ignore',allow_input_downcast=True)
                self.hid_out= theano.function([x_range],[pred],on_unused_input='ignore',allow_input_downcast=True)
                self.nnn=l_range_probas
        if if_cont==1:
            l_range_dense2_origin=lasagne.layers.ConcatLayer((l_range_dense2_origin,l_range_label),axis=2)
            self.h_dim+=self.n_classes
        # l_range_hidden=lasagne.layers.ReshapeLayer(l_range_dense2_origin2,[self.batch_size*self.path_length,1,self.h_dim])
        l_range_hidden=lasagne.layers.ReshapeLayer(l_range_dense2_origin,[self.batch_size*self.path_length,1,tmp_h_dim])

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
            # l_range_hidden=lasagne.layers.ReshapeLayer(l_range_dense2_origin,[self.batch_size*self.path_length,1,self.h_dim])
            if if_cont==1:
                l_range_status=ContChoiceLayer((l_range_memory,l_range_hidden),D3,D3,D3,self.h_dim-self.n_classes,self.n_classes,nonlinearity=lasagne.nonlinearities.tanh) #[bs*pl,(n_class+1),dim]
            else:
                l_range_status=ChoiceLayer((l_range_memory,l_range_hidden),D3,D3,D3,nonlinearity=lasagne.nonlinearities.tanh) #[bs*pl,(n_class+1),dim]
            l_range_mu = lasagne.layers.ReshapeLayer(l_range_status,[self.batch_size,self.path_length,self.n_classes])

        '''模型的总体参数和更新策略等'''
        hidden = lasagne.layers.helper.get_output(l_range_dense2_origin, {l_range_in: x_range,l_range_label:x_label})
        probas_range = lasagne.layers.helper.get_output(l_range_mu, {l_range_in: x_range,l_range_memory_in:x_memory,l_range_label:x_label})
        params=lasagne.layers.helper.get_all_params(l_range_mu,trainable=True)
        # params=params[-1:]#相当于只更新最后一个参数，别的不参与更新了
        params=[]#相当于只更新最后一个参数，别的不参与更新了
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

        if 0 and hid:
            # load_params = pickle.load(open('params/params_nnn0.701041666667_110_1_64_2016-11-17 15:38:19'))
            load_params = pickle.load(open('params/params_nnn_0.90125_7_6_2016-12-07 20:17:49'))
            lasagne.layers.set_all_param_values(self.nnn, load_params)
            print 'load succeed!'
            global hid
            hid=0


        self.network=l_range_mu
        self.rotate_label=l_range_dense2_origin1
        # self.ppp=l_range_status.W_o

    def sample_one_path(self,now_state,now_memory_label, prob,this_label,idx_path_length):
        new_memory_state=np.zeros([self.n_classes,self.h_dim])
        new_memory_label=np.zeros([self.n_classes,self.path_length])
        reward=0.0

        now_memory_label_count=np.int32(now_memory_label!=-1).sum(axis=1)#计算当前memory每个slot各有多少个数据存储
        action=np.random.choice(range(self.n_classes),p=prob)
        index_to_insert=int(np.argwhere(now_memory_label[action]==-1)[0])
        #定义reward
        if now_memory_label[action][0]==-1:#证明是新开的一个slot
            reward=-1
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
                if this_label in label_count_dict and len(label_count_dict)==1:#如果这个样本存放的action里面和它都是同label的
                    if idx_path_length==self.path_length-1:
                        reward=10
                        # print '100~!\n'
                    else:
                        reward=10
                else:
                    reward=-3
            if 0:
                if this_label in label_count_dict and len(label_count_dict)==1:#如果这个样本存放的action里面和它都是同label的
                    if idx_path_length==self.path_length-1:
                        reward=1000
                        for xx in now_memory_label:
                            if xx[0]!=-1 and len(set(xx))>2:
                                reward=-3
                                break
                        if reward==10:
                            print '100~!\n'

        #更新状态，加权平均，更新label记录
        # now_state[action]=now_state[-1]+(now_state[action])
        now_state[action]=now_state[-1]+(now_state[action]*now_memory_label_count[action])
        now_state[action]=now_state[action]/(now_memory_label_count[action]+1)
        new_memory_state=now_state[:-1]
        now_memory_label[action][index_to_insert]=this_label
        new_memory_label=now_memory_label


        return action,new_memory_state,reward,new_memory_label

    def test_acc(self,xx,yy):
        if len(xx)>2000:
            xx,yy=xx[:2000],yy[:2000]
        batch_size,path_length,n_classes=self.batch_size,self.path_length,self.n_classes
        x_dim,h_dim=self.x_dim,self.h_dim
        acc,ttt=0.,0.
        acc2,ttt2=0.,0.
        acc3,ttt3=0.,0.
        acc4,ttt4=0.,0.
        acc5,ttt5=0.,0.
        acc_end=0.
        batch_total_number=len(xx)/batch_size
        for idx_batch in tqdm(range(batch_total_number)):#对于每一个batch
            xx_batch = xx[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            yy_batch = yy[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            yy_batch_vector=action_to_vector(yy_batch,self.n_classes,0)
            # yy_batch_vector=action_to_vector_real(yy_batch,self.n_classes)
            yy_batch_vector_real=action_to_vector_real(yy_batch,self.n_classes)

            total_state = np.zeros([batch_size, path_length, n_classes+1,h_dim])
            total_memory_label=np.zeros([batch_size,path_length,n_classes,path_length],dtype=np.int32)-1 #取作-1，标志着还没有存放过样本
            total_action = np.zeros([batch_size, path_length])
            total_probas = np.zeros([batch_size, path_length,n_classes])


            memory_t_repeat=np.zeros((batch_size,path_length,self.n_classes,h_dim))#初始状态的memory

            for t in range(path_length):  # 进行了path_length步
                '''每一步的状态应该是，前面都知道标签信息且完美存放，这个时候不知道标签信息来预测。
                先设定t时候的x，和memory，然后得到probas,然后更新memory'''
                # print memory_t_repeat[0][0],'\n'
                xx_batch_t=xx_batch[:,t].reshape([batch_size,1,x_dim[0],x_dim[1]])
                xx_batch_t_repeat=xx_batch_t.repeat(path_length,axis=1)

                self.x_range_shared.set_value(xx_batch_t_repeat)
                # self.x_range_label.set_value(np.zeros_like(yy_batch_vector))#去除标签信息来预测
                self.x_range_label.set_value(yy_batch_vector[:,t].reshape(batch_size,1,n_classes).repeat(path_length,axis=1))
                self.x_range_memory.set_value(memory_t_repeat)
                probbb_t = self.output_model_range()[0][:,0]
                total_action[:,t]=np.argmax(probbb_t,axis=1)
                total_probas[:,t]=probbb_t
                # print xx_batch_t_repeat[0,0,8:12,8:12],'\n'
                # print 'idx_batch:{},idx_path:{}'.format(idx_batch,t)
                # print memory_t_repeat[0,0],'\n'
                # print probbb_t[0],'\n'

                # state_t=self.output_hidden(xx_batch_t_repeat,yy_batch_vector[:,t].reshape(batch_size,1,n_classes).repeat(path_length,axis=1))[0]#这个state是包含标签信息的
                state_t=self.output_hidden(xx_batch_t_repeat,yy_batch_vector_real[:,t].reshape(batch_size,1,n_classes).repeat(path_length,axis=1))[0]#这个state是包含标签信息的
                # print yy_batch[0]
                # print state_t[0],'\n'
                for i in range(batch_size):#对于batch里的每一个
                    memory_label=total_memory_label[i,t].copy()
                    state=total_state[i,t].copy()
                    action=int(yy_batch[i,t])
                    if t!=path_length-1:
                        insert_idx=int(np.argwhere(memory_label[action]==-1)[0])
                        memory_label[action,insert_idx]=action
                        total_memory_label[i,t+1]=memory_label
                        # print state_t[i]-state[action],'hhh',i,t
                        # state[action]=state[action]+state_t[i]
                        state[action]=state[action]*insert_idx+state_t[i]
                        state[action]/=(insert_idx+1)
                        total_state[i,t+1]=state
                if t!=path_length-1:
                    memory_t=total_state[:,t+1,:n_classes].copy()
                    memory_t=memory_t.reshape(batch_size,1,n_classes,h_dim)
                    memory_t_repeat=memory_t.repeat(path_length,axis=1)

            '''开始比对total_action和yy_batch'''
            for idx,line in enumerate(yy_batch):
                # print line
                # print total_action[idx],'\n'
                # print total_probas[idx],'\n'
                dict={}
                for jdx,jj in enumerate(line):
                    if jj not in dict:#第一次见到
                        dict[jj]=1
                    else:
                        if dict[jj]==1:#第二次见到
                            ttt+=1
                            if total_action[idx,jdx]==yy_batch[idx,jdx]:
                                acc+=1
                        if dict[jj]==2:#第三次见到
                            ttt2+=1
                            if total_action[idx,jdx]==yy_batch[idx,jdx]:
                                acc2+=1
                        if dict[jj]==3:#第四次见到
                            ttt3+=1
                            if total_action[idx,jdx]==yy_batch[idx,jdx]:
                                acc3+=1
                        if dict[jj]==4:#第五次见到
                            ttt4+=1
                            if total_action[idx,jdx]==yy_batch[idx,jdx]:
                                acc4+=1
                        if dict[jj]==5:#第六次见到
                            ttt5+=1
                            if total_action[idx,jdx]==yy_batch[idx,jdx]:
                                acc5+=1
                        dict[jj]+=1

                if total_action[idx,-1]==yy_batch[idx,-1]:
                    acc_end+=1

                # print 'acc:',acc,'ttt:',ttt

        try:
            print 'average ttt is :',float(ttt)/(batch_size*batch_total_number)
            print 'average ttt2 is :',float(ttt2)/(batch_size*batch_total_number),acc2/ttt2
            print 'average ttt3 is :',float(ttt3)/(batch_size*batch_total_number),acc3/ttt3
            print 'average ttt4 is :',float(ttt4)/(batch_size*batch_total_number),acc4/ttt4
            print 'average ttt5 is :',float(ttt5)/(batch_size*batch_total_number),acc5/ttt5
        except ZeroDivisionError:
            pass
        return acc,ttt,acc_end,batch_total_number*batch_size,

    def train(self):
        high_acc,high_acc_end=0,0
        batch_size=self.batch_size
        n_paths=self.n_paths
        path_length=self.path_length
        x_dim=self.x_dim
        h_dim=self.h_dim
        n_classes=self.n_classes
        save_path=self.save_path
        total_test=1000
        if 1:
            pre_finished=1
            # prev_weights_stable=pickle.load(open('params/params_nnn_0.953024193548_1_10_2017-02-10 11:16:30'))
            # prev_weights_stable=pickle.load(open('params/params_nnn_0.95_re'))
            # prev_weights_stable=pickle.load(gzip.open('params/params_rotate_95.pklz'))
            # prev_weights_stable=pickle.load(gzip.open('params/params_rotate_triplet.pklz'))

            # 1200版本的，还未加triplet
            # prev_weights_stable=pickle.load(gzip.open('params/weights_1200_perfect.pklz'))
            # prev_weights_stable=pickle.load(gzip.open('params/weights_1200_perfect_7.pklz'))
            # prev_weights_stable=pickle.load(gzip.open('params/cnn_triplet/weights_triplet_1200_0.1432.pklz'))
            # prev_weights_stable=pickle.load(gzip.open('params/cnn_triplet/weights_triplet_1200_0.1234.pklz'))
            # prev_weights_stable=pickle.load(gzip.open('params/cnn_triplet/weights_1200_valid0795.pklz'))

            #　1200 valid_best的
            # prev_weights_stable=pickle.load(gzip.open('params/cnn_triplet/validbest_0_0.90421875_2017-05-21 01:13:38.pklz'))
            # prev_weights_stable=pickle.load(gzip.open('params/cnn_triplet/validbest_2_0.949375_2017-05-21 03:09:28.pklz'))
            # prev_weights_stable=pickle.load(gzip.open('params/cnn_triplet/weights_0.0397784282249_cnn_rotate_1_triplet_2017-05-22 04:13:08.pklz'))
            # prev_weights_stable=pickle.load(gzip.open('params/cnn_triplet/weights_0.0358228755117_cnn_rotate_1_triplet_2017-05-22 17:47:33.pklz'))
            # prev_weights_stable=pickle.load(gzip.open('params/cnn_triplet/weights_0.0340624707658_cnn_rotate_1_triplet_2017-05-22 22:19:27.pklz'))
            # prev_weights_stable=pickle.load(gzip.open('params/cnn_triplet/weights_0.0329940586727_cnn_rotate_1_triplet_2017-05-23 07:22:44.pklz'))
            # prev_weights_stable=pickle.load(gzip.open('params/cnn_triplet/weights_0.031799513899_cnn_rotate_1_triplet_2017-05-23 22:17:25.pklz'))
            # prev_weights_stable=pickle.load(gzip.open('params/cnn_triplet/weights_0.0297732540409_cnn_rotate_1_triplet_2017-05-25 14:12:13.pklz'))
            prev_weights_stable=pickle.load(gzip.open('params/cnn_triplet/weights_0.258011473472_last1_cnn_rotate_aplha4_triplet_2017-07-21 10:25:33.pklz'))
        else:
            pre_finished=0
        # xx, yy = get_dataset(x_dim,path_length,n_classes) #xx是[sample,path_length,dimension]，yy是[sample.path_length]
        x_train,y_train,x_test,y_test=image_all.x_train,image_all.y_train,image_all.x_test,image_all.y_test
        yy_train,yy_test=image_all.y_train_shuffle,image_all.y_test_shuffle
        xx,yy=x_train,yy_train
        x_test,y_test,yy_test=x_test[:total_test],y_test[:total_test],yy_test[:total_test]
        if 0:
            load_params=pickle.load(open('params/params_119_19_99_525.694008567_2016-11-21 06:20:43'))
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
                print 'New path.\n'
                # if self.lr<0.01:
                #     self.lr*=1.05
                #     print "now lr is:",self.lr
                tmp_cost, tmp_result, tmp_reward = 0, 0, 0
                for idx_batch in range(batch_total_number):#对于每一个batch
                    batch_start_time=time.time()
                    if pre_finished:
                        lasagne.layers.set_all_param_values(self.nnn,prev_weights_stable)
                        # ccc=[ooo[0] for ooo in lasagne.layers.get_all_param_values(self.network)]
                        # del ooo
                        # for ooo in ccc:
                        #     print ooo,'\n'
                        print 'load succeed!\n'
                        pre_finished=0
                        hid=0
                # 初始化两个循环的参数，state和概率
                    xx_batch = xx[idx_batch * batch_size:(idx_batch + 1) * batch_size]
                    yy_batch = yy[idx_batch * batch_size:(idx_batch + 1) * batch_size]
                    # yy_batch_vector=action_to_vector_real(yy_batch,self.n_classes)
                    yy_batch_vector=action_to_vector(yy_batch,self.n_classes,1)

                    output_label = open('acc_label/label').read().strip()
                    output_fre = int(open('acc_label/fre').read().strip())
                    print 'label:',output_label,'fre:',output_fre
                    try:
                        fre=repeat_time%output_fre
                    except:
                        fre=0
                    if 0 and output_label=='1' and fre==0:
                        # print 'Begin to test.'
                        # if pre_finished:
                        #     lasagne.layers.set_all_param_values(self.nnn, prev_weights_stable)
                        acc,ttt,acc_end,ttt_end=self.test_acc(x_test,yy_test)
                        if (float(acc)/ttt)>high_acc:
                            high_acc=float(acc)/ttt
                        if (float(acc_end)/ttt_end)>high_acc_end:
                            high_acc_end=float(acc_end)/ttt_end
                        print 'Test one-shot acc:{}'.format(float(acc)/ttt),'\t',acc,ttt,'highest acc:',high_acc
                        print 'Test one-shot acc_end:{}'.format(float(acc_end)/ttt_end),'\t',acc_end,ttt_end,'highest acc end:',high_acc_end
                    print '\n\n\n'



                    global hid,slice_label
                    y_batch = np.int32(y_train)[idx_batch * batch_size:(idx_batch + 1) * batch_size]
                    if hid==1:
                        # self.x_range_shared.set_value(xx_batch)
                        # self.x_range_label.set_value(action_to_vector(y_batch,len(image_all.)))
                        if slice_label:
                            ccc,pred,ppp=self.hid(xx_batch,action_to_vector_real(y_batch,lll)[:,0])
                            print 'The hidden classification cost is :',ccc
                            errors=np.count_nonzero(np.int32(pred==y_batch[:,0]))
                            acc=float(errors)/len(y_batch)
                            print pred[0:100]
                            print y_batch[:,0][0:100]
                            print 'right rate:',acc
                        else:
                            ccc,pred,ppp=self.hid(xx_batch,action_to_vector_real(y_batch,lll).reshape([-1,lll]))
                            print 'The hidden classification cost is :',ccc
                            errors=np.count_nonzero(np.int32(pred==y_batch.reshape(-1)))
                            acc=float(errors)/len(pred)
                            print pred[0:30]
                            print y_batch.reshape(-1)[0:30]
                            print 'right rate:',acc
                        continue

                    global same_batch
                    if same_batch:
                        inn=batch_size/same_batch#一个batch里面有same_batch个样本，每个重复inn次
                        xxx=xx_batch.copy()
                        yyy=yy_batch.copy()
                        yyy_vector=yy_batch_vector.copy()
                        for line in range(same_batch):
                            rr=np.random.randint(0,xxx.shape[0])
                            xx_batch[(line)*inn:(line+1)*inn]=xxx[rr]
                            yy_batch[(line)*inn:(line+1)*inn]=yyy[rr]
                            yy_batch_vector[(line)*inn:(line+1)*inn]=yyy_vector[rr]

                    xx_batch_0=xx_batch[:,0,:].reshape([xx_batch.shape[0],1,xx_batch.shape[-2],xx_batch.shape[-1]])
                    xx_batch_0_repeat=xx_batch_0.repeat(path_length,axis=1)
                    memory_0=np.zeros((batch_size,self.n_classes,h_dim))
                    memory_0_repeat=np.zeros((batch_size,path_length,self.n_classes,h_dim))

                    # 但是第一步的初始化状态都是一样的
                    self.x_range_shared.set_value(xx_batch_0_repeat)
                    self.x_range_label.set_value(yy_batch_vector)
                    self.x_range_memory.set_value(memory_0_repeat)
                    probbb = self.output_model_range()[0]
                    probs_sample_0 = probbb[:, 0]

                    total_state = np.zeros([batch_size, path_length, n_classes+1,h_dim])#整个state是memory和当前x的hidden的结合
                    total_memory_label=np.zeros([batch_size,path_length,n_classes,path_length],dtype=np.int32)-1 #取作-1，标志着还没有存放过样本
                    total_reward = np.zeros([batch_size, path_length])
                    total_action = np.zeros([batch_size, path_length])
                    total_probs = np.zeros([batch_size, path_length, n_classes])

                    total_state[:, 0, -1] =self.output_hidden(xx_batch_0_repeat,yy_batch_vector)[0]#这里是把当前（第一个）x的hidden计算出来，然后填给state
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
                    self.x_range_action.set_value(action_to_vector_real(total_action, n_classes))
                    self.x_range_reward.set_value(reward_count(total_reward, length=path_length, discout=self.discount))
                    aver_reward = np.mean(np.sum(np.float32(total_reward), axis=1))
                    espect_reward = np.mean(np.float32(reward_count(total_reward,path_length,discout=self.discount)), axis=0)
                    _, cost = self.output_model_range_updates()[:-1]
                    tmp_cost += cost
                    tmp_result += aver_reward
                    tmp_reward += espect_reward
                    print 'iter:',epoch,repeat_time,idx_batch,'\tcost:{},average_reward:{},time:{}'.format(cost,aver_reward,time.time()-batch_start_time),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print total_state[0][-1][:,-20:]
                    print yy_batch[0]
                    print total_action[0]
                    print total_memory_label[0][-1]
                    print total_reward[0]
                    # print '\nparams:'
                    # for para in lasagne.layers.get_all_param_values(self.network):
                    #     if len(para.shape)==1:
                    #         print para[0]
                    #     else:
                    #         print para[0][0]
                    # print self.ppp.eval()
                    print '\n',_[0]
                    # print '\n'
                    # print total_probs[0]

                    output_label = open('acc_label/label').read().strip()
                    output_fre = int(open('acc_label/fre').read().strip())
                    print 'label:',output_label,'fre:',output_fre
                    try:
                        fre=repeat_time%output_fre
                    except:
                        fre=0
                    # if 1 or output_label=='1' and fre==0:
                    if output_label=='1' and fre==0 and idx_batch%5==0:
                    # if output_label=='1' and fre==0 and idx_batch>50 and idx_batch%5==0:
                        # print 'Begin to test.'
                        # if pre_finished:
                        #     lasagne.layers.set_all_param_values(self.nnn, prev_weights_stable)
                        acc,ttt,acc_end,ttt_end=self.test_acc(x_test,yy_test)
                        if (float(acc)/ttt)>high_acc:
                            high_acc=float(acc)/ttt
                        if (float(acc_end)/ttt_end)>high_acc_end:
                            high_acc_end=float(acc_end)/ttt_end
                        print 'Test one-shot acc:{}'.format(float(acc)/ttt),'\t',acc,ttt,'highest acc:',high_acc
                        print 'Test one-shot acc_end:{}'.format(float(acc_end)/ttt_end),'\t',acc_end,ttt_end,'highest acc end:',high_acc_end
                    print '\n\n\n'

                global hid
                if hid:
                    acc=0
                    for idx_batch in range(batch_total_number):#对于每一个batch
                    # 初始化两个循环的参数，state和概率
                        xxx = xx[idx_batch * batch_size:(idx_batch + 1) * batch_size]
                        yyy = np.int32(y_train)[idx_batch * batch_size:(idx_batch + 1) * batch_size]
                        pred =self.hid_out(xxx)
                        if slice_label:
                            acc += np.count_nonzero(np.int32(pred ==yyy[:,0]))
                        else:
                            acc += np.count_nonzero(np.int32(pred ==yyy.reshape(-1)))
                    if slice_label:
                        acc=float(acc)/(batch_size*batch_total_number)
                    else:
                        acc=float(acc)/(batch_size*batch_total_number*path_length)

                    print 'iter:', epoch,repeat_time, '|Acc:',acc,'\n\n'
                    if acc>0.5 and repeat_time%5==0:
                        acc_oneshot,ttt,acc_oneshot_end,ttt_end=self.test_acc(x_test,yy_test)
                        if (float(acc_oneshot)/ttt)>high_acc:
                            high_acc=float(acc_oneshot)/ttt
                        if (float(acc_oneshot_end)/ttt_end)>high_acc_end:
                            high_acc_end=float(acc_oneshot_end)/ttt_end
                        print 'Hid Test one-shot acc:{}'.format(float(acc_oneshot)/ttt),'\t',acc_oneshot,ttt,'highest acc_oneshot:',high_acc
                        print 'Hid Test one-shot acc_end:{}'.format(float(acc_oneshot_end)/ttt_end),'\t',acc_oneshot_end,ttt_end,'highest acc_oneshot end:',high_acc_end
                        print '\n\n\n'
                    if acc>0.95:
                        hid=0
                        pre_finished=1
                        prev_weights_stable = lasagne.layers.helper.get_all_param_values(self.nnn)
                        pickle.dump(prev_weights_stable,open('params/params_nnn_{}_{}_{}_{}'.format(acc,epoch,repeat_time,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),'wb'))

            try:
                print 'cost:{},average_reward:{},espect_reward:{},save_folder:{}'.format(tmp_cost /batch_total_number, tmp_result /batch_total_number, tmp_reward/batch_total_number, save_path)
                print 'epoch:{},time:{}'.format(epoch, time.time() - begin_time)
                prev_weights = lasagne.layers.helper.get_all_param_values(self.network)
                pickle.dump(prev_weights,open('params/params_{}_{}_{}_{}_{}'.format(save_path,epoch,repeat_time,tmp_reward/batch_total_number,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),'wb'))
            except:
                pass

test_mode=0
if_cont=0
global hid,slice_label
hid=1
slice_label=1
# lll=964*4
lll=1200*4
global same_batch
# same_batch=32
same_batch=0
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, default=1, help='Task#')
    if test_mode:
        parser.add_argument('--x_dimension', type=int, default=10, help='Dimension#')
    else:
        parser.add_argument('--x_dimension', type=tuple, default=(20,20), help='Dimension#')
        # parser.add_argument('--x_dimension', type=tuple, default=(30,30), help='Dimension#')
    parser.add_argument('--h_dimension', type=int, default=300, help='Dimension#')
    parser.add_argument('--tmp_h_dim', type=int, default=300, help='tmp_h_Dimension#')
    parser.add_argument('--n_classes', type=int, default=10, help='Task#')
    parser.add_argument('--batch_size', type=int, default=32, help='Task#')
    parser.add_argument('--n_epoch', type=int, default=100, help='Task#')
    parser.add_argument('--path_length', type=int, default=11, help='Task#')
    parser.add_argument('--n_paths', type=int, default=30, help='Task#')
    parser.add_argument('--max_norm', type=float, default=50, help='Task#')
    parser.add_argument('--lr', type=float, default=0.0000005, help='Task#')
    parser.add_argument('--discount', type=float, default=0.999, help='Task#')
    parser.add_argument('--std', type=float, default=0.1, help='Task#')
    parser.add_argument('--update_method', type=str, default='rmsprop', help='Task#')
    parser.add_argument('--save_path', type=str, default='1200allup', help='Task#')
    args=parser.parse_args()
    print '*' * 80
    print 'args:', args
    print '*' * 80
    model=Model(**args.__dict__)
    model.train()
