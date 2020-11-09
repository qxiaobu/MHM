# -*- coding: utf-8 -*-

# Author: Zhen Zhang <13161411563@163.com>


from  keras.layers import  *
from layer.crossNetLayer import crossNetLayer
from layer.HierarchicalAttention import HierarchicalAttention
from layer.lastGRU import lastGRU
from layer.sumVector import sumVector
from keras.models import Model
from keras import optimizers
from keras.layers import BatchNormalization
import keras.backend as K
from keras.callbacks import  EarlyStopping,ModelCheckpoint
import pickle
import operator
import heapq
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import configparser
config = configparser.ConfigParser()
config.read('../data/config.ini')
os.environ["CUDA_VISIBLE_DEVICES"]="0"

diagnose_dim = int(config['len']['diagnose3_len'])
proc_dim = int(config['len']['proc_len'])
num_class1 = int(config['len']['diagnose1_len'])
num_class2= int(config['len']['diagnose2_len'])
max_epochs = 100
batchSize = 100
hidden_size = 582
timseries_dim = 59*3
#评价指标，包括准确率和召回率
def EvaluationTop(y_true, y_pred, rank=[10, 20, 30]):
    recall = list()
    precision = list()
    for i in range(len(y_pred)):
        thisOne = list()
        thisTwo = list()
        codes = y_true[i]
        tops = y_pred[i]
        print(i,len(set(codes)))
        print( len(set(codes).intersection(set(tops[:10])))*1.0/len(set(codes)))

        for rk in rank:
            thisOne.append(len(set(codes).intersection(set(tops[:rk])))*1.0/len(set(codes)))
            thisTwo.append(len(set(codes).intersection(set(tops[:rk])))*1.0/min(rk,len(set(codes))))
        recall.append( thisOne )
        precision.append( thisTwo )
    return (np.array(recall)).mean(axis=0).tolist(),(np.array(precision)).mean(axis=0).tolist()
#载入数据，按照数据长度排序。
def load_data(diagnoseFile,procFile,timeseries_file,labelFile1,labelFile2,labelFile3,step):
    train_proc_x = pickle.load(open(procFile + step, 'rb'))
    train_timeseries_x = pickle.load(open(timeseries_file + step, 'rb'))
    train_diagnose_x = pickle.load(open(diagnoseFile + step, 'rb'))
    train_y1 = pickle.load(open(labelFile1 + step, 'rb'))
    train_y2 = pickle.load(open(labelFile2 + step, 'rb'))
    train_y3 = pickle.load(open(labelFile3 + step, 'rb'))
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))
    train_sorted_index = len_argsort(train_proc_x)
    train_proc_x = [train_proc_x[i] for i in train_sorted_index]
    train_timeseries_x = [train_timeseries_x[i] for i in train_sorted_index]
    train_diagnose_x = [train_diagnose_x[i] for i in train_sorted_index]
    train_y1 = [train_y1[i] for i in train_sorted_index]
    train_y2 = [train_y2[i] for i in train_sorted_index]
    train_y3 = [train_y3[i] for i in train_sorted_index]
    return train_diagnose_x, train_proc_x,train_timeseries_x,train_y1,train_y2,train_y3
#将数据转化为mutihot格式
def padMatrixWithoutTime(diagnoses, procs,timeseriess,labels1,labels2,labels3,maxlen):
    n_samples = len(diagnoses)
    x_diagnose = np.zeros(( n_samples,maxlen, diagnose_dim)).astype(np.float32)
    x_proc = np.zeros(( n_samples,maxlen, proc_dim)).astype(np.float32)
    x_timeseries1 = np.zeros(( n_samples,maxlen,59)).astype(np.float32)
    x_timeseries2 = np.zeros(( n_samples,maxlen, 59)).astype(np.float32)
    x_timeseries3 = np.zeros(( n_samples,maxlen, 59)).astype(np.float32)
    y1 = np.zeros((n_samples,num_class1)).astype(np.float32)
    y2 = np.zeros((n_samples,num_class2)).astype(np.float32)
    y3 = np.zeros(( n_samples, diagnose_dim)).astype(np.float32)
    for idx, (diagnose,proc,timeseries,label1,label2,label3) in enumerate(zip(diagnoses,procs,timeseriess,labels1,labels2,labels3)):
        for labelvalue in label1:
            y1[idx][labelvalue] = 1
        for labelvalue in label2:
            y2[idx][labelvalue] = 1
        for labelvalue in label3:
            y3[idx][labelvalue] = 1
        for xvec, subseq in zip(x_diagnose[idx,:,:], diagnose):
            xvec[subseq] = 1.
        for xvec, subseq in zip(x_proc[idx, :, :], proc):
            xvec[subseq] = 1.
        for i,timeseries_each in enumerate(timeseries):
            x_timeseries1[idx,i] = timeseries_each[:59]
            x_timeseries2[idx,i] = timeseries_each[59:59*2]
            x_timeseries3[idx,i] = timeseries_each[59*2:]
    return x_diagnose,x_proc,x_timeseries1,x_timeseries2,x_timeseries3,y1,y2,y3
def entropy_loss(num_class):
    def focal_loss_fixed(y_true, y_pred):
        return K.categorical_crossentropy(y_true,y_pred)/num_class
    return focal_loss_fixed
#网络结构模型
def myModel(maxlen):
    input_1 = Input(shape=(maxlen, diagnose_dim))  # diagnose潉¹彾A
    input_2 = Input(shape=(maxlen, proc_dim))  # procdure潉¹彾A
    input_3 = Input(shape=(maxlen, 59))  # 彗¶幾O潉¹彾A1
    input_4 = Input(shape=(maxlen, 59))  # 彗¶幾O潉¹彾A2
    input_5 = Input(shape=(maxlen, 59))  # 彗¶幾O潉¹彾A3

    t1 = Masking(mask_value=0.)(input_3)
    t2 = Masking(mask_value=0.)(input_4)
    t3 = Masking(mask_value=0.)(input_5)
    t1 = Dense(64)(t1)
    t2 = Dense(64)(t2)
    t3 = Dense(64)(t3)

    t = concatenate([t1, t2, t3])
    cross = crossNetLayer()(t)

    dense = Dense(64 * 3)(t)
    con1 = concatenate([cross, dense])

    con1 = BatchNormalization()(con1)
    time = Dense(hidden_size, kernel_regularizer=regularizers.l1(0.2))(con1)

    medcine = concatenate([input_1, input_2])
    medcine = Masking(mask_value=0.)(medcine)
    medcine = Dense(hidden_size)(medcine)

    con = concatenate([medcine, time])
    con = Dense(hidden_size, kernel_regularizer=regularizers.l2(0.05))(con)


    h = Bidirectional(GRU(int(diagnose_dim / 2), return_sequences=True))(con)
    alpha = Dense(1, activation='softmax')(h)
    c = multiply([alpha, h])
    c = sumVector()(c)
    h = lastGRU()(h)
    con = concatenate([c, h])
    ht = Dense(hidden_size, activation='tanh')(con)

    AG1 = Dense(hidden_size, activation='sigmoid')(ht)
    AG2 = Dense(hidden_size, activation='sigmoid')(multiply([AG1, ht]))
    PG = Dense(diagnose_dim, activation='sigmoid')(multiply([AG2, ht]))

    AL1 = Dense(hidden_size, activation='sigmoid')(AG1)
    PL1 = Dense(diagnose_dim, activation='softmax')(AL1)

    AL2 = Dense(hidden_size, activation='sigmoid')(AG2)
    PL2 = Dense(diagnose_dim, activation='softmax')(AL2)

    shared_dense = Dense(1, activation='sigmoid', name='shared_dense')
    alpha1 = shared_dense(PL1)
    alpha2 = shared_dense(PL2)
    alpha3 = shared_dense(PG)
    output, alpha1, alpha2, alpha3 = HierarchicalAttention(alpha1, alpha2, alpha3)([PL1, PL2, PG])
    output1 = Dense(num_class1, activation='softmax')(AL1)
    output2 = Dense(num_class2, activation='softmax')(AL2)
    output3 = Dense(diagnose_dim, activation='softmax')(PG)

    model = Model([input_1, input_2, input_3, input_4, input_5], [output, output1, output2, output3])
    adam = optimizers.adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', loss_weights=[1,0.05,0.05, 0.05],
                  optimizer=adam)
    return model


def get_topk(y_pred, y_true, k):
    r_topk = 0
    count = 0
    for pred,true in zip(y_pred,y_true):
        if (np.sum(true) != 0):
            count +=1
            value = np.zeros(len(pred))
            index = heapq.nlargest(k, range(len(pred)), pred.__getitem__)
            value[index] = 1
            right_sum = np.sum(value * true)
            r_topk +=  right_sum / np.sum(true)
    return r_topk/count

def custom_generator(train_diagnose_x,train_proc_x,train_timeseries_x,train_y1,train_y2,train_y3,maxlen):
    n_batches = int(np.ceil(float(len(train_diagnose_x)) / float(batchSize)))
    while True:
        for index in random.sample(range(n_batches), n_batches):
            batch_diagbose_X = train_diagnose_x[index * batchSize:(index + 1) * batchSize]
            batch_proc_X = train_proc_x[index * batchSize:(index + 1) * batchSize]
            batch_timeseries_X = train_timeseries_x[index * batchSize:(index + 1) * batchSize]
            batchY1 = train_y1[index * batchSize:(index + 1) * batchSize]
            batchY2 = train_y2[index * batchSize:(index + 1) * batchSize]
            batchY3 = train_y3[index * batchSize:(index + 1) * batchSize]
        x_diagnose, x_proc,x_timeseries1,x_timeseries2,x_timeseries3,y1,y2, y3= padMatrixWithoutTime(batch_diagbose_X,batch_proc_X,batch_timeseries_X,batchY1,batchY2,batchY3,maxlen)
        yield [x_diagnose,x_proc,x_timeseries1,x_timeseries2,x_timeseries3], [y3,y1,y2,y3]

def train():
    train_diagnose_x, train_proc_x, train_timeseries_x, train_y1,train_y2,train_y3 = load_data('../data/NIPS/real.each.step.diagnose.input',
                                                                              '../data/NIPS/real.each.step.proc.input',
                                                                              '../data/NIPS/real.each.step.timeseries.input',
                                                                              '../data/NIPS/real.each.step.diagnose1.output',
                                                                            '../data/NIPS/real.each.step.diagnose2.output',
                                                                            '../data/NIPS/real.each.step.diagnose3.output',
                                                                              '.train')
    test_diagnose_x, test_proc_x, test_timeseries_x, test_y1,test_y2,test_y3  = load_data('../data/NIPS/real.each.step.diagnose.input','../data/NIPS/real.each.step.proc.input','../data/NIPS/real.each.step.timeseries.input',
                                                                         '../data/NIPS/real.each.step.diagnose1.output',
                                                                          '../data/NIPS/real.each.step.diagnose2.output',
                                                                          '../data/NIPS/real.each.step.diagnose3.output',
                                                                          '.test')
    valid_diagnose_x, valid_proc_x,valid_timeseries_x,valid_y1,valid_y2,valid_y3  = load_data('../data/NIPS/real.each.step.diagnose.input', '../data/NIPS/real.each.step.proc.input','../data/NIPS/real.each.step.timeseries.input',
                                                                             '../data/NIPS/real.each.step.diagnose1.output',
                                                                           '../data/NIPS/real.each.step.diagnose2.output',
                                                                           '../data/NIPS/real.each.step.diagnose3.output','.valid')
    train_lengths = np.array([len(diagnose) for diagnose in train_diagnose_x])
    test_lengths = np.array([len(diagnose) for diagnose in test_diagnose_x])
    valid_lengths = np.array([len(diagnose) for diagnose in valid_diagnose_x])
    maxlen = np.max([np.max(train_lengths),np.max(test_lengths),np.max(valid_lengths)])
    # print('最大的长度为:',maxlen)
    model = myModel(maxlen)
    print(model.summary())
    checkpoint = ModelCheckpoint('../model_file/withHierarchal.hdf5', monitor='val_hierarchical_attention_1_loss', verbose=1,
                                 save_best_only=True, mode='min')
    early = EarlyStopping(monitor='val_hierarchical_attention_1_loss', mode='min', patience=5)
    callbacks_list = [checkpoint, early]
    x_diagnose_valid,x_proc_valid,x_timeseries_valid1,x_timeseries_valid2,x_timeseries_valid3,y_valid1,y_valid2,y_valid3 = padMatrixWithoutTime(valid_diagnose_x, valid_proc_x,valid_timeseries_x,valid_y1,valid_y2,valid_y3 ,maxlen)
    # print(y_valid.shape)
    # print(np.array(y_valid))
    x_valid = np.concatenate((x_diagnose_valid, x_proc_valid), axis=2)
    # x_valid = np.concatenate((x_diagnose_valid,x_proc_valid,x_timeseries_valid), axis=2)
    # model.fit_generator(custom_generator( train_diagnose_x, train_proc_x, train_timeseries_x, train_y1,train_y2,train_y3,maxlen),validation_data=([x_diagnose_valid,x_proc_valid,x_timeseries_valid1,x_timeseries_valid2,x_timeseries_valid3],[y_valid3,y_valid1,y_valid2,y_valid3]),samples_per_epoch=20,epochs=max_epochs,callbacks=callbacks_list)
    model.load_weights('../model_file/withHierarchal.hdf5')
    test_diagnose_x, test_proc_x, test_timeseries_x1,test_timeseries_x2,test_timeseries_x3, _,_,_ = padMatrixWithoutTime( test_diagnose_x, test_proc_x, test_timeseries_x,test_y1 ,test_y2 ,test_y3 , maxlen)
    # test_x = np.concatenate((test_diagnose_x, test_proc_x,test_timeseries_x), axis=2)
    test_x = np.concatenate((test_diagnose_x, test_proc_x), axis=2)
    preds_x = model.predict([test_diagnose_x, test_proc_x, test_timeseries_x1,test_timeseries_x2,test_timeseries_x3])
    predVec = []
    for i in range(len(preds_x[0])):
        predVec.append(list(zip(*heapq.nlargest(30, enumerate(preds_x[0][i]), key=operator.itemgetter(1))))[0])
    recall,precision =  EvaluationTop(test_y3, predVec)
    print(recall)
    print(precision)

    #
    # predVec = []
    # for i in range(len(preds_x[1])):
    #     predVec.append(list(zip(*heapq.nlargest(30, enumerate(preds_x[1][i]), key=operator.itemgetter(1))))[0])
    # recall,precision =  EvaluationTop(test_y1, predVec)
    # print(recall)
    # print(precision)
    #
    # predVec = []
    # for i in range(len(preds_x[2])):
    #     predVec.append(list(zip(*heapq.nlargest(30, enumerate(preds_x[2][i]), key=operator.itemgetter(1))))[0])
    # recall,precision =  EvaluationTop(test_y2, predVec)
    # print(recall)
    # print(precision)
    #
    # predVec = []
    # for i in range(len(preds_x[3])):
    #     predVec.append(list(zip(*heapq.nlargest(30, enumerate(preds_x[3][i]), key=operator.itemgetter(1))))[0])
    # recall,precision =  EvaluationTop(test_y3, predVec)
    # print(recall)
    # print(precision)

if __name__ == '__main__':
    # print(np.log(1280))
    train()