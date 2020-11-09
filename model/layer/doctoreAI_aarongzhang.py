from  keras.layers import  *
from layer.HierarchicalAttention import HierarchicalAttention
from layer.Reverse import Reverse
from layer.RepeateVector import RepeateVector
from layer.sumVector import sumVector
from keras.models import Model
import keras.backend as K
from keras.callbacks import  EarlyStopping,ModelCheckpoint
import pickle
import operator
import heapq
import random
diagnose_dim = 764
proc_dim = 462
max_epochs = 100
batchSize = 100
hidden_size = 256
timseries_dim = 59*7
num_class1 = 10
num_class2 = 50
def recallTop(y_true, y_pred, rank=[10, 20, 30]):
    recall = list()
    for i in range(len(y_pred)):
        thisOne = list()
        codes = y_true[i]
        tops = y_pred[i]
        for rk in rank:
            thisOne.append(len(set(codes).intersection(set(tops[:rk])))*1.0/len(set(codes)))
        recall.append( thisOne )
    return (np.array(recall)).mean(axis=0).tolist()

def load_data(diagnoseFile,procFile,timeseries_file,labelFile,step):
    train_proc_x = pickle.load(open(procFile + step, 'rb'))
    train_timeseries_x = pickle.load(open(timeseries_file + step, 'rb'))
    train_diagnose_x = pickle.load(open(diagnoseFile + step, 'rb'))
    train_y = pickle.load(open(labelFile + step, 'rb'))
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))
    train_sorted_index = len_argsort(train_proc_x)
    train_proc_x = [train_proc_x[i] for i in train_sorted_index]
    train_timeseries_x = [train_timeseries_x[i] for i in train_sorted_index]
    train_diagnose_x = [train_diagnose_x[i] for i in train_sorted_index]
    train_y = [train_y[i] for i in train_sorted_index]
    return train_diagnose_x, train_proc_x,train_timeseries_x,train_y

def padMatrixWithoutTime(diagnoses, procs,timeseriess,labels,maxlen):
    n_samples = len(diagnoses)
    x_diagnose = np.zeros(( n_samples,maxlen, diagnose_dim)).astype(np.float32)
    x_proc = np.zeros(( n_samples,maxlen, proc_dim)).astype(np.float32)
    x_timeseries = np.zeros(( n_samples,maxlen, timseries_dim)).astype(np.float32)
    y = np.zeros(( n_samples, diagnose_dim)).astype(np.float32)
    for idx, (diagnose,proc,timeseries,label) in enumerate(zip(diagnoses,procs,timeseriess,labels)):
        for labelvalue in label:
            y[idx][labelvalue] = 1
        for xvec, subseq in zip(x_diagnose[idx,:,:], diagnose):
            xvec[subseq] = 1.
        for xvec, subseq in zip(x_proc[idx, :, :], proc):
            xvec[subseq] = 1.
        for i,timeseries_each in enumerate(timeseries):
            x_timeseries[idx,i] = timeseries_each
    return x_diagnose,x_proc,x_timeseries,y
def focal_loss(alpha):
    def focal_loss_fixed(y_true, y_pred):
        return alpha*K.categorical_crossentropy(y_true,y_pred)
    return focal_loss_fixed
def retain_withoutTime(maxlen,inputDimSizd):
    input_x = Input(shape=(maxlen, diagnose_dim + proc_dim))
    x = Masking(mask_value=0.)(input_x)
    v = Dense(hidden_size,activation=None)(x)
    # c = AttentionStep(hidden_size)(v)
    reverse = Reverse()(v)
    reverse_h_a = GRU(hidden_size, return_sequences=True)(reverse)
    a_probs = Dense(1, activation='softmax')(reverse_h_a)
    alpha = Reverse()(a_probs)
    alpha = RepeateVector(hidden_size)(alpha)
    reverse_h_b = GRU(hidden_size, return_sequences=True)(reverse)
    reverse_h_b = Dense(hidden_size, activation='tanh')(reverse_h_b)
    beta = Reverse()(reverse_h_b)
    e = multiply([multiply([beta, alpha]),v])
    c = sumVector()(e)
    AG1 = Dense(hidden_size, activation='relu')(c)
    AG2 = Dense(hidden_size, activation='relu')(multiply([AG1, c]))
    PG = Dense(diagnose_dim, activation='sigmoid')(AG2)
    AL1 = Dense(hidden_size, activation='relu')(AG1)
    PL1 = Dense(diagnose_dim, activation='sigmoid')(AL1)
    AL2 = Dense(hidden_size, activation='relu')(AG2)
    PL2 = Dense(diagnose_dim, activation='sigmoid')(AL2)
    shared_dense = Dense(1,activation=None,name='shared_dense')
    alpha1 = shared_dense(PL1)
    alpha2 = shared_dense(PL2)
    alpha3 = shared_dense(PG)
    output,alpha1,alpha2,alpha3= HierarchicalAttention(alpha1,alpha2,alpha3)([PL1,PL2,PG])

    model= Model(input_x,output)
    # model= Model([input_x,input_t],output_x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam')
    return model
def Doctor_Ai_withoutTime(maxlen,inputDimSizd):
    input_x = Input(shape=(maxlen, diagnose_dim + proc_dim ))
    x = Dense(hidden_size,activation=None)(input_x)
    x = GRU(hidden_size,dropout=0.5,return_sequences=True,
            kernel_initializer=initializers.uniform(-0.01, 0.01),bias_initializer='zeros')(x)
    x = Dense(hidden_size)(x)
    x = Dense(hidden_size)(x)
    output_x = Dense(diagnose_dim, activation='softmax',name='x_out')(x)
    model= Model(input_x,output_x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam')
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

def custom_generator(train_diagnose_x,train_proc_x,train_timeseries_x,train_y,maxlen):
    n_batches = int(np.ceil(float(len(train_diagnose_x)) / float(batchSize)))
    while True:
        for index in random.sample(range(n_batches), n_batches):
            batch_diagbose_X = train_diagnose_x[index * batchSize:(index + 1) * batchSize]
            batch_proc_X = train_proc_x[index * batchSize:(index + 1) * batchSize]
            batch_timeseries_X = train_timeseries_x[index * batchSize:(index + 1) * batchSize]
            batchY = train_y[index * batchSize:(index + 1) * batchSize]
        x_diagnose, x_proc,x_timeseries, y= padMatrixWithoutTime(batch_diagbose_X,batch_proc_X,batch_timeseries_X,batchY,maxlen)
        x = np.concatenate((x_diagnose,x_proc),axis=2)
        train_x = [x,x_timeseries]
        x = np.concatenate((x_diagnose,x_proc),axis=2)
        yield x, y

def train():
    train_diagnose_x, train_proc_x, train_timeseries_x, train_y = load_data('../data/NIPS/real.each.step.diagnose.input',
                                                                              '../data/NIPS/real.each.step.proc.input',
                                                                              '../data/NIPS/real.each.step.timeseries.input',
                                                                              '../data/NIPS/real.each.step.diagnose.output',
                                                                              '.train')
    test_diagnose_x, test_proc_x, test_timeseries_x,  test_y  = load_data('../data/NIPS/real.each.step.diagnose.input','../data/NIPS/real.each.step.proc.input','../data/NIPS/real.each.step.timeseries.input',
                                                                         '../data/NIPS/real.each.step.diagnose.output','.test')
    valid_diagnose_x, valid_proc_x,valid_timeseries_x,valid_y  = load_data('../data/NIPS/real.each.step.diagnose.input', '../data/NIPS/real.each.step.proc.input','../data/NIPS/real.each.step.timeseries.input',
                                                                             '../data/NIPS/real.each.step.diagnose.output', '.valid')
    train_lengths = np.array([len(diagnose) for diagnose in train_diagnose_x])
    test_lengths = np.array([len(diagnose) for diagnose in test_diagnose_x])
    valid_lengths = np.array([len(diagnose) for diagnose in valid_diagnose_x])
    maxlen = np.max([np.max(train_lengths),np.max(test_lengths),np.max(valid_lengths)])
    # print('最大的长度为:',maxlen)
    model = retain_withoutTime(maxlen, diagnose_dim)
    checkpoint = ModelCheckpoint('../model_file/retain.hdf5', monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
    early = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    callbacks_list = [checkpoint, early]
    x_diagnose_valid,x_proc_valid,x_timeseries_valid,y_valid = padMatrixWithoutTime(valid_diagnose_x, valid_proc_x,valid_timeseries_x,valid_y ,maxlen)
    # print(y_valid.shape)
    # print(np.array(y_valid))
    x_valid = np.concatenate((x_diagnose_valid, x_proc_valid), axis=2)
    print(model.summary())
    # x_valid = np.concatenate((x_diagnose_valid,x_proc_valid,x_timeseries_valid), axis=2)
    model.fit_generator(custom_generator( train_diagnose_x, train_proc_x, train_timeseries_x, train_y,maxlen),validation_data=(x_valid,y_valid),samples_per_epoch=20,epochs=max_epochs,callbacks=callbacks_list)
    model.load_weights('../model_file/retain.hdf5')
    test_diagnose_x, test_proc_x, test_timeseries_x, test_result = padMatrixWithoutTime( test_diagnose_x, test_proc_x, test_timeseries_x,test_y , maxlen)
    # test_x = np.concatenate((test_diagnose_x, test_proc_x,test_timeseries_x), axis=2)
    test_x = np.concatenate((test_diagnose_x, test_proc_x), axis=2)
    preds_x = model.predict(test_x)
    predVec = []
    for i in range(len(preds_x)):
        predVec.append(list(zip(*heapq.nlargest(30, enumerate(preds_x[i]), key=operator.itemgetter(1))))[0])
    print(recallTop(test_y,predVec))
if __name__ == '__main__':
    # print(np.log(1280))
    train()
    # print(calculate_acc(data))