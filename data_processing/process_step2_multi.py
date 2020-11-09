#################################################################
# Code written by aarongzzhang (aarongzzhang@tencent.com)
# For bug report, please contact author using the email address
#################################################################
import pickle
import numpy as np
def load_data():
    proc_all = pickle.load(open('../data/new_proc_seqs', 'rb'))
    pids = pickle.load(open('../data/pids', 'rb'))
    diagnoses_3 = pickle.load(open('../data/new_diagnose_seqs_3', 'rb'))
    diagnoses_2 = pickle.load(open('../data/new_diagnose_seqs_2', 'rb'))
    diagnoses_1 = pickle.load(open('../data/new_diagnose_seqs_1', 'rb'))
    timeseries_all = pickle.load(open('../data/new_timeseries', 'rb'))
    if len(proc_all)!=len(diagnoses_3):
        print('Exception: len_proc is not equals with len-diagnoses')
        raise SystemExit
    diagnoses_output1 = []
    diagnoses_output2 = []
    diagnoses_output3 = []
    diagnoses_output_docterAi = []
    diagnoses_input = []
    proc_input = []
    timeseries_input = []
    for procs, diagnoses1,diagnoses2,diagnoses3,pid,timeseries in zip(proc_all, diagnoses_1,diagnoses_2,diagnoses_3,pids,timeseries_all):
        proc_input.append(procs[:-1])
        timeseries_input.append(timeseries[:-1])
        diagnoses_input.append(diagnoses3[:-1])
        diagnoses_output1.append(diagnoses1[-1])
        diagnoses_output2.append(diagnoses2[-1])
        diagnoses_output3.append(diagnoses3[-1])
        diagnoses_output_docterAi.append(diagnoses3[1:])
    return proc_input,diagnoses_input,diagnoses_output1,diagnoses_output2,diagnoses_output3,diagnoses_output_docterAi,timeseries_input
def split_data(data,shuffle_ix,outFile,uniquename):
    '''
    :param data:pids or newdiagonse_seqs or newproc_seqs and so on
    :param outFile:
    :return:
    '''
    train_shuffle_ix = shuffle_ix[:int(len(shuffle_ix) * 0.8)]
    test_shuffle_ix = shuffle_ix[int(len(shuffle_ix) * 0.8):int(len(shuffle_ix) * 0.9)]
    valid_shuffle_ix = shuffle_ix[int(len(shuffle_ix) * 0.9):]
    train = [data[x] for x in train_shuffle_ix]
    test = [data[x] for x in test_shuffle_ix]
    valid = [data[x] for x in valid_shuffle_ix]
    pickle.dump(train, open(outFile + uniquename+'.train', 'wb'), protocol=2)
    pickle.dump(test, open(outFile +  uniquename+'.test', 'wb'), protocol=2)
    pickle.dump(valid, open(outFile + uniquename +'.valid', 'wb'), protocol=2)
if __name__ == '__main__':
    proc_input, diagnoses_input, diagnoses_output1,diagnoses_output2,diagnoses_output3, diagnoses_output_docterAi,timeseries_input = load_data()
    shuffle_ix = np.random.permutation(np.arange(len(proc_input)))
    print("\033[1;31;40m Pay Attention!!!!!!!!!!!!! your shuffle_ix is changed,please check your data is the same shuffle_ix!!!!!!!!!!!!!!!!!'\033[0m")
    print("\033[1;31;40m Pay Attention!!!!!!!!!!!!! your shuffle_ix is changed,please check your data is the same shuffle_ix!!!!!!!!!!!!!!!!!'\033[0m")
    print("\033[1;31;40m Pay Attention!!!!!!!!!!!!! your shuffle_ix is changed,please check your data is the same shuffle_ix!!!!!!!!!!!!!!!!!'\033[0m")
    split_data(proc_input,shuffle_ix,'../data/NIPS','/real.each.step.proc.input')
    split_data(timeseries_input,shuffle_ix,'../data/NIPS','/real.each.step.timeseries.input')
    split_data(diagnoses_input,shuffle_ix,'../data/NIPS','/real.each.step.diagnose.input')
    split_data(diagnoses_output1,shuffle_ix,'../data/NIPS','/real.each.step.diagnose1.output')
    split_data(diagnoses_output2,shuffle_ix,'../data/NIPS','/real.each.step.diagnose2.output')
    split_data(diagnoses_output3,shuffle_ix,'../data/NIPS','/real.each.step.diagnose3.output')
    split_data(diagnoses_output_docterAi,shuffle_ix,'../data/NIPS','/real.each.step.diagnoses.output.docterAi')