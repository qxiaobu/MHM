# This script processes MIMIC-III dataset and builds longitudinal diagnosis records for patients with at least two visits.
# The output data are cPickled, and suitable for step2
# Written by zhangzhen(aarongzhang@tencent.com)
# Usage: Put this script to the foler where MIMIC-III CSV files are located. Then execute the below command.
# python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv <output file>

# Output files
# <output file>.pids: List of unique Patient IDs. Used for intermediate processing
# <output file>.dates: List of List of Python datetime objects. The outer List is for each patient. The inner List is for each visit made by each patient
# <output file>.seqs: List of List of List of integer diagnosis codes. The outer List is for each patient. The middle List contains visits made by each patient. The inner List contains the integer diagnosis codes that occurred in each visit
# <output file>.types: Python dictionary that maps string diagnosis codes to integer diagnosis codes.
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import chain
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
def calcuate_len(inputfile):
    df = pd.read_csv(inputfile,sep='\t')
    df['len'] = df['NBPm'].map(lambda x:len(x.split(',')))

    plt.figure(figsize=(15,10))
    plt.hist(df['len'], bins=50, range=[0, 25],  label='word_len')
    plt.legend()
    plt.xlabel('timeseries length', fontsize=15)
def featureConnecte(SpO2File,NBPmFile ):
# def featureConnecte(hrvFile ,RRFile,SpO2File,NBPmFile ,NBPsFile,NBPdFile,ABPsFile):
#     df_hrv = pd.read_csv(hrvFile,sep='\t')
#     df_rr = pd.read_csv(RRFile,sep='\t')[['HADM_ID','RR_feature']]
    df_spo2 = pd.read_csv(SpO2File,sep='\t')[['HADM_ID','SpO2']]
    df_nbpm = pd.read_csv(NBPmFile,sep='\t')[['HADM_ID','NBPm']]
    # df_nbps = pd.read_csv(NBPsFile,sep='\t')[['HADM_ID','NBPs']]
    # df_nbpd = pd.read_csv(NBPdFile,sep='\t')[['HADM_ID','NBPd']]
    # df_abps = pd.read_csv(ABPsFile,sep='\t')[['HADM_ID','ABPs']]
    # df_list = [df_spo2,df_rr,df_nbpd,df_nbpm,df_nbps,df_abps]
    df_list = [df_spo2,df_nbpm]
    for df in df_list:
        df_hrv = df_hrv.merge(df, on='HADM_ID')
    df_hrv.to_csv('../data/time_series/feature_merge_noscale.csv',sep='\t',index=False)
def normalizeData(inputFile,featurename):
    df = pd.read_csv(inputFile,sep='\t')
    scale_transfer = StandardScaler()
    feature = np.array(list( x.replace('inf','0').split(',') for x  in df[featurename].values))
    scale_transfer_fit = scale_transfer.fit(feature)
    #
    feature = scale_transfer_fit.transform(feature)
    feature_str = []
    for x in feature:
        res = ''
        for y in x:
            res += str(y)+','
        feature_str.append(res.strip(','))

    df[featurename] = feature_str
    df.to_csv('../data/time_series/feature_%s_scale_end.csv'%featurename,index=False,sep='\t')
    # scale_transfer_fit = scale_transfer.fit(feature)

def convert_to_3digit_icd9(dxStr):
    # if dxStr.startswith('E'):
    # 	if len(dxStr) > 3: return dxStr[:3]
    # 	else: return dxStr
    # else:
    if len(dxStr) > 3:
        return dxStr[:3]
    else:
        return dxStr
def build_admission_date_mapping(inputfile):
    '''
    :param inputfile: admissions.csv
    :return: pidAdmMap:dict,key:pids,value;Hadm
    admDateMap:dict,key:Hadm,value:date
    '''
    pidAdmMap = {}
    admDateMap = {}
    infd = open(inputfile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        admId = int(tokens[2])
        admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
        admDateMap[admId] = admTime
        if pid in pidAdmMap:
            pidAdmMap[pid].append(admId)
        else:
            pidAdmMap[pid] = [admId]
    infd.close()
    return pidAdmMap,admDateMap
def bulid_mapping(inputfile):
    '''
    :param inputfile:inputfile for mapping dict
    :return: dic,key:admin value:ICD-9
    '''
    admCodeMap = {}
    infd = open(inputfile, 'r',encoding='utf-8')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[2])
        dxStr = 'D_' + convert_to_3digit_icd9(tokens[4][1:-1])
        if admId in admCodeMap:
            if dxStr not in admCodeMap[admId]:
                admCodeMap[admId].append(dxStr)
        else:
            admCodeMap[admId] = [dxStr]
    infd.close()
    return admCodeMap
def bulid_hrv_mapping(inputfile):
    '''
    :param inputfile:inputfile for mapping dict
    :return: dic,key:admin value:ICD-9
    '''
    admCodeMap = {}
    infd = open(inputfile, 'r',encoding='utf-8')
    infd.readline()
    maxlen=0
    for line in infd:
        tokens = line.strip().split('\t')
        try:
            admId = int(tokens[1])
            timeSeriesList = tokens[2:]
            featureTimeseriex = ','.join(timeSeriesList)

        except:
            print('Exception, %s not exit'%line.strip())

        if admId in admCodeMap:
                admCodeMap[admId].append(featureTimeseriex)
        else:
            admCodeMap[admId] = [featureTimeseriex]
        maxlen = len(featureTimeseriex.split(','))
        if (len(featureTimeseriex.split(',')) > maxlen):
            maxlen = len(featureTimeseriex.split(','))
    infd.close()


    return admCodeMap
def build_types(seqs):
    '''
    :param seqs:list,the list of the icd-9
    :return:
    types:dict,key:icd-9,value:mutihot,eg dict('D01':1)
    newSeqs: list,the list of mutihot
    '''
    types = {}
    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in types:
                    newVisit.append(types[code])
                else:
                    types[code] = len(types)
                    newVisit.append(types[code])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)
    return types,newSeqs
def bulid_timeSpans(dates):
    '''
    :param dates: list,time  of visit
    :return: newDates:list ,span time of visit
    '''
    newDates = []
    for date in dates:
        newDate = []
        currenttime = 0
        for time in date:
            if (currenttime == 0):
                newDate.append(0)
            else:
                newDate.append((time - currenttime).days)
            currenttime = time
        newDates.append(newDate)
    return newDates
def build_pid_sortedVisits_mapping(pidAdmMap,admDateMap,admDxMap,admPxMap,admTimeseriesMap):
    '''
    pidAdmMap:dict{pid:adam}
    :return:
    '''
    pidSeqMap = {}
    for pid, admIdList in pidAdmMap.items():
        if len(admIdList) < 3:
            continue
        else:
            for admId in admIdList:
                if admId not in admPxMap:
                    admPxMap[admId] = list('')
                if admId not in admTimeseriesMap:
                    admTimeseriesMap[admId] = list('')
            sortedList = sorted([(admId,admDateMap[admId], admDxMap[admId], admPxMap[admId],admTimeseriesMap[admId]) for admId in admIdList])
            pidSeqMap[pid] = sortedList
    return pidSeqMap
def bulid_each_part(pidSeqMap,outputfile,featureHead):
    '''
    :param pidSeqMap: list ,contain pids.dates,diagnose_seqs,proc_seqs
    :param outputfile: wirte pidSeqMap into outputfile
    :return:pids.dates,diagnose_seqs,proc_seqs
    '''
    fwrite = open(outputfile,'w',encoding='utf-8')
    pids = []
    dates = []
    proc_seqs = []
    diagnose_seqs= []
    timeseries_seqs = []
    fwrite.write('\t'.join(featureHead)+'\n')
    for pid, visits in pidSeqMap.items():
        pids.append(pid)
        diagnose_seq = []
        date = []
        proc_seq = []
        timeseries_seq = []
        for visit in visits:
            date.append(visit[1])
            diagnose_seq.append(visit[2])
            proc_seq.append(visit[3])
            timeseries_seq.append(list(chain.from_iterable(list(x.split(',') for x in visit[4]))))
            fwrite.write(str(pid) +'\t'+str(visit[0])+ '\t' + str(visit[1]) + '\t' + ','.join(visit[2]) + '\t' + ','.join(visit[3]))
            fwrite.write( '\t'+','.join(visit[4]))
            fwrite.write('\n')
        dates.append(date)
        diagnose_seqs.append(diagnose_seq)
        proc_seqs.append(proc_seq)
        timeseries_seqs.append(timeseries_seq)
    fwrite.close()
    return pids,dates,proc_seqs,diagnose_seqs,np.array(timeseries_seqs)
##pid:subject_id 	HADM_id:admID  admTime:AdmiTime dxStr:ICD-9




if __name__ == '__main__':
    # normalizeData('../data/time_series/feature_SpO2_scale.csv','SpO2')
    # normalizeData('../data/time_series/feature_NBPm_scale.csv','NBPm')
    # normalizeData('../data/time_series/feature_NBPs_scale.csv','NBPs')
    # normalizeData('../data/time_series/feature_NBPd_scale.csv','NBPd')
    # normalizeData('../data/time_series/feature_ABPs_scale.csv','ABPs')
    #
    admissionFile = '../data/ADMISSIONS.CSV'
    proceduresFile = '../data/PROCEDURES_ICD.CSV'
    diagnosisFile = '../data/DIAGNOSES_ICD.CSV'
    # timeSeriesFile = "../data/time_series/feature_merge_noscale.csv"
    # hrvFile = '../data/time_series/feature_hr_scale.csv'
    # RRFile = '../data/time_series/feature_RR_scale.csv'
    # SpO2File = '../data/time_series/feature_SpO2_step1.csv'
    NBPmFile = '../data/time_series/NBPm_step1.csv'
    # NBPsFile = '../data/time_series/feature_NBPs_scale.csv'
    # NBPdFile = '../data/time_series/feature_NBPd_scale.csv'
    # ABPsFile = '../data/time_series/feature_ABPs_scale.csv'
    # featureConnecte(hrvFile ,RRFile,SpO2File,NBPmFile ,NBPsFile,NBPdFile,ABPsFile)
    # featureConnecte(SpO2File,NBPmFile)
    calcuate_len(NBPmFile)
    print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap, admDateMap = build_admission_date_mapping(admissionFile)
    print('Building admission-diagnosis-ICD9List mapping')
    admDxMap = bulid_mapping(diagnosisFile)

    print('Building admission-procedures List mapping')
    admPxMap = bulid_mapping(proceduresFile)

    print('Building admission-hrv List mapping')
    admTimeseriesMap = bulid_hrv_mapping(NBPmFile)

    # print('Building admission-RR List mapping')
    # admRRMap = bulid_hrv_mapping(RRFile)
    # print('Building admission-SpO2 List mapping')
    # admSpO2Map = bulid_hrv_mapping(SpO2File)
    #
    # print('Building admission-NBPmFile List mapping')
    # admNBPmMap = bulid_hrv_mapping(NBPmFile)
    #
    # print('Building admission-NBPsFile List mapping')
    # admNBPsMap = bulid_hrv_mapping(NBPsFile)
    #
    # print('Building admission-NBPdFile List mapping')
    # admNBPdMap = bulid_hrv_mapping(NBPdFile)
    #
    # print('Building admission-NBPdFile List mapping')
    # admABPsMap = bulid_hrv_mapping(ABPsFile)
    #
    print('Building pid-sortedVisits mapping')
    pidSeqMap = build_pid_sortedVisits_mapping(pidAdmMap,admDateMap,admDxMap,admPxMap,admTimeseriesMap)
    #
    print('Building pids, dates, diagnose_seqs,proc_seqs')
    featureHead = ['SUBJECT_ID', 'HADM_ID' , 'ADMITTIME', 'Diagnose_ICD', 'Procedure_ICD','timeSeries']
    pids, dates, proc_seqs, diagnose_seqs, timeseries_seqs = bulid_each_part(pidSeqMap,'../data/time_series/feature_merge_noscale.csv',featureHead)
    print('Converting strSeqs to intSeqs, and making types')
    diagnose_types,new_diagnose_seqs = build_types(diagnose_seqs)
    proc_types, new_proc_seqs = build_types(proc_seqs)
    print(proc_types)
    print(diagnose_types)
    print('Converting time to time span')
    newdates = bulid_timeSpans(dates)
    print('save file')
    pickle.dump(diagnose_types, open('../data/diagose_types', 'wb'),protocol=2)
    pickle.dump(proc_types, open('../data/poc_types', 'wb'), protocol=2)
    pickle.dump(pids, open('../data/pids', 'wb'), protocol=2)
    pickle.dump(newdates, open('../data/newdates', 'wb'), protocol=2)
    pickle.dump(new_diagnose_seqs, open('../data/new_diagnose_seqs', 'wb'), protocol=2)
    pickle.dump(new_proc_seqs, open('../data/new_proc_seqs', 'wb'), protocol=2)
    pickle.dump(timeseries_seqs, open('../data/new_timeseries', 'wb'), protocol=2)