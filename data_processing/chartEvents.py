# -*- coding: utf-8 -*-

# Author: Zhen Zhang <13161411563@163.com>

from datetime import datetime
def extract_item(inputfile,outputfile,itemID):
    outfd = open(outputfile,'w')
    with open(inputfile, 'r') as infd:
        infd.readline()
        for line in infd:
            tokens = line.strip().split(',')
            if(int(tokens[4]) in itemID):
                outfd.write(line)
    outfd.close()
def build_admission_date_mapping(inputfile):
    '''
    :param inputfile: admissions.csv
    :return: pidAdmMap:dict,key:pids,value;Hadm
    admDateMap:dict,key:Hadm,value:date
    '''
    pidAdmMap = {}
    admDateMap = {}
    infd = open(inputfile, 'r')
    lineNum = 0
    infd.readline()
    for line in infd:
        lineNum = lineNum+1
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        admId = int(tokens[2])
        # if(pid==109 and tokens[5]=='2141-06-15 09:00:00'):
        #
        #     print('ss')
        admTime = datetime.strptime(tokens[5], '%Y-%m-%d %H:%M:%S')
        if admId in admDateMap:
            admDateMap[admId].append(admTime)
        else:
            admDateMap[admId] = [admTime]
        if pid in pidAdmMap:
            if admId not in pidAdmMap[pid]:
                pidAdmMap[pid].append(admId)
        else:
            pidAdmMap[pid] = [admId]
        # if (lineNum == 10000):
        #     break
    infd.close()
    return pidAdmMap,admDateMap

def bulid_mapping(inputfile):
    '''
    :param inputfile:inputfile for mapping dict
    :return: dic,key:admin value:ICD-9
    '''
    admhrvMap = {}
    lineNum = 0
    infd = open(inputfile, 'r',encoding='utf-8')
    infd.readline()
    for line in infd:
        lineNum = lineNum+1
        tokens = line.strip().split(',')
        admId = int(tokens[2])
        hrv_value = (tokens[9])
        if admId in admhrvMap:
                admhrvMap[admId].append(hrv_value)
        else:
            admhrvMap[admId] = [hrv_value]
        # if(lineNum == 1000):
        #     break
    infd.close()
    return admhrvMap

def build_pid_sortedVisits_mapping(pidAdmMap,admDateMap,admhrvMap):
    '''
    pidAdmMap:dict{pid:adam}
    :return:
    '''
    pidSeqMap = {}
    for pid, admIdList in pidAdmMap.items():
        # if len(admIdList) < 3:
        #     continue
        # else:
            # sortedList = sorted([(admDateMap[admId], admhrvMap[admId]) for admId in admIdList])
            sortedList = []
            for admId in admIdList:
                # print(admId,len(admDateMap[admId]),len(admhrvMap[admId]))
                if len(admDateMap[admId])!= len(admhrvMap[admId]):
                    assert "%s the length of hrv and date is not equal" %admId
                list1,list2 = ((list(t) for t in zip(*sorted(zip(admDateMap[admId], admhrvMap[admId])))))
                sortedList.append([admId,list1,list2])
            # print(sortedList)
    # print(list(t) for t in zip(*sorted(zip(admDateMap[admId], admhrvMap[admId]) for admId in admIdList)))
            pidSeqMap[pid] = sortedList
    return pidSeqMap
def bulid_each_part(pidSeqMap,outputfile):
    '''
    :param pidSeqMap: list ,contain pids.dates,diagnose_seqs,proc_seqs
    :param outputfile: wirte pidSeqMap into outputfile
    :return:pids.dates,diagnose_seqs,proc_seqs
    '''
    fwrite = open(outputfile,'w',encoding='utf-8')
    fwrite.write('pid' + '\t' + 'adam' + '\t' + 'NBPd' +'\n')
    pids = []
    dates = []
    hrv_seqs = []
    for pid, visits in pidSeqMap.items():
        pids.append(pid)
        hrv_seq = []
        date = []
        for visit in visits:
            date.append(visit[1])
            heartRate = [i for i in visit[2] if i != '']
            hrv_seq.append(heartRate)
            fwrite.write(str(pid) +'\t' +str(visit[0])+'\t' + ','.join(heartRate))
            fwrite.write('\n')
        dates.append(date)
        hrv_seqs.append(hrv_seq)
    fwrite.close()
    return pids,dates,hrv_seqs
def run(inputfile,outputfile):
    print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap, admDateMap = build_admission_date_mapping(inputfile)

    print('Building admission-hrv-List mapping')
    admhrvMap = bulid_mapping(inputfile)


    print('Building pid-sortedVisits mapping')
    pidSeqMap = build_pid_sortedVisits_mapping(pidAdmMap, admDateMap, admhrvMap)

    print('Building pids, dates, diagnose_seqs,proc_seqs')
    bulid_each_part(pidSeqMap, outputfile)
if __name__ == '__main__':
    featureName = 'ABPs'
    featureId = [220050]
    extract_item('E:\work\qiao\data\CHARTEVENTS\CHARTEVENTS.csv', '../data/time_series/%s.csv'%featureName, featureId)
    run('../data/time_series/%s.csv'%featureName,'../data/time_series/%s_step1.csv'%featureName)