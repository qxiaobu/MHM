import pandas as pd
import matlab
import matlab.engine
class featureTimeSeriesScale(object):
    def __init__(self,inputfile,outputfile,featureName):
        self.inputfile = inputfile
        self.outputfile = outputfile
        self.featureName = featureName
    def processNan(self):
        df = pd.read_csv(self.inputfile,sep='\t')
        df[ self.featureName] = df.loc[df[self.featureName].notnull(),  self.featureName].map(lambda x: str(x).replace('nan', '0'))
        df.loc[df[ self.featureName].isnull(),  self.featureName] = ','.join([str(0)] * 59)
        df[['SUBJECT_ID', 'HADM_ID',  self.featureName]].to_csv(self.outputfile, sep='\t', index=False)

class featureTimeSeriesExtract(object):
    def __init__(self,inputFile):
        self.engine = matlab.engine.start_matlab()
        self.data = pd.read_csv(inputFile,sep='\t')
    def extract_hrv(self,stri):
            hrv = [float(i) for i in stri.split(',')]
            res = []
            try:
                    hrv_feature = self.engine.hrv_means(matlab.double(hrv))

                    for (d, x) in hrv_feature.items():
                            if(isinstance(x, dict)):
                                    for (k,v) in x.items():
                                            res.append(str(v))
                            else:
                                  res.append(str(x))
            except:
                    print('Exception in %s'%stri.strip())
            return ','.join(res)

# featureName = 'ABPs'
# print(data[featureName])
# data.loc[data[featureName].notnull(), featureName] = data[data[featureName].notnull()][featureName].map(extract_hrv)
# data[['SUBJECT_ID', 'HADM_ID', featureName]].to_csv('../data/time_series/%s_step3.csv'%featureName, sep='\t', index=False)



# data['SpO2_feature'] = data.loc[data['SpO2'].notnull(), 'SpO2'].map(lambda x:str(x).replace('nan','0'))
# data.loc[data['SpO2_feature'].isnull(), 'SpO2_feature'] = ','.join([str(0)]*l)
#
# data[['SUBJECT_ID', 'HADM_ID', 'SpO2_feature']].to_csv('../data/time_series/SpO2_step3.csv', sep='\t', index=False)
if __name__ == '__main__':
    featureNames = ['SpO2','ABPs','NBPs','NBPd','NBPm']
    for featureName in featureNames:
        inputfile = '../data/time_series/%s_step3.csv'%featureName
        outputfile = '../data/time_series/feature_%s_scale.csv'%featureName
        featureScale = featureTimeSeriesScale(inputfile,outputfile,featureName)
        featureScale.processNan()
    # processNan('../data/time_series/feature_RR.csv','../data/time_series/feature_RR1.csv','RR_feature')
# processNan('../data/feature_hr1.csv','../data/time_series/feature_hr1.csv','hr_feature')