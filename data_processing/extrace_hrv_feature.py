import pandas as pd

def processNan(inputFile,outputFile,featureName):
    df = pd.read_csv(inputFile)
    df[featureName] = df.loc[df[featureName].notnull(), featureName].map(lambda x: str(x).replace('nan', '0'))
    df.loc[df[featureName].isnull(), featureName] = ','.join([str(0)] * 59)
    df[['SUBJECT_ID', 'HADM_ID', featureName]].to_csv(outputFile, sep='\t', index=False)
# processNan('../data/time_series/feature_RR.csv','../data/time_series/feature_RR1.csv','RR_feature')
# processNan('../data/feature_hr1.csv','../data/time_series/feature_hr1.csv','hr_feature')
import matlab
import matlab.engine
engine = matlab.engine.start_matlab()
data = pd.read_csv('../data/time_series/Heart Rate_step1.csv',sep='\t')
def extract_hrv(stri):
        hrv = [float(i) for i in stri.split(',')]
        res = []
        try:
                hrv_feature = engine.hrv_means(matlab.double(hrv))

                for (d, x) in hrv_feature.items():
                        if(isinstance(x, dict)):
                                for (k,v) in x.items():
                                        res.append(str(v))
                        else:
                              res.append(str(x))
        except:
                print('Exception in %s'%stri.strip())
        return ','.join(res)

featureName = 'HeartRate'
print(data[featureName])
data.loc[data[featureName].notnull(), featureName] = data[data[featureName].notnull()][featureName].map(extract_hrv)
data[['SUBJECT_ID', 'HADM_ID', featureName]].to_csv('../data/time_series/%s_step3.csv'%featureName, sep='\t', index=False)



# data['SpO2_feature'] = data.loc[data['SpO2'].notnull(), 'SpO2'].map(lambda x:str(x).replace('nan','0'))
# data.loc[data['SpO2_feature'].isnull(), 'SpO2_feature'] = ','.join([str(0)]*l)
#
# data[['SUBJECT_ID', 'HADM_ID', 'SpO2_feature']].to_csv('../data/time_series/SpO2_step3.csv', sep='\t', index=False)