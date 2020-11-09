function sum = hrv_means(x)
addpath('E:\\work\\qiao\\hctsa-master\\hctsa-master\\Operations');
addpath('E:\\work\\qiao\\hctsa-master\\hctsa-master\\PeripheryFunctions');
addpath('E:\work\qiao\hctsa-master\hctsa-master\Toolboxes\Physionet');
addpath('E:\work\qiao\hctsa-master\hctsa-master\Toolboxes\MatlabCentral');

sum.dn_compareKSFit =  DN_CompareKSFit(x,'norm');

sum.dn_fitkernelSmooth = DN_FitKernelSmooth(x);
sum.CO_Embed2 = CO_Embed2(x);
sum.DN_OutlierTest = DN_OutlierTest(x);
sum.DN_RemovePoints = DN_RemovePoints(x);
sum.CO_HistogramAMI = CO_HistogramAMI(x);
sum.CO_trev = CO_trev(x);
end