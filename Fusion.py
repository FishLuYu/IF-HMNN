import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import path


filepath=r'XXX.csv'
data=pd.read_csv(filepath)
data.drop('LineNo',axis=1,inplace=True)
# data.drop('field.header.stamp',axis=1,inplace=True)
shape=data.shape


data = data.loc[:, (data != data.iloc[0]).any()]
# data.drop('timestamp_sample',axis=1,inplace=True)

def myInsert(data,time,featureName):
    i=0
    insTime=[]
    insValue=[]
    while i<data.shape[0]:
        x=data[time]
        y=data[featureName]
        f1=interp1d(x,y,kind='linear')
        x_pred=np.linspace(x[i],x[i+1],num=3) 
        insTime.extend(x_pred)
        y_pred=f1(x_pred)
        insValue.extend(y_pred)
        i=i+1
    return insTime,insValue


insTim1,insVal1=myInsert(data)

result = pd.concat([], axis=1)
result.to_csv(r'xxx.csv',index=False)