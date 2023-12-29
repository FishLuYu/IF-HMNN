import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import path


# def find_files_with_name(directory, keyword):
#     matching_files = []
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if keyword in file:
#                 matching_files.append(os.path.join(root, file))
#     return matching_files

# File=find_files_with_name(r"XXXX","XXX.csv")

# filepath=File[5]
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
    while i<data.shape[0]-1:
        x=data[time]
        y=data[featureName]
        f1=interp1d(x,y,kind='linear')
        x_pred=np.linspace(x[i],x[i+1],num=3) 
        insTime.extend(x_pred)
        y_pred=f1(x_pred)
        insValue.extend(y_pred)
        i=i+1
    return insTime,insValue

# insTim1,insVal1=myInsert(data,'timestamp','delta_angle[0]')
# insTim2,insVal2=myInsert(data,'timestamp','delta_angle[1]')
# insTim3,insVal3=myInsert(data,'timestamp','delta_angle[2]')
# insTim4,insVal4=myInsert(data,'timestamp','delta_velocity[0]')
# insTim5,insVal5=myInsert(data,'timestamp','delta_velocity[1]')
# insTim6,insVal6=myInsert(data,'timestamp','delta_velocity[2]')
# insTim7,insVal7=myInsert(data,'timestamp','delta_angle_dt')
# insTim8,insVal8=myInsert(data,'timestamp','delta_velocity_dt')
# insTim9,insVal9=myInsert(data,'timestamp','magnetometer_ga[0]')
# insTim10,insVal10=myInsert(data,'timestamp','magnetometer_ga[1]')
# insTim11,insVal11=myInsert(data,'timestamp','magnetometer_ga[2]')

insTim1,insVal1=myInsert(data,'TimeUS','DesRoll')
insTim2,insVal2=myInsert(data,'TimeUS','Roll')
insTim3,insVal3=myInsert(data,'TimeUS','DesPitch')
insTim4,insVal4=myInsert(data,'TimeUS','Pitch')
insTim5,insVal5=myInsert(data,'TimeUS','DesYaw')
insTim6,insVal6=myInsert(data,'TimeUS','Yaw')
insTim7,insVal7=myInsert(data,'TimeUS','ErrRP')
insTim8,insVal8=myInsert(data,'TimeUS','ErrYaw')
insTim9,insVal9=myInsert(data,'TimeUS','MagX')
insTim10,insVal10=myInsert(data,'TimeUS','MagY')
insTim11,insVal11=myInsert(data,'TimeUS','MagZ')

df1 = pd.DataFrame(insTim1, columns=['timestamp'])
df2 = pd.DataFrame(insVal1, columns=['DesRoll'])
df3 = pd.DataFrame(insVal2, columns=['Roll'])
df4 = pd.DataFrame(insVal3, columns=['DesPitch'])
df5 = pd.DataFrame(insVal4, columns=['Pitch'])
df6 = pd.DataFrame(insVal5, columns=['DesYaw'])
df7 = pd.DataFrame(insVal6, columns=['Yaw'])
df8 = pd.DataFrame(insVal7, columns=['ErrRP'])
df9 = pd.DataFrame(insVal7, columns=['ErrYaw'])
df10 = pd.DataFrame(insVal9, columns=['MagX'])
df11 = pd.DataFrame(insVal10, columns=['MagY'])
df12 = pd.DataFrame(insVal10, columns=['MagZ'])

result = pd.concat([df1, df2, df3,df4,df5,df6,df7,df8,df9,df10,df11,df12], axis=1)
result.to_csv(r'xxx.csv',index=False)