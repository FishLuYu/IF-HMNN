
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import path
data=pd.read_csv(r'XXX.csv')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
orgData_x=data['timestamp']
orgData_y=data['delta_angle[0]']

plt.grid(color='lightgray', linestyle='--', linewidth=0.5)

plt.title('Data points for interpolation', fontsize=12)
plt.plot(orgData_x, orgData_y,color='#6EC8C8',label='orgCompassData')
plt.plot(insTim1, insVal1,'x',color='#6496D2',label='aftCompassData')
plt.xlabel("TimeStamp")
plt.ylabel("Compass Data")
plt.legend()
plt.show()
plt.savefig(r'xxx.svg',format='svg')