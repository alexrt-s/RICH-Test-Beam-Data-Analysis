"""
@author: alexrt-s
https://github.com/alexrt-s
Time-Resolution-HV 
"""

import matplotlib.pyplot as plt
import numpy as np

path = '/Users/alexthomson-strong/Desktop/Undergraduate Physics/Senior Honours/SHP/ROOT Files/download/'


HV = [1000,950,900,850,800]

sigma_before = []
sigma_after = []

for v in HV:
    with open(path +'/HV='+str(v)+ '/sigma before after.txt','r') as f:
        sigma = f.read().split(sep=('\n'))
    sigma_before_v = []
    sigma_after_v = []
    for i in sigma:
        try:
            sigma_before_v.append(float(i.split(sep=',')[0]))
            sigma_after_v.append(float(i.split(sep=',')[1]))
        except:
            pass
    sigma_before.append(sigma_before_v)
    sigma_after.append(sigma_after_v)
sigma_after = np.array(sigma_after,dtype=float).T

sigma_before = np.array(sigma_before,dtype=float).T

ch = 25

plt.scatter(HV,sigma_before[ch],label='Before TWC')
plt.scatter(HV,sigma_after[ch],label='After TWC')
plt.legend()
plt.grid()
plt.title('Time Resolution vs HV')
plt.xlabel('HV')
plt.ylabel('Time-Resolution')
plt.show()

plt.plot(HV,1 - sigma_after[ch]/sigma_before[ch])
plt.grid()
plt.title('TWC Efficiency vs HV')
plt.xlabel('HV')
plt.ylabel('Time Walk Correction Efficiency')
plt.show()


    



'''


plt.plot(HV,sigma_before,label='Before TWC')
plt.plot(HV,sigma_after,label='After TWC')
plt.legend()
plt.grid()
plt.title('Time Resolution vs HV')
plt.xlabel('HV')
plt.ylabel('Time-Resolution')
plt.show()

plt.plot(HV,1-np.array(sigma_after)/np.array(sigma_before))
plt.grid()
plt.title('TWC Efficiency vs HV')
plt.xlabel('HV')
plt.ylabel('Time Walk Correction Efficiency')
plt.show()
'''