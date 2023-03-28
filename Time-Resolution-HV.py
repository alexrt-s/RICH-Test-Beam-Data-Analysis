"""
@author: alexrt-s
https://github.com/alexrt-s
Time-Resolution-HV 
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["errorbar.capsize"] = 4

path = '/Users/alexthomson-strong/Desktop/Undergraduate Physics/Senior Honours/SHP/Report Files/210323/'
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

error_before = []
error_after = []

for v in HV:
    with open(path +'/HV='+str(v)+ '/uncertainty before after.txt','r') as f:
        error = f.read().split(sep=('\n'))
    error_before_v = []
    error_after_v = []
    for i in error:
        try:
            error_before_v.append(float(i.split(sep=',')[0]))
            error_after_v.append(float(i.split(sep=',')[1]))
        except:
            pass
    error_before.append(error_before_v)
    error_after.append(error_after_v)
error_after = np.array(error_after,dtype=float).T
error_before = np.array(error_before,dtype=float).T

for ch in range(0,32):
    
    plt.errorbar(HV,sigma_before[ch],yerr=error_before[ch],label='Before TWC',ls='')
    plt.errorbar(HV,sigma_after[ch],yerr=error_before[ch],label='After TWC',ls='')
    plt.legend()
    plt.grid()
    plt.title('Time Resolution vs HV channel ' + str(ch) )
    plt.xlabel('Bias/ V')
    plt.ylabel('Time-Resolution/ ns')
    #plt.savefig(path + 'Time Resolution vs HV/ch' + str(ch) + '/Time Resolution vs HV')
    plt.close()

    
    plt.scatter(HV,1 - sigma_after[ch]/sigma_before[ch])
    plt.grid()
    plt.title('TWC Efficiency vs HV channel ' + str(ch))
    plt.xlabel('Bias /V')
    plt.ylabel('Time Walk Correction Efficiency')
    #plt.savefig(path + 'Time Resolution vs HV/ch' + str(ch) + '/TWC Efficiency vs HV')
    plt.close()


    
for v in range(0,len(HV)):
    plt.errorbar(np.arange(0,32,1),sigma_before.T[v],yerr=error_before.T[v],label='Before TWC',ls='')
    plt.errorbar(np.arange(0,32,1),sigma_after.T[v],yerr=error_after.T[v],label='After TWC',ls='')
    plt.legend()
    plt.grid()
    plt.xlabel('channel number')
    plt.ylabel('Time Resolution/ ns')
    plt.title('Time Resolution vs Channel Number HV = ' + str(HV[v]))
    #plt.savefig(path +'/HV='+str(HV[v])+ '/Time Resolution vs Channel Number')
    plt.close()
    
on_beam = [1,2,8,9,19,22,25,28]

for v in range(0,len(HV)):
    plt.errorbar(on_beam,sigma_before.T[v][on_beam],yerr=error_before.T[v][on_beam],label='Before TWC',ls='')
    plt.errorbar(on_beam,sigma_after.T[v][on_beam],yerr=error_after.T[v][on_beam],label='After TWC',ls='')
    plt.legend()
    plt.grid()
    plt.xlabel('channel number')
    plt.ylabel('Time Resolution/ ns')
    plt.title('Time Resolution vs Channel Number on Ring HV = ' + str(HV[v]))
    plt.savefig(path +'/HV='+str(HV[v])+ '/Time Resolution vs Channel Number on beam only')
    plt.show()
