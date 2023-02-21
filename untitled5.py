"""
@author: alexrt-s
https://github.com/alexrt-s
"""

import uproot4 as uproot
import matplotlib.pyplot as plt
import numpy as np
import math
import hist
import mplhep

from scipy.optimize import curve_fit
import scipy.signal as signal
from scipy.signal import deconvolve


path = "/Users/alexthomson-strong/Desktop/Undergraduate Physics/Senior Honours/SHP/ROOT Files/"

def gaussian(x, sigma, r,N):
    return np.array(1./(sigma * np.sqrt(2*math.pi))*np.exp(-(1/(2 * sigma**2))*np.power((x - r), 2.))) * N

def import_data(root_name,tree_name):
    
    root = uproot.open(path + root_name)
    tree = root[tree_name]
    
    h = uproot.models.TH.Model_TH2D_v4.to_hist(tree)
    
    w,x,y = h.to_numpy()
    return w,x,y,h

def convolution(y):
    mean = np.sum(y*np.arange(-224,225,1))/y.sum()
    std = 0.013
    if std == 0:
        std += 1
    x = gaussian(np.arange(-224,225,1),std,-1* mean,1)
    g = np.convolve(x,y,mode='same')
    g /= g.sum()
    g *= y.sum()
    return g

def time_walk_correction(w):
    twc = []
    for i in w:
        twc.append(convolution(i))
    twc = np.asarray(twc)
    twc = np.nan_to_num(twc)
    return(twc)

def file_names():
    
    file_names = [[None]*1]*32
    for i in range (0,32):
        if i <10:
            file_names[i] = 'ToA_ToT_board2_ch0' + str(i) + '_MCP_NOCUTS'
        else:
            file_names[i] = 'ToA_ToT_board2_ch' + str(i) + '_MCP_NOCUTS'
    return  file_names

def main(root_name,tree_name):

    w, x, y ,h = import_data(root_name,tree_name)
    twc = time_walk_correction(w)
    return w,x,y,twc,h


files = ['RunData-20221017-83800751_PLOTS.root','RunData-20221018-1289979_PLOTS.root',
         'RunData-20221018-4410040_PLOTS.root','RunData-20221018-5262581_PLOTS.root',
         'RunData-20221018-6807935_PLOTS.root']




#for name in file_names():
for i in range(0,1):
    name = 'ToA_ToT_board2_ch28_MCP_NOCUTS'
    file = files[2]

    
    w,x,y,twc,h = main(file,name)
    


    
    plt.pcolormesh(x, y, w.T,cmap='Blues')
    plt.xlabel(h.axes[0].metadata)
    plt.ylabel(h.axes[1].metadata)
    plt.ylim([-50,50])
    plt.xlim([0,100])
    plt.title('Data ' + name)
    plt.legend(labels=['data events = ' + str(w.sum())
                       ])
    plt.grid()
    plt.colorbar()
    plt.show()
    
    plt.pcolormesh(x, y, twc.T,cmap='Blues')
    plt.xlabel(h.axes[0].metadata)
    plt.ylabel(h.axes[1].metadata)
    plt.ylim([-50,50])
    plt.xlim([0,100])
    plt.title('Time Walk Correction '+ name)
    plt.legend()
    plt.colorbar()
    plt.grid()
    plt.show()
    
    plt.plot(y[1:],np.sum(twc.T,axis=1),label='After TWC')
    plt.xlabel('ToA')
    plt.ylabel('Frequency')
    plt.title('Frequency vs ToA After TWC')
    plt.xlim([-50,50])
    #plt.yscale('log')
    
     
      
    
    plt.plot(y[1:],np.sum(w.T,axis=1),label='Before TWC')
    plt.xlabel('ToA')
    plt.ylabel('Frequency')
    plt.title('Frequency vs ToA Before TWC')
    
    #plt.yscale('log')
    
    
    
    plt.grid()
    plt.legend()
    plt.show()
    
 
    
    fit,cov = curve_fit(gaussian,y[1:] ,np.sum(twc.T,axis=1 ),p0=[5,0,1750])
    
    print(fit)
    
    plt.plot(y[1:],gaussian(y[0:-1],fit[0],fit[1],fit[2]),label='fit')
    plt.plot(y[1:],np.sum(twc.T,axis=1),label='After TWC')
    plt.grid()
    plt.xlim([-50,50])
    plt.legend()
    plt.show()
    
    print('Mean = ' + str(fit[1]))
    print('Sigma = ' + str(fit[0]))
    print('error = ' + str(np.sqrt(np.diag(cov))))
    
    print('data events = ' + str(w.sum()))
    print('fit events = ' + str(twc.sum()))
    print('\n')
    print('data x mean = ' + str(np.sum(x[0:-1]*w.transpose())/w.sum()))
    print('data y mean = ' + str(np.sum(y[0:-1]*w)/w.sum()))
    print('\n')
    print('fit x mean = ' + str(np.sum(x[0:-1]*twc.transpose())/twc.sum()))
    print('fit y mean = ' + str(np.sum(y[0:-1]*twc)/twc.sum()))
    print('\n')
