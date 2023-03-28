"""
@author: alexrt-s
https://github.com/alexrt-s
RICH-Test-Beam-Data-Analysis-Time
"""

import uproot4 as uproot
import matplotlib.pyplot as plt
import numpy as np
import math
import hist
import mplhep

import scipy

from scipy.optimize import curve_fit

my_cmap = plt.cm.get_cmap('viridis').copy()
my_cmap.set_under('w')
plt.rcParams['image.cmap'] = my_cmap
plt.rcParams["errorbar.capsize"] = 4

path = '/Users/alexthomson-strong/Desktop/Undergraduate Physics/Senior Honours/SHP/ROOT Files/download/'


root_name = ['RunData-20221017-83800751_PLOTS_Time.root',
             'RunData-20221018-1289979_PLOTS_Time.root',
             'RunData-20221018-4410040_PLOTS_Time.root',
             'RunData-20221018-5262581_PLOTS_Time.root',
             'RunData-20221018-6807935_PLOTS_Time.root']

def tree_names():    
    tree_names = [[None]*1]*32
    for i in range (0,32):
        if i <10:
            tree_names[i] = 'ToA_ToT_board2_ch0' + str(i) + '_MCP_pixOn_Time;1'
        else:
            tree_names[i] = 'ToA_ToT_board2_ch' + str(i) + '_MCP_pixOn_Time;1'
    return  tree_names

def import_data(root_name,tree_name):
    '''
    a function which imports data from the CERN ROOT file,
    given the name of the root file & the tree that you 
    want to extract data from.
    '''
    root = uproot.open(root_name)
    tree = root[tree_name]
    
    h = uproot.models.TH.Model_TH2D_v4.to_hist(tree)
    
    w,x,y = h.to_numpy()
    return w,x,y,h

#Importing the data from the root file. The index passed into
#tree_names() is the channel number, eg. ch28
#Change the numbers in square brackets to change the run and pixel
#that the analysis is applied to.


RT = int(input('Which HV run do you want to analyse?'))

with open(path +'/HV='+str(RT)+ '/sigma before after.txt','w') as f:
    f.write('')
    f.close()

for CH in range(0,32):


    if RT ==1000:
        ROOT = root_name[0]
    elif RT ==950:
        ROOT = root_name[1]
    elif RT ==900:
        ROOT = root_name[2]
    elif RT ==850:
        ROOT = root_name[3]
    elif RT ==800:
        ROOT = root_name[4]
    else:
        pass
        
        
    
    name = tree_names()[CH]
    
    w,ToT,ToA,h = import_data(path + ROOT,name)
    #print('no of photons in this channel = ' + str(w.sum()))
    
    #Plotting a 2D histogram of the data before applying any corrections.
    
    plt.pcolormesh(ToT, ToA, w.T,vmin=1)
    plt.xlabel('ToT')
    plt.ylabel('ToA/ns')
    plt.ylim([-15,15])
    #plt.xlim([0,40])
    plt.title('Data ' + name)
    plt.grid()
    plt.colorbar()
    plt.savefig(path +'/HV='+str(RT)+'/ch' + str(CH) + '/ToAvsToT Before TWC')
    plt.close()
    
    #Calculating the mean values of ToA at each ToT slice
    mean_ToA = []
    
    for y in w:
    
    
            mean_ToA.append(np.sum(y*np.arange(-15,15,0.2))/y.sum())
    
    mean_ToA = np.asarray(mean_ToA)
    mean_ToA = np.nan_to_num(mean_ToA)
    
    #Removing all the empty columns and rows, where there are 0 events,
    #from the data. '_popped' is a reference to list.pop() which removes
    #elements from a list
    
    indices =[]   
    ToA_popped = []
    for i in range(0,len(y)-1):
        if w.T[i][:].sum() != 0:
            indices.append(i)
        else:
            pass
    for i in indices:
        ToA_popped.append(y[i])
        
    indices =[]   
    ToT_popped = []
    mean_popped = []
    for i in range(0,len(ToT)-1):
        if w[i][:].sum() != 0:
            indices.append(i)
        else:
            pass
    for i in indices:
        ToT_popped.append(ToT[i])
        mean_popped.append(mean_ToA[i])
    
    ToT_popped = np.array(ToT_popped) 
    ToA_popped = np.array(ToA_popped)
    mean_popped = np.array(mean_popped)
    
    
    Result = w[:,~np.all(w == 0, axis = 0)]
    Result = Result.T[:,~np.all(Result.T == 0, axis = 0)]
    w_popped = Result
    
    # Without removing the empty values from the data,
    #there would be a lot of data points along the 
    #Mean ToA = 0 axis which would affect the fitting of a function
    #to the data.
 
    
    #Defining three trial functions  and fitting it to the
    #mean ToA vs ToT data for this channel. The form of the function which needs
    #to be fitted will not necessarily be the same each time, so a Chi squared test is used
    #to evaluate which function is the best
    
  
    def polynomial(x,f,a,b,c,d,e,g,h,j):
        return  a + b * x + c*x**2 + d * x**3  + e *x**4 + j * x**5
  
    fit,cov = curve_fit(polynomial,
                        ToT_popped,
                        mean_popped,
                        sigma = np.sqrt((w_popped.sum(axis=0)))/w_popped.sum(axis=0),
                        maxfev=100000000)
    
  
    
    #Plotting the data and the fitted curve on the same axes.
    
    plt.plot(ToT,polynomial(ToT,fit[0],fit[1],fit[2],fit[3],fit[4],fit[5],fit[6],fit[7],fit[8]))
    plt.scatter(ToT_popped,mean_popped)
    plt.ylim([-5,5])
    
    plt.xlabel('ToT')
    plt.ylabel('Mean ToA')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.title('Mean ToA vs ToT ' + name)
    
    
    
    #plt.legend()
    plt.grid()
    plt.savefig(path +'/HV='+str(RT)+'/ch' + str(CH) + '/meanToAvsToT before TWC fit')
    plt.close()
    
    #Subtracting the time walk from the Mean ToA and plotting this against 
    #the ToT. If the mean ToA is roughly centred on 0 which is encouraging
    
    plt.scatter(ToT_popped,mean_popped - polynomial(ToT_popped,
                                                    fit[0],
                                                    fit[1],
                                                    fit[2],
                                                    fit[3],
                                                   fit[4],fit[5],fit[6],fit[7],fit[8])
                                                   )
    
    
    plt.xlabel('ToT')
    plt.ylabel('Mean ToA - time walk')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.title('Mean ToA - TWC vs ToT ' + name)
    
    #plt.legend()
    plt.grid()
    plt.savefig(path +'/HV='+str(RT)+'/ch' + str(CH) + '/MeanToAvsToT with TWC')
    plt.close()
    
    #Taking a vertical slice of the data & subtracting the time walk,
    #then repeating for each vertical slice and plotting plt.pcolormesh
    #really didn't want to cooperate and so it took way longer than it
    #should have to get this to work. Finally, I'm also plotting ToA vs
    #Frequency before and after the time walk is subtracted. Each slice
    #is plotted individually. It is clear to see that the means is 
    #shifted closer to zero, but it appears that a tail still remains.
    
    time_walks = []
    for i in range (0,w.shape[0]):
        time_walk = polynomial(ToT[i],fit[0],fit[1],fit[2],fit[3],fit[4],fit[5],fit[6],fit[7],fit[8])
        time_walks.append(time_walk)
        
    twc = np.empty([w.shape[0],w.shape[1]])
    for i in range(0,w.shape[0]):
        indx = np.digitize(ToA + time_walks[i],ToA) 
        for j in range(0,len(indx)):
            if indx[j]>149:
                indx[j] = 149
        indx = indx[0:-1]
        twc[i] = w[i][indx]
        
    plt.pcolormesh(ToT, ToA, twc.T,vmin=1)
    plt.xlabel('ToT')
    plt.ylabel('ToA')
    
    plt.title('Time Walk Correction ' + name)
    plt.grid()
    plt.colorbar()
    plt.savefig(path +'/HV='+str(RT)+'/ch' + str(CH) + '/ToAvsToT After TWC')
    plt.close()
        
    
    plt.plot(ToA[0:-1],w.sum(axis=0),label='Before TWC')
    plt.xlim([-5,5])
    plt.title('Frequency vs ToA before & after time walk correction')
    plt.xlabel('ToA')
    plt.ylabel('Frequency')
    
    plt.plot(ToA[0:-1],twc.sum(axis=0),label='After TWC')
    plt.xlim([-5,5])
    plt.xlabel('ToA')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.savefig(path +'/HV='+str(RT)+'/ch' + str(CH) + '/Frequency vs ToA before & after TWC')
    plt.close()
    
    def gaussian(x,r,sigma,N):
        return np.array(scipy.stats.norm.pdf(x,r,sigma)) * N
    
    fit2,cov2 = curve_fit(gaussian,ToA[0:-1],w.sum(axis=0))
    plt.plot(ToA[0:-1],gaussian(ToA[0:-1],fit2[0],fit2[1],fit2[2]),label='Gaussian Fit')
    plt.plot(ToA[0:-1],w.sum(axis=0),label='Original Data')
    plt.xlim([-5,5])
    plt.legend()
    plt.title('ToA vs Frequency before TWC with Gaussian fit')
    plt.xlabel('ToA')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(path +'/HV='+str(RT)+'/ch' + str(CH) + '/Frequency vs ToA before TWC with fit')
    plt.close()
    
    fit3,cov3 = curve_fit(gaussian,ToA[0:-1],twc.sum(axis=0),p0=(0,5,1000))
    plt.plot(ToA[0:-1],gaussian(ToA[0:-1],fit3[0],fit3[1],fit3[2]),label='Gaussian Fit')
    plt.plot(ToA[0:-1],twc.sum(axis=0),label='Time-Walk Corrected Data')
    plt.xlim([-5,5])
    plt.legend()
    plt.title('ToA vs Frequency after TWC with Gaussian fit')
    plt.xlabel('ToA')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(path +'/HV='+str(RT)+'/ch' + str(CH) + '/Frequency vs ToA after TWC with fit')
    plt.close()
    
    print('time resolution without the time walk correction is ' + str(fit2[1]))
    print('time resolution with the time walk correction is ' + str(fit3[1]))
    

    with open(path +'/HV='+str(RT)+ '/sigma before after.txt','a') as f:
        f.write(str(fit2[1])+ ',' + str(fit3[1]) + '\n')

    with open(path +'/HV='+str(RT)+'/ch' + str(CH) + '/no. of photons.txt','w') as f:
        f.write(str(w.sum()))
