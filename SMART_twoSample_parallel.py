# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 11:29:04 2018

@author: Jonathan van Leeuwen
"""
#==============================================================================
#==============================================================================
# # Run the SMART pipeline between conditions (paired sample)
#==============================================================================
#==============================================================================
# Note that this implementation is optimized for readability and not for
# performance!!!! 
# 
# To optimize for performance, the section with smoothing and permutation
# can be parallelized. 

#==============================================================================
# Assumes that the data is in a pickle file and structered identical to 
# 'ExampleDataSMUG.p'.
# 
# A pandas dataframe with:
#       Each participant in its own row 
#       Time var1 in 1 column, all the data in 1 cell
#       Dep. var1 in 1 column, all the data in 1 cell
#       Time var2 in 1 column, all the data in 1 cell
#       Dep. var2 in 1 column, all the data in 1 cell
#
# Example:
#       Index/pp	            TimeVar1	                    DepVar1	                     TimeVar2                      DepVar2
#           0	    [155 192 279 ..., 143 142 149]	    [0 0 1 ..., 0 1 0]	    [159 163 201 ..., 149 229 154]	    [0 1 0 ..., 1 0 1]
#           1	    [379 312 272 ..., 278 288 267]	    [1 1 1 ..., 0 1 1]	    [386 437 422 ..., 226 319 237]	    [1 1 1 ..., 1 1 0]
#           2	    [192 208 236 ..., 175 268 171]	    [0 0 0 ..., 0 0 0]	    [180 227 189 ..., 172 180 205]	    [1 1 1 ..., 1 1 1]
#           3	    [397 291 412 ..., 457 408 366]	    [1 1 1 ..., 1 1 1]	    [392 452 459 ..., 378 342 444]	    [1 1 0 ..., 1 1 1]

#==============================================================================
import time
t = time.time()
#==============================================================================
# # Define parameters (Make changes here)
#==============================================================================
# File params
dataFile = 'ExampleDataSMART.p' 
# Variable of interest 1
timeVar1 = 'TimeVar1' # The name of the column with he time variables in dataFile
depVar1 = 'DepVar1' # The name of the column with he depVar variables in dataFile

# Variable of interest 2
timeVar2 = 'TimeVar2' # The name of the column with he time variables in dataFile
depVar2 = 'DepVar2' # The name of the column with he depVar variables in dataFile


# Smoothing params
kernelSize = 10
timeMin = 100
timeMax = 501 # Excludes the last value, e.g. 301 returns vect until 300 
stepSize = 1

# Permutation params
nPerms = 1000

# Statistics params
sigLevel = 0.05

# Ploting params
lineColorVar1 = [250,0,0]
lineColorVar2 = [0,0,255]
lineWidth = 2
markerOffset = 0.75
markerSize = 10
xLabelSize = 20
yLabelSize = 20
yMin = -0.5
yMax = 1.5
histRes = 100 # number of bars for permutation distribution

#Parallel settings
nJobs = 4 # Number of cores to use

#==============================================================================
#==============================================================================
# # Everything after this should not need to be changed!!!!!!!!
#==============================================================================
#==============================================================================

#==============================================================================
# Import modules 
#==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import SMART_Funcs as SF
import numpy as np
from joblib import Parallel, delayed
plt.close('all')


#==============================================================================
# Load data and initiate vectors
#==============================================================================
data = pd.read_pickle(dataFile)
timeVect = np.arange(timeMin, timeMax, stepSize, dtype=float)
nPP = len(data)


#==============================================================================
# Prealocate and initiate data structures
#==============================================================================
# Smoothing
smData1 = np.zeros((nPP, len(timeVect)))
smWeights1 = np.zeros((nPP, len(timeVect)))
smData2 = np.zeros((nPP, len(timeVect)))
smWeights2 = np.zeros((nPP, len(timeVect)))

# Permutations
permData1 = np.zeros((nPP, len(timeVect), nPerms))
permWeight1 = np.zeros((nPP, len(timeVect), nPerms))
permData2 = np.zeros((nPP, len(timeVect), nPerms))
permWeight2 = np.zeros((nPP, len(timeVect), nPerms))

#==============================================================================
# Run smoothing and permutations for each participant
# * This part can be parallelized for speed if required *
#==============================================================================
def runSmoothing(i):
    #  Extract data for participant
    times1 = data[timeVar1][i]
    depV1 = data[depVar1][i]
    times2 = data[timeVar2][i]
    depV2 = data[depVar2][i]

    # Run Smoothing
    smData1, smWeights1 = SF.gaussSmooth(times1, depV1, timeVect, kernelSize)
    smData2, smWeights2 = SF.gaussSmooth(times2, depV2, timeVect, kernelSize)
    
    # Run Permutations Between conditions
    permData1, permWeight1, permData2, permWeight2 = SF.permute(times1, depV1, times2, depV2, timeVect, kernelSize, nPerms)

    return smData1, smWeights1, smData2, smWeights2, permData1, permWeight1, permData2, permWeight2

if __name__ == "__main__":
    # Run parallel processing
    AllResults = Parallel(n_jobs=nJobs,verbose=9)(delayed(runSmoothing)(i) for i in range(nPP))
    for i in range(nPP):
        smData1[i, :], smWeights1[i,:] = AllResults[i][0], AllResults[i][1]
        smData2[i, :], smWeights2[i,:] = AllResults[i][2], AllResults[i][3]
        # Run Permutations against baseline
        permData1[i,:,:], permWeight1[i,:,:] = AllResults[i][4], AllResults[i][5]
        permData2[i,:,:], permWeight2[i,:,:]= AllResults[i][6], AllResults[i][7]
    del AllResults

    #==============================================================================
    # Run statistics
    #==============================================================================
    # Weight the data and extract clusters
    weighDataAv1 = np.average(smData1, weights = smWeights1, axis=0)
    weighDataAv2 = np.average(smData2, weights = smWeights2, axis=0)
    
    sigCL, sumTvals = SF.clusterStat_rel(smData1, smData2, smWeights1, smWeights2, sigLevel)
    
    # Calculate permutation distributions and significance thresholds
    permDistr = SF.permuteClusterStat(permData1, permData2, permWeight1, permWeight2, sigLevel)
    
    # Get significant cluster size threshold
    sigThres = np.percentile(permDistr, 100-(sigLevel*100))
    
    # Calculate 95 confidence intervals
    conf95 = SF.weighPairedConf95(smData1, smData2, smWeights1, smWeights2)
    
    #==============================================================================
    # Plot data
    #==============================================================================
    # Transform color
    lineColorVar1 =([i/255.0 for i in lineColorVar1])
    lineColorVar2 =([i/255.0 for i in lineColorVar2])
    
    # Initiate plot
    fig, (ax1, ax2) = plt.subplots(2,1)
    plt.suptitle('Smoothing Method for Analysis of Response Time-course: SMART', fontsize=xLabelSize)
    # Plot smoothed data
    ax1.plot(timeVect, weighDataAv1, color=lineColorVar1, linewidth=lineWidth)
    ax1.plot(timeVect, weighDataAv2, color=lineColorVar2, linewidth=lineWidth)
    
    # Plot confidence intervals
    ax1.fill_between(timeVect, weighDataAv1-conf95, weighDataAv1+conf95, color=lineColorVar1, alpha=0.25)
    ax1.fill_between(timeVect, weighDataAv2-conf95, weighDataAv2+conf95, color=lineColorVar2, alpha=0.25)
    
    # Plot significant time points
    for ind, i in enumerate(sigCL):
        ax1.plot(timeVect[i], weighDataAv1[i], 'k-', lineWidth=lineWidth*1.5)
        ax1.plot(timeVect[i], weighDataAv2[i], 'k-', lineWidth=lineWidth*1.5)
        # Plot asterix for signficant clusters
        if sumTvals[ind] >= sigThres:
            xPos = np.average(timeVect[i])
            yPos = np.max([weighDataAv1[int(np.mean(i))], weighDataAv2[int(np.mean(i))]])+markerOffset+np.max(conf95[int(np.mean(i))])
            if yPos >= yMax:
                yPos = yMax-0.5
            ax1.plot(xPos,yPos, 'k*', ms=markerSize)
    
    ax1.set_xlim(timeMin, timeMax-1)
    ax1.set_ylim(yMin, yMax)
    ax1.legend([depVar1, depVar2,'Sig. difference'],loc=1)
    ax1.set_xlabel('Time', fontsize=xLabelSize)
    ax1.set_ylabel('Dep.var', size=yLabelSize)
    
    # Plot permutation distributions
    ax2.hist(permDistr, histRes)
    ax2.plot([sigThres,sigThres], ax2.get_ylim(), color='r', linewidth=2)
    for idx, tV in enumerate(sumTvals):
        ax2.plot([tV, tV], ax2.get_ylim(), color='k', linewidth=2)
    ax2.legend(['Sig Threshold: '+str(sigLevel), 'Clusters'],loc=1)
    # Set xaxis
    ax2.set_xlabel('Sum of cluster t-values', fontsize=xLabelSize)
    ax2.set_ylabel("Frequency (log)", size=yLabelSize)
    ax2.set_yscale('log')

    # Plot kernel density estimation KDE
    sTimes1, unqT1, countT1 = SF.getKDE(np.hstack(data[timeVar1]),timeVect, kernelSize)
    sTimes2, unqT2, countT2 = SF.getKDE(np.hstack(data[timeVar2]),timeVect, kernelSize)
    maxT = np.max(np.hstack([sTimes1, sTimes2]))*8
    ax1_1 = ax1.twinx()
    ax1_1.plot(timeVect, sTimes1, '--k', alpha = 0.6)
    ax1_1.plot(timeVect, sTimes2, '-k', alpha = 0.3)
    ax1_1.bar(unqT1, countT1, color='k', alpha = 0.6)
    ax1_1.bar(unqT2, countT2, color='k', alpha = 0.3)
    ax1_1.set_ylim(0, maxT)
    ax1_1.legend(['KDE_1', 'KDE_2'],loc=4)    
    ax1_1.set_yticks(np.linspace(0,np.max(np.hstack([sTimes1, sTimes2])),3, dtype=int))
    
    print('Duration:', time.time() - t)



