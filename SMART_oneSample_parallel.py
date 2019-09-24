# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 11:29:04 2018

@author: Jonathan van Leeuwen
"""
#==============================================================================
#==============================================================================
# # Run the SMART pipeline against baseline
#==============================================================================
#==============================================================================
# Note that this implementation is optimized for readability and not for
# performance!!!!
#
# To optimize for performance, the section with smoothing and permutation
# can be parallelized.

#==============================================================================
#
# Short format data
#
# Assumes that the data is in a pickle file and structered identical to
# 'ExampleDataSMUG.p'.
#
# A pandas dataframe with:
#       Each participant in its own row
#       Time var in 1 column, all the data in 1 cell
#       Dep. var in 1 column, all the data in 1 cell
#
# Example:
#      Index/pp                 TimeVar                      DepVar
#           0	       [155 192 279 ..., 143 142 149]	    [0 0 1 ..., 0 1 0]
#           1	       [379 312 272 ..., 278 288 267]	    [1 1 1 ..., 0 1 1]
#           2	       [192 208 236 ..., 175 268 171]	    [0 0 0 ..., 0 0 0]
#           3	       [397 291 412 ..., 457 408 366]	    [1 1 1 ..., 1 1 1]
#
#
#==============================================================================
# Long format data
#
# Assumes that the data is in a comma sepperated file, .csv and structered identical to
# File should include a header for each column. (in this example there are 2
# conditions logged in the file)
# 'ExampleDataSMUG.csv'.
#
# A pandas dataframe with:
#       Each trial in its own row
#       One columns with participant number
#       One column with the time variable
#       One column with the dependent variable 
#       One column with the condition label
#
# Example:
#         Index     participantNr       TimeVar         DepVar      Condition
#           0	          1               155             0             1
#           1	          1               192             0             1
#           3	          1               279             1             1
#           .	          .                 .             .             .
#           700           3               180             1             2
#           701           3               211             0             2
#           702           3               182             1             2
#           .	          .                 .             .             .
#           2005         10               475             1             1 
#           2006         10               437             1             1
#           2007         10               402             1             1
#
#==============================================================================
import time
t = time.time()
#==============================================================================
# # Define parameters (Make changes here)
#==============================================================================
# File params
dataFile = 'ExampleDataSMART' # Exclude the extension, it will infere extension
dataFormat = 'long' # 'long' or 'short' # See instructions above

# If short format
timeVar = 'TimeVar1' # The name of the column with he time variables in dataFile
depVar = 'DepVar1' # The name of the column with he depVar variables in dataFile

# If long format
participantColumn = 'participantNr'
timeVarColumn = 'TimeVar'
depVarColumn = 'DepVar'
conditionColumn = 'Condition' 
conditionLabel = 1 # The value for the condition we are interested in, in the condition column

# Smoothing params
kernelSize = 10
timeMin = 100
timeMax = 501 # Excludes the last value, e.g. 301 returns vect until 300
stepSize = 1

# Permutation params
nPerms = 1000
baseline = 0.5

# Statistics params
sigLevel = 0.05

# Ploting params
lineColor = [250,0,0]
lineWidth = 2
markerOffset = 0.75
markerSize = 10
xLabelSize = 20
yLabelSize = 20
yMin = -.5
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
timeVect = np.arange(timeMin, timeMax, stepSize, dtype=float)

# Load data based on format
if dataFormat == 'short':
    data = pd.read_pickle(dataFile+'.p')
    nPP = len(data)
    allTimes = np.hstack(data[timeVar])
elif dataFormat == 'long':
    data = pd.read_csv(dataFile+'.csv')
    pp = np.unique(data[participantColumn])
    nPP = len(pp)
    condBool = data[conditionColumn].values == conditionLabel
    allTimes = data[timeVarColumn].values
    
#==============================================================================
# Prealocate and initiate data structures
#==============================================================================
# Smoothing
smData = np.zeros((nPP, len(timeVect)))
smWeights = np.zeros((nPP, len(timeVect)))

# Permutations
permD1 = np.zeros((nPP, len(timeVect), nPerms))
permW1 = np.zeros((nPP, len(timeVect), nPerms))
permD2 = np.zeros((nPP, len(timeVect), nPerms))
permW2 = np.zeros((nPP, len(timeVect), nPerms))

#==============================================================================
# Run smoothing and permutations for each participant
# * This part can be parallelized for speed if required *
#==============================================================================
def runSmoothing(i):
    #  Extract data for participant based on data format
    if dataFormat == 'short': 
        times = data[timeVar][i]
        depV = data[depVar][i]
    elif dataFormat == 'long':
        ppBool = data[participantColumn].values == pp[i]
        dataBool = np.logical_and(condBool, ppBool)
        times = data[timeVarColumn][dataBool].values
        depV = data[depVarColumn][dataBool].values

    # Run Smoothing
    smData, smWeights = SF.gaussSmooth(times, depV, timeVect, kernelSize)

    # Run Permutations against baseline
    permData1, permWeight1, permData2, permWeight2 = SF.permute(
            times, 
            depV, 
            newX=timeVect, 
            sigma=kernelSize, 
            nPerms=nPerms, 
            baseline=baseline
            )
    return smData, smWeights, permData1, permWeight1, permData2, permWeight2

if __name__ == "__main__":
    
    # Run parallel processing
    AllResults = Parallel(n_jobs=nJobs,verbose=9)(delayed(runSmoothing)(i) for i in range(nPP))
    for i in range(nPP):
        smData[i, :], smWeights[i,:] = AllResults[i][0], AllResults[i][1]
    
        # Run Permutations against baseline
        permD1[i,:,:], permW1[i,:,:], permD2[i,:,:], permW2[i,:,:] = AllResults[i][2], AllResults[i][3], AllResults[i][4], AllResults[i][5]
    del AllResults
    
    #==============================================================================
    # Run statistics
    #==============================================================================
    # Weight the data and extract clusters
    weighDataAv = np.average(smData, weights = smWeights, axis=0)
    sigCL, sumTvals = SF.clusterStat_oneSamp(smData, smWeights, baseline, sigLevel)
    # Calculate 95 confidence intervals

    conf95 = SF.weighConfOneSample95(smData, smWeights)
    # Calculate permutation distributions and significance thresholds
    permDistr = SF.permuteClusterStat(permD1, permD2, permW1, permW2, sigLevel)
    sigThres = np.percentile(permDistr, 100-(sigLevel*100))
    
    #==============================================================================
    # Plot data
    #==============================================================================
    # Transform color
    lineColor =([i/255.0 for i in lineColor])
    baselineX = np.arange(timeMin, timeMax, 1)
    baselineY = np.zeros(len(baselineX))+baseline
    # Initiate plot
    fig, (ax1, ax2) = plt.subplots(2,1)
    plt.suptitle('Smoothing Method for Analysis of Response Time-course: SMART', fontsize=xLabelSize)
    # Plot smoothed data
    ax1.plot(timeVect, weighDataAv, color=lineColor, linewidth=lineWidth)
    ax1.plot(baselineX, baselineY,'k', ls='--', linewidth=1)
    
    # Plot confidence intervals
    ax1.fill_between(timeVect, weighDataAv-conf95, weighDataAv+conf95, color=lineColor, alpha=0.25)
    
    # Plot significant time points
    for ind, i in enumerate(sigCL):
        ax1.plot(timeVect[i], weighDataAv[i], 'k-', lineWidth=lineWidth*1.5)
        # Plot asterix for signficant clusters
        if sumTvals[ind] >= sigThres:
            xPos = np.average(timeVect[i])
            yPos = np.max(weighDataAv[int(np.mean(i))])+markerOffset+conf95[int(np.mean(i))]
            if yPos >= yMax:
                yPos = yMax-0.5
            ax1.plot(xPos,yPos, 'k*', ms=markerSize)
    
    ax1.set_xlim(timeMin, timeMax-1)
    ax1.set_ylim(yMin, yMax)
    ax1.legend([depVar,'Baseline'],loc=1)
    ax1.set_xlabel(timeVar, fontsize=xLabelSize)
    ax1.set_ylabel(depVar, size=yLabelSize)
    
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
    sTimes, unqT, countT = SF.getKDE(allTimes,timeVect, kernelSize)
    maxT = np.max(sTimes)*8
    ax1_1 = ax1.twinx()
    ax1_1.plot(timeVect, sTimes, '--k', alpha = 0.3)
    ax1_1.bar(unqT, countT, color='k', alpha = 0.3)
    ax1_1.set_ylim(0, maxT)
    ax1_1.legend(['KDE'],loc=4)    
    ax1_1.set_yticks(np.linspace(0,np.max(sTimes),3, dtype=int))
    
    print('Duration:', time.time() - t)



