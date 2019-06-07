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
# To optimize for performance, the section with smoothing and pemrutation
# can be parallelized.

#==============================================================================
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

#==============================================================================
import time
t = time.time()
#==============================================================================
# # Define parameters (Make changes here)
#==============================================================================
# File params
dataFile = 'ExampleDataSMART.p'
timeVar = 'TimeVar1' # The name of the column with he time variables in dataFile
depVar = 'DepVar1' # The name of the column with he depVar variables in dataFile

# Smoothing params
kernelSize = 10
timeMin = 100
timeMax = 501 # Excludes the last value, e.g. 301 returns vect until 300
stepSize = 1

# Permutation params
nPerms = 1000
baseline = .5

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
smData = np.zeros((nPP, len(timeVect)))
smWeights = np.zeros((nPP, len(timeVect)))

# Permutations and permutation weights
permD1 = np.zeros((nPP, len(timeVect), nPerms))
permW1 = np.zeros((nPP, len(timeVect), nPerms))
permD2 = np.zeros((nPP, len(timeVect), nPerms))
permW2 = np.zeros((nPP, len(timeVect), nPerms))

#==============================================================================
# Run smoothing and permutations for each participant
# * This part can be parallelized for speed if required *
#==============================================================================
for i in range(nPP):
    #  Extract data for participant
    times = data[timeVar][i]
    depV = data[depVar][i]

    # Run Smoothing
    smData[i, :], smWeights[i,:] = SF.gaussSmooth(times, depV, timeVect, kernelSize)
    
    # Run Permutations against baseline
    permD1[i,:,:], permW1[i,:,:], permD2[i,:,:], permW2[i,:,:] = SF.permute(
            times, 
            depV, 
            newX=timeVect, 
            sigma=kernelSize, 
            nPerms=nPerms, 
            baseline=baseline,
            binary = True
            )

#==============================================================================
# Run statistics
#==============================================================================
# Weight the data and extract clusters
weighData = SF.weighArraysByColumn(smData, smWeights)
weighDataAv = np.nansum(weighData, axis=0)
sigCL, sumTvals = SF.clusterStat_oneSamp(smData, smWeights, baseline, sigLevel)
# Calculate 95 confidence intervals
conf95 = SF.weighConfOneSample95(smData, smWeights)

# Do cluster stats for the permutations 
permDistr = SF.permuteClusterStat(permD1, permD2, permW1, permW2, sigLevel)
# determine the 95th percentile threshold
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
sTimes, unqT, countT = SF.getKDE(np.hstack(data[timeVar]),timeVect, kernelSize)
maxT = np.max(sTimes)*8
ax1_1 = ax1.twinx()
ax1_1.plot(timeVect, sTimes, '--k', alpha = 0.3)
ax1_1.bar(unqT, countT, color='k', alpha = 0.3)
ax1_1.set_ylim(0, maxT)
ax1_1.legend(['KDE'],loc=4)    
ax1_1.set_yticks(np.linspace(0,np.max(sTimes),3, dtype=int))

print 'Duration:', time.time() - t

                            
                            
                            
                            

