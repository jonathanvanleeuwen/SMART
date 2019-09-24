# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:34:36 2018

@author: Jonathan van Leeuwen
"""
#==============================================================================
#==============================================================================
# # Run the SMART pipeline between conditions (within subject)
#==============================================================================
#==============================================================================
import time
t = time.time()
#==============================================================================
# Assumes that the data is in a pickle file and structered identical to 
# 'ExampleDataSMART.p'.
# 
# A pandas dataframe with:
#       Each participant in its own row 
#       Time var1 in 1 column, all the data in 1 cell
#       Dep. var1 in 1 column, all the data in 1 cell
#       Time var2 in 1 column, all the data in 1 cell # if testing between conditions
#       Dep. var2 in 1 column, all the data in 1 cell # if testing between conditions
#
# Example:
#       Index/pp	            TimeVar1	                    DepVar1	                     TimeVar2                      DepVar2
#           0	    [155 192 279 ..., 143 142 149]	    [0 0 1 ..., 0 1 0]	    [159 163 201 ..., 149 229 154]	    [0 1 0 ..., 1 0 1]
#           1	    [379 312 272 ..., 278 288 267]	    [1 1 1 ..., 0 1 1]	    [386 437 422 ..., 226 319 237]	    [1 1 1 ..., 1 1 0]
#           2	    [192 208 236 ..., 175 268 171]	    [0 0 0 ..., 0 0 0]	    [180 227 189 ..., 172 180 205]	    [1 1 1 ..., 1 1 1]
#           3	    [397 291 412 ..., 457 408 366]	    [1 1 1 ..., 1 1 1]	    [392 452 459 ..., 378 342 444]	    [1 1 0 ..., 1 1 1]

from SMARTClass import SMART

# Settings
fName = 'ExampleDataSMART.p'
depVar1 = 'DepVar1'
timeVar1 ='TimeVar1'
depVar2 = 'DepVar2'
timeVar2 = 'TimeVar2'
krnSize = 10
minTime = 100
maxTime = 501
stepTime = 1
nPerm = 1000
baseline = 0.5
sigLevel = 0.05

#==============================================================================
#==============================================================================
# # # Usage one sample
#==============================================================================
#==============================================================================

#==============================================================================
# Run smoothing
#==============================================================================
oneSamp = SMART(fName, depVar1, timeVar1)
oneSamp.runSmooth(krnSize, minTime, maxTime, stepTime)
oneSamp.runPermutations(nPerm, baseline)
oneSamp.runStats(sigLevel)
oneSamp.runPlot()

##==============================================================================
## Extract data
##==============================================================================
## Smoothing
#oneSamp.smooth_dv1 # Individually smoothed data
#oneSamp.weights_dv1 # Individual weights
#oneSamp.weighDv1Average # Average smothed data
#
## Permutations
#oneSamp.permData1 #Permutations for each participant
#oneSamp.permWeight1 # Permutation weights for each participant. 
#oneSamp.weighPerm1 # Weighted permutations
#
## Stats
#oneSamp.sigCL # Significant timepoints in data
#oneSamp.sumTvals # The sum of tvalues for the each cluster
#oneSamp.permDistr # Clusters in permutations
#oneSamp.sigThres # Cluster size threshold
#   


#==============================================================================
#==============================================================================
# # # Usage two samples (paired)
#==============================================================================
#==============================================================================

#==============================================================================
# Run smoothing
#==============================================================================
pairedSamp = SMART(fName, depVar1, timeVar1, depVar2, timeVar2)
pairedSamp.runSmooth(krnSize, minTime, maxTime, stepTime)
pairedSamp.runPermutations(nPerm)
pairedSamp.runStats(sigLevel)
pairedSamp.runPlot()

#==============================================================================
# Extract data
#==============================================================================
## Smoothing
#pairedSamp.smooth_dv1 # Individually smoothed data variable 1
#pairedSamp.weights_dv1 # Individual weights  variable 1
#pairedSamp.weighDv1Average # Average smothed data variable 1
#pairedSamp.smooth_dv2 # Individually smoothed data variable 2
#pairedSamp.weights_dv2 # Individual weights  variable 2
#pairedSamp.weighDv2Average # Average smothed data variable 2
#
## Permutations
#pairedSamp.permData1 #Permutations for each participant variable 1
#pairedSamp.permWeight1 # Permutation weights for each participant variable 1
#pairedSamp.weighPerm1 # Weighted permutations variable 1
#pairedSamp.permData2 #Permutations for each participant variable 2
#pairedSamp.permWeight2 # Permutation weights for each participant variable 2
#pairedSamp.weighPerm2 # Weighted permutations variable 2
#
## Stats
#pairedSamp.sigCL # Significant timepoints in data
#pairedSamp.sumTvals # The sum of tvalues for the each cluster
#pairedSamp.permDistr # Clusters in permutations
#pairedSamp.sigThres # Cluster size threshold
   
print(time.time() - t)


