# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:53:01 2018

@author: Jonathan van Leeuwen
"""
import numpy as np
import scipy
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import SMART_Funcs as SF


class SMART:
    def __init__(self, fileN, dv1, t1, dv2=None,t2=None):
        # Set defaults
        self.testType= 'bl'
        self.fileN = fileN
        self.dv1 = dv1
        self.t1 = t1
        self.dv2 = dv2
        self.t2 = t2
        if dv2 and t2:
            self.testType='cond'
        # Smoothing inputs
        self.krnSize = None
        self.timeMin = None
        self.timeMax = None
        self.stepSize = None  
        # Smoothing containers
        self.smooth_dv1 = None
        self.weights_dv1 = None
        self.smooth_dv2 = None
        self.weights_dv2 = None
        # Permutation params
        self.nPerms = None
        self.baseline = None
        # Permutation containers
        self.permData1 = None
        self.permWeight1 = None 
        # Stats inputs
        self.sigLevel = 0.05
        self.statTest = scipy.stats.ttest_rel
        # Plotting inputs
        self.lineColor1 = [250,0,0]
        self.lineColor2 = [0,0,255]
        self.lineWidth = 2
        self.markerOffset = 0.75
        self.markerSize = 10
        self.xLabelSize = 20
        self.yLabelSize = 20
        self.yMin = -0.5
        self.yMax = 1.5
        self.histRes = 100 
        # Plotting vars
        self.baselineX = None
        self.baselineY = None
        # Load data 
        self.data = pd.read_pickle(self.fileN)
        self.nPP = len(self.data)        
        
    def runSmooth(self, krnSize, timeMin, timeMax, stepSize):
        self.krnSize = krnSize
        self.timeMin = timeMin
        self.timeMax = timeMax
        self.stepSize = stepSize
        self.timeVect = np.arange(timeMin, timeMax, stepSize, dtype=float)
        self.smooth_dv1 = np.zeros((self.nPP, len(self.timeVect)))
        self.weights_dv1 = np.zeros((self.nPP, len(self.timeVect)))
        if self.testType == 'bl':
            self.__oneSamp__()
        else:
            self.smooth_dv2 = np.zeros((self.nPP, len(self.timeVect)))
            self.weights_dv2 = np.zeros((self.nPP, len(self.timeVect)))
            self.__twoSamp__()
        
    def runPermutations(self, nPerms=1000, baseline=0): 
        self.nPerms = nPerms
        self.baseline = baseline
        self.permData1 = np.zeros((self.nPP, len(self.timeVect), self.nPerms))
        self.permWeight1 = np.zeros((self.nPP, len(self.timeVect), self.nPerms))
        self.permData2 = np.zeros((self.nPP, len(self.timeVect), self.nPerms))
        self.permWeight2 = np.zeros((self.nPP, len(self.timeVect), self.nPerms))
        if self.testType == 'bl':
            self.__oneSampPerm__()
        else:
            self.__twoSampPerm__()

    def runStats(self, sigLevel=0.05):
        self.sigLevel = sigLevel
        if self.testType == 'bl':
            self.__oneSampStats__()
        else:
            self.__twoSampStats__()
        
    def runPlot(self, lineColor1='',lineColor2='',lineWidth='', markerOffset='', 
             markerSize='', xLabelSize='', yLabelSize='',yMin='', yMax='',
             histRes=''):
        # Ploting params
        self.lineColor1 = self.lineColor1 if not lineColor1 else lineColor1
        self.lineColor2 = self.lineColor2 if not lineColor1 else lineColor2
        self.lineWidth = self.lineWidth if not lineWidth else lineWidth
        self.markerOffset = self.markerOffset if not markerOffset else markerOffset
        self.markerSize = self.markerSize if not markerSize else markerSize
        self.xLabelSize = self.xLabelSize if not xLabelSize else xLabelSize
        self.yLabelSize = self.yLabelSize if not yLabelSize else yLabelSize
        self.yMin = self.yMin if not yMin else yMin
        self.yMax = self.yMax if not yMax else yMax
        self.histRes = self.histRes if not histRes else histRes
        # Make plot variables
        self.lineColor1 =([i/255.0 for i in self.lineColor1])
        self.lineColor2 =([i/255.0 for i in self.lineColor2])
        # Initiate plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1)
        plt.suptitle('Smoothing Method for Analysis of Response Time-course: SMART', fontsize=self.xLabelSize)

        if self.testType == 'bl':
            self.baselineX = np.arange(self.timeMin, self.timeMax, 1)
            self.baselineY = np.zeros(len(self.baselineX))+self.baseline
            self.__oneSamplePlot__()
        else:
            self.__twoSamplePlot__()

    #==============================================================================
    # One sample functions
    #==============================================================================
    def __oneSamp__(self):
        # Run one samp smoothing
        for i in range(self.nPP):
            #  Extract data for participant
            depV = self.data[self.dv1][i]
            times = self.data[self.t1][i]
            self.smooth_dv1[i, :], self.weights_dv1[i,:] = SF.gaussSmooth(times, depV, self.timeVect, self.krnSize)    
        # Weigh the data
        self.weighDv1 = SF.weighArraysByColumn(self.smooth_dv1, self.weights_dv1)
        self.weighDv1Average = np.nansum(self.weighDv1, axis=0)
        
    def __oneSampPerm__(self):
        # Run one samp pemrutation
        for i in range(self.nPP):
            #  Extract data for participant            
            depV1 = self.data[self.dv1][i]            
            times1 = self.data[self.t1][i]
            # Run Permutations Between conditions
            self.permData1[i,:,:], self.permWeight1[i,:,:], self.permData2[i,:,:], self.permWeight2[i,:,:] = SF.permute(
                    times1, 
                    depV1, 
                    newX= self.timeVect, 
                    sigma=self.krnSize, 
                    nPerms=self.nPerms, 
                    baseline=self.baseline
                    )
       
    def __oneSampStats__(self):
        # Extract clusters
        self.sigCL, self.sumTvals = SF.clusterStat_oneSamp(self.smooth_dv1, self.weights_dv1, self.baseline, self.sigLevel)

        # Calculate permutation distributions and significance thresholds
        self.weighPerm1 = SF.weighArraysByColumn(self.permData1, self.permWeight1)
        self.permDistr = SF.permuteClusterStat(self.permData1, self.permData2, self.permWeight1, self.permWeight2, self.sigLevel)
        self.sigThres = np.percentile(self.permDistr, 100-(self.sigLevel*100))
        
        # Calculate 95 confidence intervals
        self.conf95 = SF.weighConfOneSample95(self.smooth_dv1, self.weights_dv1)

    def __oneSamplePlot__(self):
        # Plot smoothed data
        self.ax1.plot(self.timeVect, self.weighDv1Average, color=self.lineColor1, linewidth=self.lineWidth)
        self.ax1.plot(self.baselineX, self.baselineY,'k', ls='--', linewidth=1)
        
        # Plot confidence intervals
        self.ax1.fill_between(self.timeVect, 
                              self.weighDv1Average-self.conf95, 
                              self.weighDv1Average+self.conf95, 
                              color=self.lineColor1, 
                              alpha=0.25)

        # Plot significant time points
        for ind, i in enumerate(self.sigCL):
            self.ax1.plot(self.timeVect[i], self.weighDv1Average[i], 'k-', lineWidth=self.lineWidth*1.5)
            # Plot asterix for signficant clusters
            if self.sumTvals[ind] >= self.sigThres:
                xPos = np.average(self.timeVect[i])
                yPos = np.max(self.weighDv1Average[int(np.mean(i))])+self.markerOffset+self.conf95[int(np.mean(i))] 
                if yPos >= self.yMax:
                    yPos = self.yMax-0.5
                self.ax1.plot(xPos,yPos, 'k*', ms=self.markerSize)

        self.ax1.set_xlim(self.timeMin, self.timeMax-1)
        self.ax1.set_ylim(self.yMin, self.yMax)
        self.ax1.legend([self.dv1,'Baseline'],loc=1)
        self.ax1.set_xlabel(self.t1, fontsize=self.xLabelSize)
        self.ax1.set_ylabel(self.dv1, size=self.yLabelSize)
        
        # Plot permutation distributions
        self.ax2.hist(self.permDistr, self.histRes)
        self.ax2.plot([self.sigThres,self.sigThres], self.ax2.get_ylim(), color='r', linewidth=2)
        for idx, tV in enumerate(self.sumTvals):
            self.ax2.plot([tV, tV], self.ax2.get_ylim(), color='k', linewidth=2)
        self.ax2.legend(['Sig Threshold: '+str(self.sigLevel), 'Cluster(s)'],loc=1)
        # Set xaxis
        self.ax2.set_xlabel('Sum of cluster t-values', fontsize=self.xLabelSize)
        self.ax2.set_ylabel("Frequency (log)", size=self.yLabelSize)
        self.ax2.set_yscale('log')

        # Plot kernel density estimation KDE
        sTimes, unqT, countT = SF.getKDE(np.hstack(self.data[self.t1]),self.timeVect, self.krnSize)
        maxT = np.max(sTimes)*8
        self.ax1_1 = self.ax1.twinx()
        self.ax1_1.plot(self.timeVect, sTimes, '--k', alpha = 0.3)
        self.ax1_1.bar(unqT, countT, color='k', alpha = 0.3)
        self.ax1_1.set_ylim(0, maxT)
        self.ax1_1.legend(['KDE'],loc=4)    
        self.ax1_1.set_yticks(np.linspace(0,np.max(sTimes),3, dtype=int))

    #==============================================================================
    # Two sample functions
    #==============================================================================
    def __twoSamp__(self):
        for i in range(self.nPP):
            #  Extract data for participant            
            depV1 = self.data[self.dv1][i]            
            times1 = self.data[self.t1][i]
            depV2 = self.data[self.dv2][i]
            times2 = self.data[self.t2][i]
            # Run Smoothing
            self.smooth_dv1[i, :], self.weights_dv1[i,:] = SF.gaussSmooth(times1, depV1, self.timeVect, self.krnSize)
            self.smooth_dv2[i, :], self.weights_dv2[i,:] = SF.gaussSmooth(times2, depV2, self.timeVect, self.krnSize)

        # Weigh the data 
        self.weighDv1 = SF.weighArraysByColumn(self.smooth_dv1, self.weights_dv1)
        self.weighDv1Average = np.nansum(self.weighDv1, axis=0)
        self.weighDv2 = SF.weighArraysByColumn(self.smooth_dv2, self.weights_dv2)
        self.weighDv2Average = np.nansum(self.weighDv2, axis=0)
    
    def __twoSampPerm__(self):
        for i in range(self.nPP):
            #  Extract data for participant            
            depV1 = self.data[self.dv1][i]            
            times1 = self.data[self.t1][i]
            depV2 = self.data[self.dv2][i]
            times2 = self.data[self.t2][i]

            # Run Permutations Between conditions
            self.permData1[i,:,:], self.permWeight1[i,:,:], self.permData2[i,:,:], self.permWeight2[i,:,:] = SF.permute(times1, depV1, times2, depV2, self.timeVect, self.krnSize, self.nPerms)
        
    def __twoSampStats__(self):
        # Extract clusters
        self.sigCL, self.sumTvals = SF.clusterStat_rel(self.smooth_dv1, self.smooth_dv2, self.weights_dv1, self.weights_dv2, self.sigLevel)
        
        # Calculate permutation distributions and significance thresholds
        # Weighted permutations
        self.permDistr = SF.permuteClusterStat(self.permData1, self.permData2, self.permWeight1, self.permWeight2, self.sigLevel)
        self.sigThres = np.percentile(self.permDistr, 100-(self.sigLevel*100))
        
        # Calculate 95 confidence intervals
        self.conf95 = SF.weighPairedConf95(self.smooth_dv1, self.smooth_dv2, self.weights_dv1, self.weights_dv2)
    
    def __twoSamplePlot__(self):
        # Plot smoothed data
        self.ax1.plot(self.timeVect, self.weighDv1Average, color=self.lineColor1, linewidth=self.lineWidth)
        self.ax1.plot(self.timeVect, self.weighDv2Average, color=self.lineColor2, linewidth=self.lineWidth)
        
        # Plot confidence intervals
        self.ax1.fill_between(self.timeVect, 
                              self.weighDv1Average-self.conf95, 
                              self.weighDv1Average+self.conf95, 
                              color=self.lineColor1, 
                              alpha=0.25)
        self.ax1.fill_between(self.timeVect, 
                              self.weighDv2Average-self.conf95, 
                              self.weighDv2Average+self.conf95, 
                              color=self.lineColor2, 
                              alpha=0.25)

        
        # Plot significant time points
        for ind, i in enumerate(self.sigCL):
            self.ax1.plot(self.timeVect[i], self.weighDv1Average[i], 'k-', lineWidth=self.lineWidth*1.5)
            self.ax1.plot(self.timeVect[i], self.weighDv2Average[i], 'k-', lineWidth=self.lineWidth*1.5)
            # Plot asterix for signficant clusters
            if self.sumTvals[ind] >= self.sigThres:
                xPos = np.average(self.timeVect[i])
                yPos = np.max([self.weighDv1Average[int(np.mean(i))], self.weighDv2Average[int(np.mean(i))]])+self.markerOffset+np.max(self.conf95[int(np.mean(i))])
                if yPos >= self.yMax:
                    yPos = self.yMax-0.5
                self.ax1.plot(xPos,yPos, 'k*', ms=self.markerSize)

        self.ax1.set_xlim(self.timeMin, self.timeMax-1)
        self.ax1.set_ylim(self.yMin, self.yMax)
        self.ax1.legend([self.dv1, self.dv2,'Sig. difference'],loc=1)
        self.ax1.set_xlabel('Time', fontsize=self.xLabelSize)
        self.ax1.set_ylabel('Dep.var', size=self.yLabelSize)
        
        # Plot permutation distributions
        self.ax2.hist(self.permDistr, self.histRes)
        self.ax2.plot([self.sigThres,self.sigThres], self.ax2.get_ylim(), color='r', linewidth=2)
        for idx, tV in enumerate(self.sumTvals):
            self.ax2.plot([tV, tV], self.ax2.get_ylim(), color='k', linewidth=2)
        self.ax2.legend(['Sig Threshold: '+str(self.sigLevel), 'Cluster(s)'],loc=1)
        # Set xaxis
        self.ax2.set_xlabel('Sum of cluster t-values', fontsize=self.xLabelSize)
        self.ax2.set_ylabel("Frequency (log)", size=self.yLabelSize)
        self.ax2.set_yscale('log')

        # Plot kernel density estimation KDE
        sTimes1, unqT1, countT1 = SF.getKDE(np.hstack(self.data[self.t1]),self.timeVect, self.krnSize)
        sTimes2, unqT2, countT2 = SF.getKDE(np.hstack(self.data[self.t2]),self.timeVect, self.krnSize)
        maxT = np.max(np.hstack([sTimes1, sTimes2]))*8
        self.ax1_1 = self.ax1.twinx()
        self.ax1_1.plot(self.timeVect, sTimes1, '--k', alpha = 0.6)
        self.ax1_1.plot(self.timeVect, sTimes2, '-k', alpha = 0.3)
        self.ax1_1.bar(unqT1, countT1, color='k', alpha = 0.6)
        self.ax1_1.bar(unqT2, countT2, color='k', alpha = 0.3)
        self.ax1_1.set_ylim(0, maxT)
        self.ax1_1.legend(['KDE_1', 'KDE_2'],loc=4)    
        self.ax1_1.set_yticks(np.linspace(0,np.max(np.hstack([sTimes1, sTimes2])),3, dtype=int))



    
    
    