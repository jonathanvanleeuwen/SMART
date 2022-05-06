# -*- coding: utf-8 -*-
"""
Created on Thu May 03 13:39:12 2018

@author: Jonathan van Leeuwen
"""

#==============================================================================
#==============================================================================
# # Functions used in van Leeuwen, Smeets & Belopolsky, 2018 (SMART)
#==============================================================================
#==============================================================================
# Required modules
import numpy as np
import scipy
import scipy.stats as st
np.set_printoptions(suppress=True)

#==============================================================================
# Gaussian smoothing function
#==============================================================================
def gaussSmooth(x, y, newX, sigma):
    '''
    Smooths data using a Gaussian kernel. 
    Assumes that x and y are linked, e.g. x[0] and y[0] come from
    the same trial. 
    
    Parameters
    ----------
    x : np.array
        The temporal variable, e.g. reaction time
    y : np.array
        The dependent variable, e.g. performance
    newX : np.array
        The new temporal time points. e.g. RT from 100 ms to 500 ms 
        in 1 ms steps
    sigma : int or float
        The width of the Gaussian kernel
    
    Returns
    -------
    smoothY : np.array
        The smoothed dependent variable as a function of newX
    weights : np.array
        The sum of weights under the Gaussian for each new time point.
        Used for weighted average across participants
    
    Example
    --------
    >>> import numpy as np
    >>> y = np.array(
      [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]
      )
    >>> x = np.array(
      [416, 359, 327, 363, 324, 267, 460, 348, 201, 475, 282, 390, 314,
       365, 385, 291, 285, 405, 306, 336, 357, 370, 445, 327, 331, 403,
       288, 373, 465, 270, 471, 349, 304, 292, 275, 233, 357, 313, 405,
       336, 367, 344, 333, 357, 369, 242, 314, 321, 384, 437, 484, 441,
       282, 410, 414, 361, 280, 388, 261, 360, 382, 306, 432, 337, 361,
       360, 415, 316, 357, 398, 323, 317, 399, 292, 287, 385, 383, 267,
       277, 303, 292, 327, 285, 402, 263, 340, 385, 319, 370, 314, 277,
       282, 336, 288, 344, 337, 395, 372, 244, 251, 284, 296, 287, 343,
       433, 357, 250, 220, 307, 313, 405, 421, 326, 287]
    )
    >>> newX = np.arange(100,501,1)
    >>> sigma = 10
    >>> smoothY, weights = gaussSmooth(x, y, newX, sigma)
    >>> print smoothY
        [ 0.          0.          0.          0.          0.          0.
      0.00000001  0.00000001  0.00000001  0.00000001  0.00000001  0.00000001
      0.00000001  0.00000001  0.00000001  0.00000001  0.00000002  0.00000002
      0.00000002  0.00000002  0.00000004  0.00000002  0.00000004  0.00000004
      0.00000004  0.00000004  0.00000011  0.00000011  0.00000011  0.00000011
      0.00000031  0.00000031  0.00000031  0.00000083  0.00000083  0.00000031
      0.00000083  0.00000083  0.00000083  0.00000226  0.00000226  0.00000083
      0.00000226  0.00000226  0.00000614  0.00000226  0.00000614  0.00000614
      0.0000167   0.00000614  0.0000167   0.0000167   0.0000167   0.0000167
      0.0000454   0.0000167   0.0000454   0.0000454   0.0000454   ....]
    >>> print weights
    [  0.           0.           0.           0.           0.           0.           0.
       0.           0.           0.           0.           0.           0.           0.
       0.           0.           0.           0.           0.           0.           0.
       0.           0.           0.           0.           0.           0.           0.
       0.           0.           0.           0.           0.           0.           0.
       0.           0.           0.           0.           0.           0.00000001
       0.00000002   0.00000002   0.00000004   0.00000004   0.00000011
       0.00000011   0.00000031   0.00000031   0.00000083   0.00000083
       0.00000226   0.00000226   0.00000614   0.00000614   0.0000167
       0.0000167    0.0000454    0.0000454    0.00012342   ....]
    '''  
    delta_x = newX[:, None] - x      
    # Calculat weights
    weights = np.exp(-delta_x*delta_x / (2*sigma*sigma))  
    dataWeight = np.nansum(weights, axis= 1)
    weights /= np.nansum(weights, axis=1, keepdims=True)
    y_eval = np.dot(weights, y)
    
    # Serial, similair to equation (very slow)
    #y_eval  = []
    #weights = []
    #for i in newX:
    #    wTemp = []
    #    for n in range(len(x)):
    #        wTemp.append(np.exp( -(x[n] - i)**2/(2.*sigma**2) ))
    #    wTemp = np.array(wTemp)
    #    weights.append(np.nansum(wTemp))
    #    #wTemp /= np.nansum(wTemp)
    #    y_eval.append(np.nansum(y*wTemp))
    #y_eval = np.array(y_eval)
    #dataWeight = np.array(weights)
    
    return y_eval, dataWeight


def getKDE(x, newX, sigma):
    """ 
    Kernel density estimation using a Gaussian kernel
    
    Parameters
    ----------
    x : np.array with ints
        The vector on which to run the KDE. Discrete values 
    newX : np.array with ints
        The new temporal time points. e.g. RT from 100 ms to 500 ms 
    sigma : int or float
        The width of the Gaussian kernel
    
    Returns
    -------
    KDE : np.array
        The Kernel density estimate for the range given in newX
    uniqueX : np.array with int
        The unique values in X
    countsX : np.array with ints
        The number of unique occurences in X

    Example
    --------
    >>> import numpy as np
    >>> x = np.random.randint(0,500,10000)
    >>> newX =  np.arange(500)
    >>> sigma = 10
    >>> 
    >>> KDE, unqX, countsX = getKDE(x, newX, sigma)  
    >>> 
    >>> plt.plot(newX, KDE)
    >>> plt.bar(unqX, countsX)
    """
    # Get counts for each unique value and return ints
    unqX, countsX = np.unique(x, return_counts=True)
    unqX = np.array(unqX, dtype=int)
    
    # Make the vector on which we want to make the KDE
    x = np.arange(int(x.min()-50), int(x.max()+50))
    y = np.zeros(x.shape)
    indices = np.where(np.in1d(x, unqX))[0]
    # Make new vector with the counts
    y[np.array(indices, dtype=int)] = countsX

    # Run the KDE
    delta_x = newX[:, None] - x
    weights = np.exp(-delta_x*delta_x / (2*sigma*sigma)) / (np.sqrt(2*np.pi) * sigma)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return np.dot(weights, y), unqX, countsX

#==============================================================================
# Permutation procedure
#==============================================================================
def heavyside(arr, cutOff = 0.5):
    """ 
    Heavyside filter: returns a vector of binary (0,1) data. 
    filters the data in "arr", all values smaller than "cutOff"
    are returned as 0 and all values equal to or larger are 
    returned as 1.
    
    Parameters
    ----------
    arr : np.array
        The on which to apply the filter 
    cutOff : float, default = 0.5
        The center point for the filter, all data smaller gets set to 0.
        All data equal or larger gets set to 1.
    
    Returns
    -------
    res : np.array
        The heavyfiltered data, containing binary data

    Example
    --------
    >>> import numpy as np
    >>> arr = np.random.normal(0.5, 0.1,10)
    >>> res = heavyside(arr)
    >>> arr
    array([ 0.48686906,  0.59942521,  0.51029602,  0.57444646,  0.57035842,
            0.56056907,  0.51168252,  0.54169111,  0.34370254,  0.67485142])
    >>> res
    array([ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.])
    """
    res = np.zeros(arr.shape)
    res[arr >= cutOff] = 1
    return res

def makeBaseline(x, baseline=0.0, binary=False):
    """ 
    Creates a data vector of data drawn from a gaussian distribution centered
    on the value in "baseline" with the STDV of the Gaussian identical to the
    STDV of the data in "x". 
    
    Parameters
    ----------
    x : np.array
        The data from which the STDV is aquired for making the baseline data
    baseline : float, default = 0.0
        The baseline value the data is centered on
    binary : Bool, Default False
        True if data should be returned as a boolean vector. If set to true
        the baseline data is heavyside filtered
    
    Returns
    -------
    base : np.array
        The data contining the baseline data

    Example
    --------
    >>> import numpy as np
    >>> arr = [ 0.48686906,  0.59942521,  0.51029602,  0.57444646,  0.57035842,
            0.56056907,  0.51168252,  0.54169111,  0.34370254,  0.67485142]
    >>> base = makeBaseline(arr, 0.5)
    >>> base
    array([ 0.44619784,  0.47692352,  0.61063693,  0.45425037,  0.48101882,
            0.50937955,  0.55339372,  0.47917587,  0.60395786,  0.63011139])
    """
    base = np.random.normal(baseline, np.std(x), x.shape)
    if binary:
        base = heavyside(base, 0.5)
    return base

def permute(x1, y1, x2=None, y2=None, newX=[None], sigma=20, nPerms=1000, baseline=None, binary=False, noise = False):
    '''
    Permutes two conditions and smoothes each permutation. 
    The function returns the permutated data for both x1 and x2 and their
    weights. 
    
    Parameters
    ----------
    x1 : np.array
        The temporal variable, e.g. reaction time for condition 1 
    y1 : np.array
        The dependent variable, e.g. performance for condition 1
    x2 : np.array
        The temporal variable, e.g. reaction time for condition 2
    y2 : np.array
        The dependent variable, e.g. performance for condition 2
    newX : np.array
        The new temporal time points. e.g. RT from 100 ms to 500 ms 
        in 1 ms steps
    sigma : float or int
        The width of the Gaussian kernel 
    nPerms : int
        The number of permutations to run
    baseline : float or int
        Used when testing vs. baseline
    
    Returns
    -------
    pData1 : 2d np.array
        The smoothed permutated dependent variable for condition 1
        Dimension 1 =  The temporal order of the smoothed data
        Dimension 2 = Each permutation
    pWeights1 : 2d np.array
        The weights for the smoothed permutated dependent variable 
        for condition 1
        Dimension 1 =  The temporal order of the smoothed data
        Dimension 2 = Each permutation
    pData2 : 2d np.array
        The smoothed permutated dependent variable for condition 2
        Dimension 1 =  The temporal order of the smoothed data
        Dimension 2 = Each permutation
    pWeights2 : 2d np.array
        The weights for the smoothed permutated dependent variable 
        for condition 2
        Dimension 1 =  The temporal order of the smoothed data
        Dimension 2 = Each permutation

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import SMART_Funcs as SF
    >>> 
    >>> data = pd.read_pickle("dataFile.p")
    >>> nPP = 10
    >>> newX = np.arange(100,501,1)
    >>> nPerms = 1000
    >>> sigma = 10
    >>> 
    >>> pData1 = np.zeros((nPP, len(timeVect), nPerms))
    >>> pWeights1 = np.zeros((nPP, len(timeVect), nPerms))
    >>> pData2 = np.zeros((nPP, len(timeVect), nPerms))
    >>> pWeights2 = np.zeros((nPP, len(timeVect), nPerms))
    >>> for i in range(nPP):
            #  Extract data for participant
            x1 = data[timeVar1][i]
            y1 = data[depVar1][i]
            x2 = data[timeVar2][i]
            y2 = data[depVar2][i]
            # Run Permutations Between conditions
            pData1[i,:,:], pWeights1[i,:,:], pData2[i,:,:], pWeights2[i,:,:] = SF.twoSamplePerm(x1, y1, x2, y2, newX, sigma, nPerms)
    >>> 
    >>> print pData1[0,:,:]
    array([[ 0.5085338 ,  0.51009754,  0.66602991, ...,  0.83174216,
             0.82077497,  0.5090912 ],
           [ 0.5146783 ,  0.51544983,  0.65095297, ...,  0.82492759,
             0.81248825,  0.51577194],
           [ 0.52098926,  0.5208732 ,  0.63551502, ...,  0.818105  ,
             0.80405216,  0.52257607],
           ..., 
           [ 1.        ,  1.        ,  1.        , ...,  0.00000137,
             1.        ,  1.        ],
           [ 1.        ,  1.        ,  1.        , ...,  0.00000129,
             1.        ,  1.        ],
           [ 1.        ,  1.        ,  1.        , ...,  0.00000122,
             1.        ,  1.        ]])
    >>> print pWeights1[0,:,:]
    array([[ 0.04983031,  0.04792607,  0.13463432, ...,  0.12180662,
             0.1136124 ,  0.04234923],
           [ 0.06731265,  0.06452391,  0.17112221, ...,  0.15483881,
             0.14340635,  0.05687515],
           [ 0.0901383 ,  0.0861008 ,  0.21566799, ...,  0.1952889 ,
             0.17949341,  0.07572002],
           ..., 
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ]])
    >>> print pData2[0,:,:]
    array([[ 0.6654956 ,  0.66291209,  0.52938495, ...,  0.26911986,
             0.3432363 ,  0.65752469],
           [ 0.65423218,  0.65185384,  0.54094658, ...,  0.27414015,
             0.3506019 ,  0.64633063],
           [ 0.64293158,  0.640866  ,  0.5525787 , ...,  0.27937073,
             0.35810055,  0.63523019],
           ..., 
           [ 1.        ,  1.        ,  1.        , ...,  1.        ,
             1.        ,  0.00000137],
           [ 1.        ,  1.        ,  1.        , ...,  1.        ,
             1.        ,  0.00000129],
           [ 1.        ,  1.        ,  1.        , ...,  1.        ,
             1.        ,  0.00000122]])
    >>> print pWeights2[0,:,:]
    array([[ 0.14279646,  0.14470069,  0.05799245, ...,  0.07082014,
             0.07901437,  0.15027754],
           [ 0.18177706,  0.18456579,  0.07796749, ...,  0.0942509 ,
             0.10568335,  0.19221456],
           [ 0.22947944,  0.23351694,  0.10394975, ...,  0.12432884,
             0.14012433,  0.24389772],
           ..., 
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ]])
    ''' 
    # Input handeling
    if newX[0]==None:
        newX = np.arange(100,501)
    if baseline!=None:
        x2 = x1.copy() 
        if noise:
            y2 = makeBaseline(y1, baseline, binary)
        else:
            y2 = np.zeros(y1.shape)+baseline
        
    # Merge vectors
    data = np.hstack((x1, x2))
    depVar = np.hstack((y1, y2))
    idx = np.arange(len(depVar))

    #Prealocate data storage
    pData1 = np.zeros((nPerms,len(newX)))
    pData2 = np.copy(pData1)
    pWeights1 = np.copy(pData1)
    pWeights2 = np.copy(pData1)

    # Run the permutation testing
    for perm in range(nPerms):
        # Here we shuffle the conditions
        np.random.shuffle(idx)
        # Split the indexes
        pIdx1 = idx[0:len(x1)]
        pIdx2 = idx[len(x1):]

        # Extract permutated data
        # Cond1
        pX1 = data[pIdx1]
        pY1 = depVar[pIdx1]
        #Cond2
        pX2 = data[pIdx2]
        pY2 = depVar[pIdx2]

        ####
        # Do the actual gaussian smoothing
        # Make gaussians Cond1
        pData1[perm,:], pWeights1[perm,:] = gaussSmooth(pX1, pY1, newX, sigma)
        # Make gaussians Cond2
        pData2[perm,:], pWeights2[perm,:] = gaussSmooth(pX2, pY2, newX, sigma)

    return pData1.T, pWeights1.T, pData2.T, pWeights2.T

#==============================================================================
#==============================================================================
# # Confidence intervals
#==============================================================================
#==============================================================================
def weighSEMOneSample(performance, weights):
    """
    Calculates the weighted SEM for one sample data along axis 0. 
    
    Parameters
    ----------
    performance : 2d np.array
        The dependent variable data 
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    weights : 2d np.array
        The weights for the dependent variable data 
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    
    Returns
    -------
    confInt95 : np.array
        The 95% confidence interval for each time point. When plotting this
        add it and subtract it from the weighted mean. 
    
    Example
    --------
    >>> import numpy as np
    >>> import SMART_Funcs as SF
    >>> 
    >>> cond = np.array(
          [[ 0.22943939,  0.23401153,  0.23875737,  0.24369002,  0.24882374,
             0.25417401,  0.25975754,  0.26559225,  0.27169717,  0.27809231],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.95921204,  0.95660974,  0.95386013,  0.95095699,  0.94789409,
             0.94466517,  0.94126402,  0.93768446,  0.93392041,  0.92996589],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.0050039 ,  0.00575406,  0.00661525,  0.0076034 ,  0.00873658,
             0.01003521,  0.01152231,  0.01322375,  0.0151685 ,  0.01738882]]
           )
    >>> weights = np.array(
          [[ 0.08644327,  0.11422464,  0.14951155,  0.19386571,  0.24903863,
             0.31695961,  0.39971352,  0.499508  ,  0.61863016,  0.75939376],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.0003915 ,  0.0005859 ,  0.00086848,  0.00127506,  0.0018542 ,
             0.00267079,  0.00381058,  0.00538539,  0.00753925,  0.0104552 ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00845718,  0.01151475,  0.01552811,  0.02074147,  0.02744353,
             0.03597045,  0.04670741,  0.06008859,  0.07659494,  0.09674976]]
           )
    >>> weightedSEM = SF.weighSEMOneSample(cond, weights)
    >>> print weightedSEM
    [ 0.02702306  0.02806442  0.02913639  0.03023952  0.03137421  0.03254093
      0.0337401   0.03497213  0.03623726  0.03753552]
    """
    Pi = performance
    NP = len(Pi) - np.sum(np.isnan(Pi), axis = 0)
    WN = weights/np.sum(weights, axis = 0)
    Pw = np.nansum(WN*Pi, axis = 0)
    SEMw = np.sqrt( (NP/(NP-1.)) * np.nansum(  (WN * (Pi - Pw))**2  ,axis = 0) )    
    return SEMw

def weighSEMPaired_new(Pa, Wa, Pb, Wb):
    """
    Calculates the weighted SEM for paired samples data along axis 0.
    
    Parameters
    ----------
    Pa : 2d np.array
        The dependent variable data for condition A
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    Wa : 2d np.array
        The weights for the dependent variable data for condition A
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    Pb : 2d np.array
        The dependent variable data for condition B
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    Wb : 2d np.array
        The weights for the dependent variable data for condition B
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    
    Returns
    -------
    SEM : np.array
        The Weighted SEMfor each time point. 
    
    Example
    --------
    >>> import numpy as np
    >>> import SMART_Funcs as SF
    >>> 
    >>> cond1 = np.array(
          [[ 0.22943939,  0.23401153,  0.23875737,  0.24369002,  0.24882374,
             0.25417401,  0.25975754,  0.26559225,  0.27169717,  0.27809231],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.95921204,  0.95660974,  0.95386013,  0.95095699,  0.94789409,
             0.94466517,  0.94126402,  0.93768446,  0.93392041,  0.92996589],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.0050039 ,  0.00575406,  0.00661525,  0.0076034 ,  0.00873658,
             0.01003521,  0.01152231,  0.01322375,  0.0151685 ,  0.01738882]]
           )
    >>> cond2 = np.array(
          [[ 0.94682624,  0.94048727,  0.93355617,  0.92602112,  0.91788159,
             0.90915058,  0.89985671,  0.89004574,  0.87978133,  0.8691449 ],
           [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
             1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
           [ 0.00000366,  0.00000488,  0.00000652,  0.0000087 ,  0.00001161,
             0.0000155 ,  0.00002071,  0.00002767,  0.000037  ,  0.00004947],
           [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
             1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
           [ 0.99999866,  0.99999827,  0.99999776,  0.9999971 ,  0.99999624,
             0.99999513,  0.9999937 ,  0.99999185,  0.99998946,  0.99998637]]
           )
    >>> weights1 = np.array(
          [[ 0.08644327,  0.11422464,  0.14951155,  0.19386571,  0.24903863,
             0.31695961,  0.39971352,  0.499508  ,  0.61863016,  0.75939376],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.0003915 ,  0.0005859 ,  0.00086848,  0.00127506,  0.0018542 ,
             0.00267079,  0.00381058,  0.00538539,  0.00753925,  0.0104552 ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00845718,  0.01151475,  0.01552811,  0.02074147,  0.02744353,
             0.03597045,  0.04670741,  0.06008859,  0.07659494,  0.09674976]]
           )
    >>> weights2 = np.array(
          [[ 0.1061835 ,  0.13486507,  0.17010619,  0.21313444,  0.26536263,
             0.32841358,  0.40414825,  0.49469664,  0.6024907 ,  0.73029753],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00308873,  0.00431787,  0.00597608,  0.0081888 ,  0.01110918,
             0.01492111,  0.01984166,  0.02612241,  0.03404919,  0.04393993],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00050288,  0.00074004,  0.00107842,  0.00155623,  0.00222394,
             0.00314738,  0.00441136,  0.00612366,  0.00841953,  0.0114664 ]]
           )
    >>> weightedSEM = SF.weighSEMPaired(cond1, cond2, weights1, weights2)
    >>> print weightedSEM
    [ 0.02328062  0.02607645  0.02907438  0.03225705  0.0356039   0.03909004
      0.04268792  0.04636779  0.05009915  0.05385182]
    """
    # Determine N for each sample
    Na = len(Pa) - np.sum(np.isnan(Pa), axis = 0)
    Nb = len(Pb) - np.sum(np.isnan(Pb), axis = 0)
    
    # Get the number of non-nan values
    if np.array(Na).shape:
        Na[Nb<Na] = Nb[Nb<Na]
    else:
        Na = np.min([Na, Nb])
    NP = Na
    
    # Normalize weights
    WaN = Wa/np.sum(Wa, axis = 0)
    WbN = Wb/np.sum(Wb, axis = 0)
    
    # Get difference score and average difference score
    dP = Pa - Pb
    dPw = np.nansum( (WaN*Pa) - (WbN*Pb), axis = 0)
    
    # Calculate SEM
    SEMw = np.sqrt( (NP/(NP-1.)) * np.nansum( WaN * WbN *(dP - dPw )**2,axis = 0) )
    return SEMw

def weighConfOneSample95(cond, weights):
    '''
    Calculates the weighted 95% confidence interval for one sample data
    along axis zero. Used to calculate the 95% confidence over time for 
    the smoothed data. 
    
    Parameters
    ----------
    cond : 2d np.array
        The dependent variable data 
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    weights : 2d np.array
        The weights for the dependent variable data 
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    
    Returns
    -------
    confInt95 : np.array
        The 95% confidence interval for each time point. When plotting this
        add it and subtract it from the weighted mean. 
    
    Example
    --------
    >>> import numpy as np
    >>> import SMART_Funcs as SF
    >>> 
    >>> cond = np.array(
          [[ 0.22943939,  0.23401153,  0.23875737,  0.24369002,  0.24882374,
             0.25417401,  0.25975754,  0.26559225,  0.27169717,  0.27809231],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.95921204,  0.95660974,  0.95386013,  0.95095699,  0.94789409,
             0.94466517,  0.94126402,  0.93768446,  0.93392041,  0.92996589],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.0050039 ,  0.00575406,  0.00661525,  0.0076034 ,  0.00873658,
             0.01003521,  0.01152231,  0.01322375,  0.0151685 ,  0.01738882]]
           )
    >>> weights = np.array(
          [[ 0.08644327,  0.11422464,  0.14951155,  0.19386571,  0.24903863,
             0.31695961,  0.39971352,  0.499508  ,  0.61863016,  0.75939376],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.0003915 ,  0.0005859 ,  0.00086848,  0.00127506,  0.0018542 ,
             0.00267079,  0.00381058,  0.00538539,  0.00753925,  0.0104552 ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00845718,  0.01151475,  0.01552811,  0.02074147,  0.02744353,
             0.03597045,  0.04670741,  0.06008859,  0.07659494,  0.09674976]]
           )
    >>> confInt95 = SF.weighConfOneSample95(cond, weights)
    >>> print confInt95
    [ 0.07502806  0.07791933  0.08089559  0.08395838  0.08710877  0.09034811
      0.09367754  0.09709821  0.10061076  0.10421532]
    '''            
    # Number of non nan values per time point
    N = len(cond) - np.sum(np.isnan(cond), axis = 0) 
    t_val = scipy.stats.t.ppf(0.975,N-1)
    return t_val * weighSEMOneSample(cond, weights)

def weighPairedConf95(cond1, cond2, weights1, weights2):
    '''
    Calculates the weighted 95% confidence interval between 
    two paired samples along axis zero. Used to calculate
    the 95% confidence over time for the smoothed data. 
    
    Parameters
    ----------
    cond1 : 2d np.array
        The dependent variable data for condition 1
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    cond2 : 2d np.array
        The dependent variable data for condition 2
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    weights1 : 2d np.array
        The weights for the dependent variable data for condition 1
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    weights2 : 2d np.array
        The weights for the dependent variable data for condition 2
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    
    Returns
    -------
    confInt95 : np.array
        The 95% confidence interval for each time point. When plotting this
        add it and subtract it from the weighted mean. 
    
    Example
    --------
    >>> import numpy as np
    >>> import SMART_Funcs as SF
    >>> 
    >>> cond1 = np.array(
          [[ 0.22943939,  0.23401153,  0.23875737,  0.24369002,  0.24882374,
             0.25417401,  0.25975754,  0.26559225,  0.27169717,  0.27809231],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.95921204,  0.95660974,  0.95386013,  0.95095699,  0.94789409,
             0.94466517,  0.94126402,  0.93768446,  0.93392041,  0.92996589],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.0050039 ,  0.00575406,  0.00661525,  0.0076034 ,  0.00873658,
             0.01003521,  0.01152231,  0.01322375,  0.0151685 ,  0.01738882]]
           )
    >>> cond2 = np.array(
          [[ 0.94682624,  0.94048727,  0.93355617,  0.92602112,  0.91788159,
             0.90915058,  0.89985671,  0.89004574,  0.87978133,  0.8691449 ],
           [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
             1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
           [ 0.00000366,  0.00000488,  0.00000652,  0.0000087 ,  0.00001161,
             0.0000155 ,  0.00002071,  0.00002767,  0.000037  ,  0.00004947],
           [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
             1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
           [ 0.99999866,  0.99999827,  0.99999776,  0.9999971 ,  0.99999624,
             0.99999513,  0.9999937 ,  0.99999185,  0.99998946,  0.99998637]]
           )
    >>> weights1 = np.array(
          [[ 0.08644327,  0.11422464,  0.14951155,  0.19386571,  0.24903863,
             0.31695961,  0.39971352,  0.499508  ,  0.61863016,  0.75939376],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.0003915 ,  0.0005859 ,  0.00086848,  0.00127506,  0.0018542 ,
             0.00267079,  0.00381058,  0.00538539,  0.00753925,  0.0104552 ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00845718,  0.01151475,  0.01552811,  0.02074147,  0.02744353,
             0.03597045,  0.04670741,  0.06008859,  0.07659494,  0.09674976]]
           )
    >>> weights2 = np.array(
          [[ 0.1061835 ,  0.13486507,  0.17010619,  0.21313444,  0.26536263,
             0.32841358,  0.40414825,  0.49469664,  0.6024907 ,  0.73029753],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00308873,  0.00431787,  0.00597608,  0.0081888 ,  0.01110918,
             0.01492111,  0.01984166,  0.02612241,  0.03404919,  0.04393993],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00050288,  0.00074004,  0.00107842,  0.00155623,  0.00222394,
             0.00314738,  0.00441136,  0.00612366,  0.00841953,  0.0114664 ]]
           )
    >>> confInt95 = SF.weighPairedConf95(cond1, cond2, weights1, weights2)
    >>> print confInt95
    [ 0.03231868  0.03619992  0.04036171  0.04477996  0.04942614  0.05426568
      0.05926033  0.06436881  0.06954877  0.07475831]
    '''   
    # Determine N
    N = len(cond1) - np.sum(np.isnan(cond1), axis = 0)
    N2 = len(cond2) - np.sum(np.isnan(cond2), axis = 0)
    if np.array(N).shape:
        N[N2<N] = N2[N2<N]
    else:
        N = np.min([N, N2])
        
    # Weighted SEM)       
    sem = weighSEMPaired_new(cond1, weights1, cond2, weights2)
    t_val = scipy.stats.t.ppf(0.975,N-1)
    confInt95 = (t_val * sem)/2.
    
    return confInt95

#==============================================================================
# Functions for running statistics
#==============================================================================
def weighted_ttest(cond, weights, baseline=0):
    '''
    Perform a weighted one sampled t-test along axis 0. 
    
    Parameters
    ----------
    cond : 2d np.array
        The dependent variable data 
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    weights : 2d np.array
        The weights for the dependent variable data 
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    baseline : float or int
        The baseline to test the data against
        
    Returns
    -------
    tvals : np.array
        The t-value for each time point
    pvals : np.array
        The p-value for each time point
        
    Example
    --------
    >>> import numpy as np
    >>> import SMART_Funcs as SF
    >>> 
    >>> baseline = 0.5
    >>> cond = np.array(
          [[ 0.22943939,  0.23401153,  0.23875737,  0.24369002,  0.24882374,
             0.25417401,  0.25975754,  0.26559225,  0.27169717,  0.27809231],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.95921204,  0.95660974,  0.95386013,  0.95095699,  0.94789409,
             0.94466517,  0.94126402,  0.93768446,  0.93392041,  0.92996589],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.0050039 ,  0.00575406,  0.00661525,  0.0076034 ,  0.00873658,
             0.01003521,  0.01152231,  0.01322375,  0.0151685 ,  0.01738882]]
           )
    >>> weights = np.array(
          [[ 0.08644327,  0.11422464,  0.14951155,  0.19386571,  0.24903863,
             0.31695961,  0.39971352,  0.499508  ,  0.61863016,  0.75939376],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.0003915 ,  0.0005859 ,  0.00086848,  0.00127506,  0.0018542 ,
             0.00267079,  0.00381058,  0.00538539,  0.00753925,  0.0104552 ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00845718,  0.01151475,  0.01552811,  0.02074147,  0.02744353,
             0.03597045,  0.04670741,  0.06008859,  0.07659494,  0.09674976]]
           )
    >>> tvals, pvals = SF.weighted_ttest(cond, weights1, baseline)
    >>> print tvals
    [-10.63835989 -10.09972889  -9.58342926  -9.08795128  -8.61189737
      -8.15390273  -7.71267951  -7.287002    -6.87573553  -6.47783182]
    >>> print pvals
    [ 0.00044207  0.00054081  0.00066249  0.00081287  0.00099929  0.00123126
      0.00152111  0.00188498  0.00234415  0.00292687]
    ''' 
    N = len(cond) - np.sum(np.isnan(cond), axis = 0)
    weighAv = np.average(cond, axis = 0, weights = weights)
    tvals = (weighAv-baseline)/weighSEMOneSample(cond, weights)
    pvals = scipy.stats.t.sf(np.abs(tvals), N-1)*2
    return tvals, pvals


def weighted_ttest_rel(cond1, cond2, weights1, weights2):
    '''
    Perform a weighted paired sampled t-test along the zero axis. 
    
    Parameters
    ----------
    cond1 : 2d np.array
        The dependent variable data for condition 1
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    cond2 : 2d np.array
        The dependent variable data for condition 2
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    weights1 : 2d np.array
        The weights for the dependent variable data for condition 1
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    weights2 : 2d np.array
        The weights for the dependent variable data for condition 2
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    
    Returns
    -------
    tvals : np.array
        The t-value for each time point
    pvals : np.array
        The p-value for each time point
        
    Example
    --------
    >>> import numpy as np
    >>> import SMART_Funcs as SF
    >>> 
    >>> cond1 = np.array(
          [[ 0.22943939,  0.23401153,  0.23875737,  0.24369002,  0.24882374,
             0.25417401,  0.25975754,  0.26559225,  0.27169717,  0.27809231],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.95921204,  0.95660974,  0.95386013,  0.95095699,  0.94789409,
             0.94466517,  0.94126402,  0.93768446,  0.93392041,  0.92996589],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.0050039 ,  0.00575406,  0.00661525,  0.0076034 ,  0.00873658,
             0.01003521,  0.01152231,  0.01322375,  0.0151685 ,  0.01738882]]
           )
    >>> cond2 = np.array(
          [[ 0.94682624,  0.94048727,  0.93355617,  0.92602112,  0.91788159,
             0.90915058,  0.89985671,  0.89004574,  0.87978133,  0.8691449 ],
           [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
             1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
           [ 0.00000366,  0.00000488,  0.00000652,  0.0000087 ,  0.00001161,
             0.0000155 ,  0.00002071,  0.00002767,  0.000037  ,  0.00004947],
           [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
             1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
           [ 0.99999866,  0.99999827,  0.99999776,  0.9999971 ,  0.99999624,
             0.99999513,  0.9999937 ,  0.99999185,  0.99998946,  0.99998637]]
           )
    >>> weights1 = np.array(
          [[ 0.08644327,  0.11422464,  0.14951155,  0.19386571,  0.24903863,
             0.31695961,  0.39971352,  0.499508  ,  0.61863016,  0.75939376],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.0003915 ,  0.0005859 ,  0.00086848,  0.00127506,  0.0018542 ,
             0.00267079,  0.00381058,  0.00538539,  0.00753925,  0.0104552 ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00845718,  0.01151475,  0.01552811,  0.02074147,  0.02744353,
             0.03597045,  0.04670741,  0.06008859,  0.07659494,  0.09674976]]
           )
    >>> weights2 = np.array(
          [[ 0.1061835 ,  0.13486507,  0.17010619,  0.21313444,  0.26536263,
             0.32841358,  0.40414825,  0.49469664,  0.6024907 ,  0.73029753],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00308873,  0.00431787,  0.00597608,  0.0081888 ,  0.01110918,
             0.01492111,  0.01984166,  0.02612241,  0.03404919,  0.04393993],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00050288,  0.00074004,  0.00107842,  0.00155623,  0.00222394,
             0.00314738,  0.00441136,  0.00612366,  0.00841953,  0.0114664 ]]
           )
    >>> tvals, pvals = SF.weighted_ttest_rel(cond1, cond2, weights1, weights2)
    >>> print tvals
    [-30.40771082 -26.66094401 -23.44662594 -20.68789375 -18.31655524
     -16.27421729 -14.51085133 -12.98410269 -11.65816583 -10.50288769]
    >>> print pvals
    [ 0.00000697  0.00001176  0.00001961  0.00003225  0.00005226  0.00008343
      0.00013115  0.00020301  0.00030947  0.00046464]
    ''' 
    # get average difference
    weighAv = np.average(cond1,axis=0, weights=weights1) - np.average(cond2,axis=0, weights=weights2)

    # Determine N
    N = len(cond1) - np.sum(np.isnan(cond1), axis = 0)
    N2 = len(cond2) - np.sum(np.isnan(cond2), axis = 0)
    if np.array(N).shape:
        N[N2<N] = N2[N2<N]
    else:
        N = np.min([N, N2])
        
    # Weighted SEM          
    sem = weighSEMPaired_new(cond1, weights1, cond2, weights2)
    tvals = weighAv/sem
    pvals = scipy.stats.t.sf(np.abs(tvals), N-1)*2

    return tvals, pvals

def weighted_ttest_ind(cond1, cond2, weights1, weights2):
    '''
    Perform an independent sample t-test along the zero axis. Assumes equal variance and same N 
    Note: this function is in development and has not been tested with a 
    boostrapping procedure 
    
    Parameters
    ----------
    cond1 : 2d np.array
        The dependent variable data for condition 1
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    cond2 : 2d np.array
        The dependent variable data for condition 2
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    weights1 : 2d np.array
        The weights for the dependent variable data for condition 1
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    weights2 : 2d np.array
        The weights for the dependent variable data for condition 2
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    
    Returns
    -------
    tvals : np.array
        The t-value for each time point
    pvals : np.array
        The p-value for each time point
        
    Example
    --------
    >>> import numpy as np
    >>> import SMART_Funcs as SF
    >>> 
    >>> cond1 = np.array(
          [[ 0.22943939,  0.23401153,  0.23875737,  0.24369002,  0.24882374,
             0.25417401,  0.25975754,  0.26559225,  0.27169717,  0.27809231],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.95921204,  0.95660974,  0.95386013,  0.95095699,  0.94789409,
             0.94466517,  0.94126402,  0.93768446,  0.93392041,  0.92996589],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.0050039 ,  0.00575406,  0.00661525,  0.0076034 ,  0.00873658,
             0.01003521,  0.01152231,  0.01322375,  0.0151685 ,  0.01738882]]
           )
    >>> cond2 = np.array(
          [[ 0.94682624,  0.94048727,  0.93355617,  0.92602112,  0.91788159,
             0.90915058,  0.89985671,  0.89004574,  0.87978133,  0.8691449 ],
           [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
             1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
           [ 0.00000366,  0.00000488,  0.00000652,  0.0000087 ,  0.00001161,
             0.0000155 ,  0.00002071,  0.00002767,  0.000037  ,  0.00004947],
           [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
             1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
           [ 0.99999866,  0.99999827,  0.99999776,  0.9999971 ,  0.99999624,
             0.99999513,  0.9999937 ,  0.99999185,  0.99998946,  0.99998637]]
           )
    >>> weights1 = np.array(
          [[ 0.08644327,  0.11422464,  0.14951155,  0.19386571,  0.24903863,
             0.31695961,  0.39971352,  0.499508  ,  0.61863016,  0.75939376],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.0003915 ,  0.0005859 ,  0.00086848,  0.00127506,  0.0018542 ,
             0.00267079,  0.00381058,  0.00538539,  0.00753925,  0.0104552 ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00845718,  0.01151475,  0.01552811,  0.02074147,  0.02744353,
             0.03597045,  0.04670741,  0.06008859,  0.07659494,  0.09674976]]
           )
    >>> weights2 = np.array(
          [[ 0.1061835 ,  0.13486507,  0.17010619,  0.21313444,  0.26536263,
             0.32841358,  0.40414825,  0.49469664,  0.6024907 ,  0.73029753],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00308873,  0.00431787,  0.00597608,  0.0081888 ,  0.01110918,
             0.01492111,  0.01984166,  0.02612241,  0.03404919,  0.04393993],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00050288,  0.00074004,  0.00107842,  0.00155623,  0.00222394,
             0.00314738,  0.00441136,  0.00612366,  0.00841953,  0.0114664 ]]
           )
    >>> tvals, pvals = SF.weighted_ttest_rel(cond1, cond2, weights1, weights2)
    >>> print tvals
    [-30.40771082 -26.66094401 -23.44662594 -20.68789375 -18.31655524
     -16.27421729 -14.51085133 -12.98410269 -11.65816583 -10.50288769]
    >>> print pvals
    [ 0.00000697  0.00001176  0.00001961  0.00003225  0.00005226  0.00008343
      0.00013115  0.00020301  0.00030947  0.00046464]
    ''' 
    # get average difference
    weighAv = np.average(cond1,axis=0, weights=weights1) - np.average(cond2,axis=0, weights=weights2)

    # Determine N
    N = len(cond1) - np.sum(np.isnan(cond1), axis = 0)
    N2 = len(cond2) - np.sum(np.isnan(cond2), axis = 0)
    if np.array(N).shape:
        N[N2<N] = N2[N2<N]
    else:
        N = np.min([N, N2])
   
    
    # compute pooled variance and sem assuming equal sample size and variance for cond1 and cond2
    sem1 = weighSEMOneSample(cond1, weights1)
    sem2 = weighSEMOneSample(cond2, weights2)
    sem_pooled = np.sqrt((sem1 **2) + (sem2 **2))
    
    # compute t-values and p-values
    tvals = weighAv/sem_pooled
    pvals = scipy.stats.t.sf(np.abs(tvals), 2*N-2)*2 # dfs for pooled variance

    return tvals, pvals



#==============================================================================
# Get clusters
#==============================================================================
def getCluster(data):
    '''
    Splits a np.array with True, False values into clusters. A cluster is 
    defined as adjacent points with the same value, e.g. True or False. 
    The output from this function is used to determine cluster sizes when
    running cluster statistics. 
    
    Parameters
    ----------
    data : np.array
        A 1d np.array containing True, False values.
    
    Returns
    -------
    clusters : list of np.arrays
        The list contains the clusters split up. Each cluster in its own
        np.array. 
    indx : list of np.arrays
        The list contains the indexes for each time point in the clusters.     
    
    Example
    --------
    >>> import numpy as np
    >>> import SMART_Funcs as SF
    >>> data = np.array(
    [ True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True]
    )
    >>> clusters, indx = SF.getCluster(data)
    >>> print clusters
    [array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,  True], dtype=bool),
     array([False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False], dtype=bool),
     array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True], dtype=bool)]
    >>> print indx
    [array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18]),
     array([19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]),
     array([ 52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
             65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
             78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
             91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
            104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
            117, 118, 119, 120, 121, 122, 123, 124, 125])]
    ''' 
    clusters = np.split(data, np.where(np.diff(data) != 0)[0]+1)
    indx = np.split(np.arange(len(data)), np.where(np.diff(data) != 0)[0]+1)
    return clusters, indx

#==============================================================================
# Cluster statistics against baseline
#==============================================================================
def clusterStat_oneSamp(cond, weights, baseline, sigThresh=0.05):
    '''
    The input is the output from the SMART gaussSmooth function. 
    Does the statistics (weighted one sample t-test for each time point) 
    against baseline and subsequently returns all clusters and sum of t-values 
    for the clusters, T-values are the sum of the absolute t-values 
    in a cluster.
    
    Parameters
    ----------
    cond : 2d np.array
        The smoothed dependent variable 
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    weights : 2d np.array
        The weights for the smoothed dependent variable
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    baseline : float or int
        The baseline to do the one sample testing against
    sigThresh : float
        The significant threshold for statistical testing, default: 0.05
    
    Returns
    -------
    sigCl : list of np.array
        Each np.array contains the significant time points for each 
        cluster (indexes). The length of the list equals the number 
        of found clusters. 
    sumTvals : list
        The sum of t-values for each of the clusters.
        
    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import SMART_Funcs as SF
    >>> 
    >>> data = pd.read_pickle("dataFile.p")
    >>> nPP = 10
    >>> timeVect = np.arange(100,501,1)
    >>> nPerms = 1000
    >>> kernelSize = 10
    >>> baseline = 0.5
    >>> 
    >>> cond = np.zeros((nPP, len(timeVect)))
    >>> weights = np.zeros((nPP, len(timeVect)))
    >>> for i in range(nPP):
            #  Extract data for participant
            times = data[timeVar1][i]
            depV = data[depVar1][i]
            # Run Smoothing
            cond[i, :], weights[i,:] = SF.gaussSmooth(times, depV, timeVect, kernelSize)
    >>> 
    >>> sigCL, sumTvals = SF.clusterStat_oneSamp(cond, weights, baseline, sigThresh=0.05)
    >>> print sigCL
    [array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18]),
     array([ 52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
             65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
             78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
             91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
            104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
            117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
            130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
            143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
            156, 157, 158, 159, 160]),
     array([206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218,
            219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
            232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244,
            245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257,
            258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
            271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283,
            284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296,
            297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
            310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322,
            323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335,
            336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348,
            349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361,
            362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374,
            375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387,
            388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400])]
    >>> print sumTvals
    [104.69642787555519, 745.05504747443968, 1530.9248496008438]
    ''' 
    tValues, pValues = weighted_ttest(cond, weights, baseline)
    sigArr = pValues < sigThresh
    cl, clInd= getCluster(sigArr)
    sigCl = [clInd[i] for i in range(len(cl)) if cl[i][0] == True]
    sumTvals = [np.sum(np.abs(tValues[i])) for i in sigCl]
    return sigCl, sumTvals

#==============================================================================
# Cluster statistics between conditions (Paired testing) 
#==============================================================================
def clusterStat_rel(cond1, cond2, weights1, weights2, sigThresh=0.05):
    '''
    The input is the output from the SMART gaussSmooth function. 
    Does the statistics (weighted within subjects t-test for each time point) 
    and subsequently returns all clusters and sum of t-values for the clusters, 
    T-values are the sum of the absolute t-values in a cluster.
    
    Parameters
    ----------
    cond1 : 2d np.array
        The smoothed dependent variable for condition 1
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    cond2 : 2d np.array
        The smoothed dependent variable for condition 1
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    weights1 : 2d np.array
        The weights for the smoothed dependent variable for condition 1
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    weights2 : 2d np.array
        The weights for the smoothed dependent variable for condition 2
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
    sigThresh : float
        The significant threshold for statistical testing, default: 0.05
    
    Returns
    -------
    sigCl : list of np.array
        Each np.array contains the significant time points for each 
        cluster (indexes). The length of the list equals the number 
        of found clusters. 
    sumTvals : list
        The sum of t-values for each of the clusters.
        
    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import SMART_Funcs as SF
    >>> 
    >>> data = pd.read_pickle("dataFile.p")
    >>> nPP = 10
    >>> timeVect = np.arange(100,501,1)
    >>> nPerms = 1000
    >>> kernelSize = 10
    >>> 
    >>> cond1 = np.zeros((nPP, len(timeVect)))
    >>> weights1 = np.zeros((nPP, len(timeVect)))
    >>> cond2 = np.zeros((nPP, len(timeVect)))
    >>> weights2 = np.zeros((nPP, len(timeVect)))
    >>> for i in range(nPP):
            #  Extract data for participant
            times1 = data[timeVar1][i]
            depV1 = data[depVar1][i]
            times2 = data[timeVar2][i]
            depV2 = data[depVar2][i]
            # Run Smoothing
            cond1[i, :], weights1[i,:] = SF.gaussSmooth(times1, depV1, timeVect, kernelSize)
            cond2[i, :], weights2[i,:] = SF.gaussSmooth(times2, depV2, timeVect, kernelSize)
    >>> 
    >>> sigCL, sumTvals = SF.clusterStat_rel(cond1, cond2, weights1, weights2, sigThresh=0.05)
    >>> print sigCL
    [array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22]),
     array([ 46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,
             59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
             72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,
             85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
             98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
            111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
            137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
            150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162,
            163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
            176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188,
            189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202]),
     array([307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319,
            320, 321, 322, 323]),
     array([342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354,
            355, 356, 357])]
    >>> print sumTvals
    [270.65316608715654, 1166.3110788069557, 44.639577887050663, 39.05781218332065]

    ''' 
    tValues, pValues = weighted_ttest_rel(cond1, cond2, weights1, weights2)
    sigArr = pValues < sigThresh
    cl, clInd= getCluster(sigArr)
    sigCl = [clInd[i] for i in range(len(cl)) if cl[i][0] == True]
    sumTvals = [np.sum(np.abs(tValues[i])) for i in sigCl]
    return sigCl, sumTvals

# =============================================================================
# Cluster stats for permutations
# =============================================================================
def permuteClusterStat(perm1, perm2, permWeights1, permWeights2, sigThresh=0.05):
    '''
    The input is the output from the SMART twoSamplePerm function. 
    Does the statistics (weighted within subjects t-test for each permutation 
    and time point) and subsequently returns all significant clusters and
    sum of t-values for the clusters, T-values are the sum of the absolute 
    t-values in a cluster.
    
    Parameters
    ----------
    perm1 : 3d np.array
        The smoothed permutations for condition 1.
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
        Dimension 3 = Each permutation.
    perm2 : 3d np.array
        The smoothed permutations for condition 2
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
        Dimension 3 = Each permutation.
    permWeights1 : 3d np.array
        The weights for the smoothed permutations for condition 1
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
        Dimension 3 = Each permutation.
    permWeights2 : 3d np.array
        The weights for the smoothed permutations for condition 2
        Dimension 1 = Participant.
        Dimension 2 = The temporal order of the smoothed data
        Dimension 3 = Each permutation.
    sigThresh : float
        The significant threshold for statistical testing, default: 0.05
    
    Returns
    -------
    permDistr : np.array
        The sum of t-values for the largest cluster of significant time points
        in each permutation. 
    
    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import SMART_Funcs as SF
    >>> 
    >>> data = pd.read_pickle("dataFile.p")
    >>> nPP = 10
    >>> timeVect = np.arange(100,501,1)
    >>> nPerms = 1000
    >>> kernelSize = 10
    >>> 
    >>> perm1 = np.zeros((nPP, len(timeVect), nPerms))
    >>> permWeights1 = np.zeros((nPP, len(timeVect), nPerms))
    >>> perm2 = np.zeros((nPP, len(timeVect), nPerms))
    >>> permWeights2 = np.zeros((nPP, len(timeVect), nPerms))
    >>> for i in range(nPP):
            #  Extract data for participant
            times1 = data[timeVar1][i]
            depV1 = data[depVar1][i]
            times2 = data[timeVar2][i]
            depV2 = data[depVar2][i]
            # Run Permutations Between conditions
            perm1[i,:,:], permWeights1[i,:,:], perm2[i,:,:], permWeights2[i,:,:] = SF.twoSamplePerm(times1, depV1, times2, depV2, timeVect, kernelSize, nPerms)
    >>> 
    >>> permDistr = SF.clusterStatPerm_rel(perm1, perm2, permWeights1, permWeights2, sigThresh=0.05)
    >>> print permDistr
    array([  16.17573373,   69.79784087,   17.80550175,   91.40111032,
         11.00196301,   62.53381537,   56.76667962,   35.34379494,
         84.14146202,   33.85304266,    9.52752666,   28.34821554,
         56.0239134 ,   36.17522964,   79.02346818,   64.30517206,
         42.48436718,   37.94787513,   22.27985815,   50.62931182,
         19.92534377,   18.35998986,   28.99579596,   47.42959113,
         19.6977887 ,   25.57918298,   28.24954274,   56.00075043,
         74.50933434,   64.841294  ,   23.2026394 ,   57.65972576,
        101.6464277 ,  157.02779659,   55.04606213,   39.58701484,
         62.15710694,   81.83197457,    8.99890508,   ....])
    ''' 
    permDistr = []
    tValues, pValues = weighted_ttest_rel(perm1, perm2, permWeights1, permWeights2)
    tValues = tValues.transpose()
    sigArr = (pValues<0.05).transpose()
    for indx in range(len(perm1[0,0,:])):
        cl, clInd= getCluster(sigArr[indx,:])
        sigCl = [clInd[i] for i in range(len(cl)) if cl[i][0] == True]
        if len(sigCl)  != 0:
            permDistr.append(np.max([np.sum(np.abs(tValues[indx,i])) for i in sigCl]))
        else:
            permDistr.append(np.max(tValues[indx,:]))
    return np.array(permDistr)


