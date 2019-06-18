#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 16:13:56 2018

@author: rwilson
"""

# This is an attempt to determine the auto-correlation correction decay rate

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from scipy import signal
import pandas as pd
import cross_correlation as cc
import scipy.stats as st



def corr_significance(a,b, verbose=False):
    ''' This function is tasked with calculating the significance of a correation
    based on the work by Hanson and Yang 2009.
    '''
    from scipy.optimize import curve_fit
    from scipy.signal import hilbert
    from scipy import special

    # Setup variables
    N = len(a)
    lags = np.arange(0, N-1)
    fitRange = slice(10,-1)
    tau_max=0

    # Perform the Autocorrelation of both variables
    a1 = (a - np.mean(a)) / (np.std(a) * N)
    a2 = (a - np.mean(a)) / np.std(a)
    b1 = (b - np.mean(b)) / (np.std(b) * N)
    b2 = (b - np.mean(b)) / np.std(b)

    # Calculation of CC for x and y
    C_xx = np.correlate(a1, a2, mode="full")[N:]
    C_yy = np.correlate(b1, b2, mode="full")[N:]


    # Calculate the effective
    C_xx_Hilb = hilbert(C_xx)
    C_xx_env = np.abs(C_xx_Hilb)
    C_yy_Hilb = hilbert(C_yy)
    C_yy_env = np.abs(C_yy_Hilb)

    # Fit exponential function to the envelope of the decay
    def _func_exp(x, a, b):
        return a*np.exp(-b * x)

    for CC in [C_xx_env, C_yy_env]:
        # Calculate the fit and store the max decauy rate tau
        p_opt, p_cov = curve_fit(_func_exp, lags[fitRange], CC[fitRange],
                                 bounds=([0.2, 0.0001], [1, 0.001]))

        tau = p_opt[0]
        if tau > tau_max:
            tau_max = tau
            CC_max = CC
            x_fit = _func_exp(lags[fitRange], *p_opt)

    N_eff = N/tau_max

    if verbose:
        # Plot Comparisons
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.plot(lags, C_xx, label='AC of xx')
        ax0.plot(lags, C_yy, label='AC of yy')
        ax0.plot(lags, C_xx_env, label='Envelope xx')
        ax0.plot(lags, C_yy_env, label='Envelope yy')
        ax0.plot(lags[fitRange], x_fit, label='fit_exp_of max AC tau')
        ax0.legend()

    ################### Calculate the Significance ###################

    # The variance and cross-correlation
    # VarCC = (N_eff-1)/N_eff**2 * (np.mean(a**2) - np.mean(a)**2) * (np.mean(b**2) - np.mean(b)**2)
    VarCC = (N_eff-1)/N_eff**2 * np.std(a)**2 * np.std(b)**2
    C_xy = np.correlate(a, b, mode="full")/N_eff

    # The Test statistic
    C_xy= C_xy
    nt = len(C_xy)
    Z_n = np.abs( np.sum(C_xy)/nt )
    c_1alpha =np.sqrt(2) * special.erfcinv(0.0001) * np.sqrt(VarCC)/np.sqrt(nt)

    if Z_n > c_1alpha:
        print('The H0 is rejected: These is a correlation')
    else:
        print('The H0 is accepted: There is no correlation')



    return (N_eff, Z_n, c_1alpha, VarCC)

def decay_rate(a,b, verbose=False):
    ''' Calculates the correlation decay rate of two time series.
    Parameters
    ----------
    a : int or float
        Reference time-series of length ``N``.

    b : int or float
        Comparison time-series of length ``N``.

    verbose : bool
        If ``True`` a plot of the decay fitting will be provided

    Returns
    -------
    Tau : float
        The decay rate of the correlation between ``a`` and ``b``.
    '''

    from scipy.optimize import curve_fit
    from scipy.signal import hilbert
    from scipy import special

    # Setup variables
    N = len(a)
    lags = np.arange(0, N-1)
    fitRange = slice(10,-1)
    tau_max=0

    # Perform the Autocorrelation of both variables
    a = (a - np.mean(a)) / (np.std(a) * N)
    b = (b - np.mean(b)) / np.std(b)


    # Calculation of CC for x and y
    C_xx = np.correlate(a, b, mode="full")[N:]

    # Calculate the effective
    C_xx_Hilb = hilbert(C_xx)
    C_xx_env = np.abs(C_xx_Hilb)


    # Fit exponential function to the envelope of the decay
    def _func_exp(x, a, b):
        return a*np.exp(-b * x)

    for CC in [C_xx_env]:
        # Calculate the fit and store the max decauy rate tau
        p_opt, p_cov = curve_fit(_func_exp, lags[fitRange], CC[fitRange],
                                 bounds=([0.2, 0.0001], [1, 0.001]))

        tau = p_opt[0]
        if tau > tau_max:
            tau_max = tau
            CC_max = CC
            x_fit = _func_exp(lags[fitRange], *p_opt)


    if verbose:
        # Plot Comparisons
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.plot(lags, C_xx, label='CC of xx')
        ax0.plot(lags, C_xx_env, label='Envelope of xx')
        ax0.plot(lags[fitRange], x_fit, label='fit_exp_of max AC tau')
        ax0.legend()

    return tau_max


# N_eff, Z_n, c_1alpha, VarCC = corr_significance(a,b, verbose=True)



def spec_corr_significance(a, b, C_xy=None, tot=2000, confidence = 0.90, maxlag=500, verbose=False):
        ''' This function applies a spectral perturbation of phase to assess the
        significance of a correlation. This is similar to the work by Ebisuzaki 1997
        The basic output is the confidence of accepting the Null Hypothesis that
        there is no correlation between two time series based on the maximum CC.
        Inputs:
        -------------------
        a:       Time series reference norm. as (a - np.mean(a)) / (np.std(a) * N)
        b:       Time series comparison norm as (b - np.mean(b)) / (np.std(b))
        C_xy:    If the normalised cross-correlation is given, avoid recalculation,
                 expected lag which yields the max correlation.
        tot:     Total number of random perturbations to check the correlation
                 Default is 100
        verbose: Default is False, if true a print out statement will be provided
                 If 2 is given a plot of the perturbed trace will be made

        Outputs:
        -------------------
        Sig:      FPR: indicating the confidence that there is a statistical
                 significant correlation between the two time series.
        '''
        import matplotlib.pyplot as plt

        # Setup
        N = round(len(a)/2)*2

        if maxlag is None:
            maxlag = N-1

        # Normalize the correlation for testing the FPR
        b = (b - np.mean(b)) / (np.std(b))
        a = (a - np.mean(a)) / (np.std(a) * N)

        # Compute the decay rate
        tau = decay_rate(a,b, verbose=verbose)



        # Compute the fft of time series a
        A = np.fft.fft(a)


        # Correlate with series b
        if C_xy == None:
            C_xy = np.correlate(a, b, mode="full")

            #C_xy = np.correlate(a, b, mode="full")
            #C_xyInx_max = C_xy.argmax()
            #C_xy = C_xy[C_xyInx_max - maxlag - 1:C_xyInx_max + maxlag]

        lag = np.abs(C_xy.argmax()-N)
        maxlag = int(maxlag/tau**2) + lag
        C_xy = C_xy[N-maxlag-1:N+maxlag]

        FP = 0
        C_xy_dist = []
        for i in range(tot):
            A_pert = [x*np.exp(np.random.uniform(0,2*np.pi)*1.0j) for x in A[0:N//2-1]] # Perterb the positive frequency components
            A_pert.append(A[N//2]*2**0.5*np.cos(np.random.uniform(0,2*np.pi)*1.0j)) # Setting the nyquist
            A_pert[0] = 0 # Remove DC component
            a_pert = np.fft.ifft(A_pert).real
            a_pert = (a_pert - np.mean(a_pert)) / (np.std(a_pert) * N)
            C_xy_test = np.correlate(a_pert, b, mode="full")[N-maxlag-1:N+maxlag]
            C_xy_dist = np.concatenate( (C_xy_dist, C_xy_test) )

        sigma = np.std(C_xy_dist)
        mean = np.mean(C_xy_dist)
        Z_dist = st.mstats.zscore(C_xy_dist)
        Z = (C_xy - mean) / sigma

        # The probability of rejecting the H0 null hypothesis "No correlation present"
        # and accepting the H1 hypothesis, a correlation is present
        # sig = st.norm.cdf(Z)
        P_H1 = st.norm.ppf(confidence)
        #H1_AR = len(Z[Z>P_H1])/len(Z) # The H1 acceptance rate
        H1_AR = len(Z[Z>P_H1])/len(Z)



        #if Z < p_crit:
         #   print('The null hypothesis that a correlation exists is rejected with %g %% confidence' % (confidence*100))

        #FPR = FP/tot # When the FPR is large, this indicates that there is no correlation between a and b
        if verbose == True:
            # Conconfidence that the time series are un-correlated
            fig = plt.figure()
            ax0 = fig.add_subplot(111)
            #ax0.plot(C_xy_dist, st.norm.pdf(C_xy_dist), label='pdf')
            ax0.hist(Z_dist, bins = 'auto', normed=True, label='Z_dist')
            ax0.hist(Z, bins = 'auto', normed=True, label='Z of Correlation')
            #ax0.hist(sig, bins = 'auto', normed=True, label='sig')
            ax0.set_xlabel('Z-score')
            ax0.set_ylabel('Density')
            ax0.legend()

        dist_dic = {'sigma': sigma,
                        'mean': mean,
                        'Z_dist': Z_dist,
                        'Z': Z,
                        'tau': tau,
                        'maxlag': maxlag}

        return H1_AR, dist_dic

# TODO
        # Normalise the max lag by the decay rate of the actual correlation between
        # a and b. For more similar time series, the max lag should be less.
        # When the wavefields a totally uncorrelated, max lag must increase.
        # The decay rate is low for dissimilar time series

#wdw_pos = slice(5000,8000)
#a = TScut_DB[1].iloc[wdw_pos].as_matrix()
#for window in range(50,60):
#
#    b = TScut_DB[window].iloc[wdw_pos].as_matrix()


#a = np.random.randn(len(TScut_DB[1]))[wdw_pos]
#b = np.random.randn(len(TScut_DB[1]))[wdw_pos]
#a = (a - np.mean(a)) / (np.std(a) * leng
#b = (b - np.mean(b)) / np.std(b)


#tau = decay_rate(a,b, verbose=True)

#H1_AR, dist_dic = spec_corr_significance(a, b, C_xy=None, tot=2000, confidence = 0.99, maxlag=16, verbose=False)
#print(window, 'The rejection rate of the null hypothesis', H1_AR)
#print('max lag:', dist_dic['maxlag'], 'tau:', dist_dic['tau'])

