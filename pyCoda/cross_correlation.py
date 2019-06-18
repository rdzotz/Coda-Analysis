#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements various cross-correlation operations

Created on Tue Jan 17 11:40:08 2017
@author: rwilson
"""
import numpy as np
import pickle
import pandas as pd
import itertools
import data as dt

from scipy import signal

class cross_correlation_survey():
    '''Perform the cross-correlation for a multi-source receiver survey

    Parameters
    ----------
    Database : str
        Relative or absolute location of the database.h5 file.
    '''

    def __init__(self, Database, verbose=False):
        self.Database  = Database
        self.verbose = verbose
        self.param = dt.utilities.DB_attrs_load(self.Database,
                                                 ['ww','ww_ol','wdwPos','CC_ref',
                                                  'CC_type', 'taper','Eng_ref',
                                                  'sig', 'survey_type',
                                                  'CC_folder',
                                                  'STA_meas', 'END_meas',
                                                  'lagOverlap', 'taperAlpha'])

    def CC_run(self):
        '''Run the overhead tasks performing the cross-correlation, based on the
        parameters found in the Database attributes. A final database containing
        the CC for each source receiver pair is then stored within the Database.5
        file.
        '''

        # Setup common parameters
        lags = self.param['CC_ref']
        # Determine the total list of processing steps to be applied
        processing= ['CC', 'CC-lag']

        if self.param['sig']:
                processing.append('sig')

        # ------------------------ Single src/rec pairs ------------------------
        if 'single' == self.param['survey_type']:
            TS_groups = ['TSdata']

            # Define energy reference windows to process if required
            if self.param['Eng_ref']:
                windows = [[wdwPos[0],wdwPos[1]] for wdwPos
                              in self.param['wdwPos']]
                # dictionary of dataframes ref for all windows
                Eng_ref = {key[0]: dt.utilities.
                           DB_pd_data_load(self.Database,'TSdata/', cols=self.param['Eng_ref'],
                                           whereList = slice(window[0],window[1]+1)) for
                                           (key, window) in
                                           zip(self.param['wdwPos'], self.param['wdwPos'])}
            else:
                Eng_ref = None

            # Default parameters
            CC_all_list = []

            # Load all Time-Series data
            TSdata = dt.utilities.DB_pd_data_load(self.Database, 'TSdata')
            sta = self.param['STA_meas'] if self.param['STA_meas'] else None
            end = self.param['END_meas'] if self.param['END_meas'] else None
            TSdata = TSdata.iloc[:, sta:end]
            surveysU = TSdata.columns.get_level_values('Survey').unique().values
            surveys = TSdata.columns.get_level_values('Survey').values
            #surveyPrev = surveysU[0]
            # For each col, lag, windowpos
            for survey, lag, wdwPos in itertools.product(surveysU,
                                                        lags,
                                                        self.param['wdwPos']):

                if survey - lag < 0: continue
                #if not self.param['lagOverlap'] and survey < surveyPrev+lag: continue

                #surveyPrev = survey
                print('* For survey %s ...' %survey)
                print('* For lag %s ...' %lag)
                print('* For window %s ...' %['index>%d' %wdwPos[0],
                                              'index<%d' %wdwPos[1]])
                window = slice(wdwPos[0], wdwPos[1])

                if Eng_ref is not None: eng_wdw = Eng_ref[wdwPos[0]]
                else: eng_wdw = None

                # Refernece
                if 'rolling' in self.param['CC_type']:
                    ref = survey-lag
                elif 'fixed' in self.param['CC_type']:
                    ref = lag-1

                # Extract time series
                TSref = TSdata.loc[window, surveys==ref]

                # Current
                TScurrent = TSdata.loc[window, surveys==survey]

                # ---------- The cross-correlation ----------
                CC, processingTmp = self.CC_dataframes(TSref, TScurrent, processing,
                                                       lag, eng_wdw,
                                                       wdwPos)
                CC_all_list.append(CC)

                if len(CC_all_list)>1:
                    # For the same survey number concatenate columns
                    if CC.index.get_level_values('Survey').unique().values == \
                        CC_all_list[-2].index.get_level_values('Survey').unique().values:
                            CC_all_list = CC_all_list[:-2]+[pd.concat(CC_all_list[-2:], axis=1)]

            CC = pd.concat(CC_all_list, axis=0)
            CC.index = CC.index.droplevel([level for level in range(3,CC.index.nlevels)])

        # -------------- Multiple src/rec paris survyes in subgroups--------------
        elif 'multiple' == self.param['survey_type']:
            # Import TS groups from database
            TS_groups = dt.utilities.DB_group_names(self.Database, group_name = 'TSdata')

            if self.param['STA_meas'] and self.param['END_meas']:
                surveyTimes = [pd.to_datetime(group.split('survey')[1]) for group in
                               TS_groups]
                mask = [(group > pd.to_datetime(self.param['STA_meas'])) and
                        (group < pd.to_datetime(self.param['END_meas'])) for group in
                        surveyTimes]

                TS_groups = list(itertools.compress(TS_groups, mask))

            # Define energy reference windows to process if required
            if self.param['Eng_ref']:
                windows = [['index>%d' %wdwPos[0], 'index<%d' %wdwPos[1]] for wdwPos
                              in self.param['wdwPos']]

                # dictionary of dataframes ref for all windows
                Eng_ref = {key[0]: dt.utilities.
                           DB_pd_data_load(self.Database,'TSdata/'+  \
                                           TS_groups[self.param['Eng_ref']],
                                           whereList = window) for (key,
                                           window) in zip(windows, windows)}
            else:
                Eng_ref = None

            CC_all_list = []
            for gp in range(0, len(TS_groups)):
                print('* For group %s ...' %gp)
                CC_list = []

                # For each lag, windowpos
                for lag, wdwPos in itertools.product(lags,self.param['wdwPos']):
                    if gp - lag < 0:
                        continue
                    print('* For lag %s ...' %lag)
                    window = ['index>%d' %wdwPos[0], 'index<%d' %wdwPos[1]]
                    print('* For window %s ...' %window)

                    if Eng_ref is not None: eng_wdw = Eng_ref['index>%s'% wdwPos[0]]
                    else: eng_wdw = None

                    # Might be better to load all data once and then slice it
                    # Refernece
                    if 'rolling' in self.param['CC_type']:
                        ref = gp-lag
                    elif 'fixed' in self.param['CC_type']:
                        ref = lag

                    # Extract time series
                    TSref = dt.utilities.DB_pd_data_load(self.Database,
                                                       'TSdata/'+ TS_groups[ref],
                                                       whereList = window)
                    # Current
                    TScurrent = dt.utilities.DB_pd_data_load(self.Database,
                                                       'TSdata/'+ TS_groups[gp],
                                                       whereList = window)

                    #---------- The cross-correlation ----------
                    CC, processingTmp = self.CC_dataframes(TSref, TScurrent, processing,
                                                           lag, eng_wdw,
                                                           wdwPos)

                    CC.index = CC.index.droplevel([level for level in
                                                         range(3,CC.index.nlevels)])
                    CC_list.append(CC)


                if len(CC_list)>1:
                    # Merge all common windows and lags together
                    for idx,df in enumerate(CC_list[:-1]):
                        CC_list[0] =  CC_list[0].merge(CC_list[idx+1],
                                                  left_index=True, right_index=True)
                    # Append Df from each group
                    CC_all_list.append(CC_list[0])
                elif CC_list: # If list is not empty
                    # Append Df from each group
                    CC_all_list.append(CC_list[0])
            CC = pd.concat(CC_all_list, axis=0)

        # Save processed data to database
        dt.utilities.DB_pd_data_save(self.Database, self.param['CC_folder'], CC)


    def CC_dataframes(self, TSdata1, TSdata2, processing, lag, eng_wdw, wdwPos):
        '''Cross-correlation between two dataframes for multiple lags, along with
        the optional calculation of the relative amplitude changes between traces
        or the spectral significance of the correlation.

        Parameters
        ----------
        TSdata1, TSdata1 : DataFrame
            The time-series dataframes, stored in columns, with multi-level index
            names ``srcNo`` and ``recNo``. Only equivelent src/rec pairs will be
            compared.
        processing : list(str)
            A list of strings containing additional processing types if requested.
            The defualt is ['CC', 'CC-lag']. If 'sig' is found in the list then the
            spectral significance of the correlation will also be calculated.
        lag : int
                lag between ``TSdata1`` and ``TSdata2`` in terms of repeat measurements.

        eng_wdw : series
                The window within the time-series corresponding to the user selected
                reference time-series.

        wdwPos : list
                The start and stop window positions within a trace


        Returns
        -------
        CC : DataFrame(len(TSdata1.shape[1]), len(processing))
        processing : list(str)
            a list of strings pertaining to the column values of ``CC``


        Note
        ----
            Might be much faster if I write my own multiple 1D cross-correlation
            function.
            Add taper to the correlation calculation
            This function currently handels the situation that some src/rec pairs
            are missing from one of the dataframes,

        C = [np.correlate(TSdata1[:,TS],TSdata2[:,TS], mode='full').max() for
        S in range(0, TSdata1.shape[1])]
        '''

        # No of traces based on min
        min_cols = np.argmin((TSdata1.shape[1], TSdata2.shape[1]))
#        max_cols = np.argmax((TSdata1.shape[1], TSdata2.shape[1]))
#        no_traces = max(TSdata1.shape[1], TSdata2.shape[1])
        trace_length = TSdata2.shape[0]

        # Apply the taper to matrix
        if self.param['taper']:
            taper = signal.tukey(trace_length,alpha=self.param['taperAlpha'])
        else:
            taper = 1

        # Only calculate Relative amplitude for the min lag
        if lag == min(self.param['CC_ref']) and eng_wdw is not None:
            processingTmp = processing[:]
            processingTmp.append('RelAmp')
            eng_wdw = (eng_wdw - eng_wdw.mean()) / \
                      (eng_wdw.std() * eng_wdw.shape[0])
        else:
            processingTmp = processing

        # Normalise the entire matrix
        TSdata1 = (TSdata1 - TSdata1.mean()) / (TSdata1.std() * TSdata1.shape[0])
        TSdata2 = (TSdata2 - TSdata2.mean()) / TSdata2.std()



        # Empty dataframe for storage row no. == to max number of cols
        cols = pd.MultiIndex.\
                           from_product([[lag],['%d-%d' % (wdwPos[0], wdwPos[1])],
                                         processingTmp],
                                         names = ['lag', 'window','Parameters'])
        CC = pd.DataFrame(index=TSdata2.T.index, columns = cols)
#        CC = np.empty([no_traces, len(processingTmp)], dtype=np.float64)

        # List of src_rec combinations for dataset with min number of combinations
        # This assumes that the other Df has all expected source receiver pairs
        src_rec_pairs = [(src, rec) for src, rec in
                   zip([TSdata1, TSdata2][min_cols].columns.get_level_values('srcNo').values,
                       [TSdata1, TSdata2][min_cols].columns.get_level_values('recNo').values )]

        for src_rec in src_rec_pairs:

            ref = TSdata1.xs(src_rec, axis=1, level = ['srcNo', 'recNo']).values[:,0]
            comp = TSdata2.xs(src_rec, axis=1, level = ['srcNo', 'recNo']).values[:,0]

            CCs = np.correlate(ref* taper, comp * taper, mode='full')
            CC.loc[src_rec, (slice(None), slice(None), processingTmp[0])] = CCs.max()
            CC.loc[src_rec, (slice(None), slice(None), processingTmp[1])] = CCs.argmax() - len(CCs)//2
            if CC.shape[1] is 2: continue

            try:
                eng_ref = eng_wdw.xs(src_rec, axis=1, level = ['srcNo', 'recNo']).values
            except ValueError:
                eng_ref = eng_wdw.values
            CC.loc[src_rec, (slice(None), slice(None), processingTmp[2])] = self.relative_amplitude(eng_ref*taper, comp*taper)
            if CC.shape[1] is 3: continue
            CC.loc[src_rec, (slice(None), slice(None), processingTmp[3])] , _ = self.spec_corr_significance(ref, comp)

        return CC, processingTmp

    def relative_amplitude(self, a, b):
            ''' Determine the relative amplitude difference between two traces.

            Parameters
            ----------
            a : int or float
                Reference time-series of length ``N``.

            b : int or float
                Comparison time-series of length ``N``.


            Returns
            -------
            Relative amplitude : float
                The rejection
            Notes
            -----
            To calculate the confidence associated with the Z-score use
                ``scipy.stats.norm.ppf(sig)``
            '''

            return np.sqrt(sum(b**2)/sum(a**2))



    def spec_corr_significance(self, a, b, C_xy=None, tot=2000, confidence = 0.99,
                                   maxlag=16, verbose=False):
            ''' Applies a spectral perturbation of phase to assess the
            significance of a correlation. This is similar to the work by Ebisuzaki 1997
            The basic output is the acceptance rate of the alternative to
            the Null Hypothesis H0 that there is no correlation between two time series
            based on the CC for the.
            Parameters
            ----------
            a : int or float
                Reference time-series of length ``N``.

            b : int or float
                Comparison time-series of length ``N``.

            C_xy : int or float
                The normalised cross-correlation coefficient of length corresponding
                to ``maxlag``.

            tot : int
                Total number of random perturbations to check the correlation
                (Default is 2000).

            confidence : float
                The confidence at which the acceptance rate ``H1_AR`` is tested,
                (Default is 0.95).

            maxlag : int
                The maximum sample lags applied in the cross-correlations,
                (Default is 64).

            verbose : bool
                If ``True`` an output of the distribution of the random phase test
                cross-correlations will be made with that of the ``C_xy``
                (Default is 64).

            Returns
            -------
            H1_AR : float
                The rejection rate of the null hypoth ``a``
                and ``b``.

            sigma : float
                The variance of the distribution of random phase perturbed time-series.

            mean : float
                The mean of the distribution of random phase perturbed time-series.

            Z_dist : float
                The distribution of the random phase z-scores.

            Z : float
                The distribution of the C_xy z-scores.

            Notes
            -----
            To calculate the confidence associated with the Z-score use
                ``scipy.stats.norm.ppf(sig)``
            '''

            import matplotlib.pyplot as plt
            import scipy.stats as st

            # Setup
            N = round(len(a)/2)*2
            if maxlag is None:
                maxlag = N-1

            # Normalize the correlation for testing the FPR
            b = (b - np.mean(b)) / (np.std(b))
            a = (a - np.mean(a)) / (np.std(a) * N)

            # Compute the decay rate
            tau = self.decay_rate(a,b, verbose=verbose)

            # Compute the fft of time series a
            A = np.fft.fft(a)

            # Correlate with series b
            if C_xy == None:
                C_xy = np.correlate(a, b, mode="full")

            lag = np.abs(C_xy.argmax()-N)
            maxlag = int(maxlag/tau**2) + lag
            C_xy = C_xy[N-maxlag-1:N+maxlag]

            C_xy_dist = []
            for i in range(tot):
                # Perterb the positive frequency components
                A_pert = [x*np.exp(np.random.uniform(0,2*np.pi)*1.0j) for x in A[0:N//2-1]]
                # Setting the nyquist
                A_pert.append(A[N//2]*2**0.5*np.cos(np.random.uniform(0,2*np.pi)*1.0j))
                # Remove DC component
                A_pert[0] = 0
                a_pert = np.fft.ifft(A_pert).real
                a_pert = (a_pert - np.mean(a_pert)) / (np.std(a_pert) * N)
                C_xy_test = np.correlate(a_pert, b, mode="full")[N-maxlag-1:N+maxlag]
                C_xy_dist = np.concatenate( (C_xy_dist, C_xy_test) )

            sigma = np.std(C_xy_dist)
            mean = np.mean(C_xy_dist)
            Z_dist = st.mstats.zscore(C_xy_dist)
            Z = (C_xy - mean) / sigma

            # Prob. of rejecting the H0 null hypothesis "No correlation present"
            # and accepting the H1 hypothesis, a correlation is present
            P_H1 = st.norm.ppf(confidence)
            H1_AR = len(Z[Z>P_H1])/len(Z) # The H1 acceptance rate

            dist_dic = {'sigma': sigma,
                        'mean': mean,
                        'Z_dist': Z_dist,
                        'Z': Z}

            if verbose == True:
                fig = plt.figure()
                ax0 = fig.add_subplot(111)
                ax0.hist(Z_dist, bins = 'auto', normed=True, label='Z_dist')
                ax0.hist(Z, bins = 'auto', normed=True, label='Z of Correlation')
                ax0.set_xlabel('Z-score')
                ax0.set_ylabel('Density')
                ax0.legend()

            return H1_AR, dist_dic

    def decay_rate(self, a,b, verbose=False):
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
        import matplotlib.pyplot as plt
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
        C_xy = np.correlate(a, b, mode="full")[N:]

        # Calculate the effective
        C_xy_Hilb = hilbert(C_xy)
        C_xy_env = np.abs(C_xy_Hilb)


        # Fit exponential function to the envelope of the decay
        def _func_exp(x, a, b):
            return a*np.exp(-b * x)

        for CC in [C_xy_env]:
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
            ax0.plot(lags, C_xy, label='CC of xy')
            ax0.plot(lags, C_xy_env, label='Envelope of xy')
            ax0.plot(lags[fitRange], x_fit, label='fit_exp_of max AC tau')
            ax0.legend()

        return tau_max
class cross_correlation:
    """This class performed the cross-correlation of a matrix of time series
    based on various user defined input parameters

    Parameters
    ----------
    TSdata:  Time series data columnweise
    param:   Dictornay of user defined parameters

    Notes
    -----
    Opencv should be used in speed becomes an issue in correlations
    """

    def __init__(self, TSdata, param):
        self.TSdata = TSdata
        self.param = param
        self.wdwPos = param['wdwPos']

    def CC_wdw(self):
        """Determines the number of windows of a certain lenght which fit within
        the TS, and outputs a list of start positions in terms of sample points
        """
        # Setup param
        TS_len = len(self.TSdata)
        ERROR_MESSAGE = 'The length of a TS is', TS_len, 'Which is < end of the last window'

        if self.wdwPos[0] is None:
            # Error checks
            if TS_len < self.param['ww'][0]:
                raise Warning(ERROR_MESSAGE)

            wdwStep = np.floor(self.param['ww'][0] *
                               (100 - self.param['ww_ol']) / 100)

            print('Length fo TSdata', len(self.TSdata))
            max_wdwPos = TS_len - self.param['ww'][0] + 1
            wdwStarts = np.arange(0, max_wdwPos, wdwStep).astype(int)
            print('wdwStep',wdwStep)
            print('max_wdwPos',max_wdwPos)
            print('wdwStarts',wdwStarts)
            self.wdwPos = [ [wdw_start, wdw_start + self.param['ww'][0]] for
                           wdw_start in wdwStarts ]
            self.param['wdwPos']
        else:
            self.wdwPos = [ [wdw_start, wdw_start + ww] for wdw_start,ww in
                           zip(self.wdwPos, self.param['ww'])]
        # Update param dictionary
        self.param['wdwPos'] = self.wdwPos

    def CC_run(self):
        """Handels the cross-correlation coefficient of all TS input for each
        correlation window position defined.

        Output:
        ------
        R_t Either a dataframe or panel of dataframes with index of start of
            window positions
        TODO:
            Need to expand capability to allow for multiple lag values. Might
            be best to build a 2D matrix for this purpose.
        """

        import time

        # ------------------------ setup -------------------------
        # Determine the number of windows
        print('---------Begin Cross-correlation---------')
        self.CC_wdw()

        # For each rolling lag value
        lags = self.param['CC_ref']
        start = time.time()
        rowIndex = ['Meas'+str(col) for col in
                    range(lags[0], self.TSdata.shape[1])]
        R_t_all = pd.DataFrame()

        # function for column list generation
        def intersperse(lst, items):

            repeat = list(itertools.chain.from_iterable(itertools.repeat(x[0], 5)
                                                        for x in lst))
            minus = 0
            for i in range(len(repeat)):

                if (i % 5 == 0)==False:
                    repeat[i] = items[i-1-minus] + str(repeat[i])
                elif i>0:
                    minus +=5
            return repeat

        columns = intersperse(self.wdwPos, ['CC_lag_', 'RelAmp_','FreqBand_', 'sig_'])

        for lag in lags:
            # Define storage frame
            R_t = pd.DataFrame(index=rowIndex, columns= columns)
            R_t.index.name = 'MeasNo'
            multiCols = list(itertools.product([lag],
                                            columns))

            if 'rolling' in self.param['CC_type']:

                # Define amplitude reference trace:
                ref = self.TSdata[:,0]
                # For each window position
                for col in range(lag, self.TSdata.shape[1]):
                    # elem:elem + self.param['ww']
                    CC = [self.norm_CC(
                         self.TSdata[elem[0]:elem[1], col-lag],
                         self.TSdata[elem[0]:elem[1], col],
                         ref[elem[0]:elem[1]], lag)
                          for elem in self.wdwPos]

                    # Flatten list of tuples
                    CC_list = [e for l in CC for e in l]
                    R_t.ix[R_t.index == rowIndex[col-lags[0]], :] = CC_list
            elif 'fixed' in self.param['CC_type']:
                i = 0
                ref = self.TSdata[:,lag]
                for col in self.TSdata[:,lag+1:].T:

                    CC = [self.norm_CC(ref[elem[0]:elem[1]],
                                       col[elem[0]:elem[1]], lag)
                          for elem in self.wdwPos]
                    # Flatten list of tuples
                    CC_list = [e for l in CC for e in l]
                    R_t.ix[R_t.index == rowIndex[lag-lag+i], :] = CC_list
                    i += 1

            R_t_temp = R_t.copy()
            R_t_temp.columns = pd.MultiIndex.from_tuples(multiCols)
            R_t_all = pd.concat([R_t_all, R_t_temp], axis=1)
            R_t_all.dropna(axis=1, how='all', inplace=True)

        end = time.time()


        print('The total cross-correlation time for all ',
              len(self.wdwPos), ' window positions and ',
              self.TSdata.shape[1],
              'time series', 'and', lag, 'rolling lag runs was',
              end - start, 'sec')
        print('---------Cross-correlation Finished---------')

        return (R_t_all.astype(float), self.param)

    def norm_CC(self, a, b, eng_ref = None, lag = None, ww = 0):
        """Performs the cross-correlation of two 1D input TS, outputting the
        normalised versions. Additional parameters are calculated and output as
        described below.

        Parameters
        ----------
        a : int or float
            Reference time-series of length ``N``.

        b : int or float
            Comparison time-series of length ``N``.

        eng_ref : int or float
            The time-series to be taken as the reference for relative
            integrated energy changes. Default is ``None``.

        lag : int
            The lag value of the given ``b`` time-series. Default is ``None``

        Returns
        -------
        CC : float
            The max Cross-correlation Coefficient for all lag values.

        Lag : int
            The lag at which max cross-correlation was found in sample points.

        Eng_comp : float
            Relative energy comparison.

        freqBandW : float
            Frequency band width in Hertz.
        """
        # Setup parameters:
        ww = len(a)

        # Create the filter taper
        length = len(a)
        if self.param['taper']:
            taper = signal.hann(length)
        else:
            taper = 1

        # Calculate the freqBand width of the smallest lag value.
        if lag == self.param['CC_ref'][0]:
            Fs = self.param['SampFreq']
            T = 1/Fs
            yf = np.fft.fft(signal.detrend(b))
            xf = np.linspace(0.0, 1.0/(2.0*T), length/2)

            yf_amp = np.abs(yf[:length//2])
            freq_dom_amp = yf_amp.max()
            pec_noiseThd = 0.30
            xf_flt = xf[yf_amp > pec_noiseThd * freq_dom_amp]

            freqBandW = (xf_flt[-1] - xf_flt[0])/(xf[-1]-xf[0])

            # freq_dom_index = np.abs(yf[:length//2]).argmax()
            # freq_dom = xf[freq_dom_index]
        else:
            freqBandW = float('NaN')

        # Convert the type to float 32 and taper
        a_tp = a.astype(np.float32)*taper
        b_tp = b.astype(np.float32)*taper

        # Set condition based on user imput and document this in the manual !!!
        if self.param['Eng_ref'] and eng_ref is not None and lag==self.param['CC_ref'][0]:
            eng_ref = eng_ref.astype(np.float32)*taper
            Eng_comp = np.sqrt(sum(b**2)/sum(eng_ref**2))
        elif lag==self.param['CC_ref'][0]:
            Eng_comp = np.sqrt(sum(b**2)/sum(a**2))
        else:
            Eng_comp = float('NaN')


        a_tp = (a_tp - np.mean(a_tp)) / (np.std(a_tp) * length)
        b_tp = (b_tp - np.mean(b_tp)) / np.std(b_tp)

        # Calculation of CC
        Rt = np.correlate(a_tp, b_tp, mode="full")

        if self.param['sig']:
            sig, _ = self.spec_corr_significance(a,b)
        else:
            sig = float('NaN')

        return (Rt.max(), Rt.argmax() - ww, Eng_comp, freqBandW, sig)

    def spec_corr_significance(self, a, b, C_xy=None, tot=2000, confidence = 0.99,
                               maxlag=16, verbose=False):
        ''' Applies a spectral perturbation of phase to assess the
        significance of a correlation. This is similar to the work by Ebisuzaki 1997
        The basic output is the acceptance rate of the alternative to
        the Null Hypothesis H0 that there is no correlation between two time series
        based on the CC for the.
        Parameters
        ----------
        a : int or float
            Reference time-series of length ``N``.

        b : int or float
            Comparison time-series of length ``N``.

        C_xy : int or float
            The normalised cross-correlation coefficient of length corresponding
            to ``maxlag``.

        tot : int
            Total number of random perturbations to check the correlation
            (Default is 2000).

        confidence : float
            The confidence at which the acceptance rate ``H1_AR`` is tested,
            (Default is 0.95).

        maxlag : int
            The maximum sample lags applied in the cross-correlations,
            (Default is 64).

        verbose : bool
            If ``True`` an output of the distribution of the random phase test
            cross-correlations will be made with that of the ``C_xy``
            (Default is 64).

        Returns
        -------
        H1_AR : float
            The rejection rate of the null hypoth ``a``
            and ``b``.

        sigma : float
            The variance of the distribution of random phase perturbed time-series.

        mean : float
            The mean of the distribution of random phase perturbed time-series.

        Z_dist : float
            The distribution of the random phase z-scores.

        Z : float
            The distribution of the C_xy z-scores.

        .. note:: To calculate the confidence associated with the Z-score use
            ``scipy.stats.norm.ppf(sig)``
        '''
        import matplotlib.pyplot as plt
        import scipy.stats as st

        # Setup
        N = round(len(a)/2)*2
        if maxlag is None:
            maxlag = N-1

        # Normalize the correlation for testing the FPR
        b = (b - np.mean(b)) / (np.std(b))
        a = (a - np.mean(a)) / (np.std(a) * N)

        # Compute the decay rate
        tau = self.decay_rate(a,b, verbose=verbose)

        # Compute the fft of time series a
        A = np.fft.fft(a)

        # Correlate with series b
        if C_xy == None:
            C_xy = np.correlate(a, b, mode="full")

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

        # Prob. of rejecting the H0 null hypothesis "No correlation present"
        # and accepting the H1 hypothesis, a correlation is present
        P_H1 = st.norm.ppf(confidence)
        H1_AR = len(Z[Z>P_H1])/len(Z) # The H1 acceptance rate

        dist_dic = {'sigma': sigma,
                    'mean': mean,
                    'Z_dist': Z_dist,
                    'Z': Z}

        if verbose == True:
            fig = plt.figure()
            ax0 = fig.add_subplot(111)
            ax0.hist(Z_dist, bins = 'auto', normed=True, label='Z_dist')
            ax0.hist(Z, bins = 'auto', normed=True, label='Z of Correlation')
            ax0.set_xlabel('Z-score')
            ax0.set_ylabel('Density')
            ax0.legend()

        return H1_AR, dist_dic

    def decay_rate(self, a,b, verbose=False):
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
        import matplotlib.pyplot as plt
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
        C_xy = np.correlate(a, b, mode="full")[N:]

        # Calculate the effective
        C_xy_Hilb = hilbert(C_xy)
        C_xy_env = np.abs(C_xy_Hilb)


        # Fit exponential function to the envelope of the decay
        def _func_exp(x, a, b):
            return a*np.exp(-b * x)

        for CC in [C_xy_env]:
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
            ax0.plot(lags, C_xy, label='CC of xy')
            ax0.plot(lags, C_xy_env, label='Envelope of xy')
            ax0.plot(lags[fitRange], x_fit, label='fit_exp_of max AC tau')
            ax0.legend()

        return tau_max
