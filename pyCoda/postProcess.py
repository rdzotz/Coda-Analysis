#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:17:07 2017

@author: rwilson
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.widgets import Slider
import matplotlib.patches as patches
import itertools

class post_utilities:
    ''' A collection of post processing utility functions
    '''


    def detect_most_linear(x, y, m, smooth = False, wdw_wdth = 25, poly_ord=3 ):
        ''' Finds the most linear portion of a line via
        Inputs:
        ------
        x : Series
            y-axis array of values
        y : Serues
            y-axis array of values
        m : Length of most linear portion of line (x,y)
        smooth : bool
            Smooth the curve before search

        Outputs:
        --------
        L_end: (x_end, y_end) values at the end of the line
        idx: Indices of the most linear portion of the line
        '''

        import operator
        from scipy.signal import savgol_filter

        # Pre-smooth the curve
        x = pd.Series(savgol_filter(x, wdw_wdth, poly_ord))
        y = pd.Series(savgol_filter(y, wdw_wdth, poly_ord))

        # Filter out data above max stress
        yMaxIdx = max(enumerate(y), key=operator.itemgetter(1))
        y = y.iloc[:yMaxIdx[0]]
        x = x.iloc[:yMaxIdx[0]]
        n = x.shape[0]

        # Plot initial data
        plt.figure()
        plt.plot(x,y)

        # find the best linear section of length m
        threshold = 0.98
        slope = -float('inf')

        for i in range(0, n-m):
            x_slice = x.iloc[i:i+m-1]
            y_slice = y.iloc[i:i+m-1]

            slopei, intercepti = np.polyfit(x_slice, y_slice, deg=1)

            end_yi = np.polyval([slopei, intercepti], x_slice.iloc[-1])

            Per_sim = 1-abs(end_yi - y_slice.iloc[-1])/np.mean([end_yi, y_slice.iloc[-1]])

            if Per_sim>threshold and slopei>slope:
                slope = slopei
                intercept = intercepti
                x_fit = x_slice
                end_y = np.polyval([slopei, intercepti], x_slice.iloc[-1])

        y_fit = pd.DataFrame(np.polyval([slope, intercept], x_fit))

        plt.plot(x_fit, y_fit)

        L_end = (x_fit.iloc[-1], end_y)

        #fit_DF = pd.concat([y_fit, x_fit.reset_index()])

        return (L_end, x_fit.index[-1])


    def _yield_search(DF):
        ''' Search for the yield point
        '''
        from scipy.signal import savgol_filter

        wdw_wdth = 25
        poly_ord = 3

        if self.newParam['LLLength']>0:
            if isinstance(self.PVdata, pd.DataFrame):
                x = self.PVdata['Strain Ax. [%]']
                y = self.PVdata['Stress [MPa]']
            else:
                x = self.PV_df['Strain Ax. [%]']
                y = self.PV_df['Stress [MPa]']

            # Pre-smooth the curve
            #x = savgol_filter(x, wdw_wdth, poly_ord)
            #y = savgol_filter(y, wdw_wdth, poly_ord)
            Yield_p, idx = pre_utilities.detect_most_linear(x, y,
                           self.newParam['LLLength'])
            self.newParam['Yield_p'] = Yield_p
            self.newParam['idx'] = idx


    def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

        """Detect peaks in data based on their amplitude and other features.

        Parameters
        ----------
        x : 1D array_like
            data.
        mph : {None, number}, optional (default = None)
            detect peaks that are greater than minimum peak height.
        mpd : positive integer, optional (default = 1)
            detect peaks that are at least separated by minimum peak distance (in
            number of data).
        threshold : positive number, optional (default = 0)
            detect peaks (valleys) that are greater (smaller) than `threshold`
            in relation to their immediate neighbors.
        edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
            for a flat peak, keep only the rising edge ('rising'), only the
            falling edge ('falling'), both edges ('both'), or don't detect a
            flat peak (None).
        kpsh : bool, optional (default = False)
            keep peaks with same height even if they are closer than `mpd`.
        valley : bool, optional (default = False)
            if True (1), detect valleys (local minima) instead of peaks.
        show : bool, optional (default = False)
            if True (1), plot data in matplotlib figure.
        ax : a matplotlib.axes.Axes instance, optional (default = None).

        Returns
        -------
        ind : 1D array_like
            indeces of the peaks in `x`.

        Notes
        -----
        The detection of valleys instead of peaks is performed internally by simply
        negating the data: `ind_valleys = detect_peaks(-x)`

        The function can handle NaN's

        See this IPython Notebook [1]_.

        References
        ----------
        .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

        Examples
        --------
        >>> from detect_peaks import detect_peaks
        >>> x = np.random.randn(100)
        >>> x[60:81] = np.nan
        >>> # detect all peaks and plot data
        >>> ind = detect_peaks(x, show=True)
        >>> print(ind)

        >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
        >>> # set minimum peak height = 0 and minimum peak distance = 20
        >>> detect_peaks(x, mph=0, mpd=20, show=True)

        >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
        >>> # set minimum peak distance = 2
        >>> detect_peaks(x, mpd=2, show=True)

        >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
        >>> # detection of valleys instead of peaks
        >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

        >>> x = [0, 1, 1, 0, 1, 1, 0]
        >>> # detect both edges
        >>> detect_peaks(x, edge='both', show=True)

        >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
        >>> # set threshold = 2
        >>> detect_peaks(x, threshold = 2, show=True)
        """

        def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
            """Plot results of the detect_peaks function, see its help."""
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print('matplotlib is not available.')
            else:
                if ax is None:
                    _, ax = plt.subplots(1, 1, figsize=(8, 4))

                ax.plot(x, 'b', lw=1)
                if ind.size:
                    label = 'valley' if valley else 'peak'
                    label = label + 's' if ind.size > 1 else label
                    ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                            label='%d %s' % (ind.size, label))
                    ax.legend(loc='best', framealpha=.5, numpoints=1)
                ax.set_xlim(-.02*x.size, x.size*1.02-1)
                ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
                yrange = ymax - ymin if ymax > ymin else 1
                ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
                ax.set_xlabel('Data #', fontsize=14)
                ax.set_ylabel('Amplitude', fontsize=14)
                mode = 'Valley detection' if valley else 'Peak detection'
                ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                             % (mode, str(mph), mpd, str(threshold), edge))
                # plt.grid()
                plt.show()

        x = np.atleast_1d(x).astype('float64')
        if x.size < 3:
            return np.array([], dtype=int)
        if valley:
            x = -x
        # find indices of all peaks
        dx = x[1:] - x[:-1]
        # handle NaN's
        indnan = np.where(np.isnan(x))[0]
        if indnan.size:
            x[indnan] = np.inf
            dx[np.where(np.isnan(dx))[0]] = np.inf
        ine, ire, ife = np.array([[], [], []], dtype=int)
        if not edge:
            ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
        else:
            if edge.lower() in ['rising', 'both']:
                ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
            if edge.lower() in ['falling', 'both']:
                ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
        ind = np.unique(np.hstack((ine, ire, ife)))
        # handle NaN's
        if ind.size and indnan.size:
            # NaN's and values close to NaN's cannot be peaks
            ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
        # first and last values of x cannot be peaks
        if ind.size and ind[0] == 0:
            ind = ind[1:]
        if ind.size and ind[-1] == x.size-1:
            ind = ind[:-1]
        # remove peaks < minimum peak height
        if ind.size and mph is not None:
            ind = ind[x[ind] >= mph]
        # remove peaks - neighbors < threshold
        if ind.size and threshold > 0:
            dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
            ind = np.delete(ind, np.where(dx < threshold)[0])
        # detect small peaks closer than minimum peak distance
        if ind.size and mpd > 1:
            ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
            idel = np.zeros(ind.size, dtype=bool)
            for i in range(ind.size):
                if not idel[i]:
                    # keep peaks with the same height if kpsh is True
                    idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                        & (x[ind[i]] > x[ind] if kpsh else True)
                    idel[i] = 0  # Keep current peak
            # remove the small peaks and sort back the indices by their occurrence
            ind = np.sort(ind[~idel])

        if show:
            if indnan.size:
                x[indnan] = np.nan
            if valley:
                x = -x
            _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

        return ind


    def MLTWA_calc(TS_DB, after_FB, Ewdth = None, wd_shift = 0,
                   ref_trace = 0,mph = None, mpd=None, threshold=0, R1_Sign=False,
                   grad_period=-50,
                   verbose=False):
        '''Apply Multi Lapse-Time Window Analysis on the input database time
        -series. Additional parameters are calculated such as the B0 or ratio of
        R1 to R2 as well as their gradient difference.

        Parameters
        ----------
        TS_DB : DataFrame
            Time-series database in cronological order.
        after_FB : DataFrame.index
            Index of ``TS_DB`` after which the S-wave max value will be found
        Ewdth : list, optional (default = [len(TS_DB)//16,len(TS_DB)//16,len(TS_DB)//4]
            The first two energy window widths in sample points
        wd_shift : int
            Shift parameter of the start of windows
        ref_trace : int
            The trace to use as a reference for calculation of R2
        mph : {None, number}, optional (default = None)
            detect peaks that are greater than minimum peak height.
        mpd : positive integer, optional (default = 1)
            detect peaks that are at least separated by minimum peak distance (in
            number of data).
        threshold : positive number, optional (default = 0)
            detect peaks (valleys) that are greater (smaller) than `threshold`
            in relation to their immediate neighbors.
        R1_Sign : bool (Default False)
            Apply sign change to R1.
        grad_period : int (Default False)
            The number of periods to use when calculating the gradient of ratios R1
            and R2.
        verbose: bool, optional (default = False)
            Provide a details output

        Returns
        -------
        DB_MLTWA : DataFrame
            Containing all of the MLTWA data with index equal to third
            expected Time column of the input ``TS_DB``.
        dict_MLTWA: dictcontaining parameters pertaining to the MLTWA processing.
              -**R1**: Ratio of ``log_10 E_1(t_i)/ E_3(t_i)``
              -**R2**: Ratio of ``log_10 E_1(t_u)/ E_1(t_i)``
              -**E1**: Integrated energy early S-wave
              -**E2**: Integrated energy mid S-wave
              -**E3**: Integrated energy late S-wave
              -**E_wdw_pos**: list of start, mid, end of ``E1,E2,E3``
              -**idx_break**: list detected peaks in search of the S-wave arrival
        verbose: bool, optional (default = False)
            Provide a detailed output

        Examples
        --------
        >>> import postProcess as pp
        >>> idx = pp.post_utilities.MLTWA_calc(TScut_DB_in, after_FB = 0.00008,
                                               wd_shift=-500, mph=None, mpd=12,
                                               threshold=0, verbose=True)
        '''

        if Ewdth is None:
            length = TS_DB.shape[0]
            Ewdth = [length//16, length//16, length//4]
        TS_DB_copy = TS_DB.copy()

        # Set all before after_FB to zero
        TS_DB.loc[TS_DB.index < after_FB] = 0

        idx_break = []
        # Detect the onset of the S-wave
        for col in TS_DB.columns.values:
            idx = post_utilities.detect_peaks(TS_DB[col], mph = mph,
                                              threshold=threshold, mpd=mpd)
            if idx.size == 0: # Catch no detections
                idx = [0]
                print('No detected peaks found with parameters in trace %d, check inputs' % col )

            # Append the max of all detected
            idx_break.append(int( idx[ np.argmax(TS_DB[col].as_matrix()[idx]) ] ) )


        # Start of E1 based on average of S-wave peak
        E1sta = int(np.mean(idx_break)) + wd_shift
        dict_MLTWA = {'idx_break': idx_break,
                      'E1sta': E1sta}

        # Empty Dataframe for MLTWA processing
        DB_MLTWA = pd.DataFrame(index=TS_DB.T.index,
                                columns = ['E1', 'E2', 'E3', 'R1', 'R2'])
        # Create the windows based on E1 start
        for no,Ewd in enumerate(Ewdth):
            sta = 'E'+str(no+1)+'sta'
            end = 'E'+str(no+1)+'end'
            dict_MLTWA[end] = dict_MLTWA[sta] + Ewd
            if no < 2:
                dict_MLTWA['E'+str(no+2)+'sta'] = dict_MLTWA[end]

            # ----------- Integrate the Energy in each window -------------
            DB_MLTWA['E'+str(no+1)] = np.sqrt(
                    (TS_DB_copy.iloc[dict_MLTWA[sta]:dict_MLTWA[end],:]**2)
                    .sum()).as_matrix()
            # Normalise each energy window
#            dict_MLTWA['E'+str(no+1)] = dict_MLTWA['E'+str(no+1)] #\
                                        #/dict_MLTWA['E'+str(no+1)][ref_trace]

        # Calculate the ratios
        DB_MLTWA['R1'] = np.log10(DB_MLTWA['E1']/DB_MLTWA['E3'])
        if R1_Sign: DB_MLTWA['R1'] = DB_MLTWA['R1'] * -1
        DB_MLTWA['R2'] = np.log10(DB_MLTWA['E1'].iloc[ref_trace]/DB_MLTWA['E1'])

        # Zero Shift the ratios and calculate B0
        DB_MLTWA['R1'] = DB_MLTWA.R1 - DB_MLTWA.R1.iloc[0]
        DB_MLTWA['R2'] = DB_MLTWA.R2 - DB_MLTWA.R2.iloc[0]
        DB_MLTWA['R3'] = DB_MLTWA.R2 - DB_MLTWA.R1
        DB_MLTWA['B0'] = (DB_MLTWA.R1 +1).values/(DB_MLTWA.R2+1).values-1

        # Calculate the gradient and differences of the ratios
        DB_MLTWA['R1_grad'] = DB_MLTWA.R1.diff(grad_period)
        DB_MLTWA['R2_grad'] = DB_MLTWA.R2.diff(grad_period)
        DB_MLTWA['R2R1_gradDiff'] = DB_MLTWA.R2_grad - DB_MLTWA.R1_grad

        # Normalise The Energy windows
        for no,_ in enumerate(Ewdth):
            DB_MLTWA['E'+str(no+1)] = DB_MLTWA['E'+str(no+1)] \
                                        /DB_MLTWA['E'+str(no+1)].iloc[ref_trace]

        # Drop levels greater that 3
        DB_MLTWA.index = DB_MLTWA.index.droplevel([level for
                                                   level in
                                                   range(3,DB_MLTWA.index.nlevels)])
        DB_MLTWA = DB_MLTWA.unstack(level=[0,1])
        DB_MLTWA.index = pd.to_datetime(DB_MLTWA.index)

        # ----------- Add interactive check of first break picking -------------
        if verbose:
            ts_slider, hdl, hdl_break, hdl_thsh = post_utilities.\
                                        TS_interactive(TS_DB_copy, idx_break,
                                          wdws=[[dict_MLTWA['E1sta'],
                                            dict_MLTWA['E1end']],
                                           [dict_MLTWA['E2sta'],
                                            dict_MLTWA['E2end']],
                                           [dict_MLTWA['E3sta'],
                                            dict_MLTWA['E3end']]])

        return DB_MLTWA, dict_MLTWA, (ts_slider, hdl, hdl_break, hdl_thsh)


    def PV_segmentation(PV_df, Segments, targets, indexName='index',
                        shiftCols = None, verbose=False):
        '''Apply a range of data parameterisation methods from segments of input
        data. Note, any row with a ``nan`` will be removed before segmentation.

        Parameters
        ----------
        PV_df : DataFrame
            Time-series database in cronological order.
        Segments: list
            A list of indicies at which between which the input ``PV_df`` will be
            segmented and paramterised.
        targets: list
            List of column names in ``PV_df``.
        indexName : Str (Default 'index')
            The index in which the ``Segments`` are defined.
        shiftCols : list (Default = None)
            List of columns to begin as zero (The first value will be subtracted
            from all)
        verbose: bool, optional (default = False)
            Provide a detailed output

        Returns
        -------
        dict_Segments: dictcontaining parameters for each segment
              -**Sigma**: Sigma of each segment
              -**R2**: R2 of each segment
              -**Mean**: mean of each segment
              -**Skewedness**: mean of each segment
        Examples
        --------
        >>> import postProcess as pp
        >>> seg_list = [[4, 19.5], [21.74, 37.2], [39.68, 55.29]] # In Hours
        >>> DF_Segments = pp.post_utilities.PV_segmentation(PV_df_in, seg_list,
                                targets = ['Pore pressure| [MPa]','R1','R2'])
        '''

        # Drop any nan row with nan before processing begins
        PV_df = PV_df.dropna().copy()

        # shift to zero all columns in shiftCols
        if shiftCols:
            PV_df.loc[:, shiftCols] = PV_df.loc[:, shiftCols] - \
                                        PV_df.loc[:, shiftCols].iloc[0]

        # DataFrame for segment stats
        SegIdx = [str(seg[0])+'-'+str(seg[1]) for seg in Segments]
        df_out = pd.DataFrame(index=SegIdx, columns=None)

        # Index from input data for masking each segment
        maskIdx = getattr(PV_df, indexName)

        for ind, seg in enumerate(SegIdx):
            mask = (maskIdx >= Segments[ind][0]) & (maskIdx < Segments[ind][1])
            PV_df_seg = PV_df[mask]
            segIdx = getattr(PV_df_seg, indexName)
            # PV_df_seg = PV_df.iloc[seg:Segments[ind+1]]
            # For each target
            for target in targets:
                df_out.loc[[seg], str(target)+'-mean'] = PV_df_seg[target].mean()
                df_out.loc[[seg], str(target)+'-std'] = PV_df_seg[target].std()
                df_out.loc[[seg], str(target)+'-skew'] = PV_df_seg[target].skew()
                df_out.loc[[seg], str(target)+'-min'] = PV_df_seg[target].min()
                df_out.loc[[seg], str(target)+'-max'] = PV_df_seg[target].max()

                # Check fit quality

                slope, intercept, r_value, p_value, std_err = stats.linregress(
                        segIdx, PV_df_seg[target])
                df_out.loc[[seg], str(target)+'-slope'] = slope
                df_out.loc[[seg], str(target)+'-intercept'] = intercept
                df_out.loc[[seg], str(target)+'-R^2'] = r_value**2
                df_out.loc[[seg], str(target)+'-pValue'] = p_value

        df_out.reset_index(inplace=True)
        return df_out


    def TS_FBP(TS_DB, noiseWD, threshold=1, threshold_shift=0, mpd=1, verbose=False):
        '''This function is intended to perform first break picking

        Parameters
        ----------
        TS_DB : DataFrame
            Columnweise DataFrame of TS data, index of time expected
        noiseWD : int
            Window width in TS_DB index which is expected to be only noise in
            number of sample points.
        threshold : int/float
            Percentage of the noise standard deviation which will define the
            detection threshold.
        threshold_shift : float
            A shift to the (threshold * noiseStd)
        mpd : int
            Minimum poit distance in number of samples
        verbose : bool
            If True, the interactive plotting of picks will be made

        Returns
        -------
        idx_break : int
            index of the first break detection.
        '''
        TS_DB_copy = TS_DB.copy() - TS_DB.mean()

        # Reset each trace to zero based on the average
        TS_DB = TS_DB - TS_DB.mean()

        # Calculate the mean amp within the noise window
        noiseStd = TS_DB.query('index <'+str(noiseWD)).std()

        # set all values in the noise window to 0
        TS_DB.loc[TS_DB.index < noiseWD] = 0

        # Run the detection for each trace
        TS_DB = abs(TS_DB)

        df_FBP = pd.DataFrame(index=TS_DB_copy.columns, columns=['FBP'])
        idx_break = []
        thresholds = []
        for col, thsh in zip(TS_DB.columns.values,
                             abs(noiseStd)*threshold + threshold_shift):
            print('Threshold:', thsh)
            idx = post_utilities.detect_peaks(TS_DB[col],
                                              mph=thsh, mpd=mpd, show=False)
            if idx.size == 0:  # Catch no detections
                idx = [0]

            idx_break.append(int(idx[0]))
            thresholds.append(thsh)

        #  ----------- Add interactive check of first break picking -----------
        ts_slider, hdl, hdl_break, hdl_thsh = post_utilities.\
            TS_interactive(TS_DB_copy, idx_break, thresholds, noiseWD)

        df_FBP.loc[:, 'FBP'] = idx_break
        return df_FBP, (ts_slider, hdl, hdl_break, hdl_thsh)


    def TS_interactive(TS_DB, idx=None, threshold=False, noiseWD=False,
                       wdws=None):
        '''Interactive plot of time-series data with ability to plot detection
        parameters for analysis.

        Parameters
        ----------
        TS_DB : dataframe
            Database of all input time series in each columns
        idx : int (default None)
            The index of the detected peak
        threshold : list (default False)
            Thresholds for each trace in terms of amplitude
        noiseWD : int (default False)
            The noise zeroed window length
        wdws : list (default None)
            A list of window index start stop positions e.g. [[sta, stp]...].
                '''

        import matplotlib.patches as patches

        fig = plt.figure()

        # Add TS data pannel below main plot
        ax1 = plt.subplot(211)
        init_pos = TS_DB.shape[1]//2
        hdl = ax1.plot(TS_DB.iloc[:, init_pos].index.values,
                       TS_DB.iloc[:, init_pos])

        # --------------------- Add TS slider -----------------------
        slider_color = 'lightgoldenrodyellow'
        ts_slider_ax = fig.add_axes([0.4, 0.4, 0.50, 0.02],
                                    facecolor=slider_color)
        ts_slider = Slider(ts_slider_ax, 'Time Series',
                           0, int(TS_DB.shape[1]-1),
                           valfmt='%d',
                           valinit=init_pos)
        # just idx and thresholds
        if idx and threshold and noiseWD:
            # Add the break detection point
            detec_x = TS_DB.index[idx[init_pos]]
            detec_y = TS_DB.iloc[idx[init_pos], init_pos]


            hdl_break = ax1.plot(detec_x, detec_y, 'r*')

            hdl_thsh = ax1.hlines(y=threshold[init_pos],
                                  xmin = TS_DB.iloc[:,init_pos].index.min(),
                                  xmax = TS_DB.iloc[:,init_pos].index.max())
            ax1.axvline(x=noiseWD,
                        ymin = -TS_DB.abs().max(axis=0).max(),
                        ymax = TS_DB.abs().max(axis=0).max())

            def slider_ts_pos_on_change(val):
                trace_no = int(ts_slider.val)
                detec_x = TS_DB.index[idx[trace_no]]
                #detec_x = TS_DB[trace_no].index.values[idx[trace_no]]
                detec_y = TS_DB.iloc[idx[trace_no], trace_no]
                hdl[0].set_ydata(TS_DB.iloc[:, trace_no])
                hdl_thsh = ax1.hlines(y=threshold[trace_no],
                                  xmin = TS_DB.iloc[:,trace_no].index.min(),
                                  xmax = TS_DB.iloc[:,trace_no].index.max())
                hdl_break[0].set_data(detec_x, detec_y)
                fig.canvas.draw_idle()
        # Just index and window positions
        elif idx and wdws:
                        # Add the break detection point
            detec_x = TS_DB.index[idx[init_pos]]
            detec_y = TS_DB.iloc[idx[init_pos], init_pos]

            hdl_break = ax1.plot(detec_x, detec_y, 'r*')
            hdl_thsh = None

            height = TS_DB.max().max()*2
            y = -height/2

            for wdw in wdws:
                x = TS_DB.index.values[wdw[0]]
                width = (TS_DB.index.values[wdw[1]]-TS_DB.index.values[wdw[0]])

                hdlPatch = ax1.add_patch(
                    patches.Rectangle(
                        (x, y),   # (x,y)
                        width,          # width
                        height,          # height
                        alpha=0.1,
                        linewidth = 3,
                        edgecolor='violet'
                    )
                )


            def slider_ts_pos_on_change(val):
                trace_no = int(ts_slider.val)
                detec_x = TS_DB.index[idx[trace_no]]
                #detec_x = TS_DB[trace_no].index.values[idx[trace_no]]
                detec_y = TS_DB.iloc[idx[trace_no], trace_no]
                hdl[0].set_ydata(TS_DB.iloc[:, trace_no])

                hdl_break[0].set_data(detec_x, detec_y)
                fig.canvas.draw_idle()
        # Only the traces
        else:
            hdl_break = None
            hdl_thsh = None

            def slider_ts_pos_on_change(val):
                trace_no = int(ts_slider.val)

                hdl[0].set_ydata(TS_DB.iloc[:, trace_no])

                fig.canvas.draw_idle()

        ts_slider.on_changed(slider_ts_pos_on_change)

        plt.show()
        return ts_slider, hdl, hdl_break, hdl_thsh

    def CC_ch_drop(CC_DB, channels=None, errors='raise'):
        '''Drops channels from standard CCdata dataframe

        Parameters
        ----------
        CC_DB : dataframe
            Database containing in the first two levels src and rec numbers
        channels : int (default =  None)
            the channels to remove from the dataframe, either a list of channels
            in which case both source and receivers will be dropped, or a list of
            channel pairs in which case only the defined pairs will be dropped.
        errors : str (dfault = 'raise')
            Raise error in attempted drop does not exist, 'ignore' to surpress
            the error.
        '''

        if isinstance(channels[0], int):
            for chan in channels:
                CC_DB.drop(chan, axis=0, level=0, inplace=True, errors=errors)
                CC_DB.drop(chan, axis=0, level=1, inplace=True, errors=errors)
        else:
            srcDrop = [ch[0] for ch in channels]
            recDrop = [ch[1] for ch in channels]

#            CC_DB.drop(pd.MultiIndex.from_arrays([srcDrop, recDrop]), inplace=True)

            m = pd.MultiIndex.from_arrays([srcDrop,recDrop])

            CC_DB = CC_DB[~CC_DB.reset_index(level=2, drop=True).index.isin(m)]

        return CC_DB

#            fn = CC_DB.index.get_level_values
#            CC_DB = CC_DB[~(fn(0).isin(As) | fn(1).isin(Bs))]
#
#            for src, rec in zip(srcDrop, recDrop):
#                CC_DB_drop_idx = CC_DB.loc[(src, rec, slice(None)), :].index
#                CC_DB.drop(CC_DB_drop_idx, inplace=True, errors=errors)


    def calcSNR(TSsurvey, Noise_channels, all_channels, wdws, noiseCutOff=0, inspect=False):
        '''Determines the channels which are above a certain SNR.

        Parameters
        ----------
        TSsurvey : dataframe
            Single survey dataframe of traces
        Noise_channels : list
            the channels on which the noise will be estimated
        all_channels : list
            All channels numbers
        wdws : list
            The windows in samples points at which the SNR is calculated
        noiseCutOff: float
            The threshold of SNR in Db to filter
        inspect: bool (default = False)
            Inspect the traces which are noted as noise.
        Returns
        -------
        noiseyChannels : float
            The noisey channels
                '''
        # Generate all combinations of two

        noise_pairs = list(itertools.product(all_channels, Noise_channels,repeat = 1))
        noise_pairs = list(set(noise_pairs + [t[::-1] for t in noise_pairs]))


        traceMask = False
        for pair in noise_pairs:
            temp = (TSsurvey.columns.get_level_values('srcNo') ==[pair[0]]) & (TSsurvey.columns.get_level_values('recNo') == pair[1])
            traceMask = temp + traceMask

        NoiseTraces =  TSsurvey.loc[:, traceMask]

        if inspect:
            post_utilities.TS_interactive(NoiseTraces)
            plt.pause(10)

        print('-------- Calculate SNR for each window --------\n')
        wdws_num = [ [int(wd.split('-')[0]), int(wd.split('-')[1])] for wd in wdws]
        for wd in wdws_num:
            print('window: %s' %wd)
            temp = 10*np.log10((TSsurvey.iloc[wd[0]:wd[1]]**2).mean().pow(1/2)) - \
                              10*np.log10((NoiseTraces.iloc[wd[0]:wd[1]]**2).\
                                          mean().pow(1/2).mean())
            try:
                SNR = pd.concat([SNR, temp], axis=1)
            except (NameError, UnboundLocalError):
                SNR = temp

        if isinstance(SNR, pd.Series):
            SNR = pd.DataFrame(SNR)

        noisyCh = SNR[SNR.mean(axis=1).between(left=-np.inf, right=noiseCutOff)].\
            index.droplevel(level=[2, 3, 4, 5, 6, 7, 8, 9]).unique().values.tolist()
        return noisyCh, SNR, NoiseTraces


    def smooth(x, window_len = 11, window = 'hanning'):
        """smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal

        example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

        see also:

        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter

        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")


        if window_len<3:
            return x


        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=numpy.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        return y[0:len(x)]

    def PV_time_shift(PVdata, PV_time_col, PV_time_col_unit, PVstart_date_time):
        '''Adjusts the Time Stamp column in the PVdata frame.

        Parameters
        ----------
        PVdata : DataFrame
            The dataframe on which operations are performed
        PV_time_col : str
            The name of the PV time column to be processed.
        PV_time_col_unit : str
            Unit of the time column ``PV_time_col``.
        PVstart_date_time : str
            The YYYY-MM-DD ..etc string defining the origin of the ``PV_time_col``.


        Examples
        --------
        >>> # Determine the origin time of the PVdata
        >>> origin = pd.to_datetime('20180212180703') - pd.Timedelta(43.6725, unit='D')
        >>> # Shift the PVdata to the new origin
        >>> pp.post_utilities.PV_time_shift(PVdata, 'Time(Days)', 'D', origin)
        >>> # Save the new PVdata back to the HDF5 database
        >>> dt.utilities.DB_pd_data_save('Ultrasonic_data_DB/Database.h5', 'PVdata', PVdata)

        '''

        # Create new Time Stamp column in PVdata
        PVdata['Time Stamp'] = pd.to_datetime(
                PVdata[PV_time_col],
                unit=PV_time_col_unit,
                origin = PVstart_date_time)



class postProcess:
    '''This class is intended to handel post processing of a database of
    CC and PV data.
    '''
    def __init__(self, param, PV_df, CC, TS = []):
        self.param = param
        self.TS = TS
        self.PV_df = PV_df
        self.CC = CC

    def PV_CC_join(self):
        ''' Combine both PV and CC data into a single dataset based on
        measurement number
        '''

        # Measurement numbers to join by
        meas = ['Meas'+str(col) for col in range(1, self.PV_df.shape[0]+1)]

        # Add to PV df
        self.PV_df['measNo'] = meas

        # Merge
        df_merged = pd.merge(self.PV_df,self.CC, how='left', left_on='measNo', right_index=True)

        return df_merged

    def TS_DB(self):
        ''' Convert TS matrix into a DataFrame with axis index in time
        '''

        TS_DB = pd.DataFrame(self.TS, index=self.param['t_axis'],
                             dtype=np.float64)

        return TS_DB


    def postProcess_run(self):
        ''' Perform expected Post Processing
        '''

        # Join the PV and CC datasets into one
        PV_CC = self.PV_CC_join()

        # Add TS data into pandas dataframe with t_axis as index
        TS_DB = self.TS_DB()

        # Perform first break picking and add to PV_CC if requested
        if self.param['FBP']:
            idx_break = post_utilities().TS_FBP(TS_DB, self.param['noiseWD'],
                                    self.param['stdThd'],
                                    self.param['thdFactor'],
                                    self.param['mpd'])
            PV_CC['TOF'] =  TS_DB.index.values[idx_break]


        return TS_DB, PV_CC
