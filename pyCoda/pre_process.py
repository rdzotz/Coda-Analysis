#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:06:31 2017

@author: rwilson
"""

"""
Implements the global import of all data

Created on Mon Dec 26 20:51:08 2016
@author: rwilson
"""

import matplotlib.pylab as plt

import numpy as np
import pandas as pd
import re
import data as dt


from scipy.signal import butter, lfilter, filtfilt, freqz
from scipy.fftpack import fft, fftfreq, rfft, irfft
from matplotlib.widgets import Slider

class frequency:
    ''' A collection of functions for the inspection and processing of the
    frequency data. Written to take in a database of TSdata formate and filter in
    terms of frequency.

    Parameters
    ----------
    TSdata : DataFrame
         A dataframe containing all time series data for frequency filtering.
    fs
    '''


    def __init__(self, TSdata, fs, verbose=True):
        self.TSdata = TSdata
        self.TSfft = TSdata.apply(rfft)
        self.TSfft.iloc[0, :] = 0  # Set zero freq
        self.TSfft.index = fftfreq(TSdata.shape[0]*2, 1/fs)[range(TSdata.shape[0])]
        self.fs = fs
        self.verbose = verbose

    def TS_DB_filter(self,lowcut, highcut, order):
        '''Perform the frequency filtering of the entire dataframe
        '''

        self.TSdata  = self.TSdata.apply(lambda x:
                                      self.butter_bandpass_filter(
                                              x, lowcut, highcut, order),
                                              axis=0 )
#        self.TSdata = self.TSfft.apply(irfft)

    def butter_bandpass(self, lowcut, highcut, order=5):
        ''' Define the butterworth frequency filter window

        Parameters
        ----------
        lowcut : float (Default None)
            Low cut Filter defining the lower cutoff frequency
        highcut : float (Default None)
            High cut Filter defining the upper cutoff frequency

        Returns
        -------
        b, a: ndarray
            Numerator (b) and denominator (a) polynomials of the IIR filter.
        '''
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, order=5):
        ''' Apply the butter bandpass filter to the data.

        Parameters
        ------
        data : array
        lowcut : float (Default None)
            Low cut Filter defining the lower cutoff frequency
        highcut : float (Default None)
            High cut Filter defining the upper cutoff frequency

        Returns
        -------
        b, a: ndarray
            Numerator (b) and denominator (a) polynomials of the IIR filter.
        '''
        b, a = self.butter_bandpass(lowcut, highcut, order=order)
        #y = lfilter(b, a, data)
        y = filtfilt(b, a, data)
        return y

    def freq_filter_insp(self, lowcut, highcut, Worder=6):
        ''' Inspect the frequency filter taper before application

        Parameters:
        ------
        lowcut : float (Default None)
            Low cut Filter defining the lower cutoff frequency
        highcut : float (Default None)
            High cut Filter defining the upper cutoff frequency
        Worder : int (Default 6)
            The order of the butterworth filter
        '''

        # Plot the frequency response for a few different orders.
        plt.figure(1)
        plt.clf()
        for orde in [Worder-2, 6, Worder+2]:
            b, a = self.butter_bandpass(lowcut, highcut, order=orde)
            w, h = freqz(b, a, worN=2000)
            plt.plot((self.fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % orde)

        plt.plot([0, 0.5 * self.fs], [np.sqrt(0.5), np.sqrt(0.5)],
                 '--', label='sqrt(0.5)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')

    def TS_FS_interactive(self):
        '''Interactive plot of time-series data in the frequency spectrum.
                '''

        fig = plt.figure()

        # Add TSfreq data pannel below main plot
        ax1 = plt.subplot(211)
        init_pos = self.TSfft.shape[1]//2
        hdl = ax1.plot(self.TSfft.iloc[:, init_pos].index.values,
                       self.TSfft.iloc[:, init_pos].abs())

        # --------------------- Add TS slider -----------------------
        slider_color = 'lightgoldenrodyellow'
        ts_slider_ax = fig.add_axes([0.4, 0.4, 0.50, 0.02],
                                    facecolor=slider_color)
        ts_slider = Slider(ts_slider_ax, 'Time Series FFT',
                           0, int(self.TSfft.shape[1]-1),
                           valfmt='%d',
                           valinit=init_pos)

        # Define the slider
        def slider_ts_pos_on_change(val):
            trace_no = int(ts_slider.val)

            hdl[0].set_ydata(self.TSfft.iloc[:, trace_no].abs())

            fig.canvas.draw_idle()

        ts_slider.on_changed(slider_ts_pos_on_change)

        plt.show()
        return ts_slider, hdl,

class preProcess:
    """This class performs various pre-processing operations on imported Time
    Series and Perturbation Vector information. The PVdata is loaded into memory
    with any modifications stored to the database at the end of processing.

    Parameters
    ----------
    self.Database : str
        Defines a common hdf5 database name ``Database.h5`` including relative path
    self.newParam : dict
        A dictionary of the most recent user input parameters
    self.verbose : bool
        If ``True`` additional print to screen will occure
    self.PVdata : DataFrame
        Database of the PVdata found in self.Database

    Notes
    -----
    """


    def __init__(self, Database, newParam, verbose = True):
        self.Database = Database
        self.newParam = newParam
        self.verbose = verbose
        self.PVdata = dt.utilities.DB_pd_data_load(self.Database, 'PVdata')

    def TSslice(self):
        '''DEPRECIATED
        The slicing is intended to truncate all TS information before FB
        and after the coda is noise dominated. Additionally, the removal of TS
        at the beginning and end of the experiment is possible thr STA_meas and
        END_meas. The PV dataframe is also sliced accordingly.
        Parameters
        ----------
        Frome user defined param DataFrame

        TODO:
            * write time matching def
        '''

        # If recNo exists in input data, then slice TS and TShdrs
        if 'recNo' in self.newParam:
            recNo_list = self.TShdrs.recNo.unique()
            recIdx = [i for i,x in enumerate(recNo_list)
                       if x == self.newParam['recNo']][0]

            row_sta = recIdx*self.newParam['sampNo']
            row_end = (recIdx + 1)*self.newParam['sampNo']
            self.TSmtx = self.TSmtx[row_sta:row_end, :]
            self.TShdrs = self.TShdrs.query(
                          'recNo =='+str(self.newParam['recNo']))
            self.TShdrs = self.TShdrs[~self.TShdrs.index.get_level_values(0).
                                      duplicated(keep='last')]

        # Time match PV and TS not same length
        if self.PVdata.shape[0] != self.TSmtx.shape[1]:
            self.TS_PV_match()

            if 'PV_STA' and 'PV_END' in self.newParam:
                self.PVdata_full = self.PVdata_full[self.newParam['PV_STA'] - 1:
                                                  self.newParam['PV_END']]
        else:
            self.newParam.update({'matched': False})


        # Redfine PV_df sliced
        self.PVdata = self.PVdata[self.newParam['STA_meas'] - 1:
                                self.newParam['END_meas']]

    # Can remove this Once PV_df contains all equal dimensional data
    #####################################################
        # If TS header data exists, then slice
        if self.TShdrs.shape[0] == self.TSmtx.shape[1]:
            self.TShdrs = self.TShdrs[self.newParam['STA_meas'] - 1:
                                    self.newParam['END_meas']]
        else:
            print('No TS header data found')
    #####################################################


        self.TSmtx = self.TSmtx[self.newParam['FB_cut']: self.newParam['END_cut']-1]
        self.TSmtx = self.TSmtx[:, self.newParam['STA_meas'] - 1:
                                self.newParam['END_meas']]

        # Normalise all TS between -1 and 1
        return (self.TSmtx, self.PVdata, self.TShdrs, self.newParam)

    def TSgain(self):
        '''DEPRECIATED: Correct the input TS data for any acquisition gain. It is expected that
        a dataframe TShdrs is available which contains a column "Vertical gain",
        the contents of which will be multiplied with each time series sample point.

        Parameters
        ----------
        self.TSmtx: corrected for vertical gain applied during aqcuisiton.

        Returns
        -------
        '''
        try:
            self.TSmtx = self.TSmtx * self.TShdrs['Vertical gain'].as_matrix()
        except KeyError:
            print('Column: \"Vertical gain\" not found in TShdrs Dataframe, \
                  no correlation applied')


    def TS_t(self):
        '''REDUNDENT: This should be generated as the index of the TS data.
        Generates the corresponding time vector based on information found
        in the TS headers. If no data is found, then the time vector is tagged
        as representing measurement numbers

        Parameters
        ----------
        t:   Time vector in seconds
        '''
        if 'SampFreq' in self.newParam:
            start = 0
            stop = self.TSmtx.shape[0] / self.newParam['SampFreq']
            t = np.linspace(start, stop, self.TSmtx.shape[0])
            t_axis = {'t_axis': t}
            self.newParam.update(t_axis)

    def PV_time_corr(self):
        '''Takes either an existing column in the PV database and creates a new
        column ``Time Stamp`` containing the absolute timestamp for each datapoint,
        in sync with the absolute timestamp data stored within the TS data headers.
        Attributes are read in from the database.

        Parameters
        ----------
        self.PVdata : DataFrame
            Database containing the PV data.


        Notes
        -----
        It is expected that within the self.Database file the following attributes
        will be found.
            DB.attrs['PVstart_date_time'] : str
                String of the absolute time of the first PV datapoint in sync
                with TS time.
            DB.attrs['PV_time_col'] : str
                Name of time column to match in PV data.
            DB.attrs['PV_time_col_unit'] : str
                The time unit (Default = 'D'), see pd.to_datetime

        Examples
        --------
        >>> # Determine the PV start date time from the PVdata
        >>> PV_time_col = 'Time(Days)'
        >>> shift = 71.3980 - PV_data['Time(Days)'][0]
        >>> PV_time_origin = pd.Timestamp(pd.to_datetime(TS_data.columns.
        >>>                               get_level_values('Time')[0]) -
        >>>                               datetime.timedelta(days = shift))
        >>>

        '''

        # Update function to deal with no-correction required


        # Read in attributes from database
#        dict_attrs = dt.utilities.DB_attrs_load(self.Database, ['PVstart_date_time',
#                                                   'PV_time_col',
#                                                   'PV_time_col_unit'])

        # Create new Time Stamp column in PVdata
        self.PVdata['Time Stamp'] = pd.to_datetime(
                self.PVdata[self.newParam['PV_time_col']],
                unit=self.newParam['PV_time_col_unit'],
                origin = self.newParam['PVstart_date_time'])



    def TS_PV_match(self):
        '''DEPRECIATED: The time matching between TS and PV data for specific
        Olympus data
        type.

        TODO
        ----
        * Update to be more general, allowing matching for generic input data
        * Also allow the passing on the non-matched PV for plotting/storage
        '''
        from datetime import timedelta, datetime
        import dateutil.parser as datePar
        import time
        #import matched

        # Nested functions
        def nearest(items, pivot):
            match = min(abs(items - pivot))
            return match  # min(items, key=lambda x: abs(x-pivot))

        def getnearpos(array,value):
            idx = (np.abs(array-value)).argmin()
            return idx


        print('--------- Time Matching PV and TS data ---------')

        # Store original PV_df
        self.PVdata = self.PVdata
        # Load setup information
        acqDate = datePar.parse(self.newParam['acqDate'], dayfirst=True)

        # If user defined TS_samp_dt then match based on this
        if self.newParam['TS_samp_dt']>0:
            # Define the TS sampling frequency
            # TS_samp_dt = 15 # sec

            # The regular expression to match
            regex = re.compile(r'\d+')

            # Read the file names from the index columns of the header DF
            TS_d_list = list(self.TShdrs.index.get_level_values(0))

            # calculate the count time based on file no.
            count_time = [int(regex.findall(file)[-1])*self.newParam['TS_samp_dt']
                          for file in TS_d_list]

            # Add offset as the count of the first file in the list
            tot_time = count_time + self.TShdrs['count[sec]'][0] + self.newParam['match_timeshift']
            # Convert sec to datetime formate
            TS_d_list = [timedelta(seconds=sec) + acqDate for
                         sec in tot_time]
        elif self.newParam['TS_file_tmatch']:
            print('Time match based on time file saved')
            TS_d_list = self.TShdrs['fileMod'].tolist()
            self.TShdrs['Date_rel'] = TS_d_list
            TS_d_list = [datetime.fromtimestamp(x) for x in TS_d_list]
        else:
            # Match based on the .par file count vector
            # Convert expected TS hrs format into relDate/time format
            TS_d_list = [timedelta(seconds=sec+ self.newParam['match_timeshift'])  + acqDate for
                     sec in self.TShdrs['count[sec]']]

        self.TShdrs['Date_rel'] = TS_d_list

        # Convert Perturbation time data into relDate/time format
        PV_time = self.PVdata['Date']
        st = datetime(1899, 12, 30)
        PV_d_list = [timedelta(days=dates)+st for dates in PV_time]
        self.PVdata['Date_rel'] = PV_d_list
        start = time.time()

        # Match the two reldate/Time formats
        match = [getnearpos(self.PVdata['Date_rel'], time) for
                   time in self.TShdrs['Date_rel']]

        end = time.time()

        PV_df_filt = self.PVdata.ix[match]

        print('Data time matched')
        self.newParam.update({'TSmatched': True})
        self.newParam.update({'matched': False})

        self.PVdata = PV_df_filt


        print('The total matching time for all was',
               end - start, 'sec')

    def _stress_strainP(self):
        '''Perform stress strain calculation based on user defined inputs.
        Either stress_strain_confined is True for Confined loading or stress_strain
        is True for UCS tests.

        TODO:
            Read in the attributes from the database as below

            dict_attrs = dt.utilities.DB_attrs_load(self.Database, ['PVstart_date_time',
                                                   'PV_time_col',
                                                   'PV_time_col_unit'])
        '''

        if self.newParam['stress_strain_confined']:

            # Deal with different expected names
            rename_dict = {}
            if 'Sax| bar' in self.PVdata.columns:
                rename_dict['Sax| bar'] = 'S ax| bar'
            if 'Ea| um' in self.PVdata.columns:
                rename_dict['Ea| um'] = 'Ea_avg| um'
            if 'Et| um' in self.PVdata.columns:
                rename_dict['Et| um'] = 'Et_avg| um'

            self.PVdata.rename(columns=rename_dict, inplace=True)

            # Default processing param
            if 'Ax_Press_corr_ADT' not in self.newParam.keys():
                self.newParam['Ax_Press_corr_ADT'] = -0.0564
            if 'Rad_press_corr_ADT' not in self.newParam.keys():
                self.newParam['Rad_press_corr_ADT'] = 0.0479
            if 'avg_Press_corr_RDT' not in self.newParam.keys():
                self.newParam['avg_Press_corr_RDT'] = -0.0012
            if 'Pax_friction' not in self.newParam.keys():
                self.newParam['Pax_friction'] = 4
            if 'Temp_corr' not in self.newParam.keys():
                self.newParam['Temp_corr'] = 0

            # Handel non-consistant channel names
            try:
                self.PVdata['TempCell| °C']
            except KeyError:
                self.PVdata['TempCell| °C'] = 20

            try:
                self.PVdata['Ea_avg| um'] = self.PVdata[['Ea1| um', 'Ea2| um']].mean(axis=1)
            except KeyError:
                print('')

            try:
                self.PVdata['AvgPorePres| bar'] = self.PVdata[['Ppore in| bar', 'Ppore out| bar']].mean(axis=1)
            except KeyError:
                print('')

            try:
                self.PVdata['ADT comp| um']
            except KeyError:
                self.PVdata['ADT comp| um'] = 0

            try:
                self.PVdata['RDT comp.| um']
            except KeyError:
                self.PVdata['RDT comp.| um'] = 0

            # Perform the calculations
            self.PVdata['Pax_fric_corr'] = self.PVdata['S ax| bar'].astype(float)
            - self.newParam['Pax_friction']

            self.PVdata['ADT_corr'] = (self.PVdata['ADT comp| um']
            + self.PVdata['Pax_fric_corr'] * self.newParam['Ax_Press_corr_ADT']
            + self.PVdata['Srad| bar'] * self.newParam['Rad_press_corr_ADT']
            + (self.PVdata['TempCell| °C'] - self.PVdata['TempCell| °C'][0])
            * self.newParam['Temp_corr'])

            self.PVdata['RDT_corr'] = (self.PVdata['RDT comp.| um']
            + self.newParam['avg_Press_corr_RDT'] * self.PVdata['Srad| bar']
            + self.newParam['Temp_corr']
            * (self.PVdata['TempCell| °C'] - self.PVdata['TempCell| °C'][0]))

            self.PVdata['samp_length| mm'] = (self.newParam['L']
            + (self.PVdata['ADT_corr'][0] - self.PVdata['ADT_corr'])/1000)

            self.PVdata['samp_dia| mm'] = (self.newParam['D']
            + (self.PVdata['RDT_corr'][0] - self.PVdata['RDT_corr'])/1000)

            self.PVdata['Strain Ax. [%]'] = (self.PVdata['ADT_corr']
            - self.PVdata['ADT_corr'][0]) / self.PVdata['samp_length| mm']

            self.PVdata['Radial Ax. [%]'] = (self.PVdata['RDT_corr']
            - self.PVdata['RDT_corr'][0]) / self.PVdata['samp_dia| mm']

            self.PVdata['Bulk_Vol. [%]'] = (self.PVdata['Strain Ax. [%]']
            + self.PVdata['Radial Ax. [%]']*2)

            self.PVdata['Stress ef ax| [MPa]'] =(self.PVdata['Pax_fric_corr']
            - self.PVdata['AvgPorePres| bar']) / 10

            self.PVdata['Stress ef radial| [MPa]'] = (self.PVdata['Srad| bar']
            - self.PVdata['AvgPorePres| bar'])/ 10

            self.PVdata['Stress ax total| [MPa]'] = self.PVdata['Pax_fric_corr']/ 10
            self.PVdata['Stress radial total| [MPa]'] = self.PVdata['Srad| bar']/ 10
            self.PVdata['Stress mean total| [MPa]'] = (self.PVdata['Pax_fric_corr']
            + self.PVdata['Srad| bar'] * 2) / 30

            self.PVdata['Stress mean eff| [MPa]'] = (self.PVdata['Stress ef radial| [MPa]'] * 2 +
            self.PVdata['Stress ef ax| [MPa]']) / 3

            self.PVdata['Stress deviatoric| [MPa]'] = self.PVdata['Stress ef ax| [MPa]']
            - self.PVdata['Stress ef radial| [MPa]']

            self.PVdata['Pore pressure| [MPa]'] = self.PVdata['AvgPorePres| bar'] / 10

        elif self.newParam['stress_strain']:
            self.PVdata['LVDT_ave'] = self.PVdata[['LVDT1', 'LVDT2']].mean(axis=1)
            self.PVdata['LVDT_ave'] = self.PVdata['LVDT_ave'] + self.PVdata.ix[
                    self.newParam['LVDT_zidx'], 'LVDT_ave'] * -1  # Zero column
            area = (self.newParam['D'] ** 2 * np.pi/4)
            length = self.newParam['L']
            self.PVdata['Stress [MPa]'] = self.PVdata['Force(kN)'] / area * 1000
            self.PVdata['Strain Ax. [%]'] = self.PVdata['LVDT_ave'] / length * 100

    def _DatetimeIndex(self):
        ''' REDUNDENT: Check if the word Date is provided in the user input
        file PV1. If so perform processing to provide pd.DatetimeIndex
        '''

        if 'Date' in self.newParam['PV1']:
            try:
                # rename columne with Date in it to just Date
                self.PV_df['Date'] = pd.to_datetime(
                        self.PV_df[self.newParam['PV1']])

                # Move current index into DF
                self.PV_df.reset_index(level=0, inplace=True)
                self.PV_df.rename(columns={'level_0': 'Input_files'},
                                  inplace=True)

                # Create datetimeIndex
                self.PV_df.set_index(pd.DatetimeIndex(self.PV_df['Date']),
                                     inplace=True)
                self.PV_df.drop(self.newParam['PV1'], axis=1, inplace=True)

                self.newParam.update({'PV1': 'Date'})
            except ValueError:
                print('Date found in PV1 PV data, but it was not possible to',
                      'convert this to datetime format.')

    def _Rename_col(self):
        ''' Rename requested columns of the input PV features based on
        the user defined input dictionary rename_dic
        '''

        dict_attrs = dt.utilities.DB_attrs_load(self.Database, ['rename_dic'])

        try:
            self.PVdata.rename(columns=dict_attrs['rename_dic'], inplace=True)
        except TypeError:
            print('* No columns renamed')



    def CC_wdw(self):
        """Determines the number of windows of a certain lenght which fit within
        the TS, and outputs a list of start positions in terms of sample points

        The length of the TS data is read from the database.
        The ``wdwPos`` updated as saved to the attributes of the database.

        TODO:
            It would be better to read the attribues of the table, there is no need
            to read in the entire database.
        """
        # Setup param
        loc = 'TSdata'
        if 'single' == self.newParam['survey_type']:
            TS_len = dt.utilities.DB_pd_data_load(self.Database, loc).shape[0]
        elif 'multiple' == self.newParam['survey_type']:
            TS_group = dt.utilities.DB_group_names(self.Database, group_name = loc)[0]
            TS_len = dt.utilities.DB_pd_data_load(self.Database, loc+'/'+TS_group).shape[0]

        param = self.newParam

        # Assign TS processing length to end_wdws if given
        if param['end_wdws']:
            TS_sig_len = param['end_wdws']
        else:
            TS_sig_len = TS_len

        ERROR_MESSAGE = 'The length of a TS signal to be processed is', TS_sig_len, \
        'which is < end of the last window'

        # Calculate wdwPos for overlapping windows of ww_ol if wdwPos is False
        if param['wdwPos'][0] is False:
            # Error checks
            if TS_sig_len < self.newParam['ww'][0]:
                raise Warning(ERROR_MESSAGE)

            wdwStep = np.floor(param['ww'][0] *
                               (100 - param['ww_ol']) / 100)

            if self.verbose: print('* Length fo TSdata', TS_len)

            max_wdwPos = TS_sig_len - param['ww'][0] + 1
            wdwStarts = np.arange(0 + param['sta_wdws'], max_wdwPos, wdwStep).astype(int)

            if self.verbose: print('* The step in window potions is %s sample points' % wdwStep)
            if self.verbose: print('* The max window postions is %s sample points'% max_wdwPos)

            param['wdwPos'] = [ [wdw_start, wdw_start + param['ww'][0]] for
                           wdw_start in wdwStarts ]

        # Only update wdwPos structure if not already done so
        elif np.array(param['wdwPos'][0]).shape == ():
            param['wdwPos'] = [ [wdw_start, wdw_start + ww] for wdw_start,ww in
                           zip(param['wdwPos'], param['ww'])]

        self.newParam['wdwPos'] = param['wdwPos']


    def _paramUpdate(self):
        """Update the database attributes with new user requested processing
        parameters
        """

        # Update the database attributes accordingly.
        dt.utilities.DB_attrs_save(self.Database, self.newParam)

    def run_PP(self):
        '''Performs Pre-processing as requested by the user, as defined in the
        attributes of the database.h5.
        '''
        print('--------- Begin Pre-Processing ---------')

        # check if date time plotting is requested
#        self._DatetimeIndex()

        # Correct the windows to be processed
        self.CC_wdw()

        self.PV_time_corr()

        # Check if column renaming is required
        self._Rename_col()

        # check if stress_strain calculation is requested
        self._stress_strainP()

        # Update the attributes of the database reflecting any user changes
        self._paramUpdate()





        # Gain correlation
#        self.TSgain()

        # check if the yield point search is required
#        self._yield_search()

        # Generate a time vector based on sample numbers and samp. freq.
#        self.TS_t() # This time-vector should not be created

        # Store PVdata to Database
        dt.utilities.DB_pd_data_save(self.Database, 'PVdata', self.PVdata)

