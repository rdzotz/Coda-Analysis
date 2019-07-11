#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:44:42 2017

@author: rwilson
"""

import pandas as pd
import data as dt
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.dates import date2num
import matplotlib.dates as mdates
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
from matplotlib.widgets import CheckButtons
from matplotlib import colors as mcolors

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import itertools

class dispUtilities:
    ''' A collection of functions which may be useful for certain display types
    '''

    def multi_wdw(param, DB, TS, lag, PV, multi_plot = None,
                  PV1 = None, PV2 = None):
        ''' Inteneded to the display of multiple correlation windows on a single
        2D plot, with the average of all CC trend lines bold. Comparison with
        two PV is possible.
        ---Inputs---
        param: output param file summarising the call
        DB: DataFrame containing all PV and CC data
        TS: The time series, pandas series trace indexed in time [sec]
        lag: The value of lag to compare to
        PV: DataFrame of PV only data. If provided then PV1 and PV2 will be
            taken from here, not DB.
        multi_plot: Either 'CC', 'CC_lag' or 'RelAmp', default 'CC'
        PV1 & PV2: name of x y2 columns to plot against, if given
        '''
        # ---------------------  Basic input check   ----------------------
        if lag not in param['CC_ref']:
            raise Warning('No lag of ',lag ,'found, please check your Database')

        # ---------------------  Input Setup   ----------------------
        if multi_plot is None:
            multi_plot = 'CC'

        if PV1 is None:
            PV1 = param['PV1']
            PV1_lbl = param['PV1']
        else:
            PV1_lbl = PV1
        if PV2 is None:
            PV2 = param['PV2']
            PV2_lbl = param['PV2']
        else:
            PV2_lbl = PV2

        # Get list of tuples (lag, wd_pos) for all lag

        if multi_plot is 'CC':
            cols = [tup for tup in DB.columns.values if isinstance(tup, tuple)
                    and tup[0] == lag
                    and 'CC_lag' not in str(tup[1])
                    and 'RelAmp' not in str(tup[1])]
        elif multi_plot is 'CC_lag':
            cols = [tup for tup in DB.columns.values if isinstance(tup, tuple)
                    and tup[0] == lag
                    and 'CC_lag' in str(tup[1])
                    and 'RelAmp' not in str(tup[1])]
        elif multi_plot is 'RelAmp':
            cols = [tup for tup in DB.columns.values if isinstance(tup, tuple)
                    and tup[0] == lag
                    and 'RelAmp' in str(tup[1])]

        wd_pos_list = [tup[1] for tup in DB.columns.values if isinstance(tup, tuple)
                  and tup[0]==lag
                  and 'CC_lag' not in str(tup[1])
                  and 'RelAmp' not in str(tup[1])]
        if cols == []:
            raise Warning('No lag values ')

        # Calculate the average of all windows
        wdw_ave_col = 'ave_wdw_CC'
        DB[wdw_ave_col] = DB[cols].mean(axis=1)
        cols.append('ave_wdw_CC') # averaged of all column
        # ---------------------  Define the input data   ----------------------
        x = DB[PV1].tolist()
        y = DB.loc[:, cols]
        y2 = DB[PV2]

        # ---------------------        Setup Figure      ----------------------
        plt.style.use('bmh')
        fig = plt.figure(figsize=(20, 30))
        plt.subplots_adjust(left=0.05, bottom=0.12,
                            right=0.80, top=0.9,
                            wspace=0.9, hspace=1.5)
        gs = gridspec.GridSpec(5, 4)
        ax1 = plt.subplot(gs[0:4, :])
        if multi_plot is 'CC':
            plt.ylim([0, 1])
        linewidth = 0.5
        markersize = 1

        # -------------------   Plot the CC & PV data   ----------------------
        # Only for axis as date..
        if 'Date' in PV1:
            hdl_cc = ax1.plot_date(x, y, linewidth=linewidth,
                                    markersize=markersize)
            print('Set data as x-axis')
            plt.gcf().autofmt_xdate()
        else:
            hdl_cc = ax1.plot(x, y, linewidth=linewidth,
                                    markersize=markersize,
                                    color='grey')
        # Add second y-axis if available
        if y2 is not None:
            ax2 = ax1.twinx()
            if isinstance(PV, pd.DataFrame):
                x_full = PV[PV1]
                y2 = PV[PV2]
                ax2.plot(x_full, y2, 'k-', linewidth=1)
            else:
                ax2.plot(x, y2, 'k-', linewidth=1)

            ax2.set_ylabel(PV2_lbl, color='k')

        # -------------------   Formate the plot   ----------------------
        hdl_cc[-1].set_linewidth(2)
        hdl_cc[-1].set_color('black')

        # -------------------    Add tiles/label..etcc  ----------------------
        ax1.set_title('Comparison of multiple window positions')
        ax1.set_xlabel(PV1_lbl)
        ax1.set_ylabel(multi_plot, color='g')
        hdl_cc[-1].set_label(cols[-1])
        legn_hdl = ax1.legend(loc='upper left', frameon=False)
        legn_hdl.get_frame().set_alpha(0.4)


        # --------------   Add TS plot below for comparison   ----------------
        ax3 = plt.subplot(gs[4:5,:])

        hdl_ts, unit_m = dispUtilities()._TS_plot(ax3, TS.index.values,
                               TS, param)
        ax3.grid(False)


        # --------------   Add Overly of window pos range   ----------------
        for wdw_pos in wd_pos_list:
            hdl_TSwdws = dispUtilities()._TS_plot_wdw(param, ax3,
                                                      TS, wdw_pos)
        height = TS.max() * 2
        width = TS.index[param['ww']] * unit_m

        # Set coords of bottom corner
        y = -height/2
        x = [TS.index[pos] * unit_m for pos in wd_pos_list]

        # annotate single window
        ax3.add_patch(patches.Rectangle((x[int(len(x)/2)], y),
                                        width, height,
                                        fill = False,
                                        linewidth=2,
                                        edgecolor='black'))
        # --------------------------------------------------------------------

        plt.show()

        return hdl_cc, ax3

    def dispCWI_pres(param, DB, TS, ccPlot_list,
                     PV_full=None, PV1=None, PV2=None, PV3=None,
                     PV4 = None, plot_dic={}):
        ''' The flexible display of multiple input vectors

        Parameters
        ----------
        param : dict
            Dictionary summarising the processing run details
        DB : DataFrame
            Database of all PV and CC data to be plotted
        TS : DataFrame
             Single TS in order to display the window positions
        PV_full : Database
            Non-filtered PV_data, used for plot if given
        ccPlot_list : list
            List of tuples in CC data to plot (lag, wdwPos)
        PV1 : None or str
            X-axis, if 'Index' is given the DB datetime index will be used
        PV: None or str
            Y2 right axis
        PV3: None or str
            Y3 right axis
        PV4: None or str
            Y4 right axis
        plot_dic: dict
            Dictionary to flexibility in ploting containing
              -**PV1_leg**: ``None`` PV1 legend name
              -**PV2_lb**: ``PV2`` PV2 axis name
              -**PV3_lb**: ``PV3`` PV3 axis name
              -**PV4_lb**: ``PV4`` PV4 axis name
              -**PV4_leg**: ``None`` PV4 legend name
              -**wd_style**: "Annotate" analysis window style
              -**wd_lbl**: ``[None]*len(param['wdwPos'])`` analysis window label
              -**CC_range**: ``[0, 1]`` CC axis range
              -**X_axis_range**: ``None`` X axis range
              -**PV2_range**: ``None`` PV2 axis range
              -**PV4_range**: ``None`` PV4 axis range
              -**elapsed_hours**: ``False`` if PV1 is a 'Date', then plot elapsed hours
              -**unit_m_4**: ``1`` multiplier of the PV4 axis :math:`\\alpha`
        '''

        # ---------------------  Input Setup   ----------------------

        if PV1==None:
            PV1 = self.param['PV1']
            PV1_lb = self.param['PV1']
        else:
            PV1_lb = PV1

        if PV2==None:
            PV2 = self.param['PV2']
            PV2_lb = self.param['PV2']

        dic_defaults = {'PV1_leg': None,
                        'PV2_lb': PV2,
                        'PV3_lb': PV3,
                        'PV4_lb': PV4,
                        'PV4_leg': None,
                        'wd_style': 'Annotate',
                        'wd_lbl': [None]*len(param['wdwPos']),
                        'CC_range': [0, 1],
                        'X_axis_range': None,
                        'PV2_range': None,
                        'PV3_range': None,
                        'PV4_range': None,
                        'markersize': 3,
                        'elapsed_hours': False,
                        'unit_m_4': 1
                        }

        # Assign the defaults plot info if not already given
        for var,default in dic_defaults.items():
            try:
                plot_dic[var]
            except (KeyError, NameError, TypeError):
                plot_dic[var] = default

        unit_m_4 = plot_dic['unit_m_4']

        # Check in data requested in in the DB, if not assign None


        try:
            y2 = DB[PV2]
        except (KeyError, NameError):
            y2 = None

        try:
            y3 = DB[PV3]
        except (KeyError, NameError, ValueError):
            y3 = None

        try:
            y4 = DB[PV4] * unit_m_4
        except (KeyError, NameError,ValueError):
            y4 = None


        import matplotlib as mpl

        # ---------------------  Define the input data   ----------------------
        y = DB.loc[:, ccPlot_list].dropna()

        if PV3 is not None:
            y3 = DB[PV3]
            if PV3 == 'TOF':
                # Check for the approriate label
                if 'TS_units' in param and param['TS_units'] == 'msec':
                    unit_m = 1000
                    PV3 = 'TOF [msec]'
                else:
                    unit_m = 1
                    PV3 = 'TOF [sec]'
                y3 = y3 * unit_m


        # Window positions
        wdw_pos_list = [wdw_pos[1] for wdw_pos
                        in ccPlot_list if isinstance(wdw_pos, tuple)]

        # ---------------------        Setup Figure      ----------------------
        plt.style.use('bmh')
        fig = plt.figure(figsize=(20, 30), facecolor='white')
        plt.subplots_adjust(left=0.07, bottom=0.12,
                            right=0.6, top=0.9,
                            wspace=0.9, hspace=1.5)
        gs = gridspec.GridSpec(5, 4)
        plt.legend(frameon=True)
        #leg.get_frame().set_alpha(0.4)
        ax1 = plt.subplot(gs[0:4, :])

        plt.ylim(plot_dic['CC_range'])

        linewidth = 2
        marker = itertools.cycle(('o', 'v', '*', 'd', 's', 'D', 'h', 'P', 'X',
                                  '^', '<', '>', '8', 'H', 'p'))

        # Define empty line handels
        hdl_cc  = []
        hdl_PV2 = []
        hdl_PV3 = []
        hdl_PV4 = []

        # mpl.style.use('classic')
        # mpl.rcParams['lines.markeredgewidth'] = 0
        # mpl.rcParams['figure.facecolor'] = 'white'

        # -------------------   Plot the CC & PV data   ----------------------
        # Only for axis as date..
        if 'Index' in PV1:
            if plot_dic['elapsed_hours']:
                # Index same as entire DB
                DB['elapsed_hours'] = DB.index - DB.index[0]
                x_all = DB['elapsed_hours'].dt.total_seconds()/3600

                # index of equal length as y
                y['elapsed_hours'] = y.index-y.index[0]
                x = y['elapsed_hours'].dt.total_seconds()/3600

                hdl_cc = ax1.plot(x, y, linewidth=linewidth,
                                  markersize=plot_dic['markersize'])
                PV1_lb = 'hrs'
            else:
                x_all = DB.index.to_pydatetime()
                x = y.index.to_pydatetime()
                hdl_cc = ax1.plot_date(x, y, linewidth=linewidth,
                                  markersize=plot_dic['markersize'])
                plt.gcf().autofmt_xdate()
        else:
            x = y[PV1]
            x_all = DB[PV1]
            hdl_cc = ax1.plot(x, y, linewidth=linewidth,
                                    markersize=plot_dic['markersize'])

        # Cycle through marker list
        [(line.set_marker(next(marker)), line.set_linestyle('-')) for line in hdl_cc]

        # Add second y-axis if available
        if y2 is not None:
            ax2 = ax1.twinx()

            hdl_PV2 = ax2.plot(x_all, y2, linewidth=1, markersize=plot_dic['markersize'])

            if isinstance(PV2, list):
                [hdl_PV2[n].set_label(leg) for leg,n in zip(PV2, range(len(PV2)))]
            else:
                hdl_PV2[0].set_label(PV2)

            ax2.set_ylim(plot_dic['PV2_range'])
            ax2.set_ylabel(plot_dic['PV2_lb'])

        # Add third y-axis if available
        if y3 is not None:
            color = 'k'
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('axes', 1.15))
            ax3.set_frame_on(True)
            ax3.patch.set_visible(False)
            hdl_PV3 = ax3.plot(x_all, y3, linewidth=1, markersize=plot_dic['markersize'])
            ax3.set_ylabel(plot_dic['PV3_lb'], color=color)
            if isinstance(PV3, list):
                [hdl_PV3[n].set_label(leg) for leg,n in zip(PV3, range(len(PV3)))]
            else:
                hdl_PV3[0].set_label(PV3)
            ax3.tick_params(axis='y')
            ax3.set_ylim(plot_dic['PV3_range'])
            ax3.grid(False)

        # Add forth y-axis if available
        if y4 is not None:
            color = 'Black'
            ax4 = ax1.twinx()
            ax4.spines['right'].set_position(('axes', 1.25))
            ax4.set_frame_on(True)
            ax4.patch.set_visible(False)
            hdl_PV4 = ax4.plot(x_all, y4, linewidth=1, markersize=plot_dic['markersize'])
            ax4.set_ylabel(plot_dic['PV4_lb'], color=color)
            if plot_dic['PV4_leg'] is not None:
                [hdl_PV4[n].set_label(leg) for leg,n in zip(plot_dic['PV4_leg'], range(len(PV4)))]
            elif isinstance(PV4, list):
                [hdl_PV4[n].set_label(leg) for leg,n in zip(PV4, range(len(PV4)))]
            else:
                hdl_PV4[0].set_label(PV4)
            ax4.tick_params(axis='y')
            ax4.set_ylim(plot_dic['PV4_range'])
            ax4.grid(False)

        # -------------------    Add tiles/label..etcc  ----------------------
        ax1.set_title('')
        ax1.set_xlabel(PV1_lb)
        ax1.set_ylabel('CC', color='k')
        ax1.set_xlim(plot_dic['X_axis_range'])
        if plot_dic['PV1_leg'] is not None:
            CC_leg = plot_dic['PV1_leg']
        else:
            CC_leg = [str(x) for x in ccPlot_list]
        [hdl_cc[n].set_label(leg) for leg,n in zip(CC_leg, range(len(CC_leg)))]

        lns = hdl_cc + hdl_PV2 + hdl_PV3 + hdl_PV4
        colors_all = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        colors = itertools.cycle(list(colors_all))
        [(l.set_color(next(colors)),l.set_marker(next(marker)))  for l in lns]

        labs = [l.get_label() for l in lns]
        led = ax1.legend(lns, labs, bbox_to_anchor=(1.5, 0), frameon=False)
        led.get_frame().set_alpha((0.9))

        [(line.set_marker(next(marker))) for line in hdl_cc]
        [(line.set_linestyle('-')) for line in hdl_cc]

        # -------------------   Add TS data Below plot   ----------------------
        ax4 = plt.subplot(gs[4, :])

        hdl_TS, unit_m = dispUtilities()._TS_plot(ax4, TS.index.values,
                               TS, param)

        ax4.get_yaxis().set_visible(False)

        # --------------------     Add CC window to TS   ----------------------
        for idx, wdw_pos in enumerate(param['wdwPos']):
            hdl_TSwdws = dispUtilities()._TS_plot_wdw(param, ax4,
                                                      TS, wdw_pos,
                                                      plot_dic['wd_lbl'][idx],
                                                      plot_dic['wd_style']
                                                      )
        plt.show()


    def _TS_plot_wdw(param, ax, TS, wd_pos, wd_lbl, wd_style='Annotate'):
        '''Private function handels the general plotting of the
        correlation window considered.

        Parameters
        ----------
        param : dict
            The param dictionary containing a number of relevant details

        ax : object
            The axis handel object

        TS : float or int
            The time series

        wd_pos : list
            list of beginning and end of windows ``[[sta,end]]``

        wd_lbl : str
             window lable

        wd_style : str
            String of possible window styles, default ``Annotate``
        '''

        # Check for the approriate label
        if 'TS_units' in param and param['TS_units'] == 'msec':
            unit_m = 1000
            ax.set_xlabel('Time [msec]')
        else:
            unit_m = 1
            ax.set_xlabel('Time [sec]')

        height = TS.max()*2
        y = -height/2

        x = TS.index.values[wd_pos[0]] * unit_m
        width = (TS.index.values[wd_pos[1]]-TS.index.values[wd_pos[0]]) * unit_m

        hdl = ax.add_patch(
            patches.Rectangle(
                (x, y),   # (x,y)
                width,          # width
                height,          # height
                alpha=0.1
            )
        )

        right = x + width
        top = y + height

        if wd_style == 'Annotate':
            hdl.set_edgecolor("black")
            hdl.set_fill(True)
            ax.text(x + width/2, y + height*1.20, wd_lbl,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10, color='black')

    def _TS_plot(ax, x_TS, TS, param):
        '''Private function handels the general plotting of TS info,
        taking as input a dataframe with index takes as the time vector
        '''

        # Check for the approriate label
        if 'TS_units' in param and param['TS_units'] == 'msec':
            unit_m = 1000
            ax.set_xlabel('Time [msec]')
        else:
            unit_m = 1
            ax.set_xlabel('Time')

        [hdl] = ax.plot(x_TS * unit_m, TS)
        hdl.set_linewidth(0.5)

        return hdl, unit_m

    def PVdata_plot(PVdb, x, y):
        '''General plotting of the PV dataset.

        TODO
        ----
        Expand the plotting function to induce annotaiton
        '''

        PVdb.plot(x=x, y=y)



class dispCWI_DB:
    ''' This class handels the interactive ploting of CC data stored within
    a single database.
    '''

    def __init__(self, param, DB, TS_DB, PV_full):
        self.param = param
        self.DB = DB
        self.TS_DB = TS_DB
        self.PV_full = PV_full
        self.lag_lbl = []     # list of lag labels
        self.hdl_cc = []      # list of handels
        self.wdw_dic = {}     # dic of window visibility status
        self.wdw_pos = []     # list of window positions
        self.hdl_TSwdws = []  # list of window patch positions

    def wdw_lag_col(self):
        '''Extracts the wdw pos from DB column names
        '''

        # Get list of tuples (lag, self.self.wdw_pos)
        tuples = [tup for tup in self.DB.columns.values
                  if isinstance(tup, tuple)
                  and 'CC_lag' not in str(tup[1])
                  and 'RelAmp' not in str(tup[1])
                  and 'FreqBand' not in str(tup[1])
                  and 'sig' not in str(tup[1])
                  ]

        return tuples

    def CC_labels(self, tuples):
        ''' Handels the assignment of figure titles and axis labels. Basic
        functionality, checks if variables is assigned in param, if not a
        a default is assigned.
        tuples: tuples of (lag, self.self.wdw_pos)
        '''

        if 'PV1_lb' in self.param:
            PV1_lb = self.param['PV1_lb']
        elif 'PV1' in self.param:
            PV1_lb = self.param['PV1']
        elif 'PV2' in self.param:
            PV1_lb = self.param['PV2']
        else:
            PV1_lb = 'Measurement No.'

        if 'PV2_lb' in self.param:
            PV2_lb = self.param['PV2_lb']
        elif 'PV2' in self.param:
            PV2_lb = self.param['PV2']
        else:
            PV2_lb = None

        if 'genTitle' in self.param:
            genTitle = self.param['genTitle']
        else:
            genTitle = 'No Title'

        myLeg = [str(x) for x in tuples]

        return (PV1_lb, PV2_lb, genTitle, myLeg)

    def _TS_plot(self, ax, x_TS, TS):
        ''' This private function handels the general plotting of TS info,
        taking as input a dataframe with index takes as the time vector
        '''

        # Check for the approriate label
        if 'TS_units' in self.param and self.param['TS_units'] == 'msec':
            unit_m = 1000
            ax.set_xlabel('Time [msec]')
        else:
            unit_m = 1
            ax.set_xlabel('Time')

        [hdl] = ax.plot(x_TS * unit_m, TS)
        hdl.set_linewidth(0.5)

        return hdl, unit_m

    def num_in_str(self, num, string, pattern):
        '''Match an exact numnber in a string based on a pattern
        '''
        pat = pattern % num
        # print(pat)
        try:
            m = re.search(pat, string).group()
            print(m)
            return True
        except AttributeError:
            return False

    def TS_plot_wdw(self, ax, wd_pos, unit_m):
        ''' This private function handels the general plotting of the
        correlation window considered. for each window in wd_pos
        Inputs:
        ------
        wd_pos: sample number of beginning of window
        TODO:
        '''

        # Set hight and width of window
        height = self.TS_DB.max().max()*2

        #width = self.TS_DB.index[self.param['ww']] * unit_m

        widths = [self.TS_DB.index[pos] * unit_m for pos in self.param['ww']]

        # print('hieght', height)
        # print('width', width)

        # Set coords of bottom corner
        y = -height/2
        x = [self.TS_DB.index[pos] * unit_m for pos in wd_pos]

        #zip_list = zip(x, widths) if len(x) == len(widths),
        #        else zip(x, itertools.cycle(widths))

        zip_list = zip(x, itertools.cycle(widths))  if len(x) > len(widths) else zip(x, widths)
        # Draw each rectangle on ax
        hdls = [ax.add_patch(patches.Rectangle(
                (x_sta, y), x_end, height, alpha=0.1))
                for x_sta, x_end in zip_list]

        [hdls[n].set_label(wd_pos[n]) for n in range(len(wd_pos))]

        return hdls

    def plot_DB(self):
        ''' Generate Interactive plot of DB
        '''
        # ---------------------  Define the input data   ----------------------
        tuples = self.wdw_lag_col()
        x = self.DB[self.param['PV1']].tolist() #self.DB.index.values
        y = self.DB.loc[:, tuples]

        if self.param['PV2'] is not None:
            y2 = self.DB[self.param['PV2']]
        else:
            y2 = None

        # ---------------------        Setup Figure      ----------------------
        plt.style.use('bmh')
        fig = plt.figure(figsize=(20, 30))
        plt.subplots_adjust(left=0.05, bottom=0.12,
                            right=0.8, top=0.9,
                            wspace=0.9, hspace=1.5)
        gs = gridspec.GridSpec(5, 4)
        ax1 = plt.subplot(gs[0:4, :])
        plt.ylim([0, 1])
        linewidth = 0.5
        markersize = 1
        marker = itertools.cycle(('o', 'v', '*', 'd', 's', 'D', 'h', 'P', 'X',
                                  '^', '<', '>', '8', 'H', 'p'))
        PV1_lb, PV2_lb, genTitle, legn = self.CC_labels(tuples)

        # -------------------   Plot the CC & PV data   ----------------------

        # Only for axis as date..
        if 'Date' in self.param['PV1']:
            self.hdl_cc = ax1.plot_date(x, y, linewidth=linewidth,
                                    markersize=markersize)
            print('Set data as x-axis')
            plt.gcf().autofmt_xdate()
        else:
            self.hdl_cc = ax1.plot(x, y, linewidth=linewidth,
                                    markersize=markersize)

        # Cycle through marker list
        [(line.set_marker(next(marker)), line.set_linestyle('-'))
         for line in self.hdl_cc]

        # Add second axis PV data if available
        if y2 is not None:
            ax2 = ax1.twinx()
            if isinstance(self.PV_full, pd.DataFrame):
                x_full = self.PV_full[self.param['PV1']].tolist()
                y2 = self.PV_full[self.param['PV2']]
                ax2.plot(x_full, y2, 'k-', linewidth=1)
            else:
                ax2.plot(x, y2, 'k-', linewidth=1)

            ax2.set_ylabel(PV2_lb, color='k')

        # -------------------    Add tiles/label..etcc  ----------------------
        ax1.set_title(genTitle)
        ax1.set_xlabel(PV1_lb)
        ax1.set_ylabel('CC', color='g')
        [self.hdl_cc[n].set_label(legn[n]) for n in range(len(legn))]
        legn_hdl = ax1.legend(loc='upper left', frameon=False)
        legn_hdl.get_frame().set_alpha(0.4)

        # -----------------   Checkboxes for Lags plotted   -------------------
        rax_lag = fig.add_axes([0.85, 0.7, 0.07, 0.10])
        lag_lbl = ['lag of '+str(lags) for lags in self.param['CC_ref']]
        lag_dft_pos = (True,)*len(self.param['CC_ref'])
        lag_dic = dict(zip(lag_lbl, lag_dft_pos))
        check_lag = CheckButtons(rax_lag, tuple(lag_lbl), lag_dft_pos)

        for r in check_lag.rectangles:
            r.set_facecolor("blue")
            r.set_edgecolor("k")
            r.set_alpha(0.2)

        def lag_update(label):
            # should only work for the wdw which are true
            for lbl in lag_lbl:
                num = int(lbl.split()[-1])  # lag to match
                print(label)
                if label == lbl:
                    [(hdl.set_visible(not hdl.get_visible()),
                      lag_dic.update({'lag of '+str(num): hdl.get_visible()}))
                     for hdl in self.hdl_cc
                     if num == eval(hdl.get_label())[0]
                     and self.wdw_dic[eval(hdl.get_label())[1]]]
                # print('wdw status', self.wdw_dic)
                # print('lag status', lag_dic)
            plt.draw()

        check_lag.on_clicked(lag_update)

        # ---------------   Checkboxes for windows plotted   ------------------
        rax_wd = fig.add_axes([0.85, 0.1, 0.10, 0.50])
        self.wdw_pos = list(np.unique([tup[1] for tup in tuples]))
        wdw_dft_pos = (True,)*len(self.wdw_pos)
        self.wdw_dic = dict(zip(self.wdw_pos, list(wdw_dft_pos)))
        check_wd = CheckButtons(rax_wd, tuple(self.wdw_pos), wdw_dft_pos)

        rax_wd.xaxis.label.set_fontsize(2)


        for r in check_wd.rectangles:
            r.set_facecolor("blue")
            r.set_edgecolor("k")
            r.set_alpha(0.2)

        def wdw_update(label):
            # should only work for the lags which are true
            for lbl in self.wdw_pos: # For each wdw position
                num = lbl # lag to match
                print(label)
                if label == str(lbl): #  if the self.wdw_pos is the same as the clicked wdwpos
                    [(hdl.set_visible(not hdl.get_visible()), #  switch visibility line
                      self.wdw_dic.update({num: hdl.get_visible()}))  #  save visibility in dic
                     for hdl in self.hdl_cc  #  for all lines
                     if num==eval(hdl.get_label())[1] #  if the wdw pos matches the handel label (lags, wdwpos)
                     and lag_dic['lag of '+str(eval(hdl.get_label())[0])]] # and the lag dic is not off
                # print('wdw status', self.self.wdw_dic)
                # print('lag status', lag_dic)
            # Switch visibility of window in trace
            [hdl.set_visible(cod)
             for hdl, cod in zip(self.hdl_TSwdws, list(self.wdw_dic.values()))]
            plt.draw()

        check_wd.on_clicked(wdw_update)

        # -------------------   Add TS data Below plot   ----------------------
        ax3 = plt.subplot(gs[4, :])


        init_ts_pos = int(self.TS_DB.shape[1]/2)
        hdl_TS, unit_m = self._TS_plot(ax3, self.TS_DB.index.values,
                               self.TS_DB.loc[:, init_ts_pos])

        # --------------------     Add CC window to TS   ----------------------
        self.hdl_TSwdws = self.TS_plot_wdw(ax3, self.wdw_pos, unit_m)

        # Show plot
        # plt.tight_layout()
        plt.show()
        return check_wd, check_lag

class dispCWI2:
    '''This class is intended to handel the display of PV/CC data stored within
    a single HDF5 database file.
    '''

    def __init__(self, Database, verbose=False):
        self.Database  = Database
        self.hdl_cc = []      # list of handels
        self.verbose = verbose
        self.param = dt.utilities.DB_attrs_load(self.Database,
                                                 ['ww','ww_ol','wdwPos','CC_ref',
                                                  'CC_type', 'PV1', 'PV2',
                                                  'CC_folder'])
    def plot_DB(self):
        ''' Generate Interactive plot of Cross-Correlation data within standard
        database.
        '''

        # Function producing a unique combination of inputs
        def columnsSelect(columnsIn, lags, windows, srcNo, recNo):
            '''Select the appropriate columns from columns based on the input lag,
            window, srcNo, and recNo.

            '''
            legn = []
            columnsOut = []
            for lag, window in itertools.product(lags, windows):
                columns = [col for col in columnsIn if col[0]==lag and
                           col[1]==window and
                           col[2]==srcNo and
                           col[3]==recNo]
                columnsOut.append(columns[0])
                legn.append('lag: %d, window: %s, srcNo: %d, recNo: %d' % \
                (lag, window, srcNo, recNo))


            return columnsOut, legn


        # ---------------------  Define the input data   ----------------------
        CCdata = dt.utilities.DB_pd_data_load(self.Database, self.param['CC_folder'])
        PVdata = dt.utilities.DB_pd_data_load(self.Database, 'PVdata')

        # list of lag, window, src and rec values
        lags = CCdata.columns.get_level_values('lag').unique().tolist()
        windows = CCdata.columns.get_level_values('window').unique().tolist()
        srcNos = CCdata.index.get_level_values('srcNo').unique().tolist()
        recNos = CCdata.index.get_level_values('recNo').unique().tolist()

        # Merge the CC and PV data toegther
        PV_CC_df, columns = dt.utilities.PV_TS_DB_merge(CCdata, PVdata)

        # only use 'CC' data
        columns = [col for col in  columns if 'CC' in col]

        # X-axis data
        if self.param['PV1']:
            x = PV_CC_df[self.param['PV1']].tolist()
        else:
            x = PV_CC_df.index.tolist()

        # Initial values to plot
        srcNo = srcNos[len(srcNos)//2]
        recNo = recNos[len(recNos)//2]
        cols, legn = columnsSelect(columns, lags, windows, srcNo, recNo)

        y = PV_CC_df.loc[:, cols]

        if self.param['PV2']:
            y2 = PV_CC_df[self.param['PV2']]
        else:
            y2 = None

        # ---------------------        Setup Figure      ----------------------
        plt.style.use('bmh')
        fig = plt.figure(figsize=(20, 30))
        plt.subplots_adjust(left=0.05, bottom=0.12,
                            right=0.8, top=0.9,
                            wspace=0.9, hspace=1.5)
        gs = gridspec.GridSpec(5, 4)
        ax1 = plt.subplot(gs[0:4, :])
        plt.ylim([0, 1])
        linewidth = 0.5
        markersize = 3
        marker = itertools.cycle(('o', 'v', '*', 'd', 's', 'D', 'h', 'P', 'X',
                                  '^', '<', '>', '8', 'H', 'p'))
        PV1_lb, PV2_lb, genTitle = self.CC_labels(columns)

        # -------------------   Plot the CC & PV data   ----------------------
        self.hdl_cc = ax1.plot(x, y, linewidth=linewidth,
                                    markersize=markersize)

        # Cycle through marker list
        [(line.set_marker(next(marker)), line.set_linestyle('-'))
         for line in self.hdl_cc]

        # Add second axis PV data if available
        if y2 is not None:
            ax2 = ax1.twinx()
            ax2.plot(x, y2, 'k-', linewidth=1)
            ax2.set_ylabel(PV2_lb, color='k')

        # -------------------    Add tiles/label..etcc  ----------------------
        ax1.set_title(genTitle)
        ax1.set_xlabel(PV1_lb)
        ax1.set_ylabel('CC', color='g')
        [self.hdl_cc[n].set_label(legn[n]) for n in range(len(legn))]
        legn_hdl = ax1.legend(loc='upper left', frameon=False)
        legn_hdl.get_frame().set_alpha(0.4)

        # ---------------    Interactive Settings  ------------------
        slider_color = 'lightgoldenrodyellow'
        sldr_w = 0.10 ; sldr_h = 0.02
        sldr_x = 0.87 ; sldr_y = 0.2

        plt.show()


        # ---------------    Add slider to update receiver pos  ------------------
        if min(recNos) is not max(recNos):
            recPos_slider_ax = fig.add_axes([sldr_x, sldr_y, sldr_w, sldr_h],
                                            facecolor=slider_color)
            recPos_slider = Slider(recPos_slider_ax, 'recNo:', min(recNos),
                                   max(recNos),
                                   valfmt='%d',
                                   valinit=recNo)

            def slider_recNo_on_change(val):
                recNo = int(recPos_slider.val)
                # Update CC plot data
                cols, legn = columnsSelect(columns, lags, windows, srcNo, recNo)
                # Update data and plot
                y = PV_CC_df.loc[:, cols]
                [self.hdl_cc[idx].set_ydata(y[col]) for idx, col in enumerate(cols)]
                [legn_hdl.get_texts()[idx].set_text(leg) for idx,leg in enumerate(legn)]
                fig.canvas.draw_idle()

            recPos_slider.on_changed(slider_recNo_on_change)
        else:
            recPos_slider = None

        # ---------------    Add slider to update source pos  ------------------
        if min(srcNos) is not max(srcNos):
            slider_color = 'lightgoldenrodyellow'
            srcPos_slider_ax = fig.add_axes([sldr_x, sldr_y + sldr_w, sldr_w, sldr_h],
                                            facecolor=slider_color)
            srcPos_slider = Slider(srcPos_slider_ax, 'srcNo:', min(srcNos),
                                   max(srcNos),
                                   valfmt='%d',
                                   valinit=srcNo)

            def slider_srcNo_on_change(val):
                srcNo = int(srcPos_slider.val)
                # Update CC plot data
                cols, legn = columnsSelect(columns, lags, windows, srcNo, recNo)
                # Update data and plot
                y = PV_CC_df.loc[:, cols]
                [self.hdl_cc[idx].set_ydata(y[col]) for idx, col in enumerate(cols)]
                [legn_hdl.get_texts()[idx].set_text(leg) for idx,leg in enumerate(legn)]
                fig.canvas.draw_idle()

            srcPos_slider.on_changed(slider_srcNo_on_change)
        else:
            srcPos_slider = None

        plt.show()
        return self.hdl_cc, recPos_slider, srcPos_slider

    def CC_labels(self, tuples):
        ''' Handels the assignment of figure titles and axis labels. Basic
        functionality, checks if variables is assigned in param, if not a
        a default is assigned.
        tuples: tuples of (lag, self.self.wdw_pos)
        '''

        if 'PV1_lb' in self.param:
            PV1_lb = self.param['PV1_lb']
        elif 'PV1' in self.param:
            PV1_lb = self.param['PV1']
        elif 'PV2' in self.param:
            PV1_lb = self.param['PV2']
        else:
            PV1_lb = 'Measurement No.'

        if 'PV2_lb' in self.param:
            PV2_lb = self.param['PV2_lb']
        elif 'PV2' in self.param:
            PV2_lb = self.param['PV2']
        else:
            PV2_lb = None

        if 'genTitle' in self.param:
            genTitle = self.param['genTitle']
        else:
            genTitle = 'No Title'

        return (PV1_lb, PV2_lb, genTitle)