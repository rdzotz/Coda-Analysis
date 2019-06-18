import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.dates import date2num
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
from matplotlib.widgets import CheckButtons
import itertools


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
                  if isinstance(tup, tuple)]

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

        [hdl] = ax.plot(x_TS, TS)
        hdl.set_linewidth(0.5)

        # Check for the approriate label
        if 'TS_units' in self.param and self.param['TS_units'] == 'msec':
            ax.set_xlabel('Time [msec]')
        else:
            ax.set_xlabel('Time')

        return hdl

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

    def TS_plot_wdw(self, ax, wd_pos):
        ''' This private function handels the general plotting of the
        correlation window considered. for each window in wd_pos
        Inputs:
        ------
        wd_pos: sample number of beginning of window
        TODO:
        '''

        # Set hight and width of window
        height = self.TS_DB.max().max()*2
        width = self.TS_DB.index[self.param['ww']]

        # print('hieght', height)
        # print('width', width)

        # Set coords of bottom corner
        y = -height/2
        x = [self.TS_DB.index[pos] for pos in wd_pos]

        # Draw each rectangle on ax
        hdls = [ax.add_patch(patches.Rectangle(
                (x_pos, y), width, height, alpha=0.1))
                for x_pos in x]
        [hdls[n].set_label(wd_pos[n]) for n in range(len(wd_pos))]

        return hdls

    def plot_DB(self):
        ''' Generate Interactive plot of DB
        '''
        # ---------------------  Define the input data   ----------------------
        tuples = self.wdw_lag_col()
        x = self.DB.index.values
        y = self.DB.loc[:, tuples]
        y2 = self.DB[self.param['PV2']]

        # ---------------------        Setup Figure      ----------------------
        plt.style.use('bmh')
        fig = plt.figure(figsize=(20, 10))
        plt.subplots_adjust(left=0.1, bottom=0.5,
                            right=0.8, top=0.9,
                            wspace=0.9, hspace=1.5)
        gs = gridspec.GridSpec(4, 4)
        ax1 = plt.subplot(gs[0:3, :])
        plt.ylim([0, 1])
        linewidth = 0.5
        markersize = 1
        marker = itertools.cycle(('o', 'v', '*', 'd', 's', 'D', 'h', 'P', 'X',
                                  '^', '<', '>', '8', 'H', 'p'))
        PV1_lb, PV2_lb, genTitle, legn = self.CC_labels(tuples)

         # -------------------   Plot the CC & PV data   ----------------------
        self.hdl_cc = ax1.plot_date(x, y, linewidth=linewidth,
                               markersize=markersize)
        plt.gcf().autofmt_xdate()

        # Add second axis PV data if available
        if y2 is not None:
            ax2 = ax1.twinx()
            if isinstance(self.PV_full, pd.DataFrame):
                x_full = self.PV_full[self.param['PV1']].tolist()
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
        rax_wd = fig.add_axes([0.85, 0.4, 0.07, 0.15])
        self.wdw_pos = list(np.unique([tup[1] for tup in tuples]))
        wdw_dft_pos = (True,)*len(self.wdw_pos)
        self.wdw_dic = dict(zip(self.wdw_pos, list(wdw_dft_pos)))
        check_wd = CheckButtons(rax_wd, tuple(self.wdw_pos), wdw_dft_pos)

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
                      hdl_patch.set_visible(not hdl_patch.get_visible()),  # switch visibility window
                      self.wdw_dic.update({num: hdl.get_visible()}))  #  save visibility in dic
                     for hdl, hdl_patch in zip(self.hdl_cc, self.hdl_TSwdws)  #  for all lines
                     if num==eval(hdl.get_label())[1] #  if the wdw pos matches the handel label (lags, wdwpos)
                     and lag_dic['lag of '+str(eval(hdl.get_label())[0])]] # and the lag dic is not off
                # print('wdw status', self.self.wdw_dic)
                # print('lag status', lag_dic)
            plt.draw()

        check_wd.on_clicked(wdw_update)

        # -------------------   Add TS data Below plot   ----------------------
        ax3 = plt.subplot(gs[3, :])
        init_ts_pos = int(self.TS_DB.shape[1]/2)
        hdl_TS = self._TS_plot(ax3, self.TS_DB.index.values,
                               self.TS_DB.loc[:, init_ts_pos])

        # --------------------     Add CC window to TS   ----------------------
        self.hdl_TSwdws = self.TS_plot_wdw(ax3, self.wdw_pos)

        # Show plot
        # plt.tight_layout()
        plt.show()
        return check_wd, check_lag