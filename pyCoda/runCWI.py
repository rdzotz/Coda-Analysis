#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:03:35 2017

@author: rwilson
"""
import userInput as ui
import data as dt
import pre_process as prp
import postProcess as pop
import dataStore as ds
import cross_correlation as cc
import dispCWI as dcwi
import pickle

'''TODO:
    * Consider saving the data in fixed format to avoid issues with time index...etc
    * Move TSgain function in pre_processing to apply directly when TS data is read
    in for the shell data formate
    * write an attribute import function which imports all info from database
'''


def printOut(message, font='starwars', attrs=['bold'], color = 'yellow',
             background = 'on_red'):
    '''Basic printing to screen.
    '''
    import sys
    from colorama import init
    init(strip=not sys.stdout.isatty()) # strip colors if stdout is redirected
    from termcolor import cprint
    from pyfiglet import figlet_format

    cprint(figlet_format(message, font=font),
       color, background, attrs=attrs)

class runCWI:
    '''The overarching CWI processing class, which allows the user to be able
    to extract certain information from the processing run on request.
    Additionaly, the resulting data is archived within the run folder. If the
    user would like display the data again, the function will check against the
    archive, and only perform the processing if the new setup.txt contains
    parameters which are different from the old setup.txt.

    Parameters
    ----------
    self.setupFile : str
        Relative or absolute path to a txt file containing mandatory and optional
        setup parameters.
    self.Database str (Default None)
        Relative or absolute path to a /Database.h5 file of the expected structure.
        Only required if parameter ``import_raw`` is False.
    '''


    printOut('CWI',font='doh')

    def __init__(self, setupFile, Database=None):
        self.setupFile = setupFile
        self.Database = Database

    def runProcess(self):
        # Import user data
        usrIn = ui.userInput(self.setupFile)
        param, re_run = usrIn.fread()


        # --------------------- Import Raw Data ---------------------
        if param['import_raw']:
            # Import raw data
            raw_data = dt.data_import(param['TSloc'], param['PVloc'],
                                      param['import_dtype'], param)
            self.Database = raw_data.run_data_import()

        if re_run or re_run is None:
            print(' ----------',
              'New critical input parameters detected, begin processing',
              '----------')

            # --------------------- Apply pre processing ---------------------
            pre_process =prp.preProcess(self.Database, param)
            pre_process.run_PP()

            # --------------------- Cross-correlations ---------------------
            dt.utilities.DB_attrs_save(self.Database, param)
            process = cc.cross_correlation_survey(self.Database, verbose=True)
            process.CC_run()


#            # --------------------- Post Process data ---------------------
#            post_process = pop.postProcess(param, PV_df, CC, TScut)
#            TScut_DB, PV_df = post_process.postProcess_run()

        if param['disp_DB']:
                plot_db = dcwi.dispCWI2(self.Database)
                hdl_cc, recPos_slider, srcPos_slider = plot_db.plot_DB()
