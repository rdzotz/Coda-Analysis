#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:37:48 2017

@author: rwilson
"""

import numpy as np
import pickle
import os
import ast
import sys

class utilities:
    '''Collection of functions tools applicable to user interaction.
    '''

    @staticmethod
    def query_yes_no(question, default="yes"):
        """Ask a yes/no question via raw_input() and return their answer.

        "question" is a string that is presented to the user.
        "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

        The "answer" return value is True for "yes" or False for "no".
        """
        valid = {"yes": True, "y": True, "ye": True,
                 "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)

        while True:
            sys.stdout.write(question + prompt)
            choice = input().lower()
            if default is not None and choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write("Please respond with 'yes' or 'no' "
                                 "(or 'y' or 'n').\n")

class userInput:
    """This class is intended to handle the direct input general information
    for the purpose of processing, display and documentation. A .txt file is
    expected with a basic, variable and value format. % comments are ignored.

    var = val % val can be string or numeric input

    All identified variables are saved into a dictionary of the format
    dic = {var1: val1, var2: val2 }

    This class is also intended to compare the initial user input info with the
    previous run, and return re_run as False if the same inputs are given.

    Parameters
    ----------
    fileIn  Location of user input txt file

    Notes
    -----
    The accepted inputs are as follows:
        Setup paramaters:
        import_raw : Request Raw data to be imported (Default: False)

        re_run : Force the reprocessing, even if no changes found in input file.

        import_dtype : The data type to import(bin_par, Shell_format, 'CSIRO'.

        survey_type : ``multiple`` (Default) if multiple src/rec pairs in subfolder structure
            is expected. ``single`` if a single src/rec pair is expected with no
            subfolders.

        TSstart_date : str
            The start of TS acquistion in format YYYY-MM-DD. Only required if no
            absolute time data available in TS header information.

        TSloc : location of folder/file containing the time series data.

        PVloc : location of folder/file containing the perturbation data.

        loadDB : If True, no attempt to reload raw data into database will be made

        ------------------- Pre-processing inputs  -------------------
        PVstart_date_time : (Default None) The time ``YYYY-MM-DD HH:MM:SS.sssss``
            the first PV measurement was made, in sync with TS time.

        PV_time_col : (required) Name of time column to match in PV data

        PV_time_col_unit : (Default = 'D') the time unit of the PV column to match,
            see pd.to_datetime

        sampNo : The number of sample points in a single trace recording, only
            required as input if not found in TS header information.

        sampFreq : The sampling frequence of TS data, [samp/sec]

        ------------------- Cross-correlation parameters -------------------
        sig : bool
            If True is given then a spectral significance test will be made for each
            correlation performed. This is an expensive calculation so will require
            considerable time to perform. See ``cross-correlation``

        lagOverlap : bool (Default True) UNDER DEVELOPMENT
            Allow overlap between rolling lag values (i.e. 1-3, 2-4). If set to
            ``False``, only none overlapping lags will be calculated (e.g. 1-3, 3-5).

        Eng_ref : int (Default None)
            Indicating the trace/survey to which all subsequent relative energy calculations
            will be referenced to.

        STA_meas : int/str (Default False)
            The first survey to include in the correlation processing, if false
            then no start will be set. Note: only for ``survey_type``
            multiple, a date string in the format YYYY-MM-DD HH:MM:sec.msec must be
            given, which corresponds to the survey folder number.

        sta_wdws : int (Optional)
            If ``wdwPos`` and ``ww_ol`` as given then the user can also define the
            start of the series of overlapping windows.

        end_wdws : int (Optional)
            If ``wdwPos`` and ``ww_ol`` as given then the user can also define the
            end of the series of overlapping windows.

        wdwPos : (int, Default = False) The start position of the windows in trace sample numbers,
            if more than one is provided then ``ww`` must be of equal length. If not
            provided then both ``ww`` and ``ww_ol`` should be given.

        ww : (int) Width of each window  in trace sample numbers,
            if more than one is provided then ``wdwPos`` must also be of equal length.

        ww_ol : (int) (Default False)
            The percentage window overlap. The max number of windows of
            ``ww`` lenght which fit within a trace will be calculated.
            This will be ignored if ``wdwPos`` is provided.

        taper : bool
            Apply a tukey window taper to each correlation window

        taperAlpha : float (Default 0.1)
            The alpha of the window taper to be applied.

        CC_folder : (str, Default = 'CCprocessed')
            The name of the folder within which the CC processed data will be saved.

         ------------------- Display parameters -------------------
        disp_DB : bool (Default True)
            If set to true, some portion of the processed database will be plotted
            to screen at the end of the Class ``runCWI``.
    """

    def __init__(self, fileIn):
        self.fileIn = fileIn
        self.param = None

    def fread(self):

        # Conversion of string True or False values to booleans
        def str_to_bool(s):
            if s == 'True':
                 return True
            elif s == 'False':
                 return False
            else:
                 return s

        usrIn = {}

        with open(self.fileIn, 'r') as f:
            for line in f:
                line = line.partition('#')[0]
                line = line.partition('=')
                var = line[0].rstrip()
                val = line[2].rstrip()

                if isinstance(val, str):
                    val = val.strip()  # Remove whitespace
                    val = str_to_bool(val) # Convert to bool if possible

                if self.is_number(val):
                    val = float(val)

                    if val.is_integer():
                        val = int(val)

                usrIn[var] = val
        # Catch empty dic keys
        try:
            del usrIn['']
        except KeyError:
            None

        # Allocate recently imported user defined param
        self.param = usrIn

        # ---------------- Allocate default param ----------------
        self._setDefaults()

        # perform basic input data checks

        # Check if and changes are found in param file only if not set by user as
        # True
        re_run = self.param['re_run']
        if not re_run:

            db_folder = self.param['TSloc'].split('/')[-2]+'_DB'  # 'database'
            usrIn_Old = self._from_pkl(db_folder + '/'+'paramInt.txt')
            if usrIn_Old is not False:
                re_run = self.checkSetup(usrIn_Old)
            else:
                re_run = True

             # If false of not existing make folder and store param file
            if re_run or re_run is None:
                if not os.path.exists(db_folder):
                    os.makedirs(db_folder)

                self._to_pkl(self.param, db_folder+'/paramInt.txt')

        return usrIn, re_run

    def _to_pkl(self, data, fname):
        ''' Save pickel files
        '''
        output = open(fname, 'w+b')

        pickle.dump(data, output)
        output.close()

    def _from_pkl(self, fname):
        ''' Load pickel files
        '''
        try:
            output = open(fname, 'rb')
            obj_dict = pickle.load(output)
            return obj_dict
        except  FileNotFoundError:
            return False

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def _setDefaults(self):
        ''' Set the default values if not specified by the user
        '''

        try:
            self.param['import_raw']
            if self.param['import_raw']:
                print('\n--------- WARNING ---------')
                question = '* Parameter import_raw was set to True, this will '+ \
                'remove any existing Database.h5 files found! \n'+ \
                'Are you sure you want to continue ?'
                answer = utilities.query_yes_no(question, default="yes")
                if not answer: raise SystemExit

        except KeyError:
            # If does not exist then set to True
            self.param['import_raw'] = False
            print('Import raw data')

        try:
            self.param['survey_type']
        except KeyError:
            self.param['survey_type'] = 'multiple'

        try:
            self.param['re_run']
        except KeyError:
            self.param['re_run'] = False

        try:
            self.param['import_dtype']
        except KeyError:
            # If does not exist then set to True
            self.param['import_dtype'] = 'bin_par'
            print('Import datatype defaulted to bin_par')

        try:
            self.param['TSstart_date']
        except KeyError:
            self.param['TSstart_date'] = 'unix'

        try:
            self.param['PV_time_col']
        except KeyError:
            print('Critical parameter PV_time_col not found \n',
                  'Please define this in the setup file.')
            raise SystemExit

        try:
            self.param['PV_time_col_unit']
        except KeyError:
            self.param['PV_time_col_unit'] = 'ns'

        try:
            self.param['PVstart_date_time']
        except KeyError:
            self.param['PVstart_date_time'] = 'unix'

        try:
            self.param['PV_file_hdr_rows']
        except KeyError:
            self.param['PV_file_hdr_rows'] = False

        try:
            self.param['disp_DB']
        except KeyError:
            # If does not exist then set to True
            self.param['disp_DB'] = True

        try:
            self.param['wdwPos']
        except KeyError:
            # If does not exist then set to True
            self.param['wdwPos'] = False

        try:
            self.param['ww_ol']
        except KeyError:
            # If does not exist then set to True
            self.param['ww_ol'] = False

        try:
            self.param['sta_wdws']
        except KeyError:
            # If does not exist then set to True
            self.param['sta_wdws'] = False

        try:
            self.param['end_wdws']
        except KeyError:
            # If does not exist then set to True
            self.param['end_wdws'] = False

        try:
            self.param['STA_meas']
            if (self.param['survey_type'] == 'multiple' and
                not isinstance(self.param['STA_meas'], str)):
                sys.exit('STA_meas must be a string when survey_type==multiple')

        except KeyError:
            # If does not exist then set to True
            self.param['STA_meas'] = 0

        try:
            self.param['END_meas']
            if (self.param['survey_type'] == 'multiple' and
                not isinstance(self.param['END_meas'], str)):
                sys.exit('END_meas must be a string when survey_type==multiple')
        except KeyError:
            # If does not exist then set to True
            self.param['END_meas'] = 0

        try:
            self.param['loadDB']
        except KeyError:
            # If does not exist then set to False
            self.param['loadDB'] = False

        try:
            self.param['Eng_ref']
        except KeyError:
            # If does not exist then set to False
            self.param['Eng_ref'] = False

        try:
            self.param['taper']
        except KeyError:
            # If does not exist then set to False
            self.param['taper'] = False

        try:
            self.param['taperAlpha']
        except KeyError:
            # If does not exist then set to False
            self.param['taperAlpha'] = 0.1

        try:
            self.param['CC_folder']
        except KeyError:
            # If does not exist then set to False
            self.param['CC_folder'] = 'CCprocessed'

        try:
            self.param['lagOverlap']
        except KeyError:
            # If does not exist then set to False
            self.param['lagOverlap'] = True

        if self.param['CC_type'] == 'fixed':
            self.param['lagOverlap'] = True

        try:
            self.param['stress_strain']
        except KeyError:
            # If does not exist then set to False
            self.param['stress_strain'] = False

        try:
            self.param['stress_strain_confined']
        except KeyError:
            # If does not exist then set to False
            self.param['stress_strain_confined'] = False

        try:
            self.param['rename_dic']
            self.param['rename_dic'] = ast.literal_eval(self.param['rename_dic'])
        except KeyError:
            # If does not exist then set to False
            self.param['rename_dic'] = False

        try:
            self.param['LLLength']
        except KeyError:
            # If does not exist then set to False
            self.param['LLLength'] = False

        try:
            self.param['FBP']
        except KeyError:
            # If does not exist then set to False
            self.param['FBP'] = False

        try:
            self.param['TS_samp_dt']
        except KeyError:
            # If does not exist then set to False
            self.param['TS_samp_dt'] = False

        try:
            self.param['TS_file_tmatch']
        except KeyError:
            # If does not exist then set to False
            self.param['TS_file_tmatch'] = 0

        try:
            self.param['sig']
        except KeyError:
            # If does not exist then set to False
            self.param['sig'] = False

        try:
            self.param['PV1']
        except KeyError:
            # If does not exist then set to False
            self.param['PV1'] = False

        try:
            self.param['PV2']
        except KeyError:
            # If does not exist then set to False
            self.param['PV2'] = False

        # Install some input processing
        for NAME in ['CC_ref', 'wdwPos', 'ww','PV_file_hdr_rows']:
            try:
                if isinstance(self.param[NAME], str):
                    self.param[NAME] = [int(e) for e in
                                            self.param[NAME].split(',')]
                else:
                    self.param[NAME] = [self.param[NAME]] # ensure is a list
            except ValueError:
                raise Exception('The input variable',NAME,'should either be a',
                                'single numeric value or a comma separated',
                                'input of lag values x1,x2,x3,..etc')

    # def _inputCheck(self):


    def checkSetup(self, check):
        ''' This functions checks the setup file to determine if any param have
        changed. If yes, the processing will be re-run, otherweise the saved
        datebases will be loaded. return True if change is detected, False if
        any critical param are found and None if only non-critical param change
        '''

        def dict_compare(d1, d2):
            d1_keys = set(d1.keys())
            d2_keys = set(d2.keys())
            intersect_keys = d1_keys.intersection(d2_keys)
            added = d1_keys - d2_keys
            removed = d2_keys - d1_keys
            modified = {o : (d1[o], d2[o]) for o in intersect_keys if
                        np.all(d1[o] != d2[o])}
            # same = set(o for o in intersect_keys if np.all(d1[o] == d2[o]))

            crit_param = ['ww', 'ww_ol', 'CC_type', 'CC_ref']
            crit_mod = [d1[x] == d2[x] for x in crit_param]

            if (len(added) == 0 and len(removed) == 0 and len(modified) == 0):
                return True
            elif sum(crit_mod)>0:
                return False
            else:
                return None



        return not dict_compare(self.param, check)
    import sys