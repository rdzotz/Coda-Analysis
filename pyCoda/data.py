#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements the global import of all data

Created on Mon Dec 26 20:51:08 2016
@author: rwilson
"""

import numpy as np
import glob
import re
import os
import csv
from itertools import repeat
import pandas as pd
import h5py
from dateutil.parser import parse
import codecs
from scipy.io import loadmat
from scipy import signal


import shelve
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

class utilities:
    '''Collection of functions intended for data related processes.
    '''

    def DB_group_names(Database, group_name = None):
        '''Read in group names found within group. If group is not provided the
        upper folder structure will be read from.

        Parameters
        ----------
        Database : str
            Relative location of database
        group_name : str
            The expected attribute name/s

        Returns
        -------
        group_names : list
            Group names found within the group

        notes
        -----
            Add some additional error checks
        '''
        with h5py.File(Database, 'r') as h5file:

            if group_name is not None:
                group = h5file.get(group_name)
                group_names = [key for key in group.keys()]
            else:
                group_names = [key for key in h5file.keys()]

        return group_names

    def DB_attrs_save(Database, dictionary):
        '''Save attribute to database head.

        Parameters
        ----------
        Database : str
            Relative location of database
        dictionary : dict
            Dictionary of attributes

        notes
        -----
            Add some additional error checks
        '''

        print('* The following %s attributes will be updated' % Database)
        with h5py.File(Database, 'r+') as h5file:
                for key,item in zip(dictionary.keys(), dictionary.values()):
                    print('Key:', key,'| item:', item)
                    h5file.attrs[key] = item

    def DB_attrs_load(Database, attrs_names):
        '''Read attribute from database head.

        Parameters
        ----------
        Database : str
            Relative location of database
        attrs_names : list(str)
            The expected attribute name/s

        Returns
        -------
        dict_attri : dict
            The returned dictionary of attribute/s from the database

        notes
        -----
            Add some additional error checks
        '''
        dict_attrs = {}
        with h5py.File(Database, 'r') as h5file:

            for attrs_name in attrs_names:
                # Load the database
                attrs = h5file.attrs[attrs_name]
                dict_attrs[attrs_name] = attrs

        return dict_attrs

    def DB_pd_data_load(Database, group, cols = None, whereList = None):
        '''Loads in a pandas dataframe stored in group from the Database.

        Parameters
        ----------
        Database : str
            Relative location of database
        group : str
            The expected group name
        cols : list(str) / list(int)
            If not None, will limit the return columns, only applicable for ``table``
            format database. For ``fixed`` format database only int accepted
        whereList : list of Term (or convertable) objects or slice(from, to)
             The conditional import of data, example ['index>11', 'index<20'],
             only applicable for ``table`` format database. For ``fixed`` format
             database only a slice object is applicable and will use the row index
             numbers not the index values (i.e. df.iloc vs df.loc)

        Returns
        -------
        group_df : DataFrame
            The PV data stored in the group ``PVdata`` as a pandas dataframe


        TSsurvey =  pd.read_hdf(h5file, 'survey20180312093545',
                    columns=[(1,1), (1,2)], # Load specific columns
                    where = ['index>11', 'index<20']) # Load index 11 -> 20
        '''

        with pd.HDFStore(Database, 'r+') as h5file:

            # Check that the expected group name is found in the database
#            group_names = [key for key in h5file.keys()]
#            expected_group_name = '/'+group
#            if expected_group_name not in group_names:
#                raise KeyError('The %s group was not found within the %s database.' \
#                               %(expected_group_name, Database))
            # Load the database
            try:
                group_df = pd.read_hdf(h5file, group, columns = cols, where = whereList)
            except TypeError:
                with pd.HDFStore(Database, 'r+') as h5file:
                    group_df = pd.read_hdf(h5file, group)
                group_df = group_df.iloc[whereList, cols]
        return group_df

    def DB_pd_data_save(Database, group, df):
        '''Saves in a pandas dataframe stored in group from the Database.

        Parameters
        ----------
        Database : str
            Relative location of database
        group : str
            The expected group name
        df : DateFrame
            Pandas DataFrame to be stored in h5 file ``Database``
        '''

        with pd.HDFStore(Database, 'r+') as h5file:
            # Save the database
             df.to_hdf(h5file, group)

    def PV_TS_DB_merge(CC, PVdata, mergeOnIndex=False):
        '''Merge/concatenate based on the time axis. The expected structures of
        'PVdata' and 'CC' DataFrames is a ``Time Stamp`` axis and a ``Time`` index
        level on which the concatenation takes place.

        Parameters
        ----------
        CC : DataFrame of list(DataFrame)
            Expected to contain the processed data or a list of DataFrame with
            processed data. The index must be a timestam.
        PVdata : DataFrame
            A single DataFrame containing the corresponding perturbation information
            and must contain a column ``Time Stamp``, which will be used during the
            concatentation/merge.
        mergeOnIndex : Default False
            Merge based on the index values. Currently only works when a single ``CC``
            DataFrame is provided, and not for a list of DF.

        Returns
        -------
        PV_CC_df : DataFrame
            Merged/concatenated dataframe of both PV and Coda processed data.
        column_values : array
            of tuples defining the multi-level indecies of the CC data.
        '''

        # Load the individual dataframes.
        # CC = utilities.DB_pd_data_load(Database, 'CCprocessed')
        # PVdata = utilities.DB_pd_data_load(Database, 'PVdata')

        # Convert the Time to datetime

        # Dealing with single CC dataframe
        if not isinstance(CC, list): CC= [CC]

        CC_list = []
        col_tup = []
        for CC_df in CC:
            CC_df.index.set_levels(pd.to_datetime(CC_df.index.levels[2]), level='Time',
                                           inplace=True)

            # Pivot the otermost row indcies to columns
            CC_test = CC_df.unstack(level=[0,1])
            CC_test = CC_test.reorder_levels(
                    ['lag', 'window', 'srcNo', 'recNo','Parameters'], axis=1)
            CC_list.append(CC_test)
            col_tup = col_tup + CC_test.columns.get_values().tolist()

        PVdata.set_index('Time Stamp', inplace=True)

        if mergeOnIndex:
            PV_CC_df = pd.merge(PVdata, CC_list[0], how='inner', left_index=True,
                            right_index=True)
        else:
            PV_CC_df = pd.concat([PVdata] + CC_list, axis=1)

        interpCols = PVdata.columns
        PV_CC_df[interpCols] = PV_CC_df[interpCols].interpolate()

        return PV_CC_df, col_tup

    def DB_COL_stats(DF, colList, baseName, stats = ['mean', 'std'], norm=False):
        '''Extract stats from multiple columns with a ``CommonKey``.

        Parameters
        ----------
        DF : DataFrame
            Dataframe from which statistics will be generated.
        colList : str
            a list of columns from which the stats will be made.
        baseName : str
            The base name of the new columns to which '_[stats]' will be appended.
        stats : list
            A list of strings containing the requested stats to be generated for
            columns with ``CommonKey``.
        norm : list
            Perform a min-max norm of the added statistic between 0 and 1.

        Returns
        -------
        DF : DataFrame
            Original dataframe plus columns containing requested stats.
        '''
        for stat in stats:

            if isinstance(baseName, tuple):
                baseName_stat = list(baseName)
                baseName_stat[-1] = baseName_stat[-1]+'_'+stat
                baseName_stat = tuple(baseName_stat)
            else:
                baseName_stat = baseName+'_'+stat

            if stat is 'mean':
                DF[baseName_stat] = DF.loc[:, colList].mean(axis=1)
            elif stat is 'std':
                DF[baseName_stat] = DF.loc[:, colList].std(axis=1)

            if norm:
                DF[baseName_stat] = (DF[baseName_stat] -
                                     DF[baseName_stat].min()) / \
                                    (DF[baseName_stat].max() -
                                     DF[baseName_stat].min())
        return DF

    def CC_lag(CC_df, period=None, units='us', relVel = True, strainCol=None):
        '''Convert CC lag or First Break Picked data from number of sample points
        to time or relative velocity change based on:
            .. math::
                \dfrac{\delta v}{v} = -\dfrac{\delta t}{t}
        Expected column names should either contain 'lag' in the last of a tuple,
        eg. col[-1] in the case of lag correction, or 'FBP' in the case of First
        Break Picking correction. If a 'FBP' correction is required, then the
        correct input initial velocity should be given.

        Parameters
        ----------
        CC_df : DataFrame
            Dataframe from which statistics will be generated. A three level
            dataframe is expected where the lowest level
        period : float (default=None)
            Seconds per sample,
        units : str
            unit of the arg (D,s,ms,us,ns) denote the unit, which is an integer
            or float number.
        relVel : bool
            Output the lag in terms of the relative velocity change.
        strainCol : str (default=None)
            The name of the strain column within ``CC_df`` which will be used
            to correct the relative velocity calculation for a change in
            separation distance.
        Returns
        -------
        DF : DataFrame
            Original dataframe with the lag columns modified as specified.
        '''

        unit_dict = {'us': 10E6, 'ms': 10E3, 'ns': 10E9, 's': 1}

        # Find all the lag/FBP columns
        cols = [col for col in CC_df.columns.tolist() if 'lag' in col[-1] or
                'FBP' in col]

        if relVel and 'FBP' in cols:
            if strainCol:
                Vint = CC_df.loc[0, strainCol]/ CC_df.loc[0, cols[0]]
                deltaV = CC_df.loc[:, strainCol]/ CC_df.loc[:, cols[0]] - Vint
            else:
                Vint = 1 / (CC_df.iloc[0, :] )
                deltaV = 1 / (CC_df.loc[:, cols]) - Vint

            CC_df.loc[:, cols] = deltaV/Vint
            return CC_df

        elif relVel:
            # Middle of each window
            t_prop = np.array([(int(wdw[1].split('-')[0]) +
                                int(wdw[1].split('-')[1]))/2
                               for wdw in cols]) * unit_dict[units] * period

            CC_df.loc[:, cols] = (CC_df.loc[:, cols] * \
                                 period * unit_dict[units] ) / t_prop*-1
            return CC_df

        t_prop = 1
        CC_df.loc[:, cols] = CC_df.loc[:, cols] * \
                                        period * unit_dict[units] / t_prop

        return CC_df

    def CC_integration(DF, dx='index', intType='trapz'):
        '''Performs the integration of multiple columns witin a dataframe which is
        expected to contain row index levels 'srcNo', 'recNo' and 'Time'. Each
        srcNo and recNo pair over time will be integrated and added back into the
        DF.

        Parameters
        ----------
        DF : DataFrame
            Dataframe of the data to be integrated which must be row indexed by
            ``datetime64[ns]``.
        dx : str (Default 'index')
            If set to 'index' than integration is performed based on the time axis
        intType : str (Default 'trapz')
            The type of integration to use. trapz for trapezoid, cumsum for pure
            cummulative summation.
        Returns
        -------
        DF_int : DataFrame
            THe same dimensions and row indicies as DF containing the cumtrapz
            integration.
        '''
        from scipy import integrate
        import itertools

        # Define empty integration dataframe
        col_list = [list(col) for col in DF.columns.tolist()]
        col_list_int = [tuple(col_int[:-1] + [col_int[-1]+'-int'])
                            for col_int in col_list]

        colIndex = pd.MultiIndex.from_tuples(col_list_int, names=DF.columns.names)
        DF_int = pd.DataFrame(columns=colIndex, index=DF.index)

        # Extract each unique src-rec pair in the DF
        src_rec_pairs = [(src, rec) for src, rec in
            itertools.product(
                    DF.index.get_level_values('srcNo').unique().values,
                    DF.index.get_level_values('recNo').unique().values)]
        for src_rec in src_rec_pairs:
            # Generate the index which will be integrated and convert to seconds
            df = DF.loc[src_rec+(slice(None),)]
            if dx=='index':
                x = df.index.astype(np.int64)/10**9
            else:
                dx = 1
                x = None

            if intType=='trapz':
                y = df.apply(pd.to_numeric).values
                y_int = integrate.cumtrapz(y, x = x, dx=1, initial=0, axis=0)
                # Add data to dataframe
                DF_int.loc[src_rec+(slice(None),)] = y_int

            elif intType=='cumsum':
                y = df.apply(pd.to_numeric)
                DF_int.loc[src_rec+(slice(None),)] = y.cumsum().values

        return DF_int

    def CC_to_K(DF):
        '''Convert all Cross-correlation coefficients to decorrelation by applying
        the simple transform:
            .. math::
                K = 1- CC


        Parameters
        ----------
        DF : DataFrame
            Dataframe with multi level columns where the cross-correlation
            coefficients are expected to be named 'CC'.

        Returns
        -------
        DF_int : DataFrame
            The dataframe is returned with only the data of the columns modified.
        '''

        CC_cols = [col for col in DF.columns if 'CC' in col]
        DF.loc[:, CC_cols] = 1 - DF.loc[:, CC_cols]

        return DF

    def Data_CSV_dump(DF, fileName, colNames = None, indices = None, CCtoK = False,
                      shiftCols=None, nthRow=None):
        '''Dumps data from pandas dataframe to csv, intended for tikz plotting.
        Note, the following char will be removed from the column names ,\'_\[\]%
        and any row of the selected ``colNames`` containing atleast one NaN will
        be filled with ``0``.

        Parameters
        ----------
        DF : DataFrame
            DataFrame to be dumped
        fileName : str
            Name of the csv file saved to current working directory.
        colNames : list
            The columns to be dumped.
        indices : slice
            The slice object of indix values to be dumped.
        CCtoK : bool (Default = False)
            Convert all cross-correlation data to decorrelation. Expected column
            names as tuples with the last entry equal to 'CC' or 'CC_mean'
        shiftCols : list (Default = None)
            List of columns to begin as zero (The first value will be subtracted
            from all)
        nthRow : list (Default = None)
            List of columns to begin as zero (The first value will be subtracted
            from all)
        '''

        # Remove NaN
        if colNames:
            DF_out = DF.loc[:, colNames].fillna(0)
        else:
            DF_out = DF.fillna(0)

        # Take every nth row
        if nthRow:
            DF_out = DF_out.iloc[::nthRow]

        # shift to zero all columns in shiftCols
        if shiftCols:
            DF_out.loc[:, shiftCols] = DF_out.loc[:, shiftCols] - \
                                        DF_out.loc[:, shiftCols].iloc[0]

        # Convert all the 'CC' columns from correlation to decorrelation
        if CCtoK:
            col_tup = [col for col in DF_out.columns if isinstance(col, tuple)]
            CCcols = [col for col in col_tup if col[-1]=='CC' or col[-1]=='CC_mean']
            DF_out.loc[:, CCcols] = 1 - DF_out.loc[:, CCcols]

        # Remove all non-compatiable columns from the database columns
        renameCheck = {col: re.sub('[\'_\[\]%,]', '',str(col)).rstrip() for
                          col in DF_out.columns}
        DF_out.rename(columns = renameCheck, inplace=True)

        DF_out.to_csv(fileName, index_label = 'index')

    def Data_atGT(DF, targetCols, outputCols, points, pointsCol, shiftCols=None):
        '''Extracts first datapoint greater than a defined column value

        Parameters
        ----------
        DF : DataFrame
            DataFrame to be dumped
        targetCols : list
            list of all target columns in ``DF``
        outputCols : list
            list of output column names to use in output dataframe ``DF_out``. Must
            be of equal length to targetCols.
        points : list
            list of points at which the first values > should be extracted from
            each entry in ``targetCols``.
        pointsCol : str
            list of output column names to use in output dataframe ``DF_out``. Must
            be of equal length to targetCols.
        shiftCols : list (Default = None)
            List of columns to begin as zero (The first value will be subtracted
            from all)

        Returns
        -------
        df_trans : DataFrame
            Output dataframe containing the requested points
        '''

        # Output values at the transition
        # trans = [2.6, 3.3, 4, 6.1, 7.1, 8.8, 10, 11, 11.9, 12.9]

        # Make the dataframe for storage
        df_trans = pd.DataFrame(columns=outputCols)

        # Remove NaN
        DF_out = DF.copy().fillna(0)

        # shift to zero all columns in shiftCols
        if shiftCols:
            DF_out.loc[:, shiftCols] = DF_out.loc[:, shiftCols] - \
                                        DF_out.loc[:, shiftCols].iloc[0]

        setattr(df_trans, pointsCol, points)
        #df_trans.pointsCol = points

        for idx, col in enumerate(points):
            mask = DF_out[pointsCol]>col
            temp_data = DF_out.loc[mask].iloc[0]

            for outputCol, targetCol in zip(outputCols,targetCols):
                if isinstance(targetCol,tuple):
                    df_trans.loc[idx, outputCol] = temp_data.loc[[targetCol]].astype(float).values[0]
                else:
                    df_trans.loc[idx, outputCol] = temp_data.loc[targetCol]

        return df_trans


    def TS_Time(DF, secPerSamp, traceSlice, resampleStr = '1 us', csvDump = True,
                wdwPos = None, fileName = 'traces.csv'):
        '''Takes raw time series dataframe, allowing the slicing, resampling and
        re-indexing. Output times are in seconds.

        Parameters
        ----------
        DF : DataFrame
            DataFrame to be modified.
        secPerSamp : float
            The sampling period or seconds per sample required to generate a time
            index for the dataframe.
        traceSlice : slice
            The slice object to extract from the DF.
        resampleStr : str (default = '1 us')
            The resample string to reduce the size of each trace.
        csvDump : bool (default = True)
            Save data to ``traces.csv`` in pwd in the order, time [sec], trace1, trace2,....
        wdwPos : list
            List of [start, stop] window positions in no. of smaples
        fileName : str (Default = 'traces.csv')
            Name of output csv file.

        Returns
        -------
        DF : DataFrame
            Output dataframe.
        wdwTimes : list
            List of lists of the window positions in seconds
        '''

        # Add time index
        TdeltaIdx = pd.timedelta_range(start='0',
                                               freq = '%.9f ms' % (secPerSamp*1000),
                                               periods = DF.shape[0])
        if wdwPos is not None:
            wdwPos[:,1] = wdwPos[:,1]-1 # Correct indexing
            wdwTimes = TdeltaIdx[wdwPos].astype(float)*1E-9
            wdwTimes = [tuple(wdw*1000) for wdw in wdwTimes]

        DF = DF.loc[:, (slice(None), slice(None),
                DF.columns.get_level_values('Time')[traceSlice])]

        DF['time'] = TdeltaIdx

        DF.set_index('time', inplace=True)

        DF = DF.resample(resampleStr).sum()

        DF.reset_index(inplace=True)

        DF['time'] = DF['time'].values.astype(float)*1E-9

        if csvDump:
            DF.to_csv(fileName, header = False, index=False)

        return DF, wdwTimes


    def hdf_csv_dump(DB_fdl):
        ''' Dumps the processed databases to CC, PV, TShdrs to csv files. Note
        this function should be run in the run folder, not the database folder

        ---inputs---
        DB_fdl: relative or absolute location to the folder where all database
        files are located
        '''
        def hdf_to_csv(hdf_DB, tbl_name):
            ''' Save hdf DB to csv
            hdf_DB: HDF5  database rel of abs path and name
            tbl_name: Name of table in database
            '''
            df = pd.read_hdf(hdf_DB, tbl_name)
            df.to_csv(DB_fdl+tbl_name+'.csv')

        # Expected HDF5 table names
        CC_tbl_name = 'CC'
        PV_tbl_name = 'PV_df'
        PV_full_tbl_name = 'PV_df_full'
        TShdrs_tbl_name = 'TShdrs'
        TS_df_tbl_name = 'TS_df'

        # Expected HDF5 db names
        DB_tbl = DB_fdl+'DB_tbl_processed.h5'
        TS_cut = DB_fdl+'TS_cut.h5'

        # Load expected param file
        output = open(DB_fdl+'param.txt', 'rb')
        param = pickle.load(output)

        # Dump all expected DB tables to csv files
        hdf_to_csv(DB_tbl, PV_tbl_name)

        if param['matched']:
            hdf_to_csv(DB_tbl, PV_full_tbl_name)
        hdf_to_csv(DB_tbl, TShdrs_tbl_name)

        hdf_to_csv(DB_tbl, CC_tbl_name)


    def run_dataLoad(DB_fdl):
        ''' Loads a previous processing session into memory ready for analysis.

        - Inputs -
        DB_fdl: input folder holding the expected databases in the form
                'DB_fld/'

        - Outputs -
        PV_df:  Main database holding PV, and CC data
        TS_DB:  Database of TS data
        PV_df_full: Database including all PV data, empty if original PV and TS
                    data was coincident already.
        '''

        def from_pkl(fname):
            ''' Load pickel files
            fname: file name rel or abs path
            '''
            try:
                output = open(fname, 'rb')
                obj_dict = pickle.load(output)
                return obj_dict
            except EOFError:
                return False

        def from_hdf5(DB_tbl, tbl_name):
            '''Save expected df to hdf5 database
            '''
            df = pd.read_hdf(DB_tbl, tbl_name)

            return df

        # ------------------ Setup ------------------ #

        # Load the param file data
        param = from_pkl(DB_fdl+'param.txt')

        # The database names
        DB_tbl = pd.HDFStore(DB_fdl+'DB_tbl_processed.h5')
        TS_cut = pd.HDFStore(DB_fdl+'TS_cut.h5')

        # tabel names
        PV_tbl_name = 'PV_df'
        PV_full_tbl_name = 'PV_df_full'
        TS_df_tbl_name = 'TS_df'

        PV_df = from_hdf5(DB_tbl, PV_tbl_name)
        if 'TSmatched' in param and param['TSmatched']:
            PV_df_full = from_hdf5(DB_tbl, PV_full_tbl_name)

        # TS_df = from_hdf5(TS_cut, TS_df_tbl_name)
        TS_DB = from_hdf5(TS_cut, TS_df_tbl_name+'DB')
        # TShdrs = from_hdf5(DB_tbl, TShdrs_tbl_name)
        # CC = from_hdf5(DB_tbl, CC_tbl_name)

        # Close the DB's
        DB_tbl.close()
        TS_cut.close()

        return PV_df, TS_DB, PV_df_full, param

class data_import:
    """This class handels the import and basic processing of all
    user imput data. The core data streams are the time series information
    (TS) and the corresponding Perturbation Vectors (PV)

    Parameters
    ----------
    TSfpath :  str
        Defines the relative or absolute location of the TS data folder
    TSlocL :  list (default = None)
        List of the relative or absolute location of the TS data files
    PVloc : str
        Defines the relative or absolute location of the PV's
    Database : str
        Defines a common hdf5 database name ``Database.h5``
    import_dtype: str
        Defines several raw data types, "bin_par": for data in a binary single
        trace per file and header data in .par files, "Shell_format": all data in a
        single csv file (both PV and TS), 'NoTShdrer_format'
        "NoTShdrer_format"``.
    notes
    -----
    The output of this class should be a single ``Database.h5`` database in the
    run directory, containing all relevant data. All user defined parameters are
    assigned to the attribues of the database head.

    Examples
    --------
    >>> import h5py
    >>> # Reading the user defined parameters from the database attributes
    >>> with h5py.File('Database.h5', 'r') as h5file:
    >>>     print(dict(h5file.attrs.items()))

    """

    def __init__(self, TSfpath, PVloc, import_dtype, param = None):
        self.TSfpath = TSfpath
        self.TSlocL = None
        self.PVloc = PVloc
        self.Database = 'Database.h5'
        self.import_dtype = import_dtype
        self.param = param

    # -------------------- Reading files in folders --------------------
    def TSfiles(self):
        '''This function lists the files in the TSfpath folder location reading
        all of the information contained within. The structure of this
        data is checked and the appropriate sub-function initiated based on the
        user defined parameter 'import_dtype'

        Parameters
        ----------
        headerDB : DataFrame
            Database of header information
        TSdataList : list
            List of TS data file relative locations

        Returns
        -------
        headerDB : DataFrame or dict(DataFrames)
            DataFrame of all header file information or dict of DataFrames. The
            structure should be
            index | srcNo | recNo | Time | "other header info"

        '''

        # Assign list of TS data files to self.TSlocL
        self.TSlocL = self.read_finfd(self.TSfpath)

        if self.import_dtype == 'CSIRO':
            headerDB = self.TSfilesCSIRO()
            print('* CSIRO survey TS data loaded')
            return headerDB

        elif self.import_dtype == 'bin_par':
            headerDB = self.TSfilesPar()
            print('* TUDelft .Par and Binary TS data loaded')
            return headerDB

        elif self.import_dtype == 'Shell_format':
            headerDB = self.TSfilesPV_TS()
            print('* Shell format TS data loaded')
            return headerDB

        elif self.import_dtype == 'HDF5':
            from shutil import copyfile

            headerDB = None
            print('* HDF5 format expected ')
            try:
                copyfile(self.TSfpath, self.Database)
            except FileNotFoundError:
                return None

            return headerDB

    def read_finfd(self, file_loc):
        '''Basic function to read files in folder and sort by numeric end digits
        Parameters
        ----------
        file_loc: str
            Path to files.
        '''

        #print('Location of the time series data file/s is:', file_loc)
        files = glob.glob(file_loc + '*')
        files.sort(key=lambda var: [int(x) if
                        x.isdigit() else x
                        for x in
                        re.findall(r'[^0-9]|[0-9]+', var)])
        return files

    def TSfilesPar(self):
        '''This function is intended to perform the TSfile function operations
        for a folder containing .par headerfiles and associated binary files.

        Parameters
        ----------
        List of all header .par and Binary files

        Returns
        -------
        hdr_df  Database of all header file information, multiple
                index file name. Must contain mandatory columns
                    ``['srcNo', 'recNo', 'Time', 'Survey']``.
        TSdataL List of data files including relative location.

        TODO:
        *  Add check for various header file types .txt ..etc
        '''

        # Split .par and data files
        TSpar = [elem for elem in self.TSlocL if ".par" in elem]
        self.TSlocL = [elem for elem in self.TSlocL if ".par" not in elem]

        regex = re.compile(r'\d+')


        # Raise error if .par no != data file no
        fileNo_par = [regex.findall(file)[-1] for file in TSpar]
        fileNo_data = [regex.findall(file)[-1] for file in self.TSlocL]
        for par, data in zip(fileNo_par, fileNo_data):
            if par != data:
                print(par, data)
                raise Warning(par, '.par not equal to data', data)

        TSdata_modTime = [os.path.getmtime(elem) for elem in self.TSlocL]

        # Load header files into database
        fileNokeys = [x.rsplit('/', 1)[-1] for x in TSpar]
        header = ['recNo', 'numSamp', 'sentv', 'sampFeq',
                  'n/a', 'offset', 'n/a', 'count[sec]']

        df_list = [pd.read_csv(file, sep='\t', names=header) for file in TSpar]
        hdr_df = pd.concat(df_list, keys=fileNokeys)
        maxNorec = np.arange(hdr_df.recNo.min(),hdr_df.recNo.max()+1)
        hdr_df['Survey'] = [int(val) for val in fileNo_data for _ in
                            maxNorec]

        hdr_df.recNo = hdr_df.recNo.astype('int')
        hdr_df['srcNo'] = int(1)
        hdr_df['Time'] = pd.to_datetime(hdr_df['count[sec]'], unit='s',
                          origin = self.param['TSstart_date'], dayfirst=True)
        # Re-order list based on expected order of columns
        header.insert(0, 'srcNo')
        header.insert(2, 'Time')
        header.insert(3, 'Survey')

        hdr_df = hdr_df[header]

#        # duplicate time create based on the noRec
#        noRec = len(hdr_df.recNo.unique()) # No of receivers
#        temp = [x for item in TSdata_modTime
#                          for x in repeat(item,noRec)]
#        hdr_df['fileMod'] = temp

        return hdr_df

    def TSfilesPV_TS(self):
        '''Loads in impuse response header data from a single folder.
        The expected format is the shell data structure.

        Parameters
        ----------
        self.TSlocL : list
            List of all files within given folder

        Returns
        -------
        df_hdr : DataFrame
            Database of all header file information for each file. Must contain
            mandatory columns ``['srcNo', 'recNo', 'Time', 'Survey']``.
        '''

        def getrows(filename, rowNos):
            '''Outputs a list of the contents of each rwoNos
            Parameters
            ----------
            filename : str
                relative or absolute path to file
            rowNos : list
                list of int of increasing order defining the row numbers to read.
            '''
            with open(filename, 'r', encoding='ISO-8859-1') as f:
              datareader = csv.reader(f)
              count = 0
              rows = []
              for row in datareader:
                  if count in rowNos:
                      rows.append(row)
                  count +=1
                  if count > rowNos[-1]:
                      return rows

        # Expected formate
        rowNos = [4,6,7,8,9,10]
        columns = ['srcNo','recNo','Time', 'Survey', 'Averages', 'Exitation Freq',
                   'Vertical gain', 'Delay','Vertical Offset','Sample interval']

        df_hdr = pd.DataFrame(columns=columns)

        for idx, file in enumerate(self.TSlocL):

            # Read in the file header rows
            file_hdr = getrows(file, rowNos)

            # Formate the rows
            items = [1, # srcNo
                     1, # recNo
                     file_hdr[0][0], # Time
                     int(idx), # Survey
                     int(file_hdr[1][1]), # Averages
                     float(file_hdr[1][3]), # Exitation Freq
                     float(file_hdr[2][1]), # Vertical gain
                     float(file_hdr[3][1]), # Delay
                     float(file_hdr[4][1]), # Vertical Offset
                     float(file_hdr[5][1])] # Sample interval

            # Store within dataframe
            df_hdr.loc[idx] = items

        return df_hdr

    def TSfilesCSIRO(self):
        '''Loads in impulse response header data from multiple subfolders throughout
        a survey. Each subfolder should be of the format "sometext"YYYYMMDDHHMMSS.
        Within each subfolder are files for each source receiver pair.

        Parameters
        ----------
        self.TSlocL : list
            List of all files within given

        Returns
        -------
        header_dict : dict
            Database of all header file information, multiple
                index file name
        TdeltaIdx : timedelta64[ns]
            Time delta index of length equal to trace length of TS data
        self.TSlocL : list of lists
            Updated list of lists of each survey folders contents.
        '''
        # list for all surveys
        TSlocL_temp = []

        # Define the columns of the dataframe
        columns = ['srcNo','recNo','Time','TracePoints','TSamp','TimeUnits',
                   'AmpToVolts','TraceMaxVolts','PTime','STime']

        # Create the dataframe for all headers to be stored
        header_dict = {}

        for survey in self.TSlocL:
            survey_files = [ file for file in self.read_finfd(survey + '/') if '.atf' in file]
            TSlocL_temp.append(survey_files)

            df1 = pd.DataFrame(columns=columns)
            count = 0
            for file in survey_files:
                with open(file, newline='') as f:
                  reader = csv.reader(f)
                  next(reader)
                  header = next(reader)
                  f.close()

                header = re.split(r"[\;]+", header[0]) # split the header up
                header_split = [re.split(r'[\=]', head) for head in header][:-1] # split the header up
                header_split = [ [head[0].split()[0], head[1]] for head in header_split] # Remove white space

                # Extract the values
                items = [ item[1] for item in header_split]

                # Combine date and time:
                items[1] = items[0]+' '+items[1]
                items = items[1:]

                # Convert datatypes
                for idx,item in enumerate(items):
                    try:
                        items[idx] = np.float(item)
                    except ValueError:
                        items[idx] = pd.to_datetime(item, dayfirst=True)  # parse(item)

                # Add srcNo and recNo to header info.
                srcNo = re.split(r'[_.]',file)[-3]
                recNo = re.split(r'[_.]',file)[-2]
                items.insert(0, int(recNo))
                items.insert(0, int(srcNo))

                df1.loc[count] = items
                count += 1
            header_dict[re.split(r'[/]',survey)[-1]] = df1

        # Create time delta array
        if df1['TimeUnits'][0] == 1.00000e-006:
            freq = 'us'
        else:
            freq = 'ms'

        TdeltaIdx = pd.timedelta_range(start = '0',
                                       freq = freq,
                                       periods = df1['TracePoints'][0])

        # Redefine the list of surveys to a list of lists of survey files
        self.TSlocL = TSlocL_temp
        return header_dict

    # -------------------- Loading TS data into memory --------------------

    def TSload(self, TShdrs):
        ''' Load the list of TS files ``TSflist`` found and output a
        matrix storing TS data columnweise.

        Parameters
        ----------
        TShdrs: DataFrame
            A database of each file name containing the corresponding header. This
            is output from the function ``TSfiles``.
        TSflist: list
            A list of the files within the ``TSfpath`` folder location. This
            is output from the function ``TSfiles``.

        Returns
        -------
        TSdataMtx : numpy array or ``None``
            A columnweise stored matrix of traces for a single receiver, or ``None``
            if a multiple receiver survey is detected. In this case all TS data
            will be saved into a hdf5 database 'TSdata.h5'.

        Note
        ----
        All imported ts data will be have any linear trend removed before storage
        to the raw database.
        '''

        if self.import_dtype == 'CSIRO':
            print('* TS data written to database.h5')
            TSdataMtx = self.TSsurveys(TShdrs)
            return TSdataMtx
        elif self.import_dtype == 'bin_par':
            TSdataMtx = self.TSloadBin(TShdrs)
            print('* Binary data TS files loaded into matrix')
            return TSdataMtx
        elif self.import_dtype == 'Shell_format':
            TSdataMtx = self.TSloadPV_TS(TShdrs)
            print('* Shell format TS csv files loaded')
            return TSdataMtx
        elif self.import_dtype == 'HDF5':
            print('* No TS data found')
            return None

    def TSsurveys(self, hdr_df):
        '''Load TS data from a folder with sub-folders for each survey. The sub-
        folders should be named 'sometext'(unique_number)'
        (e.g. 'survey20180312093545'). The individual csv files must be
        named 'sometext'(unique_number)_(sourceNo)_(receverNo).(csv type formate)
        (e.g. 'survey20180312093545_0001_01.csv'). A hdf5 database will be saved
        with the following group structure.

            Database.h5
            │
            TSdata
                │
                └───survey20180312093545
                │       Table
                └───survey20180312093555
                │       Table
                :               :

        Each table includes header data from the csv files imported from the second
        line of the csv files as Date=12-03-2018; Time=09:35:45.833000;
        TracePoints=4096; TSamp=0.10000; TimeUnits=1.00000e-006;
        AmpToVolts=1.0000; TraceMaxVolts=5.0000; PTime=0.00000; STime=0.00000;

        Parameters
        ----------
        hdf_df : dict of DataFrames
            A database of header information for each Time-series recorded. As a
            minimum must contain the first three columns of 'srcNo', 'recNo' and
            'time'. The remaining header info can be any order.
        self.TSlocL : list of lists
            List of lists of the files within each survey

        Returns
        -------
        _ : None
            None indicating that data is stored in a HDF5 file.

        notes
        -----
        No longer is the TdeltaIdx added to each dataframe, thus I should figure
        out how best to store this information into the h5 file.

        Examples
        --------
        >>> import pandas as pd
        >>> import h5py
        >>> with h5py.File('Database.h5', 'a') as h5file:
        >>>     TSsurvey =  pd.read_hdf(h5file, 'survey20180312093545') # basic load
        >>>     # Load specific source-receiver pair for window between 11 and 500
        >>>     TSsurvey =  pd.read_hdf(h5file, 'survey20180312093545',
                    columns=[(1,1), (1,2)], # Load specific columns
                    where = ['index>11', 'index<20']) # Load index 11 -> 20

        '''

        # Folder setup structure
        file = self.Database
        group = 'TSdata/'

        # Read header keys from the first file
        header_attributes =  list(hdr_df[list(hdr_df.keys())[0]].keys())


        # Open link to hdf5 database
        with pd.HDFStore(file,'w') as h5file: # open the hdf data store link

            for survey in self.TSlocL:
                TS_data_list = []
                survey_no = re.split(r'[/._]',survey[0])[-4]
                # For each file in survey
                for file in survey:
                    # read source receiver numbers from file name
                    srcNo = int(re.split(r'[_.]',file)[-3])
                    recNo = int(re.split(r'[_.]',file)[-2])

                    #----- Read from dictionary the surveys headers -----#
                    head_df = hdr_df[survey_no]
                    head_df = head_df.loc[(head_df['srcNo']==srcNo) & (head_df['recNo']==recNo)]

                    #----- Load in the trace -----#
                    temp1 = [head_df[col].astype(int).values for col in header_attributes[0:2]]
                    temp2 = [head_df[col].astype(int).values for col in [header_attributes[2]]]
                    temp3 = [head_df[col].astype(float).values for col in header_attributes[3:]]
                    header_vals = temp1 + temp2 + temp3
                    TSsurvey = pd.read_csv(file,skiprows=3, names=['temp'])
                    # Place header into the multiindex
                    TSsurvey.columns = pd.MultiIndex.from_product(header_vals,
                                                           names = header_attributes)
                    TS_data_list.append(TSsurvey)

                #----- Join all into the full survey -----#
                TSsurvey = pd.concat(TS_data_list, axis=1, join_axes=[TSsurvey.index])

                # Set index to timedelta and drop the default
                #TSsurvey.set_index(TdeltaIdx, inplace=True, drop=True)
                #TSsurvey['Time Stamp'] = TdeltaIdx

                # store into the database under name of survey and header data
                h5file.put(group + survey_no, TSsurvey, format='table')

        return None

    def TSloadBin(self, TShdrs):
        '''Load in the binary file formate as expected from the
        'import_dtype'= 'bin_par'.
        '''
        # Load TS data into matrix
        TS_len = int(len(TShdrs.recNo.unique())*TShdrs.numSamp.max())
        hdr_col = TShdrs.columns[:7].tolist()

        #recNo_list = TShdrs.recNo.unique().tolist()


        df_list = []
        for file in self.TSlocL:
            with open(file, 'r') as fid:
                trace = np.fromfile(fid, np.int16)[-TS_len:] #* v_sens

            # For each header entry
            for hdr in TShdrs.loc[(file.split('/')[1]+'.par',
                                   slice(None)), :].index.values:

                # Create dataframe for storage with header info:
                values = [ [TShdrs.loc[hdr, col]] for col in hdr_col]
                cols = pd.MultiIndex.from_product(values, names = hdr_col)
                recSlice = slice(int(hdr[1] * TShdrs.loc[hdr, 'numSamp']),
                                 int((hdr[1]+1) * TShdrs.loc[hdr, 'numSamp']))
                df = pd.DataFrame(trace[recSlice], columns = cols) * TShdrs.loc[hdr,'sentv']/3200 # Vsensitivty correction
                df_list.append(df)

        TSdata = pd.concat(df_list, axis=1, join_axes=[df.index])
        with pd.HDFStore(self.Database, 'w') as h5file:
            TSdata.to_hdf(h5file, 'TSdata')

        return None

    def TSloadPV_TS(self, TShdrs):
        '''Read in time-series data and save to database
        '''

        TS_data_list = []
        for idx, survey in enumerate(self.TSlocL):
            hdr_values = [[item] for item in TShdrs.loc[idx].values]

            TSsurvey = pd.read_csv(survey, skiprows = 11, names = ['temp'])
            TSsurvey.columns = pd.MultiIndex. \
                from_product(hdr_values, names = list(TShdrs.columns.values))
            TS_data_list.append(TSsurvey)

        TSsurvey = pd.concat(TS_data_list, axis=1, join_axes=[TSsurvey.index])
        TSsurvey = TSsurvey * TShdrs['Vertical gain'].values # Correct for gain
        # TSsurvey = TSsurvey - TShdrs['Delay'].values TODO: confrim the correction for the delay time !!!!
        TSsurvey = TSsurvey.apply(signal.detrend, axis=0) # Remove linear trend from traces

        with pd.HDFStore(self.Database,'w') as h5file:
            h5file.put('TSdata', TSsurvey, format='fixed')

        return None

    # -------------------- Loading PV data into memory --------------------
    def PVload(self):
        '''Load Perturbation Vectors into a single database and store in
        'Database.h5' within group 'PVdata'.

        Database.h5
            │
            PVdata
                Table


        Parameters
        ----------
        self.PVloc : str
            Path to PV data file/files
        self.import_dtype : str
            Indication of the data type, either ``['.xls', 'bin_par', 'CSIRO']``
            or the 'Shell_format'.
        '''

        print('* Location of the perturbation data file/s is:', self.PVloc)


        if '.xls' in self.PVloc:
            df = pd.read_excel(self.PVloc)
        elif self.import_dtype == 'bin_par':
            df = pd.read_csv(self.PVloc, sep=';', header=[1, 2])
            df.columns = df.columns.get_level_values(0)
        elif self.import_dtype == 'CSIRO':
            if not self.param['PV_file_hdr_rows'][0]:
                self.param['PV_file_hdr_rows'] = [14, 15]

            if not self.param['PV_file_skip_rows'][0]:
                self.param['PV_file_skip_rows'] = [16, 17]

            df = pd.read_csv(self.PVloc, sep = '\t', header= self.param['PV_file_hdr_rows'],
                             skiprows=self.param['PV_file_skip_rows'])
            df.columns = ['%s%s' % (a, '%s' % b if b else '')
                          for a, b in df.columns]
        elif self.import_dtype == 'Shell_format':
            df = self.PVloadPV_TS()
        elif self.import_dtype == 'HDF5':
            return None

        group = 'PVdata'
        with pd.HDFStore(self.Database, 'r+') as h5file:
            h5file.put(group, df, format='table')

        return df

    def PVloadPV_TS(self):
        '''Loads in the PV data from the shell format csv files

        Parameters
        ----------
        self.TSfiles() : list of files of containing both PV and TS data
        '''

        fileNokeys = [x.rsplit('/', 1)[-1] for x in self.TSlocL]

        df_list = [pd.read_csv(file, sep=',', header=[1, 2], engine='python',
                               encoding='ISO-8859-1') for file in self.TSlocL]
        df_list = [x.query('index<1') for x in df_list]

        df = pd.concat(df_list, keys=fileNokeys)

        df.columns = ['%s%s' % (a, '|%s' % b if b else '')
                          for a, b in df.columns]

        df.columns = [a.strip() for a in df.columns]

        return df

    def run_data_import(self):
        '''Runs the class functions for importing the raw data, and stores this all
        within a database 'Database.h5'
        '''

        # Datebase setup
        db_folder = self.TSfpath.split('/')[-2]+'_DB'  # 'database'

        # add the relative path to the database
        self.Database = db_folder+'/' + self.Database

        # Make path to database folder if it does not exist
        if not os.path.exists(db_folder):
            os.makedirs(db_folder)
            print('* No existing database folder found, new folder created')
        if os.path.exists(self.Database):
            print('* Existing %s database found' %self.Database,
                  '\n This database will now be overwritten')

        # Load raw data and save to hdf5 database
        print('* load from raw files')
        TShdrs = self.TSfiles()
        self.TSload(TShdrs)
        self.PVload()

        # Save paramter file to the database attribues
        # TODO: check why param has an empty key and val
        with h5py.File(self.Database, 'a') as h5file:
            for key,item in zip(self.param.keys(), self.param.values()):
                print('Key:', key,'| item:', item)
                h5file.attrs[key] = item

        return self.Database
