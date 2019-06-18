#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:28:03 2017

@author: rwilson
"""

import shelve
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

class utilities:
    '''A logical collection of functions for interacting with 
    '''
    
    @staticmethod
    def DB_pd_data_load(Database, group):
        '''Loads in a pandas dataframe stored in group from the Database.
        
        Parameters
        ----------
        Database : str
            Relative location of database
        group : str
            The expected group name
            
        Returns
        -------
        group_df : DataFrame
            The PV data stored in the group ``PVdata`` as a pandas dataframe
        '''
    
        with pd.HDFStore.File(Database, 'r') as h5file:
    
            # Check that the expected group name is found in the database
            group_names = [key for key in h5file.keys()]
            expected_group_name = group
            if expected_group_name not in group_names:
                raise KeyError('The %s group was not found within the %s database.' \
                               %(expected_group_name, Database))
            # Load the database
            group_df = pd.read_hdf(h5file, expected_group_name)
        return group_df

    @staticmethod
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

    @staticmethod
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


class dataStore:
    '''This class is intended to handel the storage of all data aquired and or
    generated during the processing.
    '''
    def __init__(self, param={}, PV_df=[], PV_df_full=[], TS_df=[],
                 TS_DB=[], TShdrs=[], CC=[]):
        self.param = param
        self.PV_df = PV_df
        self.PV_df_full =PV_df_full
        self.TS_df = TS_df
        self.TS_DB = TS_DB
        self.TShdrs = TShdrs
        self.CC = CC
        self.CC_tbl_name = 'CC'
        self.PV_tbl_name = 'PV_df'
        self.PV_full_tbl_name = 'PV_df_full'
        self.TShdrs_tbl_name = 'TShdrs'
        self.TS_df_tbl_name = 'TS_df'
        self.DB_fdl = param['TSloc'].split('/')[-2]+'_DB/'
        self.DB_tbl = self.DB_fdl+'DB_tbl_processed.h5'
        self.TS_cut = self.DB_fdl+'TS_cut.h5'

    def pre_process(self):
        ''' check inputs are in the correct or expected format, process if
        required.
        '''

    def post_process(self):
        ''' check loaded DB is of the expected format, process if
        required.
        '''
        self.TS_df = self.TS_df.as_matrix()
#        self.PV_df.set_index('Date', drop=True, inplace=True)
#        self.TS_df.drop('index', axis=1, inplace=True)
#        self.TS_df = self.TS_df.as_matrix()
#        self.TShdrs.set_index(['level_0', 'level_1'], inplace=True)

    def checkSetup(self):
        ''' This functions checks the setup file to determine if any param have
        changed. If yes, the processing will be re-run, otherweise the saved
        datebases will be loaded. return True if change is detected
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

            if len(added) == 0 & len(removed) == 0 & len (modified) == 0:
                return True
            else:
                return False

        check = self.from_pkl('param.txt')
        return not dict_compare(self.param, check)

    def to_hdf5(self, DB_tbl,df,tbl_name,form):
        '''Save expected df to hdf5 database
        '''
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
            df.to_hdf(DB_tbl, tbl_name, format=form)
        else:
            df.to_hdf(DB_tbl, tbl_name, format=form)

    def from_hdf5(self, DB_tbl,tbl_name):
        '''Save expected df to hdf5 database
        '''
        df = pd.read_hdf(DB_tbl, tbl_name)

        return df

    def from_sql_DB(self, tbl_name, DB_egn):
        ''' Load data from the sql_DB uponsd
        '''
        df = pd.read_sql_table(tbl_name, DB_egn)
        return df


    def from_pkl(self, fname):
        ''' Load pickel files
        fname: file name rel or abs path
        '''
        try:
            output = open(fname, 'rb')
            obj_dict = pickle.load(output)
            return obj_dict
        except  EOFError:
            return False


    def to_pkl(self, data, fname):
        ''' Save pickel files
        data: data to pickel
        fname: file name rel or abs path
        '''
        output = open(fname, 'w+b')

        pickle.dump(data, output)
        output.close()

    def hdf_to_csv(self, hdf_DB, tbl_name):
        ''' Save hdf DB to csv
        hdf_DB: HDF5  database rel of abs path and name
        tbl_name: Name of table in database
        '''

        df = self.from_hdf5(DB_tbl,tbl_name)
        df.to_csv(tbl_name)


    def run_dataStore(self):
        ''' run the expected data storage workflow
        '''
        # The database names
        DB_tbl = pd.HDFStore(self.DB_tbl)
        TS_cut = pd.HDFStore(self.TS_cut)

        # Save to hdf5 databases
        self.to_hdf5(DB_tbl,self.PV_df,'PV_df','t')

        if 'TSmatched' in self.param and self.param['TSmatched']:
            self.to_hdf5(DB_tbl, self.PV_df_full, self.PV_full_tbl_name,'t')

        self.to_hdf5(TS_cut, self.TS_df, self.TS_df_tbl_name,'f')
        self.to_hdf5(TS_cut, self.TS_DB, self.TS_df_tbl_name+'DB','f')
        self.to_hdf5(DB_tbl, self.TShdrs, self.TShdrs_tbl_name,'f')
        self.to_hdf5(DB_tbl, self.CC, self.CC_tbl_name,'t')

        DB_tbl.close()
        TS_cut.close()

        # Pickle the param file
        self.to_pkl(self.param, self.DB_fdl+'param.txt')


    def run_dataLoad(self):
        ''' run the expected data loading workflow
        '''

        # Pickle data
        self.param = self.from_pkl(self.DB_fdl+'param.txt')

        # The database names
        DB_tbl = pd.HDFStore(self.DB_tbl)
        TS_cut = pd.HDFStore(self.TS_cut)


        self.PV_df = self.from_hdf5(DB_tbl,self.PV_tbl_name)
        if 'TSmatched' in self.param and self.param['TSmatched']:
            self.PV_df_full = self.from_hdf5(DB_tbl, self.PV_full_tbl_name)


        self.TS_df = self.from_hdf5(TS_cut, self.TS_df_tbl_name)
        self.TS_DB = self.from_hdf5(TS_cut, self.TS_df_tbl_name+'DB')
        self.TShdrs = self.from_hdf5(DB_tbl, self.TShdrs_tbl_name)
        self.CC = self.from_hdf5(DB_tbl, self.CC_tbl_name)

        self.post_process()

        # Close the DB's
        DB_tbl.close()
        TS_cut.close()

        return self.param, self.PV_df, self.PV_df_full, self.TS_df, self.TS_DB, self.TShdrs, self.CC
