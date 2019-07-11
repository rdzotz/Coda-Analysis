#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:42:02 2018

@author: rwilson
"""

import pandas as pd
import numpy as np
import scipy.linalg as linalg
import random
import os
import h5py
import matplotlib.pyplot as plt
import itertools
from numba import njit
from numba import prange
import os
import shutil

class utilities():
    ''' Some helper functions
    '''

    def src_rec_pairs(channels, exclude=None, reciprocity=False, randSample=None):
        '''Generate a list of source receiver pairs for all excluding a certain
        channels.

        Parameters
        ----------
        channels : list
            list of channels from which src rec pairs should be generated
        exclude : list (Default = None)
            list of channels which should be excluded from the list of channels
        reciprocity : bool (Default = False)
            Include reciprocal pairs.
        randSample : int (Default = None)
            Extract a random subset from the list of length ``randSample``

        Returns
        -------
        src_rec : list
            list of unique source receiver pairs
        '''

        if reciprocity:
            src_rec = [(i, j) for i in channels for j in channels if i!=j and
                                            i!=np.all(exclude) and
                                            j!=np.all(exclude)]
        elif not reciprocity:
            src_rec = [(i, j) for i in channels for j in channels if i!=j and
                                            i!=np.all(exclude) and
                                            j!=np.all(exclude) and
                                            i<j]

        if randSample:
            return random.sample(src_rec, randSample)
        else:
            return src_rec

    def read_channelPos(file, dimensions):
        '''Read in csv containing each channel position. Currently expecting that
        the channel position csv is of a specific type and needs shifting to bottom
        zeroed coord. system.

        Parameters
        ----------
        file : str
            Location of csv containing the channel locations
        dimensions : dict
            The ``height`` of the mesh.

        Returns
        -------
        dfChan : DataFrame
            Database of each channel location
        '''

        dfChan = pd.read_csv(file,
                         delim_whitespace=True, skiprows=2, usecols=[0,1,2,3,4])
        dfChan.index = dfChan.index.droplevel()
        dfChan.drop(inplace=True, columns=dfChan.columns[-2:].tolist())
        dfChan.columns = ['x','y','z']

        # Shift coords to mesh bottom zeroed
        dfChan.z = dfChan.z + np.abs(dfChan.z.min()) + dimensions['height']/2 - np.abs(dfChan.z.min())


        print('Channel Positions:\n', [(dfChan.iloc[i].x, dfChan.iloc[i].y, dfChan.iloc[i].z)
                for i in range(dfChan.shape[0])])
        print('Channel index:\n',[str(chan)
                for _,chan in enumerate(dfChan.index.values)])

        return dfChan

    def HDF5_data_save(HDF5File, group, name, data, attrb={'attr': 0}, ReRw='w'):
        '''Saves data into a hdf5 database, if data name already exists, then an
        attempt to overwrite the data will be made

        Parameters
        ----------
        HDF5File : str
            Relative location of database
        group : str
            The expected group name
        name : str
            The name of the data to be saved within group
        attrb : dict
            attribute dictionary to store along with the database.
        ReRw : str (Default = 'w')
            The read/write format
        '''

        toscreen = '----- Attributes added to database %s %s, table %s ----- \n' \
                    %(HDF5File,group, name)

        with h5py.File(HDF5File, ReRw) as f:
            try:
                dset = f.create_dataset(os.path.join(group, name), data=data, dtype='f')

                print(toscreen)
                for key,item in zip(attrb.keys(), attrb.values()):
                    print('Key:', key,'| item:', item)
                    dset.attrs[key] = item
            except RuntimeError:
                del f[os.path.join(group, name)]
                dset = f.create_dataset(os.path.join(group, name), data=data, dtype='f')

                print(toscreen)
                for key,item in zip(attrb.keys(), attrb.values()):
                    print('Key:', key,'| item:', item)
                    dset.attrs[key] = item

    def HDF5_data_read(HDF5File, group, name, ReRw='r'):
        '''Saves data into a hdf5 database

        Parameters
        ----------
        HDF5File : str
            Relative location of database
        group : str
            The expected group name
        attrb : tuple/list
            attribute to store along with the database.
        ReRw : str (Default = 'w')
            The read/write format

        Returns
        -------
        dset : ()
            Data contained within group/name

        '''

        with h5py.File(HDF5File, ReRw) as f:
            dset = f[os.path.join(group,name)].value

        return dset

    def HDF5_attri_read(HDF5File, group, name, ReRw='r'):
        '''Read keys and attributes from hdf5 database.

        Parameters
        ----------
        HDF5File : str
            Relative location of database
        group : str
            The expected group name
        attrb : tuple/list
            attribute to store along with the database.
        ReRw : str (Default = 'w')
            The read/write format

        Returns
        -------
        dic : dict
            A dictionary of all the attributes stored within the group/name.
        '''

        with h5py.File(HDF5File, ReRw) as f:
            return {item[0]:item[1] for item in f[os.path.join(group,name)].attrs.items()}

    def WindowTcent(TS, wdws):
        '''Determine the centre of each correlation window in time from the input
        time-series database.

        Parameters
        ----------
        TS : float
            Sampling period
        wdws : list(str)
            Containing the windows range in sample points separated by -
        '''

        wdws_cent = [int(np.mean([int(wdw.split('-')[0]),
                     int(wdw.split('-')[1]) ])) for wdw in wdws]
        wdws_cent = np.array(wdws_cent) * TS

        return wdws_cent

    def DiffRegress(Tseries, dfChan, Emaxt0, plotOut=False):
        '''Perform linear regression to fit the 1D diffusion equation to an input
        time series. The output of this function is an estimation of the
        diffusivity and dissipation. (P. Anguonda et. al. 2001)

        Parameters
        ----------
        Tseries : array-like
            The input time series
        dfChan : DataFrame
            Containing the channel positsion columns x, y, z
        Emaxt0 : int
            The index corresponding to the arrival time (onset of) maximum energy

        Returns
        -------
        popt[1] : float
            The diffusitivty determined from the least squared fit.
            Units depends upon input t and z units check units
        popt[2] : float
            The Dissipation
        '''

        from scipy import optimize

        # Determine absolute distance between source and receiver
        recPos = dfChan.loc[Tseries['recNo']]
        srcPos = dfChan.loc[Tseries['srcNo']]

        absDist = np.sqrt(abs(recPos.x - srcPos.x)**2 +
                          abs(recPos.y - srcPos.y)**2 +
                          abs(recPos.z - srcPos.z)**2)

        # Define the 1D diffusion equation, logE(z,t)
        def diffusivity(t, z, D, sigma):
            return np.log(1/(2*np.sqrt(np.pi*D))) \
                   - 0.5*np.log(t) - z**2/(4*D*t) - sigma*t

        # The energy density
        y_data = np.log(Tseries['Tseries']**2)[Emaxt0:]

        # The time axis zeroed to the onset of Emaxt0
        x_data = (np.arange(0, Tseries['TracePoints']) *
                  Tseries['TSamp'])[Emaxt0-1:]
        x_data = (x_data-x_data[0])[1:]

        popt, pcov = optimize.curve_fit(diffusivity,
                                        x_data,
                                        y_data,
                                        p0=[absDist, 1, 1],
                                        bounds=([absDist*0.9, 0.1, 0.1],
                                                [absDist*1.1, np.inf, np.inf]))
        if plotOut:
            # Plot the resulting fit
            plt.figure(figsize=(6, 4))
            plt.scatter(x_data, y_data, label='Data')
            plt.plot(x_data, diffusivity(x_data, popt[0], popt[1], popt[2]),
                     label='Fitted function', color='red')

            plt.legend(loc='best')
            plt.show()

        return popt[1], popt[2]

    def src_recNo(CCdata):
        '''Extract the source receiver paris within CCdata, excluding common pairs.

        Parameters
        ----------
        CCdata : dataframe
            CCdata dataframe

        Returns
        -------
        src_recNo : list
            List of the source receiver numbers
        '''

        src_rec = list(sorted(set(
                [(srcNo, recNo) for srcNo, recNo in
                 zip(CCdata.index.get_level_values('srcNo'),
                     CCdata.index.get_level_values('recNo')) if
                 srcNo != recNo]
                )))

        return src_rec

    def traceAttributes(SurveyDB, Col):
        '''Extract a single trace and its attributes from single survey dataframe,
        into a dictionary.

        Parameters
        ----------
        TStrace : DataFrame
            Containing all traces for a single survey.
        Col : int
            The column to extract from the database.

        Returns
        -------
        traceDict: dict
            Containing the trace along with all header information.
        '''

        traceDict = {key: SurveyDB.columns.get_level_values(key)[Col] for key in
                     SurveyDB.columns.names}

        traceDict['Tseries'] = SurveyDB.iloc[:, Col].values
        return traceDict

    def d_obs_time(CCdata, src_rec, lag, window, parameter='CC', staTime=None, stopTime=None):
        '''Construct the d_obs dataframe over time from the input CCdata, for a
        select list of source-receiver pairs.

        Parameters
        ----------
        CCdata : dataframe
            CCdata dataframe
        src_rec : list(tuples)
            A list of tuples for each source receiver pair.
        lag : list(tuples)
            The lag value from which the extraction is made.
        window : str/list(str)
            string of windows or list of str of windows.
        parameter : str (Default='CC')
            Parameter from which to extract from the dataframe
        staTime : str
            The start time from which ``d_obs`` is extracted
        stopTime : str
            The stop time before which ``d_obs`` is extracted.

        Returns
        -------
        d_obs_time : dataframe
            dataframe containing the in each row the d_obs vector for all requested
            source-receiver pairs, increasing with time.
        '''

        if staTime and stopTime:
            mask = (pd.to_datetime(CCdata.index.get_level_values('Time')) >
                    pd.to_datetime(staTime)) & \
                   (pd.to_datetime(CCdata.index.get_level_values('Time')) <
                    pd.to_datetime(stopTime))
            CCdata = CCdata.copy().loc[mask]
        elif staTime:
            mask = (pd.to_datetime(CCdata.index.get_level_values('Time')) >
                    pd.to_datetime(staTime))
            CCdata = CCdata.copy().loc[mask]
        elif stopTime:
            mask = (pd.to_datetime(CCdata.index.get_level_values('Time')) <
                   pd.to_datetime(stopTime))
            CCdata = CCdata.copy().loc[mask]

        # Time index for each survey based on first src_rec pair.
        time_index = pd.to_datetime(CCdata.loc[([src_rec[0][0]], [src_rec[0][1]]),
                                               (lag, window[0], parameter)].
                                    unstack(level=[0, 1]).index)

        if len(window)>1:
            temp = []

            for wdw in window:
                df = pd.concat([CCdata.loc[([sr[0]], [sr[1]]),
                                               (lag, wdw, parameter)].
                                unstack(level=[0, 1]).
                                reset_index(drop=True) for
                                sr in src_rec], axis=1)
                temp.append(df)
            d_obs_time = pd.concat(temp, axis=1)

        else:
            d_obs_time =  pd.concat([CCdata.loc[([sr[0]], [sr[1]]),
                                                (lag, window, parameter)].
                                    unstack(level=[0, 1]).
                                    reset_index(drop=True) for
                                    sr in src_rec], axis=1)
        d_obs_time.index = time_index

        return d_obs_time.dropna().astype(float)

    def measNorange(CCdata, staTime, endTime):
        '''Determines the measurement survey number between given time interval.
        This function is intended to allow the user to quickly determine the measurement
        number range of interest, thereby allowing the reporcessing of the raw data
        over this region only. This requires that the user passes a CCdata which
        represents the entire raw dataset.

        Parameters
        ----------
        CCdata : dataframe
            CCdata dataframe
        staTime : str
            The start time of the interval/.
        endTime : str
            The start time of the interval/.

        Returns
        -------
        None : tuple
            Measurement survey numbers within the range given.
        '''

        mask = (pd.to_datetime(CCdata.index.get_level_values('Time').values) > pd.to_datetime(staTime)) & \
        (pd.to_datetime(CCdata.index.get_level_values('Time').values) < pd.to_datetime(endTime))

        measNo = [i for i, x in enumerate(mask) if x]


        return (measNo[0], measNo[-1])

    def surveyNorange(TSsurveys, staTime, endTime):
        '''Determines the survey number between given time interval.
        This function is intended to allow the user to quickly determine the survey
        number range of interest, thereby allowing the reporcessing of the raw data
        over this region only. This requires that the user passes a CCdata which
        represents the entire raw dataset.

        Parameters
        ----------
        TSsurveys : list
            The survey folder numbers
        staTime : str
            The start time of the interval/.
        endTime : str
            The start time of the interval/.

        Returns
        -------
        None : tuple
            Measurement survey numbers within the range given.
        '''

        surveyTimes = [pd.to_datetime(group.split('survey')[1]) for group in
                       TSsurveys]
        mask = [(group > pd.to_datetime(staTime)) and
                (group < pd.to_datetime(endTime)) for group in
                surveyTimes]
        TSsurveys = list(itertools.compress(TSsurveys, mask))

        surveyNo = [i for i, x in enumerate(mask) if x]

        return (surveyNo[0], surveyNo[-1])

    def Data_on_mesh(mesh_obj, data, loc=''):
        '''Place data onto the mesh, and store each mesh file in subfolder.

        Parameters
        ----------
        mesh_obj : mesh.object
            mesh object as generated by the class mesh, containing the mesh as well
            as the functions for setting and saving that mesh.
        data : array
            Data containing mesh data stored in rows for each time step in columns.
            Separate mesh file will be generated for each ``n`` column.
        loc : str (Default = '')
            Location to which the output vtu files will be saved.
        '''

        if loc=='':
            loc = 'inv_on_mesh/'

        exists = os.path.isdir(loc)

        if exists:
            shutil.rmtree(loc)
            os.makedirs(loc)
        else:
            os.makedirs(loc)


        # Extract the Kt values for a single src rev pair and save to mesh
        for idx, tstep in enumerate(data.T):
            # Calculate the decorrelation values
            mesh_obj.setCellsVal(tstep)

            # Save mesh to file
            mesh_obj.saveMesh(os.path.join(loc,'mesh%s' % idx))
#            self.mesh_obj.saveMesh(os.path.join(loc,'Kernel%s_%s_No%s' % (srcrec[0], srcrec[1],idx)))

    def inversionSetup(mesh_param, channelPos, noise_chan, drop_ch,noiseCutOff,database,
                       CCdataTBL, Emaxt0, TS_idx, inversion_param, verbose=False):
        '''Run function intended to cluster the steps required to setup the
        inversion mesh and the associated sensitivity kernels. If the mesh_param
        dictionary key "makeMSH" is true, a new mesh will be constructed,
        otherweise an attempt to load it from disk will be made.

        Parameters
        ----------
        mesh_param : dict
            The height, radius, char_len, makeMSH (bool) of the mesh
        channelPos : str
            Location of the csv contraining the channel positions, same units as
            provided in the ``mesh_param``
        noise_chan : list/None
            Channels which are pure noise, and will be used to determine the SNR.
            if ``None`` is given no attempt to calculate the SNR will be made.
        drop_ch : list
            Channels which should be dropped. Pass ``None`` or ``False`` to skip.
        noiseCutOff : float (Default=10.91)
            The noise cutoff in dB
        database : str
            The location of the processed data ``database.h5``.
        CCdataTBL : str
            The name of the database table eg:``CCprocessedFixed``
        Emaxt0 : int
            The arrival time in number of sample points at which max. s-wave energy
            arrives. Used to perfrom the regression to determine the diffusivity.
        TS_idx : int
            The index of the survey which is to be used for SNR calculation.
        inversion_param : dict
            sigma "The scattering cross-section perturbation size in [area]",
            c "velocity of the core". If sigma is provided than the value
            determined from diffusion regression will be overwritten.
            lag "The lag value (fixed or rolling) intended to feed into ``d_obs``
            calcKernels "If true calculate the sens. kernels, otherweise skip and database won't be
            overwritten."
        verbose : Bool (default = False)
            True for the most verbose output to screen.

        Returns
        -------
        CWD_Inversion.h5 : hdf5 database
            The output G matrix holding all sensntivity kernels for each src/rec
            pair are written to the database for use in the inversion.
        setupDict : dict
            Containing modified or produced data for the inversion.
        '''

        import postProcess as pp
        import mesh as msh
        import data as dt

        # ------------------- Mesh the cylinder -------------------#
        if mesh_param['makeMSH']:
            clyMesh = msh.mesher(mesh_param)    # Declare the class
            clyMesh.meshIt()                    # Create the mesh
            clyMesh.meshOjtoDisk()              # Save mesh to disk
        else:
            clyMesh = msh.mesher(mesh_param)    # Declare the class
            clyMesh = clyMesh.meshOjfromDisk()  # Load from desk,

        # ------------------- Calculate the Kij -------------------#
        # Read in channel datafile
        dfChan = utilities.read_channelPos(channelPos, mesh_param)

        # Load in single survey details
        TSsurveys = dt.utilities.DB_group_names(database, 'TSdata')
        TSsurvey = dt.utilities.\
                   DB_pd_data_load(database,
                                   os.path.join('TSdata',
                                                TSsurveys[TS_idx]))

        # Read in the src/rec pairs
        CCdata = dt.utilities.DB_pd_data_load(database, CCdataTBL)


        # Calculate the window centre of each window in time
        TS = TSsurvey.columns.get_level_values('TSamp').unique().tolist()[0]
        wdws = CCdata.columns.get_level_values('window').unique().tolist()
        wdws_cent = utilities.WindowTcent(TS, wdws)

        # Remove noisy channels
        if noise_chan:
            noiseyChannels, SNR, NoiseTraces = pp.\
                                           post_utilities.\
                                           calcSNR(TSsurvey,
                                                   noise_chan,
                                                   dfChan.index.values,
                                                   wdws,
                                                   noiseCutOff, inspect=verbose)
            pp.post_utilities.CC_ch_drop(CCdata, noiseyChannels, errors='ignore')

        if drop_ch:
            pp.post_utilities.CC_ch_drop(CCdata, drop_ch, errors='ignore')

        src_rec = utilities.src_recNo(CCdata)


        # The linear regression for D and sigma using a trace selected mid-way
        trace = utilities.traceAttributes(TSsurvey, TSsurvey.shape[1]//2)

        if verbose:
            plt.figure(figsize=(7, 2))
            plt.plot(np.log(trace['Tseries']**2))

        D, sigma_temp = utilities.DiffRegress(trace, dfChan,
                                              Emaxt0, plotOut=verbose)

        if inversion_param['sigma'] is None:
            inversion_param['sigma'] = sigma_temp

        # Unity medium parameters
        inversion_param['D'] = D

        print('\n-------- Applied Kernel Parameters --------\n',
              'D = %g : sigma = %g, : c = %g \n' %(D,
                                                   inversion_param['sigma'],
                                                   inversion_param['c']))

        if 'calcKernels' in inversion_param.keys() and inversion_param['calcKernels']:
            # decorrelation ceofficient for each tet centre, each src/rec
            Kth = decorrTheory(src_rec, dfChan, clyMesh.cell_cent,
                               wdws_cent, inversion_param, clyMesh)

            # Generate the required kernel matrix for inversion
            Kth.Kt_run()

            # Place the kernels on the mesh
            Kth.K_on_mesh()

        # ------------------- determine d_obs -------------------#
        d_obs = utilities.d_obs_time(CCdata, src_rec, inversion_param['lag'],
                                         wdws, staTime=None,
                                         stopTime=None)

        utilities.HDF5_data_save('CWD_Inversion.h5', 'Inversion', 'd_obs',
                                 d_obs.values, {'wdws': [x.encode('utf-8') for x in wdws],
                                         'lag': inversion_param['lag'],
                                         'Times': [a.encode('utf8') for a in d_obs.index.strftime("%d-%b-%Y %H:%M:%S.%f").values.astype(np.str)]},
                                 ReRw='r+')

        # ------------------- Calculate initial tuning param -------------------#
        print('\n-------- Initial turning parameters --------')
        inversion_param['lambda_0'] = inversion_param['c'] / inversion_param['f_0']
        if 'L_0' not in inversion_param.keys():
            inversion_param['L_0'] = 8 * inversion_param['lambda_0']  # Planes 2015

        print('lambda_0 = %g' % inversion_param['lambda_0'] )
        print('L_0 = %g' % inversion_param['L_0'])
        print('L_c = %g' % inversion_param['L_c'])

        try:
            print('The user defined sigma_m = %g will be applied'
                  % inversion_param['sigma_m'])
        except KeyError:
            inversion_param['sigma_m'] = 1e-4 * \
                inversion_param['lambda_0']/10**2  # Planes 2015

            print('Calculated sigma_m = %g will be applied'
                  % inversion_param['sigma_m'])

        # Store various info in dict
        return {'CCdata' :CCdata, 'src_rec' : src_rec, 'dfChan' : dfChan,
                'd_obs' : d_obs, 'wdws': wdws, 'cell_cents': clyMesh.cell_cent,
                **inversion_param}

    def inv_on_mesh(hdf5File):
        '''Read in all the ``m_tilde`` groups within the provided hdf5 database file and places them
        onto the vtu mesh. of multiple groups are found then each will be saved into a different
        folder. Each folder will contain the inversion results for each time step

        Parameters:
        -----------
        mesh_param : dict
            The height, radius, char_len, makeMSH (bool) of the mesh
        hdf5File : str
            The name and relative location of the database containing all inversion results.
        '''

        import postProcess as pp
        import mesh as msh
        import data as dt

        # Load the mesh
#        clyMesh = msh.mesher(mesh_param)    # Declare the class
        clyMesh = msh.utilities.meshOjfromDisk()  # Read the mesh from disk

        # Load in the model param
        groups = dt.utilities.DB_group_names(hdf5File, 'Inversion')
        m_tildes_groups = [s for s in groups if "m_tildes" in s]

        for folder in m_tildes_groups:

            # Read in the inversion
            m_tilde = utilities.HDF5_data_read(hdf5File, 'Inversion', folder)

            # Place onto mesh
            utilities.Data_on_mesh(clyMesh, m_tilde, 'inv_'+folder+'/' )

    def l1_residual(hdf5File, someInts):
        '''Write function that calculates the l1 norm and residual from each m_tilde inversion.
        Should include ability to plot the resulting L_curve, and append new results to a database


        Parameters
        ----------
        hdf5File : dict
            The name and relative location of the database containing all inversion results.
        someInts : str
            The ....
        '''

        # Calculate the l1 norm
        l1[idx] = np.sum(np.abs(m_tilde))

        # Calculate the residual error
        ErrorR[idx] = np.sqrt(np.sum((d_obs - G.dot(m_tilde))**2))


    def inv_plot_tsteps(hdf5File, zslice_pos=1, subplot=True, plotcore=False):
        '''Creates plot of each time-step from different inversion results, for all inversion
        folders found within root directory.


        Parameters
        ----------
        hdf5File : str
            The hdf5 file from which each inversion result will be read
        mesh_param : str
            The hdf5 file from which each inversion result will be read
        sliceOrthog : bool (default=True)
            Perform orthogonal slicing of the mesh.
        subplot : bool (default=True)
            Place all time-steps on single subplot, else each time step will be placed on an
            individual figure.
        plotcore : bool (default=False)
            Plot the core surface

        Notes
        -----
        '''

        import pyvista as pv
        from pyvista import examples
        import numpy as np
        import matplotlib.pyplot as plt
        import glob
        from os.path import join
        import mesh as msh
        import data as dt

#        print('The inversion folder to be plotted are:\n', invFolders)

        # Load mesh object
        clyMesh = msh.utilities.meshOjfromDisk()

        # Load in the model param
        groups = dt.utilities.DB_group_names(hdf5File, 'Inversion')
        invFolders = [s for s in groups if "m_tildes" in s]

        for invfolder in invFolders:

#            invinFolder = glob.glob(invfolder+"*.vtu")
#            meshes = [pv.read(inv) for inv in invinFolder]

            m_tildes = utilities.HDF5_data_read(hdf5File, 'Inversion', invfolder)

            # Load in the attributes,
            attri = utilities.HDF5_attri_read(hdf5File, 'Inversion', invfolder)

            # TODO:
            # Plot the standard attribute common to all , and then update the time for each
            attri['Times'] = [time.decode('UTF-8') for time in attri['Times']]


            plot_rows = m_tildes.shape[1]

            # Create mesh for each time-step
            meshes = []
            for time, m_tilde in enumerate(m_tildes.T):

                # The apply data to the mesh
                clyMesh.setCellsVal(m_tilde)

#                clyMesh.cell_data['tetra']['gmsh:physical'] = m_tilde[1]
                clyMesh.saveMesh('baseMesh')

                # Load in base mesh
                meshes.append(pv.read('baseMesh.vtu'))

#                meshes['mesh'+str(time)] = baseMesh
#                meshes['slice'+str(time)] = baseMesh.slice_orthogonal(x=None, y=None, z=zslice_pos)


#

            all_dataRange = (min([mesh.get_data_range()[0] for mesh in meshes]),
                 max([mesh.get_data_range()[1] for mesh in meshes]))

            # Slice each mesh
            slices = [mesh.slice_orthogonal(x=None, y=None, z=zslice_pos) for mesh in meshes]


            col = 2 if plotcore else 1

            if subplot:
                p = pv.Plotter(shape=(plot_rows, col), border=False, off_screen=True)
                p.add_text(invfolder)

                for time, _ in enumerate(m_tildes.T):
                    print('Time:',time)

                    p.subplot(0, time)
                    p.add_mesh(slices[time], cmap='hot', lighting=True,
                                      stitle='Time  %g' % time)
                    p.view_isometric()

                    if plotcore:
                        p.subplot(1, time)
                        p.add_mesh(slices[time], cmap='hot', lighting=True,
                                          stitle='Time  %g' % time)
                        p.view_isometric()

                p.screenshot(invfolder.split('/')[0]+".png")

            else:
                for time, _ in enumerate(m_tildes.T):
                    p = pv.Plotter(border=False, off_screen=True)
                    p.add_text('Time: %s\n L_c: %g\n sigma_m: %g\n rms_max: %g' \
                               % (attri['Times'][time],
                                  attri['L_c'],
                                  attri['sigma_m'],
                                  attri['rms_max']),
                               font_size=12)
                    p.add_text(str(time), position=(10,10))
                    p.add_mesh(slices[time], cmap='hot', lighting=True,
                                        stitle='Time  %g' % time,
                                        clim=all_dataRange,
                                        scalar_bar_args=dict(vertical=True))
                    p.screenshot(invfolder.split('/')[0]+'_'+str(time)+".png")

class decorrTheory():
    '''Calculate the theoretical decorrelation ceofficient between each source
    receiver pair, based on the Code-Wave Diffusion method (Rossetto2011).

    Parameters
    ----------
    src_rec : list(tuples)
        List of each source-receiver pair.
    channels : DataFrame
        Index of channel numbers and the corresponding x,y,z (columns) positions
        corresponding to the 'src' and corresponding 'rec' numbers. Units should be
        according to the physical parameters of ``D`` and ``sigma``.
    X_b : array(x,y,z)
        Location coords of perturbation within model domain in SI units.
    t : float or list(float)
        Centre time of the correlation windows with which the comparison is to be made.
    param : dict
        Expected parameters of medium, for example ``D`` (diffusion coefficient) and
        ``sigma`` (scattering cross-section).
    mesh_obj : object
        Mesh object generated from the ``mesh`` class.
    srRx_df : DataFrame
        Calculated based on the input ``src_rec``, ``channels``, and ``X_b`` data.
        Multi-index df of s,r, and R, and Kt columns for each src, rec, tetraNo.
    G : np.array
        Matrix G of row.No = len(src_rec) * len(t).

    Notes
    -----

    '''

    def __init__(self,src_rec, channels, X_b, t, param, mesh_obj=None):
        self.src_rec = src_rec
        self.channels = channels
        self.X_b = X_b
        self.t = t
        self.param = param
        self.mesh_obj = mesh_obj
        self.srRx_df = None
        self.G = None

    def Kt_run(self):
        '''Run sequence of functions to produce the required inputs for inversion.

        Note
        ----
            The theoretical decorrelation coefficient will be calculated for each
            ``self.t`` provided, with the result appended to the matrix self.G.
        '''

        # Generate the inputs for each source receiver pair
        self.src_rec_srR()

        # Calculate the sensitivity kernal values for each cell.
        for idx, _ in enumerate(self.t):

            # The kernel
            self.Kt(t_no=idx)

            # make the matrix
            self.Kt_mat()

        # store the matrix
        utilities.HDF5_data_save('CWD_Inversion.h5', 'Inversion', 'G',
                                 self.G, {'src_rec': self.src_rec, **self.param})

    def Kt(self, t_no=0):
        '''The theoretical decorrelation coefficient between a single source-receiver
        pair.

        Parameters
        ----------
        t_no : int (Default=0)
            The time at which the kernel is calculated

        Notes
        -----
        The '_b' notation signifies a vector in cartesian coordinates
        '''

        try:
            t = self.t[t_no]
        except IndexError:
            t = self.t

        # should perform for all S/R paris in matrix notation.
        self.srRx_df['Kt'] = self.param['c']*self.param['sigma']/2 * \
            1/(4 * np.pi * self.param['D']) * \
            (1/self.srRx_df.s + 1/self.srRx_df.r) * np.exp((self.srRx_df.R**2 - \
            (self.srRx_df.s + self.srRx_df.r)**2)/(4 * self.param['D'] * t))

    def Kt_mat(self):
        '''Construct the kernel matrix or rows for each source/receiver pair and
        columns for each tet. cell centre.

        Notes
        -----
        '''

        # should perform for all S/R paris in matrix notation.
        if self.G is None:
            self.G = self.srRx_df.Kt.unstack(level=[2]).values
        else:
            self.G = np.append(self.G, self.srRx_df.Kt.unstack(level=[2]).values, axis=0)

    def src_rec_srR(self):
        '''Calculates the corresponding ``s``, ``r``, and ``R`` for all source
        receiver pairs found in ``self.src_rec``.

        s = |S_b-X_b|
        r = |r_b-X_b|
        R = |S_b-r_b|
        '''

        # List of src and rec
        src = [src[0] for src in self.src_rec]
        rec = [rec[1] for rec in self.src_rec]

        # Create df with the src rec positions used (i.e. found in self.src_rec)
        srR_df = pd.DataFrame({'src': src, 'rec': rec})

        srR_df = pd.concat([srR_df, self.channels.loc[srR_df['src']]. \
                   rename(columns = {'x':'sx', 'y':'sy', 'z':'sz'}). \
                   reset_index(drop=True)], axis=1)

        srR_df = pd.concat([srR_df, self.channels.loc[srR_df['rec']]. \
                   rename(columns = {'x':'rx', 'y':'ry', 'z':'rz'}). \
                   reset_index(drop=True)], axis=1)

        # Calculate for every x location the s and r values and R values
        # Construct the milti index

        tuples = [ srcrec + (tet,) for srcrec in self.src_rec for
                                tet in np.arange(0,len(self.X_b))]
        multi_index = pd.MultiIndex.from_tuples(tuples,
                                                  names = ['src', 'rec', 'tetraNo'])

        # Define empty dataframe for storage
        src_pos = ['sx', 'sy', 'sz']
        rec_pos = ['rx', 'ry', 'rz']

        srRx_df = pd.DataFrame(columns = ['s', 'r', 'R'] + src_pos + rec_pos,
                               index = multi_index, dtype=float)

        # Assign the R value
        R_vals =np.linalg.norm(self.channels.loc[src].values -
                               self.channels.loc[rec].values, axis=1)

        for idx, R in enumerate(R_vals):
            srRx_df.loc[self.src_rec[idx], 'R'] = R
            srRx_df.loc[self.src_rec[idx], src_pos + rec_pos] = \
                srR_df.loc[idx, src_pos + rec_pos].values

        srRx_df['s'] = np.linalg.norm(srRx_df.loc[:, src_pos].values - \
                            [self.X_b[tet] for tet in
                             srRx_df.index.get_level_values('tetraNo').\
                             values], axis=1)

        srRx_df['r'] = np.linalg.norm(srRx_df.loc[:, rec_pos].values - \
                            [self.X_b[tet] for tet in
                             srRx_df.index.get_level_values('tetraNo').\
                             values], axis=1)

        #Drop non-required columns
        srRx_df.drop(src_pos + rec_pos, axis=1, inplace=True)

        self.srRx_df = srRx_df

    def K_on_mesh(self, loc=''):
        '''Place each source-receiver kernel onto the mesh and save to file.

        Parameters
        ----------
        loc : str (Default = '')
            Location to which the output vtu files will be saved.
        '''

        if loc=='':
            loc = 'k_on_mesh/'

        exists = os.path.isdir(loc)

        if exists:
            shutil.rmtree(loc)
            os.makedirs(loc)
        else:
            os.makedirs(loc)

        # Extract the Kt values for a single src rev pair and save to mesh
        for idx, srcrec in enumerate(self.src_rec):
            # Calculate the decorrelation values
            self.mesh_obj.setCellsVal(self.srRx_df.loc[srcrec, 'Kt'].values)

            # Save mesh to file
            self.mesh_obj.saveMesh(os.path.join(loc,'Kernel%s' % idx))
#            self.mesh_obj.saveMesh(os.path.join(loc,'Kernel%s_%s_No%s' % (srcrec[0], srcrec[1],idx)))


class decorrInversion():
    '''Perform the inversion(Rossetto2011).

    Parameters
    ----------
    G : np.array
        Matrix containing each sensitivity kernel stored roweise for each mesh cell
        stored columnweise.
    d_obs : np.array
        Data vector or length equal to the number or source/receiver pairs by the
        number of windows implemented.
    L_0 : float
        Should correspond to the size of a model cell.
    L_c : float
        The correlation distance between cells
    sigma_m : float
        The standard deviation of the model cells
    cell_cents : float
        The correlation distance between cells

    Notes
    -----

    '''

    def __init__(self, G, d_obs, L_0, L_c, sigma_m, cell_cents, verbose=True):
        self.G = G
        self.d_obs = d_obs.astype(np.float32)
        self.L_0 = L_0
        self.L_c = L_c
        self.sigma_m = sigma_m
        self.cell_cents = cell_cents
        self.C_D = None
        self.C_M = None
        self.vecNorm = None
        self.m_tilde = None
        self.verbose = verbose

    def _C_D(self):
        '''Determine the covarience matrix of the data
        '''

        self.C_D = np.diag((self.d_obs*0.3)**2)

    def paramUpdate(self, V_0, f_0, factor=8):
        '''Update the L_0 of material parameters. output in [mm]
        Paerameters
        -----------
        V_0 : float
            Velocity of the bulk material [m/s]
        f_0 : float
            The dominante frequency of the wavefield [Hz]
        '''

        self.L_0 = factor * V_0/f_0*1000

    def _vecNorm(self):
        '''Calculate the relative distance between each cell.
        '''

        M = np.stack(self.cell_cents, axis=0).T

        self.vecNorm = np.empty([M.shape[1], M.shape[1]])

        for i, vec in enumerate(M.T):
            temp_M = M.copy()
            temp_M[0, :] = temp_M[0, :] - vec[0]
            temp_M[1, :] = temp_M[1, :] - vec[1]
            temp_M[2, :] = temp_M[2, :] - vec[2]

            self.vecNorm[:, i] = linalg.norm(temp_M, axis=0).T

    @staticmethod
#    @njit(parallel=True, fastmath=True)
    def C_M_speed(vecNorm, G, sigma_m, L_c, L_0):
        '''Write jitted function for increasing the speed of L_curve operations
        '''
        n = G.shape[1]
        k = len(sigma_m)
        C_M = np.zeros([n, n, k]).astype(np.float32)

        for i in range(k):
            C_M[:, :, i]  = (sigma_m[i] * L_0/L_c[i])**2 * \
                                  np.exp(-vecNorm/L_c[i])

        return C_M

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def inv_Lcurve(d_obs, G, C_M):
        '''The L-curve dedicated incersion jit'ed for increased speed. Note, this function
        does not store the output inversion model, and for large number ``C_M`` matrix RAM
        will become an issue.
        '''

        # Itterater over set of turning param
        n = C_M.shape[2]

        # Allocate containers for l1 and ErrorR
        l1 = np.zeros(n).astype(np.float32)
        ErrorR = np.zeros(n).astype(np.float32)

        for idx in range(n):

            # Calculate C_D
            C_D = np.diag((d_obs*0.3)**2).astype(np.float32)

            C_M_tilde = np.linalg.inv(
                           G.T.dot(np.linalg.inv(C_D)).dot(G) + \
                           np.linalg.inv(C_M[:,:,idx]))

            # Calculate the model
            m_tilde = C_M_tilde.dot(G.T).dot(np.linalg.inv(C_D)).dot(d_obs)

            # Calculate the l1 norm
            l1[idx] = np.sum(np.abs(m_tilde))

            # Calculate the residual error
            ErrorR[idx] = np.sqrt(np.sum((d_obs - G.dot(m_tilde))**2))

        return l1, ErrorR

    def _C_M(self):
        '''Determine the covarience matrix of the model. If multiple param are
        found then C_M will be calculated for each.
        '''


        if isinstance(self.L_c, np.ndarray) and isinstance(self.sigma_m, np.ndarray):
            vecNorm = self.vecNorm.astype(np.float32)
            G = self.G
            sigma_m = self.sigma_m.astype(np.float32)
            L_c = self.L_c.astype(np.float32)
            L_0 = np.float32(self.L_0)

            self.C_M = self.C_M_speed(vecNorm, G, sigma_m, L_c, L_0)

        else:
            self.C_M = (self.sigma_m * self.L_0/self.L_c)**2 * np.exp(-self.vecNorm/self.L_c).astype(np.float32)

    def invTuning(self, L_crange, sigma_mrange, runs, d_obs_idx=0):
        '''Perform the L-curve trade-off turning base of free param L_0 and sigma_m

        Parameters
        ----------
        L_0range : tuple
            Range of L_0 parameters (min, max, num).
        sigma_mrange : tuple
            Range of sigma_m parameters (min, max, num).
        runs : int
            The number of ramdom samples extracted from each tuning param
        d_obs_idx : int
            The index of the d_obs step on which turning is performed

        Returns
        -------

        '''

        # Establish the range of each and randomly sample
        L_c_s = np.linspace(L_crange[0], L_crange[1], L_crange[2])
        self.L_c = np.random.choice(L_c_s, runs)

        sigma_m_s = np.linspace(sigma_mrange[0], sigma_mrange[1], sigma_mrange[2])
        self.sigma_m = np.random.choice(sigma_m_s, runs)

        # Calculate the covarience matrix of the model for each combination
        self._vecNorm()  # The relative distance between cells
        self._C_M()

        # Now set up the L-curve method to estimate M
        d_obs = self.d_obs[d_obs_idx].astype(np.float32)
        G = self.G.astype(np.float32)
        C_M = self.C_M.astype(np.float32)

        l1_s, ErrorR_s = self.inv_Lcurve(d_obs, G, C_M)

        return l1_s, ErrorR_s, self.L_c, self.sigma_m

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def inv(d_obs, G, C_M, m_prior, m_tildes, error_thresh, no_iteration):
        '''Estimation of the model parameters, for each time-step found in d_obs,
        where each row of 'd_obs' contains a single time step.

        Parameters
        ----------
        d_obs : array
            rows of observed data at each time-step
        G : array
            Sensntivity matrix G
        C_M : array
            Covarience matrix of the model param
        m_prior : array
            Prior model param
        m_tildes : array
            empty matrix into which each time-step generated model param is stored
            along the columns.
        error_thresh : float.32
            The error threashold below which the inversion resultes are accepted
        no_interation : int
            The max. number of repeat inversion steps.

        Returns
        -------
        m_tildes : array
            The model response from each of the time-step inversions with rows: of
            the number of model cells and columns: of the number of time-steps.
        rms : float
            The root-mean-square diference after inforcing the positivity constraint.

        TODO
        ----
        Impletment the subiteration process by which the negative change value are
        progresivly removed. This should be possible on the entire inversion set.
        '''


        n = d_obs.shape[0]                      # Number of time-steps
        mshape = m_tildes.shape                 # Shape of mtildes [No.cells, dobs]

        # Flattened array for positivity constraint enforcement
        m_tildes_zero = np.zeros(mshape[0]*mshape[1], dtype=np.float32)
        rms = np.float32(0)
        iteration = int(0)

        error_thresh = np.float32(error_thresh)
        no_iteration = int(no_iteration)

        # Define outer loop for initially a set number of interations
#        while rms > error_thresh and iteration<no_iteration:

        # TODO: ATTEMPT TO BACK IMPLETMENT TO GET CONDITIONS WORKING

        while iteration <= no_iteration:
            iteration += 1
            for idx in range(n):
                #cwd_inv = cwd.decorrInversion(G, d_obs[idx], L_0, L_c, sigma_m, cell_cent, verbose=True)

                # Calculate C_D
                C_D = np.diag((d_obs[idx]*0.3)**2).astype(np.float32)

                C_M_tilde = np.linalg.inv(
                               G.T.dot(np.linalg.inv(C_D)).dot(G) + \
                               np.linalg.inv(C_M))

                # Calculate the model
                m_tildes[:,idx] = m_prior[:,idx] + C_M_tilde.dot(G.T).dot(np.linalg.inv(C_D)).dot(d_obs[idx] - G.dot(m_prior[:, idx]))

            # Perform zero operation
            m_tildes_ravel = np.ravel(m_tildes)
            for i in range(m_tildes_ravel.shape[0]):
                val = m_tildes_ravel[i]
                if val < 0:
                    val = 0
                m_tildes_zero[i] = val

            m_tildes = np.reshape(m_tildes_zero, mshape)

            rms = np.sqrt((m_prior - m_tildes)**2).mean()

            # Update param for next step
            C_M = C_M_tilde.copy()
            m_prior = m_tildes.copy()
#            print('Iteration No,: %d, rms: %g' % (iteration, rms))

        return m_tildes, rms


    def invRun(self, L_c=None, sigma_m=None, Database='CWD_Inversion.h5', m_tildes=None,
               error_thresh=100.0, no_iteration=10, chunk=False, d_obs_time=None,
               down_sample=None, runs=50):
        '''Overhead inversion function to perform time-lapse inversion operations
        for each time-step in parallel

        Parameters
        ----------
        L_c : float/list(floats)/tuple (default = None)
            Define the final ``L_c`` to use during the entire inversion. If list is given multiple
            inversions will be run for all time-step for each ``L_c`` given. Currently this
            freature is not compatible when ``chunk`` is true. If ``L_c`` is a list, then
            ``sigma_m`` must be a list of equal lenght. If a tuple is provided, a range of values
            between ``L_c[0]`` and ``L_c[1]`` every ``L_c[2]``  will be calculated.
        sigma_m : float/list(floats)/tuple (default = None)
            Define the final sigma_m to use during the entire inversion. If list is given multiple
            inversions will be run for all time-step for each sigma_m given. Currently this
            freature is not compatible when ``chunk`` is true. If a tuple is provided, a range of values
            between ``sigma_m[0]`` and ``sigma_m[1]`` every ``sigma_m[2]``  will be calculated.
        Database : h5 object
            The database to which all inversion related param are stored.
        m_tildes : floats (default = None)
            model space vector
        error_thresh : float.32 (default = 100.0)
            The error threashold below which the inversion resultes are accepted
        no_interation : int (default = 10)
            The max. number of repeat inversion steps enforcing non-negative cond.
        chunk : bool or int (default = False)
            Defines the t-step chunks which to perform the inversion. This is
            intended to reduce the RAM requirements when many time-steps given.
        d_obs_time : list[str]/array[str] (default = None)
            The time corresponding to each inversion, should be equal to the number of rows in
            ``self.d_obs``.
        down_sample : int (default = None)
            Reduce the number of times steps in the inversion an integer (e.g. every 2nd time step).
            The intended use case is where computational time is an issue.
        runs : int (default = 50)
            The number of random combinations of ``L_c`` and ``sigma_m`` turning param runs.
        '''

        # ------------------ SETUP PARAM. ------------------

        # Update the tuning param (if provided) and ensure dtype
        if isinstance(sigma_m, tuple) and isinstance(L_c, tuple):
            # Establish the range of each and randomly sample
            L_c_s = np.linspace(L_c[0], L_c[1], L_c[2])
            L_c = np.random.choice(L_c_s, runs)

            sigma_m_s = np.linspace(sigma_m[0], sigma_m[1], sigma_m[2])
            sigma_m = np.random.choice(sigma_m_s, runs)


        # Deal with only one list(float) and one float
        if isinstance(sigma_m, list) and not isinstance(L_c, list) and L_c:
            L_c = [L_c]*len(sigma_m)

        if isinstance(L_c, list) and not isinstance(sigma_m, list) and sigma_m:
            sigma_m = [sigma_m]*len(L_c)

        if isinstance(L_c, list):
            L_c = np.float32(L_c)
            self.L_c = L_c[0]
        elif L_c is not None:
            L_c = np.float32(L_c)
            self.L_c = L_c
        else:
            self.L_c = np.float32(self.L_c)

        if isinstance(sigma_m, list):
            sigma_m = np.float32(sigma_m)
            self.sigma_m = sigma_m[0]
        elif sigma_m is not None:
            sigma_m = np.float32(sigma_m)
            self.sigma_m = sigma_m
        else:
            self.sigma_m = np.float32(self.sigma_m)

        error_thresh = np.float32(error_thresh)
        no_iteration = int(no_iteration)

        # Determine the t-steps to be performed
        if down_sample:
            d_obs = self.d_obs[0::down_sample, :]
            d_obs_time = d_obs_time[0::down_sample]
        else:
            d_obs = self.d_obs

        # Define prior model
        m_prior = np.zeros([self.G.shape[1], d_obs.shape[0]],
                           dtype=np.float32)

        if m_tildes is None:
            m_tildes = np.zeros([self.G.shape[1], d_obs.shape[0]],
                                dtype=np.float32)

        self._vecNorm()

        #  ------------------ Begin ------------------
        if chunk:
            print('\n----------- Begin chuncked Inversion -----------')
            # Re-calculate the covariance model matrix
            self._C_M()

            tsteps = d_obs.shape[0]
            wholes = tsteps//chunk

            remainder = tsteps - wholes*chunk

            if chunk > tsteps:
                raise ValueError('In input chunk = %g is greater than the number of time steps: %g'
                                 % (chunk, tsteps))

            if remainder:
                chunks = [wholes]*chunk + [remainder]
            else:
                chunks = [wholes]*chunk

            step = 0
            for idx, ch in enumerate(chunks):
                print(idx, "from %g - to %g" % (step, step+ch))

                m_tildes, rms = self.inv(d_obs[step:step+ch, :],
                                         self.G, self.C_M, m_prior, m_tildes,
                                         error_thresh, no_iteration)

                # Store the data to the database along wtih some attributes
                attrb = {'L_c': self.L_c, 'sigma_m': self.sigma_m, 'rms_max': rms, 'tstep': step,
                         'Times': d_obs_time[step:step+ch]}
                utilities.HDF5_data_save(Database, 'Inversion', 'm_tildes'+str(step), m_tildes,
                                         attrb=attrb, ReRw='r+')
                step += ch

        elif sigma_m.shape or L_c.shape:  # perform for multiple sigma
            print('\n----------- Begin Inversion for Multiple sigma_m or L_c -----------')
            for sigmam, Lc in zip(sigma_m, L_c):
                self.sigma_m = sigmam
                self.L_c = Lc
                print('sigma m: %g' % sigmam)
                print('L_c : %g' % Lc)
                self._C_M()
                m_tildes, rms = self.inv(d_obs, self.G, self.C_M, m_prior, m_tildes,
                                         error_thresh, no_iteration)

                # Store the data to the database along wtih some attributes
                attrb = {'L_c': self.L_c, 'sigma_m': self.sigma_m, 'rms_max': rms, 'Times': d_obs_time}
                utilities.HDF5_data_save(Database, 'Inversion', 'm_tildes'+str(sigmam)+"_"+str(Lc), m_tildes,
                                         attrb=attrb, ReRw='r+')

        else:
            print('\n----------- Begin Inversion -----------')
            m_tildes, rms = self.inv(d_obs, self.G, self.C_M, m_prior, m_tildes,
                                     error_thresh, no_iteration)
            # Store the data to the database along wtih some attributes
            attrb = {'L_c': self.L_c, 'sigma_m': self.sigma_m, 'rms_max': rms, 'Times': d_obs_time}
            utilities.HDF5_data_save(Database, 'Inversion', 'm_tildes', m_tildes,
                                     attrb=attrb, ReRw='r+')

