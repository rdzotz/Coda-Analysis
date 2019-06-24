#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 14:37:38 2018

@author: rwilson
"""
import postProcess as pp
import pre_process as prp
import data as dt
import dispCWI as disp
#import matplotlib.pyplot as plt
import re
import glob
import csv
import pandas as pd
import imp
import numpy as np
import scipy.linalg as linalg
from scipy import signal
import random
import os
import matplotlib.pyplot as plt
import itertools
#from matplotlib2tikz import save as tikz_save\

imp.reload(disp)
imp.reload(pp)
imp.reload(prp)
imp.reload(dt)

########################### Scratch Pad ###########################

''' TODO:
# Save mesh to disk, and enable read of mesh details from disk for processing
# Add init param to save all figures to disk in prep for server side operations
# Afterwards, continue to setup the submission of multiple jobs
'''
# %%---------------------- Plotting the data ------------------------
database ='Ultrasonic_data_DB/Database.h5'
#
## Testing plot function from database.
#plot_db = disp.dispCWI2(database)
#plot_db.plot_DB()
#
#CCdata.loc[([1,2], [5,10],[slice(None)]*2), :].index
#
## Need to do this multiple times
#CCdata.loc[(1,5,slice(None)), :].index
#
#CCdata.drop( index=(1,5,slice(None)) )
#
#
#CCdata_drop_idx = CCdata.loc[(1, 2, slice(None)), :].index
#
#CCdata.drop(index=CCdata_drop_idx)
#
#mask = CCdata.index.isin( [1,2,3], level=0 ) & CCdata.index.isin( [10,12,13], level=1 )
#
#
#CCdata.loc[~mask]


# %%---------------------- Determine the subset of Meas No's ----------------------
#import CWD as cwd
#import imp
#
##imp.reload(cwd)
##database ='Ultrasonic_data_active_DB/Database.h5'
PVdata = dt.utilities.DB_pd_data_load(database, 'PVdata')
CCdata = dt.utilities.DB_pd_data_load(database, 'CCprocessedFixed')
#CCdata = dt.utilities.DB_pd_data_load(database, 'CCprocessedFixedMW') # Targeted section

# Determine a list of TS surveys
TSsurveys = dt.utilities.DB_group_names(database, 'TSdata')

#surveyNosRange = cwd.utilities.surveyNorange(TSsurveys, '2018-03-14 19:00:00', '2018-03-14 23:00:00')

# Slice the TS surveys based on the range used
#TSsurveys = TSsurveys[surveyNosRange[0]: surveyNosRange[1]]

# Load in first of these surveys
TSsurvey = dt.utilities.DB_pd_data_load(database, os.path.join('TSdata', TSsurveys[1000]))

#measNosRange = cwd.utilities.measNorange(CCdata, '2018-03-14 19:00:00', '2018-03-14 23:00:00')

import numpy as np
import pandas as pd
def mklbl(prefix, n):
     return ["%s%s" % (prefix, i) for i in range(n)]

miindex = pd.MultiIndex.from_product([mklbl('A', 4),
                                   mklbl('B', 4),
                                   mklbl('C', 10)])

dfmi = pd.DataFrame(np.arange(len(miindex) * 2)
               .reshape((len(miindex), 2)),
                index=miindex).sort_index().sort_index(axis=1)

As = ['A0', 'A2']
Bs = ['B1', 'B3']

#for a,b in zip(As, Bs):
#    dfmi_drop_idx = dfmi.loc[(a, b, slice(None)), :].index
#    dfmi.drop(dfmi_drop_idx, inplace=True, errors='ignore')


dfmi.drop(pd.MultiIndex.from_arrays([As,Bs]), inplace=True)

#fn = dfmi.index.get_level_values
#dfmi[~(fn(0).isin(As) | fn(1).isin(Bs))]

# %%---------------------- Test shift PV and save back to DB hdf5 ----------------------

#origin = pd.to_datetime('20180212180703') - pd.Timedelta(43.6725, unit='D')
#
#pp.post_utilities.PV_time_shift(PVdata, 'Time(Days)', 'D', origin)

# %%---------------------- Inspect the  ----------------------

#TSsurvey = dt.utilities.DB_pd_data_load(database, os.path.join('TSdata', TSsurveys[1000]))

# %% ------------- Calculate the Theoretical Decorrelation on mesh ---------------

import CWD as cwd
import imp

imp.reload(cwd)

mesh_param = {'height': 85.25, 'radius': 18.84, 'char_len': 3.5, 'makeMSH' : True}
channelPos = '2762_Sensors_D38.1_Array1_16P-4S.csv'
noise_chan = [9, 19, 20]
drop_ch = [17, 18, 16]
noiseCutOff = 1.91 # dB
database = 'Ultrasonic_data_DB/Database.h5'
CCdataTBL = 'CCprocessedFixed'
Emaxt0 = 400
TS_idx = 1000
inversion_param = {'sigma': 5, 'c': 3.5, 'lag': 60,  # D [mm^2/us], C [mm/us], sigma [mm2]
                   'f_0': 1, 'L_c': 4.0, 'sigma_m': 0.5}  # f_0 [1/us], L_c [mm]

setupDict = cwd.utilities.inversionSetup(mesh_param, channelPos, noise_chan,
                                         drop_ch, noiseCutOff, database,
                                         CCdataTBL, Emaxt0, TS_idx, inversion_param,
                                         verbose=False)


# %% ------------- Run the inversion on each time-step ---------------

hdf5File = "CWD_Inversion.h5"
G = cwd.utilities.HDF5_data_read(hdf5File, group='Inversion', name='G')

# Declare the inversion parameters
cwd_inv = cwd.decorrInversion(G, setupDict['d_obs'],
                              setupDict['L_0'],
                              setupDict['L_c'],
                              setupDict['sigma_m'],
                              setupDict['cell_cents'])

cwd_inv.invRun(Database = hdf5File, error_thresh = np.float32(100.0), no_iteration=int(5))
# %%
chunk = 4
tsteps = d_obs.shape[0]
wholes = tsteps//chunk

remainder = tsteps - wholes*chunk

if remainder:
    chunks = [wholes]*chunk +[remainder]
else:
    chunks = [wholes]*chunk

sta = 0
for idx, ch in enumerate(chunks):
    print(idx, "from %g - to %g " %(sta, sta+ch ))
    sta+= ch



# %%---------------------- Setup the inversion ------------------------
hdf5File = "CWD_Inversion.h5"
G = cwd.utilities.HDF5_data_read(hdf5File, group='Inversion', name='G')

# Data observed read just a subset
#sta = '2018-03-13 06:00:00' # Actual start
#end = '2018-03-13 07:00:00' # End for small subset

#sta = '2018-03-14 22:00:00' # active region sta
#end = '2018-03-14 23:00:00' # active region end

# Lag value
lag=10

#end = '2018-03-15 05:00:00' # Actual end
d_obs = cwd.utilities.d_obs_time(CCdata, src_rec, lag, wdws, staTime=None,
                                 stopTime=None).values

lambda_0 = 3000/1e6*1000

L_0 = 8 * lambda_0 # As per Planes 2015

L_c = 2.0 # The correlation distance between cells [mm]

sigma_m = 1e-4 *lambda_0/10**2

#from numba import jit, prange

# %%---------------------- L-curve tuning ------------------------
import CWD as cwd
imp.reload(cwd)

# Declare the inversion parameters
cwd_inv = cwd.decorrInversion(G, d_obs, L_0, L_c, sigma_m, clyMesh.cell_cent)

# range within which the turning param can be sampled (min, max, number)
l1_s = np.empty(0)
ErrorR_s = np.empty(0)
L_cs = np.empty(0)
sigma_ms = np.empty(0)

L_crange = (lambda_0*2,lambda_0*50,50)
sigma_mrange = (sigma_m*0.1, sigma_m/0.0002**2, 50)
runs = 100 # Number of random combination runs

l1_temp, ErrorR_temp, L_ctemp, sigma_mtemp = cwd_inv.invTuning(L_crange, sigma_mrange, runs,d_obs_idx=-20)

l1_s = np.append(l1_s,l1_temp)
ErrorR_s = np.append(ErrorR_s,ErrorR_temp)
L_cs = np.append(L_cs, L_ctemp)
sigma_ms = np.append(sigma_ms, sigma_mtemp)

LcurveDB = pd.DataFrame({'l1_s' : l1_s, 'ErrorR_s' : ErrorR_s,
                         'L_cs': L_cs, 'sigma_ms': sigma_ms})

# Load the server generated data
LcurveDB2 = pd.read_hdf(hdf5File, 'Lcurve')
#LcurveDB.to_hdf("CWD_Inversion.h5", key='Lcurve/', mode='r+')

# Scatter plot the two
plt.scatter(np.log10(LcurveDB2.l1_s), np.log10(LcurveDB2.ErrorR_s))
#plt.scatter(LcurveDB.l1_s, LcurveDB.ErrorR_s)

for i, txt in enumerate(LcurveDB2.sigma_ms):
    plt.annotate(txt, (np.log10(LcurveDB2.l1_s)[i], np.log10(LcurveDB2.ErrorR_s)[i]))

# The L-curve analysis yields:
print(LcurveDB2.iloc[23].L_cs)

L_c = 36.73495714285715
sigma_m = 10408.16326540408

# %% Run the inversion for all data over time
import h5py
import CWD as cwd
imp.reload(cwd)

lag=50
d_obs = cwd.utilities.d_obs_time(CCdata, src_rec, lag, wdws, staTime=None,
                                 stopTime=None).values

lambda_0 = 3000/1e6*1000
L_0 = 8 * lambda_0 # As per Planes 2015

# Declare the inversion parameters
cwd_inv = cwd.decorrInversion(G, d_obs, L_0, L_c, sigma_m, clyMesh.cell_cent)
cwd_inv.invRun(L_c, sigma_m, hdf5File, error_thresh = np.float32(100.0), no_iteration=int(10))

# %% Save the inversion resuts to mesh

# Load the model inversion results
m_tilde = cwd.utilities.HDF5_data_read(hdf5File, 'Inversion', 'm_tildes')

# Load the mesh
clyMesh = clyMesh.meshOjfromDisk()

# Place onto mesh
cwd.utilities.Data_on_mesh(clyMesh, m_tilde)

# %% Numba cliping function

from numba import njit
from numba import prange

@njit()
def myclip(val):
    """zeroing of value

    Parameters
    ----------
    val: float
        The value to be clamped.

    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """

    if val < 0:
        return 0
    else:
        return val

shape = m_tilde.shape
m_tilde_ravel = np.ravel(m_tilde)
m_tilde_zero = np.zeros(shape[0]*shape[1])

for i in range(shape[0]*shape[1]):
    m_tilde_zero[i] = myclip(m_tilde_ravel[i])


m_tilde_zero = np.reshape(m_tilde_zero, shape)

# %% Investigations with numba speed up
import time
import numba as nb
from numba import njit
from numba import prange

#spec = [
#    (nb.float32[:, ::1]),               # G
#    (nb.float64[::1, :]),          # d_obs
#    (nb.float64),               # L_0
#    (nb.float64),               # L_c
#    (nb.float64),               # sigma_m
#    (nb.float64[::1]),               # cell_cents
#]

@njit(parallel=True, fastmath=True)
def inv_monitoring(d_obs, G, C_M, m_prior):
    # Define the inversion class instance

    n = d_obs.shape[0]
    for idx in range(n):
#        cwd_inv = cwd.decorrInversion(G, d_obs[idx], L_0, L_c, sigma_m, cell_cent, verbose=True)

        # Calculate C_D
        C_D = np.diag((d_obs[idx]*0.3)**2).astype(np.float32)

        C_M_tilde = np.linalg.inv(
                       G.T.dot(np.linalg.inv(C_D)).dot(G) + \
                       np.linalg.inv(C_M))

        # Calculate the model
        m_tilde = m_prior + C_M_tilde.dot(G.T).dot(np.linalg.inv(C_D)).dot(d_obs[idx] - G.dot(m_prior))

        return m_tilde
        #t1= time.time()
        # Save the mesh
#        clyMesh.setCellsVal(cwd_inv.m_tilde)
#        clyMesh.saveMesh('Inversion_%d' %idx)
# %%

# Define the initial m_prior
m_prior = np.zeros([G.shape[1]]).astype(np.float32)

# Prepar the C_M matrix
M = np.stack(clyMesh.cell_cent, axis=0).T
C_M = np.empty([M.shape[1], M.shape[1]] )

for i, vec in enumerate(M.T):
    temp_M = M.copy()
    temp_M[0, :] = temp_M[0, :] - vec[0]
    temp_M[1, :] = temp_M[1, :] - vec[1]
    temp_M[2, :] = temp_M[2, :] - vec[2]

    C_M[:, i] = linalg.norm(temp_M, axis=0).T


C_M = ((sigma_m * L_0/L_c)**2 * np.exp(-C_M/L_c)).astype(np.float32)

# %%
t0= time.time()
m_tilde= inv_monitoring(d_obs.astype(np.float32), G, C_M, m_prior)

t1= time.time()

print('Total time: %g' %(t1 - t0))

#C_D = np.diag((d_obs*0.3)**2)
#
#m_tilde = (G.T.dot(np.linalg.inv(C_D)).dot(G) + np.linalg.inv(C_M))

# %%---------------------- Scratch Pad ------------------------


# %%----------------------------------------------


# %%---------------------- Check ------------------------

'''TODO:
    Write up a function in the CWD class to calculate the direct inversion loss
    problem.
'''