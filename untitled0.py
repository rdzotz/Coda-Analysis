#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:12:46 2019

@author: rwilson
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 14:37:38 2018

@author: rwilson
"""


import re
import CWD as cwd
#from matplotlib2tikz import save as tikz_save\
imp.reload(cwd)

########################### Scratch Pad ###########################
''' TODO:
# Save mesh to disk, and enable read of mesh details from disk for processing
# Add init param to save all figures to disk in prep for server side operations
# Afterwards, continue to setup the submission of multiple jobs
'''
# %%---------------------- Loading in mesh for pyvista plotting ------------------------
# Read in single m_tilde onto mesh
hdf5File = "CWD_Inversion.h5"

PV_plot_dict = dict(database='Ultrasonic_data_active_DB/Database.h5', y_data='Ax.Load SN 0323(kN)')

cwd.utilities.inv_plot_tsteps(hdf5File, subplot=False, PV_plot_dict=PV_plot_dict)