Coda Analysis
=============

Introduction
------------

Coda Analysis, is a collection of python code intended for the processing of the coda or diffuse portion of recorded waveforms. The package currently consists of two modules;

pyCoda:
The basic structure of the raw input data is a series of recorded Time-Series (TSdata), and the corresponding time-matched Perturbation Vectors (PVdata). The intended application of this package of code is for the monitoring of changes in material properties due to some perturbation.

pyCWD:
The implementaiton of Coda-Wave Decorrelation as described in "Imaging multiple local changes in heterogeneous media with diffuse waves, Planes et. al. 2015". This module takes as input the processed correlation coefficients for differnet time windows within the coda. Currently this package is set up to perform the inversion on a 3D cylindrical mesh. 


A basic workflow tutorial will be provided shortly

Citing Coda Analysis
--------------------

If you use the modules contained within the package Coda Analysis, please cite the current release as;

.. image:: https://zenodo.org/badge/192512410.svg
   :target: https://zenodo.org/badge/latestdoi/192512410

Author
------

* `Reuben Zotz-Wilson <https://orcid.org/0000-0001-6223-2825>`_

  *Delft University of Technology, Department of Civil Engineering and Ceosciences, Delft, Netherlands*

License
-------
Coda Analysis is distributed under the terms of the **Apache 2.0** license. Details on
the license agreement can be found `here
<https://www.apache.org/licenses/LICENSE-2.0>`_.
