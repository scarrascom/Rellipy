# Rellipy
Python code to extract Rayleigh-wave ellipticity and phase-shift from earthquake recordings. Developed and first applied for:

_Carrasco et al. (2023), "Constraints for the Martian crustal structure from Rayleigh waves ellipticity of large seismic events", submitted to GRL._

This repository contains the following scripts:

**1. Rellipy.py**
 - Functions for plotting and extracting ellipticity and phase shifts of selected Rayleigh waves.
 - Dictionaries are returned with the extracted data, which can also be saved into pickle files (default).

**2. get_ellipdata_fdsn.py**
 - Implementation of Rellipy for any station accesible via FDSN services or stored locally. A GCMT global catalog is utlized (source can be modified).

**3. get_ellipfinal.py**
 - Example script on how to extract the data from all the events (for one station) and calculate a final ellipticity curve for the 

Examples of the output using station BQ.DREG on Earth are provided under directory DREG. The analysis of two events (Turkey, 06.02.2023) is exemplified.

The code has been tested for different stations on Earth and works satisfactorily, but further improvements can definitively (and will) be made.
In case of bugs, comments or issues, don't hesitate to contact me.

[Sebastian Carrasco](mailto:acarrasc@uni-koeln.de?subject=[GitHub]%20Source%20Han%20Sans)
