#!/usr/bin/env python
'''
.. script::
    :language: Python Version 3.7.4
    :platform: Windows 10
    :synopsis: download shoreline, TSS, and NAIS data

.. moduleauthor:: Maura Rowell <mkrowell@uw.edu>
'''


# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------
from os.path import abspath, dirname, join
import sys
import yaml

sys.path.append(abspath('.'))
import src


# ------------------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------------------
city = 'seattle'
year = '2017' 
months = [str(i).zfill(2) for i in range(1, 3)]


# ------------------------------------------------------------------------------
# DOWNLOAD DATA
# ------------------------------------------------------------------------------
# Download a shapefile representation of the United States shoreline
# and save it to the data directory.
shore = src.Shapefile_Download('shore')
shore.download()

# Download a shapefile representation of the US Traffic Separation Scheme
# and save it to the data directory.
tss = src.Shapefile_Download('tss')
tss.download()

# Download raw NAIS data from MarineCadastre for the given city and year 
# and save it to the data directory.
nais = src.NAIS_Download(city, year)
for month in months:
    nais.month = month
    nais.download()
    nais.clean_raw()
# Remove temporary folders
nais.clean_up()
