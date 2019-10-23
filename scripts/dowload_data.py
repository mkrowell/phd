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

sys.path.append(abspath('.'))
import src


# ------------------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------------------
folder_shore = 'shoreline'
url_shore = 'https://coast.noaa.gov/htdata/Shoreline/us_medium_shoreline.zip'

folder_tss = 'tss'
url_tss = 'http://encdirect.noaa.gov/theme_layers/data/shipping_lanes/shippinglanes.zip'

city = 'seattle'
year = '2017' 
projection = "+proj=utm +zone=10 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
epsg = '32610'
months = ['{:02d}'.format(m) for m in range(1,2)]



# ------------------------------------------------------------------------------
# DOWNLOAD DATA
# ------------------------------------------------------------------------------
# Download a shapefile representation of the United States shoreline
# and save it to the data directory.
shore = src.Shapefile_Download(folder_shore, url_shore)
shore.download()

# Download a shapefile representation of the US Traffic Separation Scheme
# and save it to the data directory.
tss = src.Shapefile_Download(folder_tss, url_tss)
tss.download()

# Download raw NAIS data from MarineCadastre for the given city and year 
# and save it to the data directory.
nais = src.NAIS_Download(city, year, epsg)
for month in months:
    nais.download(month)
    nais.clean_raw(month)
# Remove temporary folders
nais.clean_up()
