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

sys.path.append('..')
import src


# ------------------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------------------
city = 'seattle'
year = '2017'
months = ['{:02d}'.format(m) for m in range(1,2)]


# ------------------------------------------------------------------------------
# DOWNLOAD DATA
# ------------------------------------------------------------------------------
shore = src.Shoreline_Download()
shore.download()

tss = src.TSS_Download()
tss.download()

nais = src.NAIS_Download(city, year)
for month in months:
    nais.download(month)
    try:
        nais.clean_raw(month)
    except IOError:
        # File has already been cleaned
        continue
nais.clean_up()
