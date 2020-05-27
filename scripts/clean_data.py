#!/usr/bin/env python
"""
.. script::
    :language: Python Version 3.7.3
    :platform: Windows 10
    :synopsis: download shoreline, TSS, and NAIS data

.. moduleauthor:: Maura Rowell <mkrowell@uw.edu>
"""


# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------
import os
import sys

sys.path.append(os.path.abspath("."))
import src.clean


# ------------------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------------------
city = "seattle"
year = "2017"
months = ["07"]

# ------------------------------------------------------------------------------
# CLEAN DATA
# ------------------------------------------------------------------------------
nais = src.clean.NAIS_Cleaner(city, year)
for month in months:
    nais.month = month
    nais.clean_raw(overwrite=True)
    nais.process(overwrite=True)
