#!/usr/bin/env python
'''
.. module:: src.download
    :language: Python Version 3.7.4
    :platform: Windows 10
    :synopsis: download shoreline, TSS, and NAIS data

.. moduleauthor:: Maura Rowell <mkrowell@uw.edu>
'''


# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------
import io
from glob import glob
import os
from os.path import abspath, basename, dirname, exists, join
import requests
from retrying import retry
import shutil
import zipfile
import yaml

import src


# ------------------------------------------------------------------------------
# DATA CLEANING
# ------------------------------------------------------------------------------
class NAIS_Cleaner(src.download.NAIS_Download):

    '''
    Clean and preprocess raw NAIS data
    '''

    def __init__(self, city, year):
        """Initialize parameters and setup directories"""
        super().__init__(city, year)
        self.city = city
        self.year = year
        self._month = '01'

        # Data directories
        self.cleaned = abspath(join('data','cleaned','ais'))
        self.processed = abspath(join('data','processed','ais'))
        os.makedirs(self.cleaned, exist_ok=True)
        os.makedirs(self.processed, exist_ok=True)      

        # City associated parameters
        self.minPoints = self.parameters['minPoints']
        self.lonMin = self.parameters['lonMin']
        self.lonMax = self.parameters['lonMax']
        self.latMin = self.parameters['latMin']
        self.latMax = self.parameters['latMax']

    @property
    def csv_cleaned(self):
        """Return path to the cleaned file"""
        return join(self.cleaned, self.name)

    @property
    def csv_processed(self):
        """Return path to the processed file"""
        return join(self.processed, self.name)

    @property
    def df_raw(self):
        """
        Return the initial df which:
        - is sorted by MMSI, BaseDateTime
        - reduced to study area
        - standardizes missing status and heading
        - maps vessel types from code to string
        - normalizes cog
        """
        return src.dataframe.Basic_Clean(
            self.csv,
            self.minPoints,
            self.lonMin,
            self.lonMax,
            self.latMin,
            self.latMax
        )

    @property
    def df_clean(self):
        """
        Returns cleaned df sorted and projected to 32610
        """
        return src.dataframe.Processor(
            self.csv_cleaned, self.month, self.minPoints
        )

    def clean_raw(self, overwrite=False):
        """Basic cleaning and reducing of data"""
        print(f'Cleaning NAIS file for month {self.month}...')
        if not exists(self.csv_cleaned) or overwrite:
            self.df_raw.clean_raw()
        else:
            print(f'NAIS file for month {self.month} has been cleaned.')        

    def process(self, overwrite=False):
        """Basic cleaning and reducing of data"""
        print(f'Processing NAIS file for month {self.month}...')
        if not exists(self.csv_processed) or overwrite:
            self.df_clean.preprocess()
        else:
            print(f'NAIS file for month {self.month} has been preprocessed.')
