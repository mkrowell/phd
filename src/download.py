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
import os
from os.path import abspath, basename, dirname, exists, join
import requests
from retrying import retry
import shutil
import zipfile
import yaml

import src


# ------------------------------------------------------------------------------
# DOWNLOADS
# ------------------------------------------------------------------------------
class Shapefile_Download(object):

    '''
    Download a shapefile representation of the United States shoreline
    and save it to the data directory.
    '''

    def __init__(self, name):
        parameters_file = join(dirname(__file__), 'shapefile_settings.yaml')
        with open(parameters_file, 'r') as stream:
            parameters = yaml.safe_load(stream)

        self.root = abspath(parameters[name]['root'])
        self.url = parameters[name]['url']
        self.filename = parameters[name]['output']
        self.output = join(self.root, self.filename)

    def download(self):
        '''Download zip file and extract to data directory.'''
        if exists(self.output):
            print(f'The {self.filename} shapefile has already been downloaded.')
            return self.output

        print(f'Downloading the {self.name} shapefile...')
        download = requests.get(self.url)
        zfile = zipfile.ZipFile(io.BytesIO(download.content))
        zfile.extractall(self.root)
        return self.output

class NAIS_Download(object):

    '''
    Download raw NAIS data from MarineCadastre for the given city and year
    and save it to the data directory.
    '''

    def __init__(self, city, year):
        self.city = city
        self.year = year

        # Data directories
        self.root = abspath(join('data','raw','ais'))
        self.processed = abspath(join('data','processed','ais'))
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.processed, exist_ok=True)

        # Download parameters
        self.download_dir = join(self.root, 'AIS_ASCII_by_UTM_Month')

        # Data files are separated by month
        self._month = '01'

        # City associated parameters
        param_yaml = join(dirname(__file__), 'settings.yaml')
        with open(param_yaml, 'r') as stream:
            self.parameters = yaml.safe_load(stream)[self.city]

        self.zone = self.parameters['zone']
        self.minPoints = self.parameters['minPoints']
        self.lonMin = self.parameters['lonMin']
        self.lonMax = self.parameters['lonMax']
        self.latMin = self.parameters['latMin']
        self.latMax = self.parameters['latMax']

        # Initialize dataframe
        self.df = None  

    @property
    def month(self):
        '''Return the month of the instance.'''
        return self._month

    @month.setter
    def month(self, month):
        '''Set the month of the instance.'''
        month = str(month).zfill(2)
        if month not in [str(i).zfill(2) for i in range(1, 13)]:
            raise UserWarning('Month must be betwen 01 and 12.')
        self._month = month

    @property
    def name(self):
        '''Return basename of the downloaded file.'''
        return f'AIS_{self.year}_{self.month}_Zone{self.zone}.csv'

    @property
    def csv(self):
        '''Return path to the raw downloaded file.'''
        return join(self.root, self.name)

    @property
    def csv_processed(self):
        '''Return path to the cleaned downloaded file.'''
        return join(self.processed, self.name)

    @property
    def url(self):
        '''Return url for the given year, month, and zone.'''
        return f'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{self.year}/AIS_{self.year}_{self.month}_Zone{self.zone}.zip'
    
    @retry(stop_max_attempt_number=5)
    def download(self):
        '''Download zip file and extract to temp directory.'''
        if exists(self.csv) or exists(self.csv_processed):
            return

        print(f'Downloading NAIS file for month {self.month}...')
        zfile = src.download_url(self.url, self.root, '.zip')
        src.extract_zip(zfile, self.root)

        # Move to top level directory
        self.extracted_file = src.find_file(self.root, self.name)
        shutil.copy(self.extracted_file, self.root)
        os.remove(zfile)

    def clean_raw(self):
        '''Basic cleaning and reducing of data.'''
        print(f'Cleaning NAIS file for month {self.month}...')
        try:
            self.raw_basic = src.dataframe.Basic_Clean(
                self.csv,
                self.minPoints,
                self.lonMin,
                self.lonMax,
                self.latMin,
                self.latMax
            )
            self.raw_basic.clean_raw()
        except IOError:
            print(f'NAIS file for month {self.month} has been cleaned.')
            pass

    def clean_up(self):
        '''Remove subdirectories created during unzipping.'''
        if exists(self.download_dir):
            shutil.rmtree(self.download_dir)