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
from glob import glob
import io
import os
from os.path import abspath, dirname, exists, join
import requests
from retrying import retry
import shutil
import zipfile
import yaml

import src


# ------------------------------------------------------------------------------
# DOWNLOADS
# ------------------------------------------------------------------------------
class Shoreline_Download(object):

    '''
    Download a GIS representation of the United States shoreline
    and save it to the data directory.
    '''

    def __init__(self):
        self.root = abspath(join('..','data','raw','shoreline'))
        self.url = 'https://coast.noaa.gov/htdata/Shoreline/us_medium_shoreline.zip'
        self.output = join(self.root, 'us_medium_shoreline.shp')

    def download(self):
        '''Download zip file and extract to data directory.'''
        if exists(self.output):
            print('The Shoreline shapefile has already been downloaded.')
            return self.output

        print('Downloading the Shoreline shapefile...')
        download = requests.get(self.url)
        zfile = zipfile.ZipFile(io.BytesIO(download.content))
        zfile.extractall(self.root)
        return self.output

class TSS_Download(object):

    '''
    Download a GIS representation of the US Traffic Separation Scheme
    and save it to the data directory.
    '''

    def __init__(self):
        self.root = abspath(join('..','data','raw','tss'))
        self.url = 'http://encdirect.noaa.gov/theme_layers/data/shipping_lanes/shippinglanes.zip'
        self.output = join(self.root, 'shippinglanes.shp')

    def download(self):
        '''Download zip file and extract to data directory.'''
        if exists(self.output):
            print('The TSS shapefile has already been downloaded.')
            return self.output

        print('Downloading the TSS shapefile...')
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
        self.root = abspath(join('..','data','raw','ais'))
        self.processed = abspath(join('..','data','processed','ais'))
        self.year = year
        self.city = city

        param_yaml = abspath(join(dirname(__file__), '..', 'src', 'settings.yaml'))
        with open(param_yaml, 'r') as stream:
            self.parameters = yaml.safe_load(stream)[self.city]

        self.zone = self.parameters['zone']
        self.minPoints = self.parameters['minPoints']
        self.lonMin = self.parameters['lonMin']
        self.lonMax = self.parameters['lonMax']
        self.latMin = self.parameters['latMin']
        self.latMax = self.parameters['latMax']

        self.name = 'AIS_{0}_{1}_Zone{2}.csv'
        self.url = 'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{0}/AIS_{1}_{2}_Zone{3}.zip'
        self.download_dir = join(self.root, 'AIS_ASCII_by_UTM_Month')

        self.df = None

        # Create the directories if they dont exist
        if not exists(self.root):
            os.makedirs(self.root)
        if not exists(self.processed):
            os.makedirs(self.processed)
            

    @retry(stop_max_attempt_number=5)
    def download(self, month):
        '''Download zip file and extract to temp directory.'''
        name = self.name.format(self.year, month, self.zone)
        csv = join(self.root, name)
        csv_processed = join(self.processed, name)
        if exists(csv) or exists(csv_processed):
            return

        print('Downloading NAIS file for month {0}...'.format(month))
        url = self.url.format(self.year, self.year, month, self.zone)

        zfile = src.download_url(url, self.root, '.zip')
        src.extract_zip(zfile, self.root)

        # Move to top level directory
        self.extracted_file = src.find_file(self.root, name)
        shutil.copy(self.extracted_file, self.root)
        os.remove(zfile)

    def clean_up(self):
        '''Remove subdirectories created during unzipping.'''
        if exists(self.download_dir):
            shutil.rmtree(self.download_dir)

    def clean_raw(self, month):
        '''Basic cleaning and reducing of data.'''
        name = self.name.format(self.year, month, self.zone)
        csv = join(self.root, name)
        self.raw = src.dataframe.Basic_Clean(
            csv,
            self.minPoints,
            self.lonMin,
            self.lonMax,
            self.latMin,
            self.latMax
        )
        self.raw.clean_raw()
