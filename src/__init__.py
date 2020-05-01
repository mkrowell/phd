#!/usr/bin/env python
'''
.. module::
    :language: Python Version 3.6.8
    :platform: Windows 10
    :synopsis: utility functions and decorators

.. moduleauthor:: Maura Rowell <mkrowell@uw.edu>
'''

# ------------------------------------------------------------------------------
# EXCEPTIONS
# ------------------------------------------------------------------------------
class fileNotFound(Exception):
    def __init__(self, directory, filename, msg = None):
        if msg is None:
            msg = ('The file {0} was not found in the directory: {1}').format(
                  filename, directory)
        super(fileNotFound, self).__init__(msg)
        self.directory = directory
        self.filename = filename


# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------
import datetime
from collections import namedtuple
import certifi
import ctypes
import datetime
from dateutil.parser import parse
import os
from os.path import abspath, basename, dirname, exists, join, splitext
import pycurl
import re
import time
import zipfile


# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------
# passenger = {i:'passenger' for i in list(range(60,70)) + [1013, 1014]}
# cargo = {i:'cargo' for i in list(range(70,80)) + [1003, 1004, 1016]}
# tanker = {i:'tanker' for i in list(range(80,90)) + [1017, 1024]}
# tug = {i:'tug' for i in [31, 32, 52, 1023, 1025]}
# fishing = {i:'fishing' for i in [30, 1001, 1002]}

# V_TYPES =  {
#     1012: 'ferry',
#     1019: 'recreational',
# }
# V_TYPES.update(passenger)
# V_TYPES.update(cargo)
# V_TYPES.update(tanker)
# V_TYPES.update(tug)
# V_TYPES.update(fishing) 



# ------------------------------------------------------------------------------
# WEB
# ------------------------------------------------------------------------------
def download_url(url, destination, extension = '.xls'):
    '''
    Write url to temp file.
    :param url: URL to download data from
    :param destination: Path to download data to
    :param fileExtension: extension to use for temp file

    :type url: string
    :type destination: string
    :type fileExtension: string
    :return: Path to temp file containing URL data
    :rtype: string
    '''
    tempFile = join(
        destination,
        'temp_data%s' % extension
    )
    with open(tempFile, 'wb') as f:
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        c.setopt(pycurl.CAINFO, certifi.where())
        c.setopt(pycurl.HTTP_VERSION, pycurl.CURL_HTTP_VERSION_1_0)
        c.setopt(c.WRITEDATA, f)
        c.perform()
        c.close()
    return tempFile


# ------------------------------------------------------------------------------
# USER
# ------------------------------------------------------------------------------
def message_box_OK_Cancel(title, msg):
    '''
    Open a message box with OK and Cancel button options.

    :param title: Title of the message box
    :param msg: Message to show in message box

    :type title: string
    :type msg: string

    :return: Message box
    :rtype: cytpes.windll.user32.MessageBox
    '''
    return ctypes.windll.user32.MessageBoxW(0, msg, title, 1)


# ------------------------------------------------------------------------------
# FILE SYSTEM
# ------------------------------------------------------------------------------
def create_folder(parent, name):
    '''
    Create a folder with given name as a subdirectory of parent, if the
    folder does not already exist, and return path to the folder.

    :param parent: Directory in which the folder will be created
    :param name: Name of the folder to be created

    :type parent: string
    :type name: string

    :return: Path to folder
    :rtype: string
    '''
    folder = abspath(join(parent, name))
    if not exists(folder):
        try:
            os.makedirs(folder)
        except IOError as err:
            folderExistsCode = 17
            if err.errno != folderExistsCode:
                raise err
    return folder

def find_file(directory, filename):
    '''
    Search for a filename in the directory and return the full path. The
    extension does not need to be included, but the first filepath that
    matches will be returned.

    :param directory: Directory within which to search for file
    :param filename: Name of file, with or without extension, to search for

    :type directory: string
    :type filename: string

    :return: Path to file
    :rtype: string
    :raises fileNotFound: Filename cannot be located within the directory
    '''
    rex = re.compile(filename.lower())
    for root, dirs, files in os.walk(directory):
        for f in files:
            result = rex.search(f.lower())
            if result:
                return join(root, f)
    raise fileNotFound(directory, filename)

def file_parts(filepath):
    '''
    Return the path components of the file.

    :param filepath: Path to the file
    :type filepath: string
    :return: Directory, name, extension, and date (if present in name) of file
    :rtype: named tuple
    '''
    directory = dirname(filepath)
    name, extension = splitext(basename(filepath))
    fileparts = namedtuple('FileParts', 'directory name extension')
    return fileparts(directory, name, extension)

def rename_file(filepath, rename):
    '''
    Rename the filepath while keeping original extension.

    :param filepath: Path or name of file to be renamed
    :param rename: New name without extension

    :type filepath: string
    :type rename: string

    :return: Path or name of file with new name
    :rtype: string
    '''
    parts = file_parts(filepath)
    return join(parts.directory, rename + parts.extension)


# ------------------------------------------------------------------------------
# ZIP FILES
# ------------------------------------------------------------------------------
def extract_zip(filepath, destination):
    '''
    Extract zip file to the destination and return list of filenames contained
    within the archive.

    :param filepath: Filepath to the zip file to be unzipped
    :param destination: Location for the archive files to be extracted to

    :type filepath: string
    :type destination: string

    :return: List of filenames that have been extracted to the destination
    :rtype: list of strings
    '''
    zfile = zipfile.ZipFile(filepath, 'r')
    zfile.printdir()
    zfile.extractall(destination)
    zfile.close()
    return zfile.namelist()

def extract_file(filepath, destination):
    '''
    Return the path to the first file extracted from a zip file.
    '''
    zf = extract_zip(filepath, destination)
    if len(zf) == 1:
        name = zf[0]
        return join(destination, name)
    raise UserWarning('There are multiple files in the zip file.')


# ------------------------------------------------------------------------------
# VALUES
# ------------------------------------------------------------------------------
def snap_value(value, l, u):
    return min([l,u], key=lambda x:abs(x-value))




# ------------------------------------------------------------------------------
# DATAFRAMES
# ------------------------------------------------------------------------------
def concat_df(func, iterateList, *args, **kwargs):
    '''
    Return a concated dataframe made of list of dataframes obtained using func.
    '''
    dfs = list()
    for item in iterateList:
        print('Concatenating {0}...'.format(item))
        df = func(item, *args, **kwargs)
        dfs.append(df)
    return pd.concat(dfs, sort = False)


# ------------------------------------------------------------------------------
# DECORATORS
# ------------------------------------------------------------------------------
def print_reduction(original_function):
    '''Print the amount of rows removed by method.'''
    def print_reduction_wrapper(self, *args,**kwargs):
        before = len(self.df)
        x = original_function(self, *args,**kwargs)
        after = len(self.df)
        rows = after - before
        percent = round(-100*rows/before, 2)
        msg = (
            f"Month {self.month}, Method {original_function.__name__}: "
            f"Removed Rows = {-rows}, Percent Reduction = {percent}"
        )
        print(msg)
        with open('reports\\logs\\basic_clean.txt', 'a') as f:
            f.write(msg + "\n")
        return x
    return print_reduction_wrapper

def print_reduction_gdf(original_function):
    '''Print the amount of rows removed by method.'''
    def print_reduction_wrapper(self, *args,**kwargs):
        before = len(self.gdf)
        x = original_function(self, *args,**kwargs)
        after = len(self.gdf)
        rows = after - before
        percent = round(-100*rows/before, 2)
        msg = (
            f"Month {self.month}, Method {original_function.__name__}: "
            f"Removed Rows = {-rows}, Percent Reduction = {percent}"
        )
        print(msg)
        with open('reports\\logs\\preprocess.txt', 'a') as f:
            f.write(msg + "\n")
        return x
    return print_reduction_wrapper

def time_this(original_function):
    '''Print the method's execution time.'''
    def time_this_wrapper(*args,**kwargs):
        before = datetime.datetime.now()
        x = original_function(*args,**kwargs)
        after = datetime.datetime.now()
        print("Method {0}: Elapsed Time = {1}".format(
            original_function.__name__,
            after-before)
        )
        return x
    return time_this_wrapper

def time_all(Cls):
    '''Apply the time_this decorator to each method in class.'''
    # https://www.codementor.io/sheena/advanced-use-python-decorators-class-function-du107nxsv
    class DecoratedClass(object):
        def __init__(self, *args, **kwargs):
            self.oInstance = Cls(*args, **kwargs)
        def __getattribute__(self, s):
            try:
                # base class implementation
                x = super(DecoratedClass, self).__getattribute__(s)
            except AttributeError:
                # base class does not have an implementation, raises AttributeError
                pass
            else:
                return x
            # Get the instance attributes
            x = self.oInstance.__getattribute__(s)
            # If it is a method, decorate it
            if type(x) == type(self.__init__):
                return time_this(x)
            else:
                return x
    return DecoratedClass

# ------------------------------------------------------------------------------
# PACKAGE IMPORTS
# ------------------------------------------------------------------------------
from .download import *
from .dataframe import *
from .database import *
