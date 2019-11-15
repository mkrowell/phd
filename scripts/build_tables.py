#!/usr/bin/env python
'''
.. script::
    :language: Python Version 3.7.4
    :platform: Windows 10
    :synopsis: build basic tables in Postgres

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
srid = 32610
year = '2017'
months = [str(i).zfill(2) for i in range(1, 13)]

# Load the parameters for the seattle region
parameters_file = abspath(join(dirname(__file__) ,'..','src','settings.yaml'))
with open(parameters_file, 'r') as stream:
    parameters = yaml.safe_load(stream)[city]


# # ------------------------------------------------------------------------------
# # BUILD SHORELINE TABLE
# # ------------------------------------------------------------------------------
# # Create a Postgres table from the shoreline shapefile
# shapefile_shore = src.Shapefile_Download('shore').output
# table_shore = src.Postgres_Table('shore')
# table_shore.create_table(filepath=shapefile_shore)

# # Transform to UTM 10 SRID
# table_shore.project_column('geom', 'MULTILINESTRING', srid)

# # Keep only the relevant shore line
# table_shore.reduce_table('regions', '!=', parameters['region'])
# table_shore.add_index('idx_geom', 'geom', gist=True)


# # ------------------------------------------------------------------------------
# # BUILD TSS TABLE
# # ------------------------------------------------------------------------------
# # Create a Postgres table from the tss shapefile'
# shapefile_tss = src.Shapefile_Download('tss').output
# table_tss = src.Postgres_Table('tss')
# table_tss.create_table(filepath=shapefile_tss)

# # Transform to UTM 10 SRID
# table_shore.project_column('geom', 'MULTILINESTRING', srid)

# # Keep only the relevant TSS
# table_tss.reduce_table('objl', '!=', parameters['tss'])
# table_tss.add_index('idx_geom', 'geom', gist=True)


# # ------------------------------------------------------------------------------
# # BUILD NAIS TABLES
# # ------------------------------------------------------------------------------
table_points = src.Points_Table('points')
# table_points.drop_table()
# table_points.create_table(columns=table_points.columns)

# # Copy processed NAIS data to table
# nais = src.NAIS_Download(city, year)
# for month in months:
#     nais.month = month
#     table_points.copy_data(nais.csv_processed)

# Add local time and time interval between rows grouped by MMSI
table_points.add_time_interval()
table_points.add_index("idx_time", "TimeInterval")

# Remove points that have too high a time interval between points
# In bounds data is no longer needed 
table_points.reduce_table('TimeInterval', '>=', '4 minutes')
table_points.drop_column('In_Bound')

# Delete odd cog/sog
table_points.drop_anomaly()

# Vaccum talbe
table_points.vaccum_table()

# Recalculate time interval
table_points.add_time_interval()


# # Add POINTM
# table_points.add_geometry()
# table_points.project_column('geom', 'POINTM', 32610)
# table_points.add_index("idx_geom", "geom")



