#!/usr/bin/env python
'''
.. module::
    :language: Python Version 3.6.8
    :platform: Windows 10
    :synopsis: process raw NAIS broadcast data points

.. moduleauthor:: Maura Rowell <mkrowell@uw.edu>
'''


# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------
import angles
import datetime
import geopandas as gpd
import numpy as np
import os
from os.path import abspath, basename, dirname, exists, join
import pandas as pd
from shapely.geometry import Point
import yaml

import matplotlib.pyplot as plt 
import pyplot_themes as themes


import src
from src import print_reduction, print_reduction_gdf, time_all


# ------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------
METERS_IN_NM = 1852
EARTH_RADIUS_KM = 6371
PLOTS_DIR = abspath(join(dirname(__file__) ,'..','reports','figures'))

# ------------------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------------------
themes.theme_paul_tol()


# ------------------------------------------------------------------------------
# SPATIAL FUNCTIONS
# ------------------------------------------------------------------------------
def azimuth_utm(point1, point2):
    """Return the azimuth from point1 to point2"""
    east = point2.x - point1.x
    north = point2.y - point1.y
    return np.degrees(np.arctan2(east, north))

def angle_difference(angle1, angle2):
    """Return the signed difference between two angles in radians"""
    angle1 = np.radians(angle1)
    angle2 = np.radians(angle2)
    y = np.sin(angle1 - angle2)
    x = np.cos(angle1 - angle2)
    return np.arctan2(y, x)


# ------------------------------------------------------------------------------
# PLOT FUNCTIONS
# ------------------------------------------------------------------------------
def save_plot(filepath):
    """Save and close figure"""
    if exists(filepath):
            os.remove(filepath) 
    plt.tight_layout()
    plt.savefig(filepath)

def plot_mmsi_by_type(df, prepend):
    """Create a chart of the number of unique vessels per type"""
    filename = f"{prepend} - Unique MMSI by Vessel Type"
    fig, ax = plt.subplots()
    unique = df[['MMSI', 'VesselType']].drop_duplicates(keep='first')
    unique['VesselType'].value_counts().plot(kind='bar', alpha=0.75, rot=0)

    ax.set_ylabel('Number of Unique MMSI')
    fig.suptitle(filename, fontsize=16)
    plt.savefig(join(PLOTS_DIR, filename))
    plt.close(fig)

def plot_status(df, prepend):
    """Create a chart of status observations"""
    filename = f"{prepend} - Navigation Status"
    fig, ax = plt.subplots()
    df['Status'].value_counts().plot(kind='bar', alpha=0.75)

    ax.set_ylabel('Number of Data Points')
    fig.suptitle(filename, fontsize=16)
    plt.savefig(join(PLOTS_DIR, filename))
    plt.close(fig)

def plot_sog(df, prepend):
    """Create a chart of SOGs observed"""
    filename = f'{prepend} - SOG.png'

    fig, ax = plt.subplots()
    df.groupby('SOG').count()['BaseDateTime'].plot()
    ax.set_ylabel('Number of Data Points')
    fig.suptitle(filename, fontsize=16)

    filepath = join(PLOTS_DIR, filename)
    save_plot(filepath)
    plt.close(fig)

def plot_cog(df, prepend):
    """Create a chart of COGs observed by type"""
    filename = f'{prepend} - COG.png'   
    groups = df.groupby('VesselType')['COG']
    fig, ax = plt.subplots()
    for k, v in groups:
        v.hist(label=k, alpha=.3, ax=ax)

    ax.legend()
    ax.set_ylabel('Number of Data Points')
    fig.suptitle(filename, fontsize=16)
    plt.savefig(join(PLOTS_DIR, filename))
    plt.close(fig)

def plot_time_interval(df, prepend):
    """Create a chart of time intervals observed"""
    filename = f'{prepend} - Time Interval.png'

    fig, ax = plt.subplots()
    df['Interval'].hist()
    ax.set_ylabel('Number of Data Points')
    fig.suptitle(filename, fontsize=16)

    filepath = join(PLOTS_DIR, filename)
    save_plot(filepath)
    plt.close(fig)

# ------------------------------------------------------------------------------
# CLEAN
# ------------------------------------------------------------------------------  
@time_all
class Basic_Clean(object):

    """
    Clean and reduce the NAIS data.
    """

    def __init__(self, csvFile, minPoints, lonMin, lonMax, latMin, latMax):
        """Initialize attributes and dataframe."""
        self.csv = csvFile
        self.cleaned = self.csv.replace("raw","cleaned")
        self.month = int(self.csv.split("_")[-2])

        # Spatial parameters
        self.minPoints = minPoints
        self.lonMin = lonMin
        self.lonMax = lonMax
        self.latMin = latMin
        self.latMax = latMax

        # Create raw NAIS dataframe
        self.df = pd.read_csv(self.csv)
        self.df['BaseDateTime'] = pd.to_datetime(self.df['BaseDateTime'])
        self.df.sort_values('MMSI', inplace=True)
        
        # Standardize missing values
        self.df['Status'].replace(np.nan, "undefined", inplace=True)
        self.df['Heading'].replace(511, np.nan, inplace=True)
        self.map_vessel_types()
        self.normalize_angles()
        self.required = [
            'MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Heading'
        ]

        # Reduce to area of interest
        self.select_spatial()

        # Calculate time interval - for plotting
        self.time_interval()
    
        # Save raw df for plotting purposes
        self.df_raw = self.df.copy()

    # MAIN FUNCTIONS
    def plot(self):
        """Plot data before and after its been cleaned"""
        plot_mmsi_by_type(self.df_raw, "Raw")
        plot_status(self.df_raw, "Raw")
        plot_sog(self.df_raw, "Raw")
        plot_cog(self.df_raw, "Raw")
        plot_time_interval(self.df_raw, "Raw")

        plot_mmsi_by_type(self.df, "Cleaned")
        plot_status(self.df, "Cleaned")
        plot_sog(self.df, "Cleaned")
        plot_cog(self.df, "Cleaned")
        plot_time_interval(self.df, "Cleaned")

    @print_reduction
    def clean_raw(self):
        """
        Select the area of interest, remove duplicate information, remove
        contradictory information, remove invalid IDs, normalize, write output.
        """
        self.drop_null()
        self.drop_duplicate_rows()

        self.drop_duplicate_keys()
        self.drop_inconsistent_info()
        self.drop_columns()
        
        self.drop_bad_mmsi()
        self.filter_sog(3)
        self.filter_time(3)
       
        self.drop_sparse_mmsi()

        self.filter_type()
        self.filter_status()
       
        self.df.sort_values(['MMSI', 'BaseDateTime'], inplace=True)
        self.df.to_csv(self.cleaned, index=False, header=True)

    # DATAFRAME CLEANING 
    def select_spatial(self):
        """Limit data to bounding box of interest."""
        self.df = self.df[
            (self.df['LON'].between(self.lonMin, self.lonMax)) &
            (self.df['LAT'].between(self.latMin, self.latMax))
        ].copy()

    @print_reduction
    def drop_null(self):
        """
        Drop rows with nulls in all the required columns. 
        No loss of information.
        """
        self.df.replace("", np.nan, inplace=True)
        self.df.dropna(how='any', subset=self.required, inplace=True)

    @print_reduction
    def drop_duplicate_rows(self):
        """
        Remove entirely duplicated rows. No loss of information.
        """
        self.df.drop_duplicates(keep='first', inplace=True)

    @print_reduction
    def drop_duplicate_keys(self):
        """
        MMSI, BaseDateTime pairs must be unique. Can't calculate step 
        calculations with duplicate timestamps. Drop both duplicate rows.
        """
        key = ['MMSI', 'BaseDateTime']
        self.df.drop_duplicates(subset=key, keep=False, inplace=True)

    @print_reduction
    def drop_inconsistent_info(self):
        """
        Confirm that a MMSI is associated with only one name, dimension.
        A mix of vessels using the same MMSI will not be included.
        This data is entered only once, so a mistake in entry will appear
        in all data points and not change over time.
        """
        mmsi = self.df.groupby(['MMSI'])
        self.df = mmsi.filter(lambda g: g['VesselName'].nunique()<=1)
        self.df = mmsi.filter(lambda g: g['Length'].nunique()<=1)
        self.df = mmsi.filter(lambda g: g['Width'].nunique()<=1)

    def drop_columns(self):
        """Remove unneccessary columns."""
        unused = ['CallSign', 'IMO', 'Cargo', 'Width', 'Draft']
        self.df.drop(columns=unused, inplace=True)

    @print_reduction
    def drop_bad_mmsi(self):
        """MMSI numbers should be 9 digits and between a given range."""
        condRange = self.df['MMSI'].between(201000000, 775999999)
        self.df = self.df[condRange]

    @print_reduction
    def filter_sog(self, limit):
        """Limit to points with > limit SOG."""
        self.df['SOG'] = self.df['SOG'].abs()
        self.df = self.df[self.df['SOG'] > limit]
    
    def time_interval(self):
        """Calculate time interval between points"""
        col = 'Interval'
        group = self.df.sort_values(['MMSI', 'BaseDateTime']).groupby('MMSI')
        
        self.df[col] = group['BaseDateTime'].diff()
        self.df[col].fillna(datetime.timedelta(seconds=60), inplace=True)
        self.df[col] = self.df[col].astype('timedelta64[s]')

    @print_reduction
    def filter_time(self, limit):
        """Limit to points less than 3 minutes from prior data point."""
        self.df = self.df[self.df['Interval'] < limit*60]

    @print_reduction
    def drop_sparse_mmsi(self):
        """Remove MMSIs with few data points."""
        self.df = self.df.groupby(['MMSI']).filter(
            lambda g: len(g)>self.minPoints
        )
    
    def normalize_angles(self):
        """Normalize COG to an angle between [0, 360)."""
        self.df['COG'] = self.df['COG'].apply(
            lambda x: angles.normalize(x, 0, 360)
        )
        self.df['Heading'] = self.df['Heading'].apply(
            lambda x: angles.normalize(x, 0, 360)
        )

    def map_vessel_types(self):
        """Map codes to categories."""
        type_dict = abspath(join(dirname(__file__), 'vessel_types.yaml'))
        with open("src\\vessel_types.yaml", 'r') as stream:
            v_map = yaml.safe_load(stream)

        self.df['VesselType'].replace("", np.nan, inplace=True)
        self.df['VesselType'] = self.df['VesselType'].map(v_map)
        self.df['VesselType'] = self.df['VesselType'].replace(np.nan, "Unknown")
        self.df['VesselType'] = self.df['VesselType'].astype('category')

    @print_reduction
    def filter_type(self):
        """Filter non-normal operating vessels"""
        types = ['tanker', 'cargo', 'ferry']
        self.df = self.df[self.df['VesselType'].isin(types)]

    @print_reduction
    def filter_status(self):
        """Filter non-normal stauts"""
        status = [
            'not under command', 
            'restricted maneuverability', 
            'engaged in fishing', 
            'power-driven vessel towing astern', 
            'reserved for future use (9)',
            'power-driven vessel pushing ahead or towing alongside'
        ]
        self.df = self.df[~self.df['Status'].isin(status)]
    
      
# ------------------------------------------------------------------------------
# QUALITY CHECK
# ------------------------------------------------------------------------------
@time_all
class Processor(object):

    """
    Remove nonsensical data points.
    """

    def __init__(self, csvFile, month, minPoints):
        """Initialize attributes and geodataframe."""
        self.csv = csvFile
        self.csv_processed = self.csv.replace("cleaned", "processed")
        self.month = int(month)
        self.minPoints = minPoints

        # Create raw NAIS dataframe
        self.df = pd.read_csv(self.csv)
        self.df['BaseDateTime'] = pd.to_datetime(self.df['BaseDateTime'])
        self.df.sort_values(['MMSI', 'BaseDateTime'], inplace=True)

        # Create and project geopandas dataframe
        self.gdf = gpd.GeoDataFrame(
            self.df, 
            geometry=gpd.points_from_xy(self.df['LON'], self.df['LAT']),
            crs={'init':'epsg:4326'}
        )
        self.gdf = self.gdf.to_crs(32610) 
        self.gdf['LAT_UTM'] = self.gdf['geometry'].y
        self.gdf['LON_UTM'] = self.gdf['geometry'].x    
      
    @property
    def grouped_mmsi(self):
        '''Return sorted dataframe grouped by MMSI.'''
        return self.gdf.sort_values(['MMSI', 'BaseDateTime']).groupby('MMSI')

    @property
    def grouped_trip(self):
        '''Return sorted dataframe grouped by MMSI.'''
        return self.gdf.sort_values(['MMSI', 'Trip', 'BaseDateTime']).groupby(
            ['MMSI', 'Trip']
        )


    # MAIN METHOD --------------------------------------------------------------
    def preprocess(self):
        '''
        Detect suspicious data.
        '''
        self.update_interval()
        self.mark_trips()
        self.drop_sparse_trips()
       
        self.mark_distance(1.25)

        self.update_interval()
        self.mark_trips()
        self.drop_sparse_trips()

        self.step_cog()
        self.acceleration()
        self.alteration()

        self.normalize_time()
        self.write_output()

                
    # PREPROCESSING ------------------------------------------------------------
    def update_interval(self):
        """Update time interval after data cleaning."""
        self.gdf['Interval'] = self.grouped_mmsi['BaseDateTime'].diff()
        self.gdf['Interval'].fillna(
            datetime.timedelta(seconds=0), inplace=True
        )
        self.gdf['Interval'] = self.gdf['Interval'].astype('timedelta64[s]')
           
    def mark_trips(self):
        """Make trips at time breaks"""
        self.gdf['Break'] = np.where(
            (self.gdf['Interval'] == 0) | (self.gdf['Interval'] > 3*60), 
            1, 
            0
        )
        self.gdf['Trip'] = self.grouped_mmsi['Break'].cumsum()
        self.gdf['Trip'] = self.gdf['Trip'] + 10000*self.month
        self.gdf.drop(columns=['Break'], inplace=True)

    @print_reduction_gdf
    def drop_sparse_trips(self):
        """Remove MMSIs with few data points."""
        self.gdf = self.grouped_trip.filter(
            lambda g: len(g)>self.minPoints
        )

    def step_distance(self):
        """Return distance between lat/lon positions."""
        def distance(df):
            df.reset_index(inplace=True)
            df['Step_Distance'] = df['geometry'].distance(
                df['geometry'].shift()
            )
            return df.set_index('index')
        self.gdf = self.grouped_trip.apply(distance)
        self.gdf['Step_Distance'].fillna(0)
        self.gdf['Step_Distance']/METERS_IN_NM

    @print_reduction_gdf
    def mark_distance(self, limit):
        """Compare step_distance to speed*time. Remove suspicious data."""
        self.step_distance()
        self.gdf['Expected_Distance'] = (
            (self.gdf['SOG']*self.gdf['Interval']*METERS_IN_NM)/3600
        )

        self.gdf['Outlier'] = np.where(
            self.gdf['Step_Distance'] >= self.gdf['Expected_Distance']*limit,
            1,
            0
        )
        self.gdf = self.gdf[self.gdf['Outlier'] == 0]
        self.gdf.drop(columns=['Expected_Distance', 'Outlier'], inplace=True)
  
    def step_cog(self):
        """Calculate the course between two position points."""
        def course_utm(df):
            df.reset_index(inplace=True)
            df['Step_Azimuth'] = azimuth_utm(
                df['geometry'].shift(),
                df.loc[1:,'geometry']
            )
            return df.set_index('index')

        # Calculate and normalize course between successive points
        self.gdf = self.grouped_trip.apply(course_utm)
        self.gdf['Step_Azimuth'].fillna(method='bfill', inplace=True)
        self.gdf['Step_Azimuth'] = round(
            self.gdf['Step_Azimuth'].apply(
                lambda x: angles.normalize(x, 0, 360)
            )
        )
        
        # Caclulate error
        self.gdf['Error_COG'] = (180 - abs(
            abs(self.gdf['COG'] - self.gdf['Step_Azimuth']) - 180
        ))
        self.gdf['Error_Heading'] = (180 - abs(
            abs(self.gdf['Heading'] - self.gdf['Step_Azimuth']) - 180
        ))
        
    def acceleration(self):
        """Add acceleration."""
        self.gdf['DS'] = self.grouped_trip['SOG'].diff()
        self.gdf['Acceleration'] = 3600*self.gdf['DS'].divide(
            self.df['Interval'], fill_value=0)
        self.gdf.drop(columns=['DS'], inplace=True)
        self.gdf['Acceleration'].fillna(method='bfill', inplace=True)

    def alteration(self):
        """Calculate change in heading."""
        def delta_heading(df):
            df.reset_index(inplace=True)
            df['Alteration'] = angle_difference(
                df['Heading'].shift(),
                df.loc[1:,'Heading']
            )
            return df.set_index('index')
        self.gdf = self.grouped_trip.apply(delta_heading)
        self.gdf['Alteration'].fillna(method='bfill', inplace=True)
        self.gdf['Alteration_Degrees'] = np.degrees(self.gdf['Alteration'])
        self.gdf['Alteration_Cosine'] = np.cos(self.gdf['Alteration'])
        
    @print_reduction_gdf
    def normalize_time(self):
        '''Round time to nearest minute.'''
        self.gdf['DateTime'] = self.gdf['BaseDateTime'].dt.round('1min')
        self.gdf.drop_duplicates(
            subset=['MMSI', 'Trip', 'DateTime'],
            keep='first',
            inplace=True
        )

    def replace_nan(self):
        """Fill undefined values with default float value."""
        for col in ['Length', 'VesselType']:
            self.gdf[col].replace("", np.nan, inplace=True)
            self.gdf[col].replace(np.nan, -1, inplace=True)

    def write_output(self):
        """Write to one processed CSV file"""
        self.replace_nan()
        self.gdf.drop(columns=['BaseDateTime', 'Interval','Step_Distance'])
        columns = [
            'MMSI',
            'Trip', 
            'DateTime', 
            'LAT',
            'LAT_UTM',
            'LON',
            'LON_UTM',
            'SOG', 
            'COG', 
            'Heading',
            'Step_Azimuth', 
            'Acceleration', 
            'Alteration', 
            'Alteration_Degrees', 
            'Alteration_Cosine',
            'VesselName', 
            'VesselType', 
            'Status', 
            'Length',
        ]
        self.gdf = self.gdf[columns]
        self.gdf.reindex(columns)
        self.gdf.sort_values(['MMSI', 'Trip', 'DateTime'], inplace=True)
        self.gdf.to_csv(self.csv_processed, index=False, header=False)


  









  







# ------------------------------------------------------------------------------
# DATAFRAMES





    def mmsi_plot(self):
        '''Plot raw trajectory for each MMSI.'''
        for name, group in self.grouped_trip:
            try:
                mmsi = group['MMSI'].unique()[0]
                print('Plotting MMSI %s' % mmsi)
                data = [
                    ['LAT'],
                    ['LON'],
                    ['SOG'],
                    ['COG'],
                    ['Time Interval']
                ]
                legend = [False, False, 'full', 'full', False]
                linestyles = ["","-", "--"]

                # Create plots
                fig, axes = plt.subplots(5, 1, sharex='col', figsize=(8, 11))
                fig.suptitle("MMSI: {0}".format(mmsi), fontsize=14)
                for i in range(5):
                    for j in range(len(data[i])):
                        sns.lineplot(
                            x='BaseDateTime',
                            y=data[i][j],
                            style=linestyle[j],
                            data=group,
                            palette=sns.color_palette("paired", len(data[i])),
                            ax=axes[i],
                            label=data[i][j],
                            legend=legend[i]
                        )
                        axes[i].set_ylabel(data[i][0])

                # Format plot area
                plt.xticks(rotation=70)
                plt.tight_layout()
                fig.subplots_adjust(top=0.9)

                # Save to directory
                plt.savefig(
                    join(DIRECTORY_PLOTS, 'MMSI', 'Raw', '{0}.png'.format(mmsi))
                )
                plt.close()
            except:
                continue


    # FIND ERRONEOUS DATA ------------------------------------------------------


    # def mmsi_plot(self):
    #     '''Plot raw trajectory for each MMSI.'''
    #     for name, group in self.grouped_mmsi:
    #         mmsi = group['MMSI'].unique()[0]
    #         data = ['LAT', 'LON', 'SOG', 'COG', 'Time Interval']
    #         # legend = ['full', False, False, False, False]
    #         legend = [False, False, False, False, False]
    #
    #         # Create plots
    #         fig, axes = plt.subplots(5, 1, sharex='col', figsize=(8, 11))
    #         fig.suptitle("MMSI: {0}".format(mmsi), fontsize=14)
    #         for i in range(0,5):
    #             axes[i].set_ylabel(data[i])
    #             sns.lineplot(
    #                 x='BaseDateTime',
    #                 y=data[i],
    #                 # hue='In_Bound',
    #                 data=group,
    #                 # palette=sns.color_palette("cubehelix", 2),
    #                 ax=axes[i],
    #                 legend=legend[i]
    #             )
    #
    #         # First plot with legend
    #         # axes[0].legend(
    #         #     loc='upper center',
    #         #     bbox_to_anchor=(0.5, 1.35),
    #         #     fancybox=True)
    #
    #         # Format plot area
    #         plt.xticks(rotation=70)
    #         plt.tight_layout()
    #         fig.subplots_adjust(top=0.95)
    #
    #         # Save to directory
    #         plt.savefig(
    #             join(DIRECTORY_PLOTS, 'MMSI', 'Raw', '{0}.png'.format(mmsi))
    #         )
    #         plt.close()
    #








    # SUB-TRAJECTORIES ---------------------------------------------------------

   

  

    # def mmsi_plot(self):
    #     '''Plot raw trajectory for each MMSI.'''
    #     # plt.style.use('dark_background')
    #     for name, group in self.grouped_mmsi:
    #         mmsi = group['MMSI'].unique()[0]
    #         data = ['LAT', 'LON', 'SOG', 'COG', 'Time Interval']
    #         legend = ['full', False, False, False, False]
    #         ntracks = len(group['Trajectory'].unique())
    #         # Create plots
    #         fig, axes = plt.subplots(5, 1, sharex='col', figsize=(8, 11))
    #         fig.suptitle("MMSI: {0}".format(mmsi), fontsize=14)
    #         for i in range(0,5):
    #             axes[i].set_ylabel(data[i])
    #             sns.lineplot(
    #                 x='BaseDateTime',
    #                 y=data[i],
    #                 hue='Trajectory',
    #                 data=group,
    #                 palette=sns.color_palette("cubehelix", ntracks),
    #                 ax=axes[i],
    #                 legend=legend[i]
    #             )
    #
    #         # First plot with legend
    #         axes[0].legend(
    #             loc='upper center',
    #             bbox_to_anchor=(0.5, 1.35),
    #             ncol=12,
    #             fancybox=True)
    #
    #         # Format plot area
    #         plt.xticks(rotation=70)
    #         plt.tight_layout()
    #         fig.subplots_adjust(top=0.90)
    #
    #         # Save to directory
    #         plt.savefig(
    #             join(DIRECTORY_PLOTS, 'MMSI', 'Raw', '{0}.png'.format(mmsi))
    #         )
    #         plt.close()










