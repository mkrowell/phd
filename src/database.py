#!/usr/bin/env python
"""
.. module::
    :language: Python Version 3.6.8
    :platform: Windows 10
    :synopsis: construct postgres database

.. moduleauthor:: Maura Rowell <mkrowell@uw.edu>
"""


# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import abspath, dirname, exists, join
import osgeo.ogr
import pandas as pd
from postgis.psycopg import register
import psycopg2
import seaborn as sns
import shutil
import subprocess
import tempfile
import time
import yaml

from src import time_this, LOGGER
from src.dataframe import save_plot


PLOTS_DIR = abspath(join(dirname(__file__), "..", "reports", "figures"))
plt.style.use("seaborn")
sns.set(color_codes=True)

# ------------------------------------------------------------------------------
# BASE CLASSES
# ------------------------------------------------------------------------------
class Postgres_Connection(object):

    """
    Open/close standard connection to the postgres database
    """

        # Create connection object to Postgres
    LOGGER.info("Connecting to postgres...")
    conn = psycopg2.connect(
        host="localhost",
        dbname="postgres",
        user="postgres",
        password=os.environ["PGPASSWORD"],
        )

        # Set standard properties and extensions
    with conn.cursor() as cur:
            cur.execute("SET timezone = 'UTC'")
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
        conn.commit()

        # Enable PostGIS extension
    register(conn)

    # Add functions to database
    sql = """
        CREATE OR REPLACE FUNCTION normalize_angle(
            angle FLOAT,
            range_start FLOAT,
            range_end FLOAT)
        RETURNS FLOAT AS $$
        BEGIN
            RETURN (angle - range_start) - FLOOR((angle - range_start)/(range_end - range_start))*(range_end - range_start) + range_start;
        END; $$
        LANGUAGE PLPGSQL;

        CREATE OR REPLACE FUNCTION angle_difference(
            angle1 FLOAT,
            angle2 FLOAT)
        RETURNS FLOAT AS $$
        BEGIN
            RETURN DEGREES(ATAN2(
                SIN(RADIANS(angle1) - RADIANS(angle2)),
                COS(RADIANS(angle1) - RADIANS(angle2))
            ));
        END; $$
        LANGUAGE PLPGSQL
    """
    with conn.cursor() as cur:
        try:
            cur.execute(sql)
            conn.commit()
        except psycopg2.DatabaseError as err:
            conn.rollback()
            raise err    

    def close_connection(self):
        """Close the database connection, if open"""
        if Postgres_Connection.conn:
            Postgres_Connection.conn.close()
            LOGGER.info("Database connection closed.")


class Postgres_Table(Postgres_Connection):

    """
    Base class for Postgres tables
    """

    def __init__(self, table):
        """
        Initialize connection and set srid"""
        super(Postgres_Table, self).__init__()
        self.table = table
        self.srid = 4326

    @time_this
    def run_query(self, query):
        """
        Execute a DDL SQL statement and commit it to the database.
        
        Args:
            query (string): An SQL statement
        """
        with self.conn.cursor() as cur:
            try:
                cur.execute(query)
                self.conn.commit()
            except psycopg2.DatabaseError as err:
                self.conn.rollback()
                raise err
                
    def create_table(self, filepath=None, columns=None):
        """
        Create table in the database.

        If filepath is provided, the shp2pgsql utility will be used 
        to create table from the file. Otherwise, an empty table will
        be created with provided columns.
        
        Args:
            filepath (string, optional): Path to the shapefile that is to 
                be created as a table. Defaults to None
            columns (string, optional): SQL statement defining the columns 
                of the table. Defaults to None
        """
        if filepath:
            cmd = f'"C:\\Program Files\\PostgreSQL\\12\\bin\\shp2pgsql.exe" -s 4326 -d {filepath} {self.table} | psql -d postgres -U postgres -q'
            LOGGER.info(f"Constructing {self.table} from {filepath}...")
            subprocess.call(cmd, shell=True)
        elif columns:
            sql = f"CREATE TABLE IF NOT EXISTS {self.table} ({columns})"
            LOGGER.info(f"Constructing {self.table} from columns...")
            self.run_query(sql)
        else:
            raise UserWarning("You must provide a filepath or column defintions.")

    def drop_table(self, table=None):
        """
        Drop the table from the database, if it exists.

        If table is None, self.table is dropped. Otherwise, the given 
        table is dropped.
        
        Args:
            table (string, optional): Name of table to be dropped
        """
        if table is None:
            table = self.table
        sql = f"DROP TABLE IF EXISTS {table}"
        LOGGER.info(f"Dropping table {table}...")
        self.run_query(sql)

    def copy_data(self, csv_file):
        """
        Copy data from CSV file to table.
        
        Args:
            csv_file (string): Filepath to the csv data 
                that is to be copied into the table
        """
        with open(csv_file, "r") as csv:
            LOGGER.info(f"Copying {csv_file} to {self.table}...")
            with self.conn.cursor() as cur:
                try:
                    cur.copy_from(csv, self.table, sep=",")
                    self.conn.commit()
                except psycopg2.DatabaseError as err:
                    self.conn.rollback()
                    raise err

    def add_column(self, name, datatype=None, geometry=False, default=None):
        """
        Add column to the table with the given datatype and default
        
        Args:
            name (string): Name of the new column.
            datatype (type, optional): Postgresql/PostGIS datatype. 
                Defaults to None
            geometry (bool, optional): Whether the datatype is a geometry. 
                Defaults to False
            default (value, optional): The default value of the column. 
                Should be the same type as datatype. Defaults to None
        """
        # Handle geometry types
        datatype_str = f"{datatype}"
        if geometry:
            datatype_str = f"geometry({datatype}, {self.srid})"
       
        # Handle default data types
        default_str = ""
        if default:
            default_str = f"DEFAULT {default}"

        # Entire SQL string
        sql = f"""
            ALTER TABLE {self.table} 
            ADD COLUMN IF NOT EXISTS {name} {datatype_str} {default_str}
        """

        LOGGER.info(f"Adding {name} ({datatype}) to {self.table}...")
        self.run_query(sql)

    def drop_column(self, column):
        """
        Drop column from the table, if column exists
        
        Args:
            column (string): Name of the column to be dropped
        """
        sql = f"ALTER TABLE {self.table} DROP COLUMN IF EXISTS {column}"
        LOGGER.info(f"Dropping column {column}...")
        self.run_query(sql)

    def add_point(self, name, lon, lat, time=None):
        """
        Make the given column a POINT/POINTM geometry
        
        Args:
            name (string): Name of existing column to be made into Point 
            lon (string): Name of the column containing the longitude of Point
            lat (string): Name of the column containing the latitude of Point
            time (string, optional): Name of the column containing the time 
                of Point. Defaults to None
        """
        if time:
            sql = f"""
                UPDATE {self.table}
                SET {name} = ST_SetSRID(
                    ST_MakePointM({lon}, {lat}, {time}), 
                    {self.srid}
                )
            """
        else:
            sql = f"""
                UPDATE {self.table}
                SET {name} = ST_SetSRID(ST_MakePoint({lon}, {lat}), {self.srid})
            """
        self.run_query(sql)

    def project_column(self, column, datatype, new_srid):
        """
        Convert the SRID of the geometry column
        
        Args:
            column (sting): Name of the column to project.
            datatype (type): Postgis geometry datatype.
            new_srid (int): Code for the SRID to project the geometry into.
        """
        sql = f"""
            ALTER TABLE {self.table}
            ALTER COLUMN {column}
            TYPE Geometry({datatype}, {new_srid})
            USING ST_Transform({column}, {new_srid})
        """
        LOGGER.info(f"Projecting {column} from {self.srid} to {new_srid}...")
        self.run_query(sql)

    def add_index(self, name, field=None, btree=False, gist=False):
        """
        Add index to table using the given field. If field
        is None, update the existing index.
        
        Args:
            name (string): Name of the new index. Default None
            field (string, optional): Column on which to build the index
        """
        if field is None:
            sql = f"REINDEX {name}"
            LOGGER.info(f"Updating index on {field}...")
        else:
            kind_str = ""
            if gist:
                kind_str = "USING GiST"
            if btree:
                kind_str = "USING btree"

            sql = f"""
                    CREATE INDEX IF NOT EXISTS {name}
                    ON {self.table} {kind_str} ({field})
                """
            LOGGER.info(f"Adding index on {field}...")
        self.run_query(sql)

    def reduce_table(self, column, relationship, value):
        """
        Drop rows that meet condition
        
        Args:
            column (string): column to test condition on
            relationship (string): =, <, > !=, etc.
            value (string, number): value to use in condition
        """
        if isinstance(value, str):
            sql_delete = "DELETE FROM {0} WHERE {1} {2} '{3}'"
        else:
            sql_delete = "DELETE FROM {0} WHERE {1} {2} {3}"
        sql = sql_delete.format(self.table, column, relationship, value)
        LOGGER.info(f"Dropping {column} {relationship} {value} from {self.table}...")
        self.run_query(sql)
        
    def remove_null(self, col):
        """
        Remove rows in the table that have a Null in the col
        """
        sql = f"""DELETE FROM {self.table} WHERE {col} IS NULL"""
        LOGGER.info(f"Deleting null rows from {self.table}...")
        self.run_query(sql)

    def add_local_time(self, timezone="america/los_angeles"):
        """
        Add column containing local time

        Args:
            timezone (string): timezone to add time for. Defaults to
                america/los_angles        
        """        
        name = "datetime_local"
        self.add_column(name, datatype="timestamptz")
        sql = f"""
            UPDATE {self.table}
            SET {name} = datetime at time zone 'utc' at time zone '{timezone}'
        """
        LOGGER.info(f"Updating {name} with timezone...")
        self.run_query(sql)

    def table_dataframe(self, table=None, select_col=None, where_cond=None):
        """Return Pandas dataframe of table"""
        if table is None:
            table = self.table
        if select_col is None:
            select_col = "*"
        sql = f"""SELECT {select_col} FROM {table}"""

        if where_cond:
            sql = sql + f""" WHERE {where_cond} """

        self.cur.execute(sql)
        column_names = [desc[0] for desc in self.cur.description]
        LOGGER.info(
            f"Constructing dataframe for {self.table} with columns "
            f"{select} and where condition {where_cond}..."
        )
        return pd.DataFrame(self.cur.fetchall(), columns=column_names)

 
# ------------------------------------------------------------------------------
# NAIS CLASSES
# ------------------------------------------------------------------------------
class Points_Table(Postgres_Table):

    """
    All processed point data from MarineCadastre
    """

    def __init__(self, table):
        """
        Connect to default database and set table schema.
        
        Args:
            table (string): Name of table.
        """
        super(Points_Table, self).__init__(table)
        self.epsg = 32610
        self.columns = """
            MMSI integer NOT NULL,
            Trip integer NOT NULL,
            DateTime timestamptz NOT NULL,
            LAT float8 NOT NULL,
            LAT_UTM float8 NOT NULL,
            LON float8 NOT NULL,
            LON_UTM float8 NOT NULL,
            SOG float(4) NOT NULL,
            COG float(4) NOT NULL,
            Heading float(4),
            Step_Distance float(4),
            Step_Azimuth float(4), 
            Acceleration float(4),
            Alteration float(4), 
            Alteration_Degrees float(4), 
            Alteration_Cosine float(4), 
            VesselName varchar(32),
            VesselType varchar(64),
            Status varchar(64),
            Length float(4),
            PRIMARY KEY(MMSI, Trip, DateTime)
        """

    def add_geometry(self):
        """
        PostGIS PointM geometry to the database.
        """
        self.add_column("geom", datatype="POINTM", geometry=True)
        self.add_point("geom", "lon", "lat", "date_part('epoch', datetime)")
        self.project_column("geom", "POINTM", self.epsg)

    def add_tss(self, tss):
        """
        Add column marking whether the point is in the TSS.

        Args:
            tss (postgres_table): table representing the TSS
        """
        name = "in_tss"
        self.add_column(name, datatype="boolean", default="FALSE")
        sql = f"""
            UPDATE {self.table}
            SET {name} = TRUE
            FROM (
                SELECT points.lat, points.lon
                FROM {self.table} AS points
                RIGHT JOIN {tss} AS polygons
                ON ST_Contains(polygons.geom, points.geom)
            ) as tssTest
            WHERE {self.table}.lat=tssTest.lat
            AND {self.table}.lon=tssTest.lon
        """
        self.run_query(sql)

    def plot_tss(self):
        """Plot count of in versus out of TSS by vessel type"""
        df = self.table_dataframe()
        sns.countplot("in_tss", data=df, palette="Paired", hue="VesselType")
        save_plot(join(PLOTS_DIR, "TSS"))
        plt.close(fig)


class Tracks_Table(Postgres_Table):

    """
    Tracks constructed from MMSI, Trip, DateTime
    """

    def __init__(self, table):
        """
        Connect to default database.
        
        Args:
            table (string): Name of table.
        """
        super(Tracks_Table, self).__init__(table)
        self.cur = self.conn.cursor()      

    def convert_to_tracks(self, points):
        """Add LINESTRING for each MMSI, TrackID."""
        LOGGER.info("Creating tracks from points...")
        sql = f"""
            CREATE TABLE {self.table} AS
            SELECT
                MMSI,
                Trip,
                VesselType,
                max(DateTime) - min(DateTime) AS duration,
                tstzrange(min(DateTime), max(DateTime)) AS period,
                ST_MakeLine(geom ORDER BY DateTime) AS track
            FROM {points}
            GROUP BY MMSI, Trip, VesselType
            HAVING COUNT(*) > 2
        """
        self.run_query(sql)

    def add_length(self):
        """Add track length"""
        self.add_column("length", "float(4)", geometry=False)
        sql = f"""
            UPDATE {self.table}
            SET length = ST_Length(track)
        """
        self.run_query(sql)


class CPA_Table(Postgres_Table):

    """
    Closest point of approach calculated between each set of concurrent
    tracks
    """

    def __init__(self, table, input_table, shore_table):
        """
        Connect to default database.
        
        Args:
            table (string): Name of table
            input_table (string): Name of tracks table
            shore_table (string): Name of shore table
        """
        super(CPA_Table, self).__init__(table)
        self.cur = self.conn.cursor()
        self.input = input_table
        self.shore = shore_table

    @time_this
    def tracks_tracks(self):
        """Make pairs of tracks that happen in same time interval."""
        sql = f"""
            CREATE TABLE {self.table} AS
            SELECT
                t1.mmsi AS mmsi_1,
                t1.vesseltype AS type_1,
                t1.trip AS trip_1,
                t1.track AS track_1,
                t2.mmsi AS mmsi_2,
                t2.vesseltype AS type_2,
                t2.trip AS trip_2,
                t2.track AS track_2,
                ST_ClosestPointOfApproach(t1.track, t2.track) AS cpa_epoch,
                ST_DistanceCPA(t1.track, t2.track) AS cpa_distance
            FROM {self.input} t1 LEFT JOIN {self.input} t2
            ON t1.period && t2.period
            AND t1.mmsi != t2.mmsi
        """
        LOGGER.info(f"Joining {self.input} with itself to make {self.table}...")
        self.run_query(sql)

    @time_this
    def cpa_time(self):
        """Add time when CPA occurs"""
        col = "cpa_time"
        self.add_column(col, "TIMESTAMP")
        sql = f"""
            UPDATE {self.table}
            SET {col} = to_timestamp(cpa_epoch)::timestamptz
        """
        LOGGER.info(f"Updating column {col}...")
        self.run_query(sql)

    @time_this
    def cpa_points(self):
        """Add track points where CPA occurs"""
        for i in [1, 2]:
            col = f"cpa_point_{i}"
            self.add_column(col, "POINTM", geometry=True)
        sql = f"""
            UPDATE {self.table}
                SET {col} = ST_Force3DM(
                    ST_GeometryN(
                        ST_LocateAlong(track_{i}, cpa_epoch),
                        1
                    )
                    )
        """
            LOGGER.info(f"Updating column {col}...")
            self.run_query(sql)
    
    @time_this
    def cpa_line(self):
        """Add line between CPA points"""
        col = "cpa_line"
        self.add_column(col, "LINESTRINGM", geometry=True)
        sql = f"""
            UPDATE {self.table}
            SET {col} = ST_MakeLine(cpa_point_1, cpa_point_2)
        """
        LOGGER.info(f"Updating column {col}...")
        self.run_query(sql)

    @time_this
    def delete_shore_cross(self):
        """Delete CPAs that line on shore."""
        LOGGER.info("Deleting shore intersects from {0}...".format(self.table))
        sql = f"""
            DELETE FROM {self.table} c
            USING {self.shore} s
            WHERE ST_Intersects(c.cpa_line, s.geom)
        """
        self.cur.execute(sql)
        self.conn.commit()


# class CPA_Table(Postgres_Table):

#     def __init__(self, conn, table, input_table):
#         '''Connect to default database.'''
#         super(CPA_Table, self).__init__(conn, table)
#         self.cur = self.conn.cursor()
#         self.input = input_table

#     def points_points(self):
#         '''Join points to itself to get closest points.'''
#         print('Joining points to points to find CPAs...')
#         sql = """
#             CREATE TABLE {table} AS
#             SELECT
#                 p.basedatetime as datetime,
#                 p.mmsi AS mmsi1,
#                 p.track AS track1,
#                 p.step_cog_degrees AS step_cog1,
#                 p.cog_cosine AS cog_cos1,
#                 p.step_acceleration AS accel1,
#                 p.point_cog AS cog1,
#                 p.heading AS heading1,
#                 p.sog AS sog1,
#                 p.vesseltype AS type1,
#                 p.length AS length1,
#                 p.in_tss AS point_tss1,
#                 p.geom AS point1,
#                 p2.mmsi AS mmsi2,
#                 p2.track AS track2,
#                 p2.step_cog_degrees AS step_cog2,
#                 p2.cog_cosine AS cog_cos2,
#                 p2.step_acceleration AS accel2,
#                 p2.point_cog AS cog2,
#                 p2.heading AS heading2,
#                 p2.sog AS sog2,
#                 p2.vesseltype AS type2,
#                 p2.length AS length2,
#                 p2.in_tss AS point_tss2,
#                 p2.geom AS point2,
#                 ST_Distance(p.geom, p2.geom)::int AS distance,
#                 min(ST_Distance(p.geom, p2.geom)::int) OVER (PARTITION BY p.mmsi, p.track, p2.mmsi, p2.track) AS cpa_distance,
#                 normalize_angle(angle_difference(DEGREES(ST_Azimuth(p.geom, p2.geom)), p.heading),0,360) AS bearing12,
#                 ROW_NUMBER() OVER (PARTITION BY p.mmsi, p.track, p2.mmsi, p2.track ORDER BY p2.basedatetime ASC) as rownum
#             FROM {points} AS p LEFT JOIN {points} AS p2
#             ON p.basedatetime = p2.basedatetime
#             AND p.mmsi != p2.mmsi
#         """.format(table=self.table, points=self.input)
#         self.cur.execute(sql)
#         self.conn.commit()
#
#     def add_duplicate_rank(self):
#         '''Rank duplicate interactions.'''
#         rank = 'rank'
#         self.add_column(rank, 'integer')
#
#         sql = """
#             UPDATE {table}
#             SET {rank} = CASE
#             WHEN mmsi1::int > mmsi2::int THEN 1 ELSE 2 END
#         """.format(table=self.table, rank=rank)
#         self.cur.execute(sql)
#         self.conn.commit()
#
#     def cpa_range(self, buffer=10):
#         '''Keep only the 20 points around the CPA.'''
#         sql = """
#             WITH current AS (
#                 SELECT mmsi1, track1, mmsi2, track2, rownum
#                 FROM {table}
#                 WHERE distance = cpa_distance
#             )
#
#             DELETE FROM {table}
#             USING current
#             WHERE {table}.mmsi1 = current.mmsi1
#             AND {table}.track1 = current.track1
#             AND {table}.mmsi2 = current.mmsi2
#             AND {table}.track2 = current.track2
#             AND {table}.rownum BETWEEN current.rownum - {buffer} AND current.rownum + {buffer}
#         """.format(table=self.table, buffer=buffer)
#         self.cur.execute(sql)
#         self.conn.commit()
#
#     def cpa_attributes(self, buffer=10):
#         '''Keep only the 20 points around the CPA.'''
#         time = 'cpa_time'
#         self.add_column(time, 'timestamp')
#
#         point1 = 'cpa_point1'
#         self.add_column(point1, datatype='pointm', geometry=True)
#
#         point2 = 'cpa_point2'
#         self.add_column(point2,  datatype='pointm', geometry=True)
#         sql = """
#             WITH current AS (
#                 SELECT mmsi1, track1, mmsi2, track2, rownum
#                 FROM {table}
#                 WHERE distance = cpa_distance
#             )
#             UPDATE {table}
#             SET
#                 {time} = {table}.datetime,
#                 {p1} = {table}.point1,
#                 {p2} = {table}.point2
#             FROM current
#             WHERE {table}.mmsi1 = current.mmsi1
#             AND {table}.track1 = current.track1
#             AND {table}.mmsi2 = current.mmsi2
#             AND {table}.track2 = current.track2
#             AND {table}.rownum = current.rownum
#         """.format(table=self.table, time=time, p1=point1, p2=point2)
#         self.cur.execute(sql)
#         self.conn.commit()
#
#     def encounter_type(self):
#         '''Add type of interaction.'''
#         conType = 'encounter'
#         self.add_column(conType, datatype='varchar')
#
#         sql = """
#             WITH first AS (
#                 SELECT *
#                 FROM {table}
#                 WHERE (mmsi1, track1, mmsi2, track2) IN (
#                     SELECT mmsi1, track1, mmsi2, track2
#                     FROM (
#                         SELECT
#                             mmsi1,
#                             track1,
#                             mmsi2,
#                             track2,
#                             ROW_NUMBER() OVER(PARTITION BY mmsi1, track1, mmsi2, track2 ORDER BY datetime ASC) as rk
#                         FROM {table}
#                     ) AS subquery
#                 WHERE rk = 1
#                 )
#             )
#
#             UPDATE {table}
#             SET {type} = CASE
#                 WHEN @first.cogdiff12 BETWEEN 165 AND 195 THEN 'head-on'
#                 WHEN @first.cogdiff12 < 15 OR @first.cogdiff12 > 345 THEN 'overtaking'
#                 ELSE 'crossing'
#                 END
#             FROM first
#             WHERE first.mmsi1 = {table}.mmsi1
#             AND first.track1 = {table}.track1
#             AND first.mmsi2 = {table}.mmsi2
#             AND first.track2 = {table}.track2
#         """.format(table=self.table, type=conType)
#         self.cur.execute(sql)
#         self.conn.commit()
#
#     def check_nearby(self):
#         '''Delete encounters that are just nearby and not really encounter.'''
#         print('Deleting near ships with no encounter...')
#         sql = """
#             DELETE FROM {0}
#             WHERE (
#                 (encounter = 'head-on' OR encounter = 'overtaking')
#                 AND (
#                     90 NOT BETWEEN min(bearing12::int) and max(bearing12::int) OR
#                     270 NOT BETWEEN min(bearing12::int) and max(bearing12::int)
#             ) OR (
#                 (encounter = 'crossing')
#                 AND (
#                     0 NOT BETWEEN min(bearing12::int)-5 and min(bearing12::int) +5 OR
#                     360 NOT BETWEEN max(bearing12::int)-5 and max(bearing12::int) +5 OR
#                     180 NOT BETWEEN min(bearing12::int) and max(bearing12::int)
#             )
#         """.format(self.table)
#         self.cur.execute(sql)
#         self.conn.commit()
#



    # int4range(
    #      min(normalize_angle(angle_difference(DEGREES(ST_Azimuth(p.geom, p2.geom)), p.heading),0,360)) OVER (PARTITION BY p.mmsi, p.track, p2.mmsi, p2.track)::int,
    #      max(normalize_angle(angle_difference(DEGREES(ST_Azimuth(p.geom, p2.geom)), p.heading),0,360)) OVER (PARTITION BY p.mmsi, p.track, p2.mmsi, p2.track)::int) AS brange,








class Encounters_Table(Postgres_Table):

    def __init__(self, table, input_points, input_cpa):
        """Connect to default database"""
        super(Encounters_Table, self).__init__(table)
        self.cur = self.conn.cursor()
        self.points = input_points
        self.cpa = input_cpa

    @time_this
    def cpa_points(self):
        """Make pairs of tracks that happen in same time interval"""
        LOGGER.info("Joining points with cpas to make encounter table...")
        sql = f"""
            CREATE TABLE {self.table} AS
            SELECT
                c.mmsi_1,
                c.trip_1,
                c.type_1,
                c.mmsi_2,
                c.trip_2,
                c.type_2,
                p.datetime,
                c.cpa_distance,
                ST_Distance(p.geom, p2.geom) AS distance_12,
                normalize_angle(angle_difference(DEGREES(ST_Azimuth(p.geom, p2.geom)), p.heading),0,360) AS bearing_12,
                angle_difference(p.heading, p2.heading) AS heading_diff_12,
                int4range(
                    min(normalize_angle(angle_difference(DEGREES(ST_Azimuth(p.geom, p2.geom)), p.heading),0,360)) OVER (PARTITION BY c.mmsi_1, c.trip_1, c.mmsi_2, c.trip_2)::int,
                    max(normalize_angle(angle_difference(DEGREES(ST_Azimuth(p.geom, p2.geom)), p.heading),0,360)) OVER (PARTITION BY c.mmsi_1, c.trip_1, c.mmsi_2, c.trip_2)::int) AS brange,
                max(abs(p.alteration_degrees))
                    OVER (PARTITION BY c.mmsi_1, c.trip_1, c.mmsi_2, c.trip_2) AS alteration_max_1,
                max(abs(p2.alteration_degrees)) 
                    OVER (PARTITION BY c.mmsi_1, c.trip_1, c.mmsi_2, c.trip_2) AS alteration_max_2, 
                AVG(p.alteration_cosine) 
                    OVER (PARTITION BY c.mmsi_1, c.trip_1, c.mmsi_2, c.trip_2) AS sinuosity_1,
                AVG(p2.alteration_cosine) 
                    OVER (PARTITION BY c.mmsi_1, c.trip_1, c.mmsi_2, c.trip_2) AS sinuosity_2,  
                c.cpa_point_1,
                c.cpa_point_2,
                c.cpa_time,  
                p.geom as point_1,
                p.sog as sog_1,
                p.heading as heading_1,
                p.cog as cog_1,
                p.acceleration as acceleration_1,
                p.alteration_degrees as alteration_degrees_1,
                p.alteration_cosine as alteration_cosine_1,
                p.in_tss as tss_1,
                p2.geom as point_2,
                p2.sog as sog_2,
                p2.heading as heading_2,
                p2.cog as cog_2,
                p2.acceleration as acceleration_2,
                p2.alteration_degrees as alteration_degrees_2,
                p2.alteration_cosine as alteration_cosine_2,      
                p2.in_tss as tss_2,
                min(p2.datetime) 
                    OVER (PARTITION BY c.mmsi_1, c.trip_1, c.mmsi_2, c.trip_2) AS period_start,
                max(p2.datetime) 
                    OVER (PARTITION BY c.mmsi_1, c.trip_1, c.mmsi_2, c.trip_2) AS period_end
            FROM {self.cpa} c
            LEFT JOIN {self.points} p
                ON p.datetime between c.cpa_time - INTERVAL '10 minutes' AND c.cpa_time + INTERVAL '10 minutes'
                AND p.mmsi = c.mmsi_1
            INNER JOIN {self.points} p2
                ON p2.mmsi = c.mmsi_2
                AND p2.datetime = p.datetime
        """
        self.run_query(sql)
    
    @time_this
    def plot_domain_type(self, tss=None):
        """Plot CPA ship domain"""
        if tss is None:
            self.df = self.table_dataframe()
            name = "Ship_Domain"
        if tss:
            self.df = self.table_dataframe(
                where_cond="tss_1 = True and tss_2 = True"
            )
            name = "Ship_Domain_Inside_TSS"
        elif tss is False:
            self.df = self.table_dataframe(
                where_cond="cpa_distance < 3000 and tss_1 = False and tss_2 = False"
            )
            name = "Ship_Domain_Outside_TSS"

        self.df["r"] = np.deg2rad(df["bearing_12"])
        try:
            g = sns.FacetGrid(
                self.df,
                col="type_2",
                row="type_1",
                subplot_kws=dict(projection="polar"),
                sharex=False,
                sharey=False,
                despine=False,
                gridspec_kws={"wspace": 1.5},
                # height=5
            )
        except Exception as err:
            LOGGER.error(err)
            
        g.map(sns.scatterplot, "r", "cpa_distance", marker=".", s=30)
       
        # Limit to lower upper triangle of grid
        g.axes[1, 2].set_visible(False)
        g.axes[2, 1].set_visible(False)
        g.axes[2, 2].set_visible(False)

        # Set north to 0 degrees
        for axes in g.axes.flat:
            axes.set_theta_zero_location("N")
            axes.set_theta_direction(-1)
            axes.title.set_position([0.5, 1.1])
            axes.yaxis.labelpad = 25

        save_plot(join(PLOTS_DIR, name), tight=False)
        plt.close(fig)

    @time_this
    def plot_domain_encounter(self, tss=None):
        """Plot CPA ship domain limited to 3nm"""
        if tss is None:
            df = self.table_dataframe(where_cond="cpa_distance < 5556")
            name = "Ship_Domain_by_Encounter"

        if tss:
            df = self.table_dataframe(
                where_cond="cpa_distance < 5556 and tss_1 = True and tss_2 = True"
            )
            name = "Ship_Domain_Inside_TSS_By_Encounter"

        elif tss is False:
            df = self.table_dataframe(
                where_cond="cpa_distance < 3000 and tss_1 = False and tss_2 = False"
            )
            name = "Ship_Domain_Outside_TSS_By_Encounter"

        df["r"] = np.deg2rad(df["bearing_12"])

        fig = plt.figure(figsize=(20, 20))
        g = sns.FacetGrid(
            df, 
            row="encounter",
            subplot_kws=dict(projection="polar"),
            sharex=False, 
            sharey=False, 
            despine=False,
            gridspec_kws={"wspace": 1.5},
            # height=5
        )
        g.map(
            sns.scatterplot, "r", "cpa_distance", marker=".", s=30,
        )
    
        g.axes[1, 2].set_visible(False)
        g.axes[2, 1].set_visible(False)
        g.axes[2, 2].set_visible(False)

        for axes in g.axes.flat:
            axes.set_theta_zero_location("N")
            axes.set_theta_direction(-1)
            # axes.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        save_plot(join(PLOTS_DIR, name), tight=False)

    @time_this
    def encounter_type(self):
        """Add type of interaction"""
        conType = "encounter"
        self.add_column(conType, datatype="varchar")

        sql = f"""
            WITH first AS (
                SELECT *
                FROM {self.table}
                WHERE (mmsi_1, trip_1, mmsi_2, trip_2) IN (
                    SELECT mmsi_1, trip_1, mmsi_2, trip_2
                    FROM (
                        SELECT
                            mmsi_1,
                            trip_1,
                            mmsi_2,
                            trip_2,
                            ROW_NUMBER() OVER(PARTITION BY mmsi_1, trip_1, mmsi_2, trip_2 ORDER BY datetime ASC) as rk
                        FROM {self.table}
                    ) AS subquery
                WHERE rk = 1
                )
            )

            UPDATE {self.table}
            SET {conType} = CASE
                WHEN @first.heading_diff_12 BETWEEN 165 AND 195 THEN 'head-on'
                WHEN @first.heading_diff_12 < 15 OR @first.heading_diff_12 > 345 THEN 'overtaking'
                ELSE 'crossing'
                END
            FROM first
            WHERE first.mmsi_1 = {self.table}.mmsi_1
            AND first.trip_1 = {self.table}.trip_1
            AND first.mmsi_2 = {self.table}.mmsi_2
            AND first.trip_2 = {self.table}.trip_2
        """
        self.run_query(sql)

    @time_this
    def giveway_info(self):
        """Add GW, SO info"""
        vessel = "ship_1_give_way"
        self.add_column(vessel, datatype="integer")

        sql = f"""
            WITH first AS (
                SELECT *
                FROM {self.table}
                WHERE (mmsi_1, trip_1, mmsi_2, trip_2) IN (
                    SELECT mmsi_1, trip_1, mmsi_2, trip_2
                    FROM (
                        SELECT
                            mmsi_1,
                            trip_1,
                            mmsi_2,
                            trip_2,
                            ROW_NUMBER() OVER(PARTITION BY mmsi_1, trip_1, mmsi_2, trip_2 ORDER BY datetime ASC) as rk
                        FROM {self.table}
                    ) AS subquery
                WHERE rk = 1
                )
            )

            UPDATE {self.table}
            SET
                {vessel} = CASE
                    WHEN {self.table}.encounter = 'overtaking' THEN CASE
                        WHEN first.bearing_12 BETWEEN 90 AND 270 THEN 0
                        ELSE 1
                        END
                    WHEN {self.table}.encounter = 'crossing' THEN CASE
                        WHEN first.bearing_12 BETWEEN 0 AND 112.5 THEN 1
                        ELSE 0
                        END
                    WHEN {self.table}.encounter = 'head-on' THEN 1
                    END
            FROM first
            WHERE first.mmsi_1 = {self.table}.mmsi_1
            AND first.trip_1 = {self.table}.trip_1
            AND first.mmsi_2 = {self.table}.mmsi_2
            AND first.trip_2 = {self.table}.trip_2
        """
        self.run_query(sql)

    @time_this
    def dcpa(self):
        """Add distance to CPA"""
        name1 = "dcpa_1"
        name2 = "dcpa_2"

        self.add_column(name1, datatype="float(4)")
        self.add_column(name2, datatype="float(4)")

        sql = """
            UPDATE {0}
            SET {1} = ST_Distance({2}, {3})
        """
        self.cur.execute(sql.format(self.table, name1, "point_1", "cpa_point_1"))
        self.cur.execute(sql.format(self.table, name2, "point_2", "cpa_point_2"))
        self.conn.commit()

    @time_this
    def tcpa(self):
        """Add time to CPA"""
        name = "tcpa"
        self.add_column(name, datatype="float(4)")
        sql = f"""
            UPDATE {self.table}
            SET {name} = EXTRACT(MINUTE FROM (datetime - cpa_time))
        """
        self.run_query(sql)

    @time_this
    def time_range(self):
        """Add time period of interaction"""
        LOGGER.info("Adding time period to {0}".format(self.table))
        period = "period"
        duration = "duration"

        self.add_column(period, datatype="tsrange")
        self.add_column(duration, datatype="time")

        sql = f"""
            UPDATE {self.table}
            SET
                {period} = tstzrange(period_start, period_end),
                {duration} = period_end - period_start
        """
        self.run_query(sql)

    def traffic(self):
        """Add the number of other vessel's the ship is interacting with"""
        name = "traffic"
        self.add_column(name, datatype="integer")
        sql = f"""
            WITH count AS (
                SELECT
                    mmsi_1,
                    trip_1,
                    period,
                    COUNT (DISTINCT mmsi_2) AS traffic
                FROM {self.table}
                GROUP BY mmsi_1, track1, period
            )

            UPDATE {self.table}
            SET {name} = count.traffic
            FROM count
            WHERE {self.table}.mmsi_1 = count.mmsi_1
            AND {self.table}.track1 = count.trip_1
            AND {self.table}.period && count.period
        """
        self.run_query(sql)

    def plot_encounters(self):

        df = self.table.table_dataframe()
        for name, group in df.groupby(['mmsi1','track1','mmsi2','track2']):
            name = 'Plot_{0}_{1}_{2}_{3}.png'.format()
            fig = plt.figure(figsize=(12, 4), dpi=None, facecolor='white')
            fig.suptitle('Track Comparison', fontsize=12, fontweight='bold', x=0.5, y =1.01)
            plt.title(' => '.join([str(i) for i in name]), fontsize=10, loc='center')
            # plt.yticks([])

# ax1 = fig.add_subplot(111)
# ax1.set_ylabel('COG')
# plt.plot('tcpa', 'cog1', data=temp, marker='o', markerfacecolor='blue', markersize=5)
# plt.plot('tcpa', 'cog2', data=temp, marker='o', color='red', linewidth=2)
# plt.legend()
# ax1.grid(b=True, which='major', color='grey', linestyle='dotted')
#
# ax2 = fig.add_subplot(211)
# ax2.set_ylabel('SOG')
# plt.plot('tcpa', 'sog1', data=temp, marker='o', markerfacecolor='blue', markersize=5)
# plt.plot('tcpa', 'sog2', data=temp, marker='o', color='red', linewidth=2)
# plt.legend()
# ax2.grid(b=True, which='major', color='grey', linestyle='dotted')
#
# ax3 = fig.add_subplot(311)
# ax3.set_ylabel('Distance to CPA')
# plt.plot('tcpa', 'dcpa1', data=temp, marker='o', markerfacecolor='blue', markersize=5)
# plt.plot('tcpa', 'dcpa2', data=temp, marker='o', color='red', linewidth=2)
# plt.legend()
# ax3.grid(b=True, which='major', color='grey', linestyle='dotted')
#
# ax4 = fig.add_subplot(411)
# ax4.set_ylabel('Bearing from Ship 1 to Ship 2')
# plt.plot('tcpa', 'bearing12', data=temp, marker='o', markerfacecolor='black', markersize=5)
# plt.legend()
# ax4.grid(b=True, which='major', color='grey', linestyle='dotted')
#
#             fig.savefig(name, format="png", dpi=100, bbox_inches='tight', pad_inches=0.4)








# class Crossing_Table(Postgres_Table):
#
#     def __init__(self, conn, table, input_analysis):
#         '''Connect to default database.'''
#         super(Crossing_Table, self).__init__(conn, table)
#         self.cur = self.conn.cursor()
#         self.input = input_analysis
#         self.colregs = '{0}_colregs'.format(self.table)
#         self.others = '{0}_others'.format(self.table)
#
#     def select_type(self):
#         '''Select only the crossing interactions.'''
#         print('Selecting crossing interactions...')
#         sql = """
#             CREATE TABLE {0} AS
#             SELECT
#                 *,
#                 avg(stand_on_cog_cos) OVER (PARTITION BY mmsi1, track1, mmsi2, track2) AS avg_cog_cos,
#                 avg(stand_on_accel) OVER (PARTITION BY mmsi1, track1, mmsi2, track2) AS avg_accel
#             FROM {1}
#             WHERE encounter = 'crossing'
#         """.format(self.table, self.input)
#         self.cur.execute(sql)
#         self.conn.commit()
#
#     def separate_colregs(self):
#         '''Create table of colregs compliant interactions.'''
#         print('Selecting colreg compliant interactions...')
#         sql_colregs = """
#             CREATE TABLE {0} AS
#             SELECT *
#             FROM {1}
#             WHERE avg_cog_cos >= 0.999
#             AND avg_accel <= abs(10)
#         """.format(self.colregs, self.table)
#         self.cur.execute(sql_colregs)
#         self.conn.commit()
#
#         print('Selecting colreg compliant interactions...')
#         sql_others = """
#             CREATE TABLE {0} AS
#             SELECT *
#             FROM {1}
#             WHERE avg_cog_cos < 0.999
#             AND avg_accel > abs(10)
#         """.format(self.others, self.table)
#         self.cur.execute(sql_others)
#         self.conn.commit()
#
# class Overtaking_Table(Postgres_Table):
#
#     def __init__(self, conn, table, input_analysis):
#         '''Connect to default database.'''
#         super(Overtaking_Table, self).__init__(conn, table)
#         self.cur = self.conn.cursor()
#         self.input = input_analysis
#         self.colregs = '{0}_colregs'.format(self.table)
#         self.others = '{0}_others'.format(self.table)
#
#     def select_type(self):
#         '''Select only the overtaking interactions.'''
#         print('Selecting overtaking interactions...')
#         sql = """
#             CREATE TABLE {0} AS
#             SELECT
#                 *,
#                 avg(stand_on_cog_cos) OVER (PARTITION BY mmsi1, track1, mmsi2, track2) AS avg_cog_cos,
#                 avg(stand_on_accel) OVER (PARTITION BY mmsi1, track1, mmsi2, track2) AS avg_accel
#             FROM {1}
#             WHERE encounter = 'overtaking'
#         """.format(self.table, self.input)
#         self.cur.execute(sql)
#         self.conn.commit()
#
#     def separate_colregs(self):
#         '''Create table of colregs compliant interactions.'''
#         print('Selecting colreg compliant interactions...')
#         sql_colregs = """
#             CREATE TABLE {0} AS
#             SELECT *
#             FROM {1}
#             WHERE avg_cog_cos >= 0.999
#             AND avg_accel <= abs(10)
#         """.format(self.colregs, self.table)
#         self.cur.execute(sql_colregs)
#         self.conn.commit()
#
#         print('Selecting colreg compliant interactions...')
#         sql_others = """
#             CREATE TABLE {0} AS
#             SELECT *
#             FROM {1}
#             WHERE avg_cog_cos < 0.999
#             AND avg_accel > abs(10)
#         """.format(self.others, self.table)
#         self.cur.execute(sql_others)
#         self.conn.commit()
#
#

#
#
# # -------------def build_grid(self):
#     # '''Create grid table.'''
#     # print('Constructing grid table...')
#     # self.grid_df = dataframes.Sector_Dataframe(
#     #     self.lonMin,
#     #     self.lonMax,
#     #     self.latMin,
#     #     self.latMax,
#     #     self.stepSize
#     # ).generate_df()
#     # self.grid_csv = join(self.root, 'grid_table.csv')
#     # self.grid_df.to_csv(self.grid_csv, index=True, header=False)
#     # self.table_grid.drop_table()
#     #
#     # self.table_grid.create_table()
#     # self.table_grid.copy_data(self.grid_csv)
#     # self.table_grid.add_points()
#     # self.table_grid.make_bounding_box()----------------------------------------------------------------
#
# # class Near_Table(Postgres_Table):
# #
# #     def __init__(self, conn, table, input_table):
# #         '''Connect to default database.'''
# #         super(Near_Table, self).__init__(conn, table)
# #         self.cur = self.conn.cursor()
# #         self.input = input_table
# #
# #         self.columns =  [
# #             'own_mmsi',
# #             'own_sectorid',
# #             'own_sog',
# #             'own_heading',
# #             'own_rot',
# #             'own_length',
# #             'own_vesseltype',
# #             'own_tss',
# #             'target_mmsi',
# #             'target_sectorid',
# #             'target_sog',
# #             'target_heading',
# #             'target_rot',
# #             'target_vesseltype',
# #             'target_tss',
# #             'azimuth_deg',
# #             'point_distance',
# #             'bearing'
# #         ]
# #         self.columnString =  ', '.join(self.columns[:-1])
# #
# #     def near_points(self):
# #         '''Make pairs of points that happen in same sector and time interval.'''
# #         print('Joining nais_points with itself to make near table...')
# #         sql = """
# #             CREATE TABLE {0} AS
# #             SELECT
# #                 n1.mmsi AS own_mmsi,
# #                 n1.sectorid AS own_sectorid,
# #                 n1.trackid AS own_trackid,
# #                 n1.sog AS own_sog,
# #                 n1.cog AS own_cog,
# #                 n1.heading AS own_heading,
# #                 n1.rot AS own_rot,
# #                 n1.vesseltype AS own_vesseltype,
# #                 n1.length AS own_length,
# #                 n1.geom AS own_geom,
# #                 n1.in_tss AS own_tss,
# #                 n2.mmsi AS target_mmsi,
# #                 n2.sectorid AS target_sectorid,
# #                 n2.trackid AS target_trackid,
# #                 n2.sog AS target_sog,
# #                 n2.cog AS target_cog,
# #                 n2.heading AS target_heading,
# #                 n2.rot AS target_rot,
# #                 n2.vesseltype AS target_vesseltype,
# #                 n2.length AS target_length,
# #                 n2.geom AS target_geom,
# #                 n2.in_tss AS target_tss,
# #                 ST_Distance(ST_Transform(n1.geom, 7260), ST_Transform(n2.geom, 7260)) AS point_distance,
# #                 DEGREES(ST_Azimuth(n1.geom, n2.geom)) AS azimuth_deg
# #             FROM {1} n1 INNER JOIN {2} n2
# #             ON n1.sectorid = n2.sectorid
# #             WHERE n1.basedatetime = n2.basedatetime
# #             AND n1.mmsi != n2.mmsi
# #         """.format(self.table, self.input, self.input)
# #         self.cur.execute(sql)
# #         self.conn.commit()
# #
# #     def near_points_dataframe(self, max_distance):
#         '''Return dataframe of near points within max distance.'''
#         cond = 'point_distance <= {0}'.format(max_distance)
#         return self.table_dataframe(self.columnString, cond)
#
#     def near_table(self, max_distance):
#         '''Create near table in pandas using max_distance = 1nm.'''
#         # 1nm = 1852 meters
#         df = self.near_points_dataframe(max_distance)
#         df['bearing'] = (df['azimuth_deg'] - df['own_heading']) % 360
#         return df[self.columns].copy()
#
#     def near_plot(self, max_distance, display_points):
#         '''Plot the near points all in reference to own ship.'''
#         self.df_near = self.near_table(max_distance).head(display_points)
#         self.df_near = self.df_near[
#             (self.df_near['own_sog'] >10) & (self.df_near['target_sog'] > 10)].copy()
#         theta = np.array(self.df_near['bearing'])
#         r = np.array(self.df_near['point_distance'])
#         tss = np.array(self.df_near['bearing'])
#         # colors = theta.apply(np.radians)
#         colors = tss
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='polar')
#         c = ax.scatter(theta, r, c=colors, cmap='hsv', alpha=0.75, s=1)
#         # plt.legend(loc='upper left')
#         plt.show()
#


# class Grid_Table(Postgres_Table):
#
#     def __init__(self, conn, table):
#         '''Connect to default database.'''
#         super(Grid_Table, self).__init__(conn, table)
#         self.cur = self.conn.cursor()
#
#         self.columns = """
#             SectorID char(5) PRIMARY KEY,
#             MinLon float8 NOT NULL,
#             MinLat float8 NOT NULL,
#             MaxLon float8 NOT NULL,
#             MaxLat float8 NOT NULL
#         """
#
#     def add_points(self):
#         '''Add Point geometry to the database.'''
#         self.add_column('leftBottom', datatype='POINT', geometry=True)
#         self.add_column('leftTop', datatype='POINT', geometry=True)
#         self.add_column('rightTop', datatype='POINT', geometry=True)
#         self.add_column('rightBottom', datatype='POINT', geometry=True)
#
#         print('Adding PostGIS POINTs to {0}...'.format(self.table))
#         self.add_point('leftBottom', 'minlon', 'minlat')
#         self.add_point('leftTop', 'minlon', 'maxlat')
#         self.add_point('rightTop', 'maxlon', 'maxlat')
#         self.add_point('rightBottom', 'maxlon', 'minlat')
#
#     def make_bounding_box(self):
#         '''Add polygon column in order to do spatial analysis.'''
#         print('Adding PostGIS POLYGON to {0}...'.format(self.table))
#         self.add_column('boundingbox', datatype='Polygon', geometry=True)
#         sql = """
#             UPDATE {0}
#             SET {1} = ST_SetSRID(ST_MakePolygon(
#                 ST_MakeLine(array[{2}, {3}, {4}, {5}, {6}])
#             ), 4326)
#         """.format(
#             self.table,
#             'boundingbox',
#             'leftBottom',
#             'leftTop',
#             'rightTop',
#             'rightBottom',
#             'leftBottom'
#             )
#         self.cur.execute(sql)
#         self.conn.commit()

    # def track_changes(self):
    #     '''Reorganize data into give way and stand on.'''
    #     print('Adding course info to {0}'.format(self.table))
    #     cog_gw = 'give_way_cog_cos'
    #     cog_so = 'stand_on_cog_cos'
    #     accel_gw = 'give_way_accel'
    #     accel_so = 'stand_on_accel'
    #     type_gw = 'give_way_type'
    #     type_so = 'stand_on_type'
    #
    #     self.add_column(cog_gw, datatype='float(4)')
    #     self.add_column(cog_so, datatype='float(4)')
    #     self.add_column(accel_gw, datatype='float(4)')
    #     self.add_column(accel_so, datatype='float(4)')
    #     self.add_column(type_gw, datatype='varchar')
    #     self.add_column(type_so, datatype='varchar')
    #
    #     sql = """
    #         UPDATE {table}
    #         SET
    #             {cog_gw} = CASE
    #                 WHEN ship1_give_way = 1 THEN cog_cos1 ELSE cog_cos2 END,
    #             {cog_so} = CASE
    #                 WHEN ship1_give_way = 1 THEN cog_cos2 ELSE cog_cos1 END,
    #             {accel_gw} = CASE
    #                 WHEN ship1_give_way = 1 THEN accel1 ELSE accel2 END,
    #             {accel_so} = CASE
    #                 WHEN ship1_give_way = 1 THEN accel2 ELSE accel1 END,
    #             {type_gw} = CASE
    #                 WHEN ship1_give_way = 1 THEN type1 ELSE type2 END,
    #             {type_so} = CASE
    #                 WHEN ship1_give_way = 1 THEN type2 ELSE type1 END
    #     """.format(
    #         table=self.table,
    #         cog_gw=cog_gw,
    #         cog_so=cog_so,
    #         accel_gw=accel_gw,
    #         accel_so=accel_so,
    #         type_gw=type_gw,
    #         type_so=type_so
    #     )
    #     self.cur.execute(sql)
    #     self.conn.commit()
    #

# ------------------------------------------------------------------------------
# MAIN CLASS
# ------------------------------------------------------------------------------
@time_all
class NAIS_Database(object):

    '''
    Build PostgreSQL database of NAIS point data.
    '''

    def __init__(self, city, year):
        # arguments
        self.city = city
        self.year = year
        self.password = password

        # file parameters
        self.root = tempfile.mkdtemp()
        self.nais_file = join(self.root, 'AIS_All.csv')

        # spatial parameters
        param_yaml = join(dirname(__file__), 'settings.yaml')
        with open(param_yaml, 'r') as stream:
            self.parameters = yaml.safe_load(stream)[self.city]

        self.zone = self.parameters['zone']
        self.lonMin = self.parameters['lonMin']
        self.lonMax = self.parameters['lonMax']
        self.latMin = self.parameters['latMin']
        self.latMax = self.parameters['latMax']
        self.srid = self.parameters['srid']

        # time parameters
        # self.months = [str(i).zfill(2) for i in range(1, 13)]
        self.months = ['01']

        # database parameters
        self.conn = psycopg2.connect(
            host='localhost',
            dbname='postgres',
            user='postgres',
            password=self.password)

        # Add functions to database
        sql = """
            CREATE OR REPLACE FUNCTION normalize_angle(
                angle FLOAT,
                range_start FLOAT,
                range_end FLOAT)
            RETURNS FLOAT AS $$
            BEGIN
                RETURN (angle - range_start) - FLOOR((angle - range_start)/(range_end - range_start))*(range_end - range_start) + range_start;
            END; $$
            LANGUAGE PLPGSQL;

            CREATE OR REPLACE FUNCTION angle_difference(
                angle1 FLOAT,
                angle2 FLOAT)
            RETURNS FLOAT AS $$
            BEGIN
                RETURN DEGREES(ATAN2(
                    SIN(RADIANS(angle1) - RADIANS(angle2)),
                    COS(RADIANS(angle1) - RADIANS(angle2))
                ));
            END; $$
            LANGUAGE PLPGSQL
        """
        self.conn.cursor().execute(sql)
        self.conn.commit()

        # Add SRID
        self.conn.cursor().execute(self.srid)
        self.conn.commit()

       

        # eda
        # self.table_eda = EDA_Table(
        #     self.conn,
        #     'eda_points_{0}'.format(self.city)
        # )
        # self.table_eda_tracks_mmsi = Tracks_Table(
        #     self.conn,
        #     'eda_tracks_mmsi_{0}'.format(self.city)
        # )
        # self.table_eda_tracks_trajectory = Tracks_Table(
        #     self.conn,
        #     'eda_tracks_trajectory_{0}'.format(self.city)
        # )



        # self.table_points = Points_Table(
        #     self.conn,
        #     'points_{0}'.format(self.city)
        # )
        # self.table_cpas = CPA_Table(
        #     self.conn,
        #     'cpa_{0}'.format(self.city),
        #     self.table_points.table
        # )


        # self.table_tracks = Tracks_Table(
        #     self.conn,
        #     'tracks_{0}'.format(self.city)
        # )
        # self.table_cpa = CPA_Table(
        #     self.conn,
        #     'cpa_{0}'.format(self.city),
        #     self.table_tracks.table,
        #     self.table_shore.table
        # )

        # # Encounters
        # self.table_encounters = Encounters_Table(
        #     self.conn,
        #     'encounters_{0}_cog'.format(self.city),
        #     self.table_points.table,
        #     self.table_cpa.table
        # )
        #
        #






        # self.table_crossing = Crossing_Table(
        #     self.conn,
        #     'crossing_{0}'.format(self.city),
        #     self.table_analysis.table
        # )
        # self.table_overtaking = Overtaking_Table(
        #     self.conn,
        #     'overtaking_{0}'.format(self.city),
        #     self.table_analysis.table
        # )

    @property
    def nais_csvs(self):
        return glob(self.root + '\\AIS*.csv')

    # BUILD DATABASE -----------------------------------------------------------
    # def build_tables(self):
    #     '''Build database of raw data.'''
    #     start = time.time()
    #     try:
    #         # Points
    #         # self.build_nais_points()

    #         # Tracks
    #         # self.build_nais_tracks()

    #         # CPAsdf
    #         # self.build_nais_cpas()

    #         # Analysis
    #         # self.build_nais_encounters()
    #         # self.build_nais_analysis()

    #     except Exception as err:
    #         print(err)
    #         self.conn.rollback()
    #         self.conn.close()
    #     finally:
    #         shutil.rmtree(self.root)
    #         end = time.time()
    #         print('Elapsed Time: {0} minutes'.format((end-start)/60))

  

  

    def build_nais_eda(self):
        '''Build nais exploratory data analysis table.'''
        # Download and process raw AIS data from MarineCadastre.gov
        # if not exists(self.nais_file):
        #     raw = NAIS_Download(self.root, self.city, self.year)
        #     for month in self.months:
        #         raw.download_nais(month)
        #     raw.clean_up()
        #     raw.preprocess_eda()

        # Create table
        self.table_eda.drop_table()
        self.table_eda.create_table()
        self.table_eda.copy_data(self.nais_file)
        self.table_eda.add_local_time()

        # Add postgis point and project to UTM 10 SRID
        self.table_eda.add_geometry()
        self.table_eda.project_column('geom', 'POINTM', 32610)
        self.table_eda.add_index("idx_geom", "geom", type="gist")

        # Make tracks on MMSI only and on MMSI, Trajectory
        points = self.table_eda.table
        self.table_eda_tracks_mmsi.drop_table()
        self.table_eda_tracks_mmsi.convert_to_tracks(points, groupby='mmsi, type')
        self.table_eda_tracks_trajectory.drop_table()
        self.table_eda_tracks_trajectory.convert_to_tracks(points, groupby='mmsi, trajectory, type')



    def build_nais_points(self):
        '''Build nais points table.'''
        # Create table
        self.table_points.drop_table()
        self.table_points.create_table()
        self.table_points.copy_data(self.nais_file)

        # Add postgis point and project to UTM 10 SRID
        self.table_points.add_geometry()
        self.table_points.project_column('geom', 'POINTM', 32610)
        self.table_points.add_index("idx_geom", "geom")
        self.table_points.add_tss(self.table_tss.table)

        # Add time index for join
        self.table_points.add_index("idx_time", "datetime")

    def build_nais_interactions(self):
        '''Join points to points to get CPA.'''
        self.table_cpas.drop_table()

        # Join points to points and select closest CPAs
        self.table_cpas.points_points()
        self.table_cpas.add_index("idx_distance", "cpa_distance")
        self.table_cpas.reduce_table('cpa_distance', '>=', 0.5*1852)

        # Add rank to duplicates
        self.table_cpas.add_index("idx_mmsi", "mmsi1")
        self.table_cpas.add_duplicate_rank()

        # Select the 20 points around the CPA
        self.table_cpas.cpa_range()
        self.table_cpas.cpa_attributes()

        # Add encounter type
        self.table_cpas.encounter_type()

    def build_nais_tracks(self):
        '''Create tracks table from points table.'''
        print('Constructing nais_tracks table...')
        self.table_tracks.drop_table()
        self.table_tracks.convert_to_tracks(self.table_points.table)

        self.table_tracks.add_index("idx_gist_period", "period", type="gist")
        self.table_tracks.add_index("idx_duration", "duration")

        self.table_tracks.reduce_table('duration', '<=', '00:30:00')

    def build_nais_cpas(self):
        '''Create table to generate pair-wise cpa.'''
        self.table_cpa.drop_table()

        # Join tracks pairwise
        self.table_cpa.tracks_tracks()
        self.table_cpa.remove_null('cpa_epoch')

        # Add CPA attributes
        self.table_cpa.cpa_time()
        self.table_cpa.cpa_points()
        self.table_cpa.cpa_distance()

        self.table_cpa.add_index("idx_distance", "point_distance")
        self.table_cpa.reduce_table('point_distance','>',  3*1852)

        # Remove cpas that cross the shore
        self.table_cpa.cpa_line()
        self.table_cpa.delete_shore_cross()

        # Add rank to duplicate interactions
        self.table_cpa.add_duplicate_rank()

    def build_nais_encounters(self):
        '''Add point data to cpa instances.'''
        self.table_encounters.drop_table()

        # Add point data to CPA
        self.table_encounters.cpa_points()

        # Get encounter type
        self.table_encounters.encounter_type()
        self.table_encounters.check_nearby()

        # Add encounter info
        self.table_encounters.giveway_info()
        self.table_encounters.dcpa()
        self.table_encounters.tcpa()








#         # plot every encounter cog v time, sog v time, step_cog v tcpa
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='polar')
# ax.set_rlabel_position(135)
# ax.set_theta_zero_location('N')
# ax.set_theta_direction(-1)
# c = ax.scatter(np.radians(overtaking['bearing12']), overtaking['distance'], c=np.radians(overtaking['bearing12']), cmap='hsv', alpha=0.75)
# #
# self.window_mmsi = '(PARTITION BY MMSI, Trip ORDER BY DateTime ASC)'

#     def window_column(self, name, window_query):
#         """
#         Allows column assignment using a window functions.
        
#         Args:
#             name (string): Name of column to update
#             select_query (string): Expression to apply the window to
#         """
#         sql = f"""
#             UPDATE {self.table}
#             SET {name} = temp.SubQueryColumn
#             FROM (
#                 SELECT 
#                     MMSI,
#                     Trip,
#                     DateTime,
#                     {window_query} OVER {self.window_mmsi} AS SubQueryColumn
#                 FROM {self.table}
#             ) AS temp
#             WHERE 
#                 temp.MMSI = {self.table}.MMSI AND
#                 temp.Trip = {self.table}.Trip AND
#                 temp.DateTime = {self.table}.DateTime        
#         """
#         return sql