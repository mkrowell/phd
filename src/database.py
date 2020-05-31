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
from os.path import abspath, dirname, exists, join
from postgis.psycopg import register
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import osgeo.ogr
import pandas as pd
import psycopg2
import seaborn as sns
import shutil
import statsmodels.api as sm
import subprocess
import tempfile
import time
import yaml
from textwrap import wrap

from src import time_this, LOGGER
from src.dataframe import save_plot

import warnings
warnings.filterwarnings("ignore")

PLOTS_DIR = abspath(join(dirname(__file__), "..", "reports", "figures", "7"))
sns.set_palette("Set1")
# sns.set_context("paper") 
sns.set(font_scale=1.25)  



def plot_domain_tss(df, name):
    """Plot the ship domain for the given df"""
    df["r"] = np.deg2rad(df["bearing_12"])
    g = sns.FacetGrid(
        df,
        col="target ship",
        row="ownship",
        subplot_kws=dict(projection="polar"),
        sharex=False,
        sharey=False,
        despine=False,
        gridspec_kws={"wspace": 1.5},
    )
    g.map(sns.scatterplot, "r", "cpa_distance", marker=".", s=30, alpha=0.3)

    # Limit to lower upper triangle of grid
    # g.axes[1, 2].set_visible(False)
    # g.axes[2, 1].set_visible(False)
    # g.axes[2, 2].set_visible(False)

    # Set north to 0 degrees
    for axes in g.axes.flat:
        axes.set_theta_zero_location("N")
        axes.set_theta_direction(-1)
        axes.title.set_position([0.5, 1.5])
        axes.yaxis.labelpad = 40

    for i, axes_row in enumerate(g.axes):
        for j, axes_col in enumerate(axes_row):
            row, col = axes_col.get_title().split('|')
            if i == 0:
                axes_col.set_title(col.strip())
            else:
                axes_col.set_title('')
            if j == 0:
                ylabel = axes_col.get_ylabel()
                axes_col.set_ylabel(row.strip() + ' | ' + ylabel)

    save_plot(join(PLOTS_DIR, name), tight=False)
    plt.close()


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

    def __init__(self, table, srid=4326):
        """
        Initialize connection and set srid"""
        super(Postgres_Table, self).__init__()
        self.table = table
        self.srid = srid

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
            cmd = f'"C:\\Program Files\\PostgreSQL\\12\\bin\\shp2pgsql.exe" -s {self.srid} -d {filepath} {self.table} | psql -d postgres -U postgres -q'
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

    @time_this
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

    def add_column(self, name, datatype=None, geometry=False, default=None, srid=None):
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
        if srid is None:
            srid = self.srid
        # Handle geometry types
        datatype_str = f"{datatype}"
        if geometry:
            datatype_str = f"geometry({datatype}, {srid})"

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

    def alter_storage(self, column):
        """Set storage for column to external (uncompressed)"""
        sql = f"""
            ALTER TABLE {self.table}
            ALTER COLUMN {column}
            SET STORAGE EXTERNAL;
        """
        LOGGER.info(f"Updating storage to external for column {column}...")
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

    @time_this
    def table_dataframe(self, table=None, select_col=None, where_cond=None):
        """Return Pandas dataframe of table"""
        if table is None:
            table = self.table
        if select_col is None:
            select_col = "*"
        sql = f"""SELECT {select_col} FROM {table}"""

        if where_cond:
            sql = sql + f""" WHERE {where_cond} """

        with self.conn.cursor() as cur:
            LOGGER.info(
                f"Constructing dataframe for {self.table} with columns "
                f"{select_col} and where condition {where_cond}..."
            )
            cur.execute(sql)
            column_names = [desc[0] for desc in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=column_names)


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
            DateTime timestamp NOT NULL,
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
        gid = "tss_gid"
        self.add_column(gid, datatype="integer")
        sql = f"""
            WITH tssTest AS (
                SELECT points.geom AS point_geom, polygons.gid
                FROM {self.table} AS points
                RIGHT JOIN {tss} AS polygons
                ON ST_Contains(polygons.geom, points.geom)
            )

            UPDATE 
                {self.table}
            SET 
                {name} = TRUE,
                {gid} = tssTest.gid
            FROM tssTest
            WHERE {self.table}.geom=tssTest.point_geom
        """
        self.run_query(sql)

    def plot_tss(self):
        """
        Plot count of in versus out of TSS by vessel type
        """
        df = self.table_dataframe(select_col="mmsi, datetime, in_tss, vesseltype")
        LOGGER.info(f"Plotting tss from {self.table}")
        fig, ax = plt.subplots()
        sns.countplot("in_tss", data=df, palette="Paired", hue="vesseltype")
        ax.set_ylabel("Number of Data Points")
        ax.set_xlabel("In TSS")
        save_plot(join(PLOTS_DIR, "TSS"), tight=True)
        plt.close()
        del df

    def add_terminal(self, terminals):
        """Add nearby ferry terminal"""
        name = "terminal"
        self.add_column(name, datatype='text')
        sql = f"""
            WITH temp AS (
                SELECT points.geom AS point_geom, ports.display
                FROM {self.table} AS points
                LEFT JOIN {terminals} AS ports
                ON ST_DWithin(ports.geom, points.geom, 1852)
            )

            UPDATE 
                {self.table}
            SET 
                {name} = temp.display
            FROM temp
            WHERE {self.table}.geom=temp.point_geom
        """
        self.run_query(sql)


class TSS_Intersection_Table(Postgres_Table):

    """
    All processed point data from MarineCadastre
    """

    def __init__(self, table, points_input, tss_input):
        """
        Connect to default database and set table schema.
        
        Args:
            table (string): Name of table.
        """
        super(TSS_Intersection_Table, self).__init__(table)
        self.points = points_input
        self.tss = tss_input
        self.srid = 32610

    def select_intersections(self):
        """Get last point outside and first point inside"""
        # seattle-bainbridge 52N, 55S
        # seattle-bremerton 51N, 50S
        sql = f"""
            CREATE TABLE {self.table} AS
            SELECT 
                tmp.mmsi, 
                tmp.trip, 
                tmp.vesseltype,
                tmp.vesselname,
                tmp.datetime, 
                tmp.geom AS point_geom, 
                tmp.heading, 
                tmp.in_tss,
                tss.geom AS tss_geom,
                tss.gid,
                ST_ClosestPoint(ST_Boundary(tss.geom), tmp.last_point) AS closest_point,
                ST_ClosestPoint(ST_Boundary(tss.geom), tmp.prior_point) AS nearby_point
            FROM (
                SELECT 
                    mmsi, 
                    trip,
                    vesseltype,
                    vesselname,
                    datetime,
                    geom,
                    lag(geom, 1) OVER (PARTITION BY mmsi, trip ORDER BY datetime) AS last_point,
                    lag(geom, 2) OVER (PARTITION BY mmsi, trip ORDER BY datetime) AS prior_point,
                    heading,
                    in_tss, 
                    lag(in_tss) OVER (PARTITION BY mmsi, trip ORDER BY datetime) AS last_tss,
                    lead(in_tss) OVER (PARTITION BY mmsi, trip ORDER BY datetime) AS next_tss
                FROM {self.points}
                ) AS tmp,
                {self.tss}
            WHERE tmp.vesseltype = 'ferry'
            AND (tmp.in_tss = True AND tmp.last_tss = False)
            AND ST_Within(tmp.geom, {self.tss}.geom)
            AND {self.tss}.gid IN (50, 51, 52, 55)
            ORDER BY mmsi, trip, datetime
            """
        self.run_query(sql)

    def add_direction(self):
        """Add N/S to tss"""
        col = "tss_direction"
        self.add_column(col, "char(1)")
        sql = f"""
            UPDATE {self.table}
            SET {col} = CASE
                WHEN gid IN (51, 52) THEN 'N' 
                WHEN gid IN (50, 55) THEN 'S'
            END
        """
        self.run_query(sql)

    def get_tss_heading(self):
        """Add heading of TSS points"""
        col = "tss_heading"
        self.add_column(col, "float(4)")
        # sql = f"""
        #     UPDATE {self.table}
        #     SET {col} = CASE
        #         WHEN tss_direction = 'N' THEN Degrees(ST_AZIMUTH(nearby_point, closest_point))
        #         WHEN tss_direction = 'S' THEN Degrees(ST_AZIMUTH(nearby_point, closest_point))
        #     END
        # """
        sql = f"""
            UPDATE {self.table}
            SET {col} = CASE
                WHEN gid = 52 THEN 354
                WHEN gid = 55 THEN 174
                WHEN gid = 51 THEN 346
                WHEN gid = 50 THEN 166
            END
        """
        self.run_query(sql)

    def get_entrance_angle(self):
        """Add entrance to TSS angle"""
        col = "entrance_angle"
        self.add_column(col, "float(4)")
        sql = f"""
            UPDATE {self.table}
            SET {col} = LEAST(normalize_angle(heading - tss_heading, 0, 180), 180 - normalize_angle(heading - tss_heading, 0, 180))
        """
        self.run_query(sql)

    def plot_angles(self):
        """
        Plot histogram of TSS entrance angles
        """
        df = self.table_dataframe(where_cond="gid in (52, 55) and vesselname LIKE 'WSF %'")
        LOGGER.info(f"Plotting entrance angles from {self.table}")
        fig, ax = plt.subplots()
        sns.distplot(df["entrance_angle"], kde=False)
        ax.set_ylabel("Number of Entrances")
        ax.set_xlabel("Entrance Angle")
        save_plot("TSS_Entrance_Angle", False)
        plt.close()
        del df


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
                VesselName,
                min(DateTime) AS start,
                max(DateTime) AS end,
                max(DateTime) - min(DateTime) AS duration,
                tstzrange(min(DateTime), max(DateTime)) AS period,
                ST_MakeLine(geom ORDER BY DateTime) AS track
            FROM {points}
            GROUP BY MMSI, Trip, VesselType, VesselName
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

    def add_displacement(self):
        """Add track displacement"""
        self.add_column("displacement", "float(4)", geometry=False)
        sql = f"""
            UPDATE {self.table}
            SET displacement = ST_Distance(ST_StartPoint(track), ST_EndPoint(track))
        """
        self.run_query(sql)
        self.add_column("straightness", "float(4)", geometry=False)
        sql = f"""
            UPDATE {self.table}
            SET straightness = displacement/length
        """
        self.run_query(sql)

    def add_od(self, points):
        origin = "origin"
        self.add_column(origin, datatype="text")
        dest = "destination"
        self.add_column(dest, datatype="text")
        sql = f"""
            UPDATE {self.table}
            SET  {origin} = {points}.terminal
            FROM {points}
            WHERE {self.table}.start = {points}.datetime
            AND {self.table}.trip = {points}.trip
        """
        self.run_query(sql)

        sql = f"""
            UPDATE {self.table}
            SET  {dest} = {points}.terminal
            FROM {points}
            WHERE {self.table}.end = {points}.datetime
            AND {self.table}.trip = {points}.trip
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
        self.srid = 32610

    def tracks_tracks(self):
        """Make pairs of tracks that happen in same time interval."""
        sql = f"""
            CREATE TABLE {self.table} AS
            SELECT
                t1.mmsi AS mmsi_1,
                t1.vesseltype AS type_1,
                t1.vesselname AS name_1,
                t1.trip AS trip_1,
                t1.length AS length_1,
                t1.straightness AS straight_1,
                t1.period AS period_1,
                t1.start AS start_1,
                t1.end AS end_1,
                t1.duration AS duration_1,
                t1.origin AS origin_1,
                t1.destination AS destination_1,
                t1.track AS track_1,
                t2.mmsi AS mmsi_2,
                t2.vesseltype AS type_2,
                t2.vesselname AS name_2,
                t2.trip AS trip_2,
                t2.length AS length_2,
                t2.straightness AS straight_2,
                t2.period AS period_2,
                t2.start AS start_2,
                t2.end AS end_2,
                t2.duration AS duration_2,
                t2.origin AS origin_2,
                t2.destination AS destination_2,
                t2.track AS track_2,
                ST_ClosestPointOfApproach(t1.track, t2.track) AS cpa_epoch,
                ST_DistanceCPA(t1.track, t2.track) AS cpa_distance
            FROM {self.input} t1 LEFT JOIN {self.input} t2
            ON t1.period && t2.period
            AND t1.mmsi != t2.mmsi
        """
        LOGGER.info(f"Joining {self.input} with itself to make {self.table}...")
        self.run_query(sql)

    def cpa_time(self):
        """Add time when CPA occurs"""
        col = "cpa_time"
        self.add_column(col, "TIMESTAMP")
        sql = f"""
            UPDATE {self.table}
            SET {col} = to_timestamp(cpa_epoch)::timestamp
        """
        LOGGER.info(f"Updating column {col}...")
        self.run_query(sql)

    def cpa_points(self):
        """Add track points where CPA occurs"""
        for i in [1, 2]:
            col = f"cpa_point_{i}"
            self.add_column(col, "POINTM", geometry=True, srid=32610)
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

    def delete_shore_cross(self):
        """Delete CPAs that line on shore."""
        LOGGER.info("Deleting shore intersects from {0}...".format(self.table))
        sql = f"""
            DELETE FROM {self.table} c
            USING {self.shore} s
            WHERE ST_Intersects(c.cpa_line, s.geom)
        """
        self.run_query(sql)


class Encounters_Table(Postgres_Table):

    """
    Create table with encounter information from cpa and point tables
    """

    def __init__(self, table, input_points, input_cpa):
        """Connect to default database"""
        super(Encounters_Table, self).__init__(table)
        self.cur = self.conn.cursor()
        self.points = input_points
        self.cpa = input_cpa
        self._df = None

    @property
    def df(self):
        """Return dataframe with columns necessary for reporting and plotting summary"""
        if self._df is not None:
            return self._df
        self._df = self.table_dataframe()
        self._df["cpa_time"] = self._df["cpa_time"].dt.round("1min")
        self._df["bearing_12"] = self._df["bearing_12"].round()
        self._df["r"] = np.deg2rad(self._df["bearing_12"])
        self._df.rename(
            columns = {
                "type_1": "ownship", 
                "type_2": "target ship",
                "tss_1": "ownship in TSS",
                "tss_2": "target ship in TSS"
            }, inplace=True
        )
        return self._df
            
    def cpa_points(self):
        """Make pairs of tracks that happen in same time interval"""
        LOGGER.info("Joining points with cpas to make encounter table...")
        sql = f"""
            CREATE TABLE {self.table} AS
            SELECT
                c.mmsi_1,
                c.trip_1,
                c.type_1,
                c.name_1,
                c.length_1,
                c.straight_1,
                c.duration_1,
                c.start_1,
                c.end_1,
                c.origin_1,
                c.destination_1,
                c.mmsi_2,
                c.trip_2,
                c.type_2,
                c.name_2,
                c.length_2,
                c.straight_2,
                c.duration_2,
                c.start_2,
                c.end_2,
                c.origin_2,
                c.destination_2,
                p.datetime,
                c.cpa_distance,
                ST_Distance(p.geom, p2.geom) AS distance_12,
                p.heading as heading_1,
                p2.heading as heading_2,
                normalize_angle(angle_difference(DEGREES(ST_Azimuth(p.geom, p2.geom)), p.heading),0,360) AS bearing_12,
                normalize_angle(angle_difference(p.heading, p2.heading),0,360) AS heading_diff_12,
                int4range(
                    min(normalize_angle(angle_difference(DEGREES(ST_Azimuth(p.geom, p2.geom)), p.heading),0,360)) OVER (PARTITION BY c.mmsi_1, c.trip_1, c.mmsi_2, c.trip_2)::int,
                    max(normalize_angle(angle_difference(DEGREES(ST_Azimuth(p.geom, p2.geom)), p.heading),0,360)) OVER (PARTITION BY c.mmsi_1, c.trip_1, c.mmsi_2, c.trip_2)::int) AS brange,
                max(abs(p.alteration_degrees))
                    OVER (PARTITION BY c.mmsi_1, c.trip_1, c.mmsi_2, c.trip_2) AS alteration_max_1,
                max(abs(p2.alteration_degrees)) 
                    OVER (PARTITION BY c.mmsi_1, c.trip_1, c.mmsi_2, c.trip_2) AS alteration_max_2, 
                c.cpa_point_1,
                c.cpa_point_2,
                c.cpa_time,  
                p.lat as lat_1,
                p.lon as lon_1,
                p2.lat as lat_2,
                p2.lon as lon_2,
                p.geom as point_1,
                p2.geom as point_2,
                p.sog as sog_1,
                p2.sog as sog_2,
                p.acceleration as acceleration_1,
                p2.acceleration as acceleration_2,
                p.alteration_degrees as alteration_1,
                p2.alteration_degrees as alteration_2,
                p.in_tss as tss_1,   
                p2.in_tss as tss_2,
                SUM(p.in_tss::int) OVER (PARTITION BY c.mmsi_1, c.trip_1, c.mmsi_2, c.trip_2) as track_tss_1,
                SUM(p2.in_tss::int) OVER (PARTITION BY c.mmsi_1, c.trip_1, c.mmsi_2, c.trip_2) as track_tss_2,
                p.tss_gid as gid_1,
                p2.tss_gid as gid_2
            FROM {self.cpa} c
            LEFT JOIN {self.points} p
                ON p.datetime BETWEEN c.cpa_time - INTERVAL '10 minutes' AND c.cpa_time + INTERVAL '20 minutes'
                AND p.mmsi = c.mmsi_1
                And p.trip = c.trip_1
            INNER JOIN {self.points} p2
                ON p2.mmsi = c.mmsi_2
                AND p2.trip = c.trip_2
                AND p2.datetime = p.datetime
        """
        self.run_query(sql)

    def tcpa(self):
        """Add time to CPA"""
        name = "tcpa"
        self.add_column(name, datatype="float(4)")
        sql = f"""
            UPDATE {self.table}
            SET {name} = EXTRACT(MINUTE FROM (datetime - cpa_time))
        """
        self.run_query(sql)

    def drop_sparse_encounters(self):
        """Drop encounters with less than 15 data points"""
        sql = f"""
            DELETE FROM {self.table}
            USING (
                SELECT mmsi_1, trip_1, mmsi_2, trip_2, count(datetime) AS num_rows
                FROM {self.table}
                WHERE tcpa < 1
                GROUP BY mmsi_1, trip_1, mmsi_2, trip_2
                HAVING count(datetime) < 10 
            ) sparse
            WHERE
                {self.table}.mmsi_1 = sparse.mmsi_1
                AND {self.table}.mmsi_2 = sparse.mmsi_2
                AND {self.table}.trip_1 = sparse.trip_1
                AND {self.table}.trip_2 = sparse.trip_2
        """
        LOGGER.info("Deleting sparse encounters...")
        self.run_query(sql)

    def encounter_type(self):
        """Add type of interaction"""
        conType = "encounter"
        self.add_column(conType, datatype="varchar")
        diff = "first_heading_diff"
        self.add_column(diff, datatype="varchar")

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
            
            UPDATE 
                {self.table}
            SET 
                {conType} = CASE
                    WHEN 
                        (@first.heading_diff_12 BETWEEN 165 AND 195) AND 
                        ((first.bearing_12 > 345) OR (first.bearing_12 < 15)) 
                        THEN 'head-on'
                    WHEN 
                        ((@first.heading_diff_12 < 15) OR (@first.heading_diff_12 > 345)) AND 
                        ((first.bearing_12 BETWEEN 165 AND 195) OR ((first.bearing_12 > 345) OR (first.bearing_12 < 15))) 
                        THEN 'overtaking'
                    WHEN 
                        ((@first.heading_diff_12 BETWEEN 15 AND 165) AND (first.bearing_12 BETWEEN 0 AND 90)) OR 
                        ((@first.heading_diff_12 BETWEEN 195 AND 345) AND (first.bearing_12 BETWEEN 270 AND 360))
                        THEN 'crossing'
                    ELSE 'none'
                    END,
                {diff} = first.heading_diff_12
            FROM first
            WHERE first.mmsi_1 = {self.table}.mmsi_1
            AND first.trip_1 = {self.table}.trip_1
            AND first.mmsi_2 = {self.table}.mmsi_2
            AND first.trip_2 = {self.table}.trip_2
        """
        self.run_query(sql)

    def give_way_info(self):
        """Add type of interaction"""
        vessel_1 = "give_way_1"
        self.add_column(vessel_1, datatype="integer")

        vessel_2 = "give_way_2"
        self.add_column(vessel_2, datatype="integer")

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
            
            UPDATE 
                {self.table}
            SET 
                {vessel_1} = CASE
                    WHEN {self.table}.encounter = 'overtaking' THEN CASE
                        WHEN first.bearing_12 BETWEEN 90 AND 270 THEN 0
                        ELSE 1
                        END
                    WHEN {self.table}.encounter = 'crossing' THEN CASE
                        WHEN first.bearing_12 BETWEEN 0 AND 112.5 THEN 1
                        ELSE 0
                        END
                    WHEN {self.table}.encounter = 'head-on' THEN 1
                    END,
                {vessel_2} = CASE
                    WHEN {self.table}.encounter = 'overtaking' THEN CASE
                        WHEN first.bearing_12 BETWEEN 90 AND 270 THEN 1
                        ELSE 0
                        END
                    WHEN {self.table}.encounter = 'crossing' THEN CASE
                        WHEN first.bearing_12 BETWEEN 0 AND 112.5 THEN 0
                        ELSE 1
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

    def mark_true(self):
        """Mark encounters that are just nearby and not really encounter"""
        LOGGER.info('Marking near ships with no encounter...')
        col = "Nearby_Only"
        self.add_column(col, datatype="boolean", default=True)
        sql = f"""
            UPDATE {self.table}
            SET {col} = CASE
                WHEN (encounter = 'head-on') AND (90 <@ brange OR 270 <@ brange) THEN False
                WHEN (encounter = 'overtaking') AND (90 <@ brange OR 270 <@ brange) THEN False
                WHEN (encounter = 'crossing')
                    AND ((int4range(350, 360) && brange) 
                    OR (int4range(0, 10) && brange)
                    OR (int4range(170, 190) && brange)) THEN False ELSE True
            END                    
        """
        self.run_query(sql)

    def mark_ho_passing(self):
        """Mark HO encounters as SS PP"""
        col="head_on_passing"
        self.add_column(col, datatype='text')
        sql = f"""
            UPDATE {self.table}
            SET {col} = CASE
                WHEN (encounter = 'head-on') AND (90 <@ brange) THEN 'S'
                WHEN (encounter = 'head-on') AND (270 <@ brange) THEN 'P'
                ELSE 'N'
            END
        """
        self.run_query(sql)

    def tss_clearance(self, tss):
        
        col_64 = "tss_clearance_64"
        col_65 = "tss_clearance_65"
        self.add_column(col_64, "float(4)")
        self.add_column(col_65, "float(4)")
        sql = f"""
            UPDATE {self.table}
            SET {col_64} = ST_Distance(point_1, {tss}.geom)
            FROM {tss}
            WHERE {self.table}.tss_1 = False
            AND ST_DWithin(point_1, {tss}.geom, 2*1852)
            AND {tss}.gid = 64
        """
        self.run_query(sql)
        sql = f"""
            UPDATE {self.table}
            SET {col_65} = ST_Distance(point_1, {tss}.geom)
            FROM {tss}
            WHERE {self.table}.tss_1 = False
            AND ST_DWithin(point_1, {tss}.geom, 2*1852)
            AND {tss}.gid = 65
        """
        self.run_query(sql)

    def encounter_table(self):
        """Return count of each encounter type by vessel type"""
        df = self.df[(self.df["encounter"] != 'none') & (self.df["nearby_only"] == False)]
        df = df.drop_duplicates(subset=["cpa_distance", "cpa_time", "encounter"])
        return pd.pivot_table(
            df, 
            values='mmsi_1', 
            index=['encounter', 'ownship', 'ownship in TSS'],
            columns=['target ship', 'target ship in TSS'], aggfunc='count'
        )

    def head_on_table(self):
        df = self.df[(self.df["encounter"] == "head-on")]
        df = df[df["head_on_passing"] == 'S']
        df = df.drop_duplicates(subset=["cpa_distance", "cpa_time", "encounter"])
        df["pair1"] = df[['ownship', 'target ship']].agg('-'.join, axis=1)
        df["pair"] = np.where(df["pair1"] == "ferry-cargo", "cargo-ferry", df["pair1"])
        sns.countplot(df["head_on_passing"], hue=df["pair"])
        save_plot(join(PLOTS_DIR, f"Head On Encounters"), tight=True)
        plt.close()

        non_ferry = df[df["pair"] != "ferry-ferry"]
        df["TSS"] = df["ownship in TSS"] + df["target ship in TSS"]
        df["Split TSS"] = np.where(df["TSS"] == 1, 1 ,0)
        sns.countplot(df["Split TSS"], hue=df["pair"])
        save_plot(join(PLOTS_DIR, f"Head On Encounters Split TSS"), tight=True)
        plt.close()

    def _df_give_way(self, encounter=None, tss=None, vtype=None):
        """Return give way vessel dataframe"""
        df = self.df[
            (self.df['give_way_1'] == 1) & 
            (self.df['give_way_2'] == 0)
        ]
        df = df[(df["encounter"] != 'none') & (df['nearby_only'] == False)]
        if encounter:
            df = df[df['encounter'] == encounter] 
        if tss in [0,1]:
            df = df[
                (df['ownship in TSS'] == tss) &
                (df['target ship in TSS'] == tss)
            ]
        if vtype:
            df = df[df['ownship'] == vtype]
        df["Type"] = "Give Way"
        return df.rename(
            columns={
                "mmsi_1": "mmsi",
                "trip_1": "trip",
                "acceleration_1": "acceleration", 
                "alteration_1": "alteration", 
                "dcpa_1": "DCPA", 
                "tcpa": "TCPA"
            }
        )
            
    def _df_stand_on(self, encounter=None, tss=None, vtype=None):
        """Return stand on vessel dataframe"""
        df = self.df[
            (self.df['give_way_1'] == 1) & 
            (self.df['give_way_2'] == 0) 
        ]
        df = df[(df["encounter"] != 'none') & (df['nearby_only'] == False)]
        if encounter:
            df = df[df['encounter'] == encounter] 
        if tss in [0,1]:
            df = df[
                (df['ownship in TSS'] == tss) &
                (df['target ship in TSS'] == tss)
            ]
        if vtype:
            df = df[df['target ship'] == vtype]
        df["Type"] = "Stand On"
        return df.rename(
            columns={
                "mmsi_2": "mmsi",
                "trip_2": "trip",
                "acceleration_2": "acceleration", 
                "alteration_2": "alteration", 
                "dcpa_2": "DCPA", 
                "tcpa": "TCPA"
            }
        )

    def _df_ship_1(self, tss=None, vtype=None):
        """Return ship 1 and ship 2 dataframe"""
        df = self.df[
            (self.df['give_way_1'] == 1) & 
            (self.df['give_way_2'] == 1) &
            (self.df['encounter'] == "head-on")
        ] 
        df = df[(df["encounter"] != 'none') & (df['nearby_only'] == False)]
        if tss in [0,1]:
            df = df[
                (df['ownship in TSS'] == tss) &
                (df['target ship in TSS'] == tss)
            ]
        if vtype:
            df = df[df['ownship'] == vtype]
        df["Type"] = "Ship 1"
        return df.rename(
            columns={
                "mmsi_1": "mmsi",
                "trip_1": "trip",
                "acceleration_1": "acceleration", 
                "alteration_1": "alteration", 
                "dcpa_1": "DCPA", 
                "tcpa": "TCPA"
            }
        )

    def _df_ship_2(self, tss=None, vtype=None):
        """Return ship 1 and ship 2 dataframe"""
        df = self.df[
            (self.df['give_way_1'] == 1) & 
            (self.df['give_way_2'] == 1) &
            (self.df['encounter'] == "head-on")
        ]
        df = df[(df["encounter"] != 'none') & (df['nearby_only'] == False)]
        if tss in [0,1]:
            df = df[
                (df['ownship in TSS'] == tss) &
                (df['target ship in TSS'] == tss)
            ]
        if vtype:
            df = df[df['target ship'] == vtype]
        df["Type"] = "Ship 2"
        return df.rename(
            columns={
                "mmsi_2": "mmsi",
                "trip_2": "trip",
                "acceleration_2": "acceleration", 
                "alteration_2": "alteration", 
                "dcpa_2": "DCPA", 
                "tcpa": "TCPA"
            }
        )

    @property
    def df_no_so_maneuver(self):
        """Return a df where the so vessel does not maneuver"""
        alt_limit = 10
        stand_on = self._df_stand_on()
        return stand_on[abs(stand_on["alteration_1"]) < alt_limit]

    # PLOTS
    def plot_ferry(self):
        """Plot length when encountering a ferry versus non-ferry"""
        df = self.df[(self.df['ownship'] == 'ferry')]
        df = df[df['nearby_only'] == False]
        df = df[((df['origin_1']=="Bainbridge Island") & (df['destination_1']=="Seattle Pier 50")) | ((df['origin_1']=="Seattle Pier 50") & (df['destination_1']=="Bainbridge Island"))]
        # df = df[(df['origin_1'].isin(["Bremerton", "Seattle Pier 50"])) & (df['destination_1'].isin(["Bremerton", "Seattle Pier 50"]))]
        df['duration_1'] = df['duration_1']/ np.timedelta64(1, 'm')
       
        fig, ax = plt.subplots()
        sns.scatterplot(x='lon_1', y='lat_1', hue="target ship", alpha=0.5, data=df)
        ax.set_ylabel("Latitude")
        ax.set_xlabel("Longitude")
        ax.set_title(f"Spatial Distribution of Ferry Trips Between Bainbridge and Seattle v Target Ship Type")
        save_plot(join(PLOTS_DIR, f"Spatial Distribution of Ferry Trips v Target Ship Type.png"), tight=False)

        fig, ax = plt.subplots()
        sns.boxplot(y="alteration_1", x="target ship", hue="give_way_1", data=df, showfliers=False) 
        ax.set_ylabel("Alteration")
        ax.set_xlabel("Target Ship")
        ax.set_title(f"Alterations of Ferry Trips Between Bainbridge and Seattle v Target Ship Type")
        save_plot(join(PLOTS_DIR, f"Alterations of Ferry Trips v Target Ship Type.png"), tight=False)
        plt.close()

        fig, ax = plt.subplots()
        sns.boxplot(y="acceleration_1", x="target ship", hue="give_way_1", data=df, showfliers=False)
        ax.set_ylabel("Acceleration")
        ax.set_xlabel("Target Ship")
        ax.set_title(f"Acceleration of Ferry Trips Between Bainbridge and Seattle v Target Ship Type")
        save_plot(join(PLOTS_DIR, f"Acceleration of Ferry Trips v Target Ship Type.png"), tight=False)
        plt.close()

        fig, ax = plt.subplots()
        sns.boxplot(y="sog_1", x="target ship", hue="give_way_1", data=df, showfliers=False)
        ax.set_ylabel("SOG")
        ax.set_xlabel("Target Ship")
        ax.set_title(f"SOG of Ferry Trips Between Bainbridge and Seattle v Target Ship Type")
        save_plot(join(PLOTS_DIR, f"SOG of Ferry Trips v Target Ship Type.png"), tight=False)
        plt.close()

        fig, ax = plt.subplots()
        sns.boxplot(y="duration_1", x="target ship", hue="give_way_1", data=df, showfliers=False)
        ax.set_ylabel("Duration")
        ax.set_xlabel("Target Ship")
        ax.set_title(f"Duration of Ferry Trips Between Bainbridge and Seattle v Target Ship Type")
        save_plot(join(PLOTS_DIR, f"Duration of Ferry Trips v Target Ship Type.png"), tight=False)
        plt.close()  

        # get single track for trip
        df.drop_duplicates(subset=["cpa_distance", "cpa_time", "encounter"], inplace=True)
        fig, ax = plt.subplots()
        sns.boxplot(y="straight_1", x="target ship", hue="give_way_1", data=df)
        ax.set_ylabel("Straightness Index")
        ax.set_xlabel("Target Ship")
        ax.set_title(f"SI of Ferry Trips Between Bainbridge and Seattle v Target Ship Type")
        save_plot(join(PLOTS_DIR, f"SI of Ferry Trips v Target Ship Type.png"), tight=False)
        plt.close()  
         
        fig, ax = plt.subplots()
        sns.boxplot(y="length_1", x="target ship", data=df, showfliers=False)
        ax.set_ylabel("Length of Trip (m)")
        ax.set_xlabel("Target Ship")
        ax.set_title(f"Length of Ferry Trips Between Bainbridge and Seattle v Target Ship Type")
        save_plot(join(PLOTS_DIR, f"Length of Ferry Trips v Target Ship Type.png"), tight=False)
        plt.close()   

    @time_this
    def plot_ship_domain(self, hue):
        """Plot CPA ship domain"""
        df = self.df
        # df = df[(df['ownship'] != 'tanker') | df['target ship'] != 'tanker']
        df["target ship in TSS"] = df["target ship in TSS"].astype(int)
        df.rename(
            columns={
                "alteration_max_2": "target ship max alteration",
                "heading_2": "target ship heading",
                "give_way_2": "target ship give way"
            },
            inplace=True
        )
        g = sns.FacetGrid(
            df,
            col="ownship",
            row="target ship",
            subplot_kws=dict(projection="polar"),
            sharex=False,
            sharey=False,
            despine=False,
            gridspec_kws={"wspace": 1.0},
            height=6
        )
        g.map(sns.scatterplot, "r", "distance_12", hue=df[hue], s=20, alpha=0.5)
        
        # Set north to 0 degrees
        for axes in g.axes.flat:
            axes.set_theta_zero_location("N")
            axes.set_theta_direction(-1)
            axes.title.set_position([0.5, 1.2])
            axes.yaxis.labelpad = 40

        g.set_axis_labels("", "Distance")
        plt.legend(loc='lower right', bbox_to_anchor=(0.1, -0.5), ncol=1)
        save_plot(join(PLOTS_DIR, f"Ship Domain by Vessel Types, {hue.title()}"), tight=False)
        plt.close()
        self._df = None

    @time_this
    def plot_tss_clearance(self):

        df = self.df[self.df['ownship in TSS'] == 0]
        #  dft = df[df['encounter'] == 'none']
        df.dropna(subset=['tss_clearance_64','tss_clearance_65'], inplace=True)
        df['clearance'] =  df[["tss_clearance_64", "tss_clearance_65"]].max(axis=1)
        df = df[df['clearance'] > 500]
        
        fig, ax = plt.subplots()
        sns.boxplot(y='clearance', x="ownship", data=df)
        save_plot(join(PLOTS_DIR, f"TSS Clearance.png"), tight=False)

    @time_this
    def plot_alteration_dcpa(self):
        """Plot give way and stand on alterations as function of DCPA""" 
        label_so = "Stand On"
        label_gw = "Give Way"

        encounter = "Crossing"
        for tss in [0, 1]:
            for stand_on_vtype in ["cargo", "ferry"]:
                for give_way_vtype in ["cargo", "ferry"]:
                    give_way = self._df_give_way(tss=tss, vtype=give_way_vtype,encounter="crossing")
                    stand_on = self._df_stand_on(tss=tss, vtype=stand_on_vtype,encounter="crossing")
                    df = pd.concat([give_way, stand_on])
                    # if encounter == "head-on":
                    #     label_so = "Ship 1"
                    #     label_gw = "Ship 2"
                    #     give_way = self._df_ship_1(tss, give_way_vtype)
                    #     stand_on = self._df_ship_2(tss, stand_on_vtype)
                    #     df = pd.concat([give_way, stand_on])
                    #     df.drop_duplicates(
                    #         subset=["mmsi", "trip", "acceleration", "alteration","DCPA", "TCPA"], 
                    #         inplace=True
                    #     )                      
                    
                    # plot the max alteration in each encounter
                    fig, ax = plt.subplots()
                    sns.distplot(stand_on.groupby(["mmsi","trip"])["alteration"].max(), label=label_so, kde=True, hist=False)
                    sns.distplot(give_way.groupby(["mmsi","trip"])["alteration"].max(), label=label_gw, kde=True, hist=False)
                    plt.legend()
                    ax.set_ylabel("Number of Encounters")
                    ax.set_xlabel("Alteration (Degrees)")
                    ax.set_title(f"Alterations during {encounter.title()} Encounters in TSS = {bool(tss)} \n{label_gw} {give_way_vtype} {label_so} {stand_on_vtype}")
                    save_plot(join(PLOTS_DIR, f"Alterations during {encounter} Encounters in TSS {str(tss)} {label_gw} {give_way_vtype} {label_so} {stand_on_vtype}"), tight=True)
                    plt.close()

                    
                    alt_limit = 15
                    try:
                        df['Altered_Course'] = np.where(abs(df['alteration']) > alt_limit, 1, 0)
                        altered = df[(df['Altered_Course'] == 1)]
                        give_way_groups = altered[altered["Type"] == label_gw].sort_values(["mmsi", "trip", "TCPA"]).groupby(["mmsi", "trip"]).first()
                        stand_on_groups = altered[altered["Type"] == label_so].sort_values(["mmsi", "trip", "TCPA"]).groupby(["mmsi", "trip"]).first()

                        fig, ax = plt.subplots()
                        sns.distplot(give_way_groups["distance_12"]/1852, label=label_gw, kde=True, hist=False)
                        sns.distplot(stand_on_groups["distance_12"]/1852, label=label_so, kde=True, hist=False)
                        # sns.distplot(give_way_groups[give_way_groups["ownship"]=="cargo"]["distance_12"]/1852, label=f"{label_gw} cargo", kde=True, hist=False)
                        # sns.distplot(stand_on_groups[stand_on_groups["ownship"]=="cargo"]["distance_12"]/1852, label=f"{label_so} cargo", kde=True, hist=False)
                        # sns.distplot(give_way_groups[give_way_groups["ownship"]=="ferry"]["distance_12"]/1852, label=f"{label_gw} ferry", kde=True, hist=False)
                        # sns.distplot(stand_on_groups[stand_on_groups["ownship"]=="ferry"]["distance_12"]/1852, label=f"{label_so} ferry", kde=True, hist=False)
                        plt.legend()
                        ax.set_ylabel("Number of Encounters")
                        ax.set_xlabel("Alter-course-distance (nm)")
                        ax.set_title(f"ACD during {encounter.title()} Encounters in TSS = {bool(tss)} \n{label_gw} {give_way_vtype} {label_so} {stand_on_vtype}")
                        save_plot(join(PLOTS_DIR, f"ACD during {encounter} Encounters in TSS {str(tss)} {label_gw} {give_way_vtype} {label_so} {stand_on_vtype}"), tight=True)
                        plt.close()

                        fig, ax = plt.subplots()
                        groups = altered.sort_values(["mmsi", "trip", "TCPA"]).groupby(["mmsi", "trip"]).first()
                        groups = groups[groups["TCPA"]<1]
                        sns.boxplot(x="TCPA", y="alteration", data=groups, hue="Type")
                        # give_way_counts = give_way_groups[give_way_groups['TCPA']<1]['TCPA'].value_counts(normalize=True)
                        # stand_on_counts = stand_on_groups[stand_on_groups['TCPA']<1]['TCPA'].value_counts(normalize=True)

                        # sns.lineplot(x=give_way_counts.index, y=give_way_counts.values, label=label_gw)
                        # sns.lineplot(x=stand_on_counts.index, y=stand_on_counts.values, label=label_so)
                        # plt.legend()
                        ax.set_ylabel("Number of Encounters")
                        ax.set_xlabel("Alter-course-time (TCPA)")
                        ax.set_title(f"ACT during {encounter.title()} Encounters in TSS = {bool(tss)} \n{label_gw} {give_way_vtype} {label_so} {stand_on_vtype}")
                        save_plot(join(PLOTS_DIR, f"ACT during {encounter.title()} Encounters in TSS {str(tss)} {label_gw} {give_way_vtype} {label_so} {stand_on_vtype}"), tight=True)
                        plt.close()
                        
                    except:
                        continue
    
    @time_this
    def plot_encounters(self):
        """Plot tracks pairwise"""
        df = self.df[(self.df["encounter"] != 'none') & (self.df["nearby_only"] == False)]
        df = df.drop_duplicates(subset=["cpa_distance", "cpa_time", "encounter", "datetime"])
        for name, group in df.groupby(['mmsi_1','trip_1','mmsi_2','trip_2']):
            group = group.sort_values("tcpa")
            temp = list(name)
            output_name = " - ".join([str(e) for e in temp])
            encounter = group["encounter"].iloc[0]
            if group["give_way_1"].iloc[0] == 1 and group["give_way_2"].iloc[0] == 0:
                ship_1_label = f"Give Way Ship - {group['ownship'].iloc[0]}"
                ship_2_label = f"Stand On Ship - {group['target ship'].iloc[0]}"
            if group["give_way_1"].iloc[0] == 0 and group["give_way_2"].iloc[0] == 1:
                ship_1_label = f"Stand On Ship - {group['ownship'].iloc[0]}"
                ship_2_label = f"Give Way Ship - {group['target ship'].iloc[0]}"
            if group["give_way_1"].iloc[0] == 1 and group["give_way_2"].iloc[0] == 1:
                ship_1_label = f"Ship 1 - {group['ownship'].iloc[0]}"
                ship_2_label = f"Ship 2 - {group['target ship'].iloc[0]}"

            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, figsize=(7,10))
            fig.suptitle(f'Track Comparison {encounter.title()} \nMMSI {name[0]} Trip {name[1]} \nand \nMMSI {name[2]} Trip {name[3]}', fontsize=12, fontweight='bold', x=0.54, y=0.99)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=0, hspace=0.3)
  
            ax1.set_title('LAT')
            sns.lineplot(x='tcpa', y='lat_1', data=group, color='blue', ax=ax1, label=ship_1_label, legend=False)
            sns.lineplot(x='tcpa', y='lat_2', data=group, color='red', dashes=True, ax=ax1, label=ship_2_label, legend=False)
            
            ax2.set_title('LON')
            sns.lineplot(x='tcpa', y='lon_1', data=group, color='blue',ax=ax2, label=ship_1_label, legend=False)
            sns.lineplot(x='tcpa', y='lon_2', data=group, color='red', dashes=True, ax=ax2, label=ship_2_label, legend=False)

            ax3.set_title('Acceleration')
            sns.lineplot(x='tcpa', y='acceleration_1', data=group, color='blue', ax=ax3, label=ship_1_label, legend=False)
            sns.lineplot(x='tcpa', y='acceleration_2', data=group, color='red', dashes=True, ax=ax3, label=ship_2_label, legend=False)

            ax4.set_title("Change in Heading")
            sns.lineplot(x='tcpa', y='alteration_1', data=group, color="blue", ax=ax4, label=ship_1_label, legend=False)
            sns.lineplot(x='tcpa', y='alteration_2', data=group, color="red", dashes=True, ax=ax4, label=ship_2_label, legend=False)
            
            ax5.set_title("Distance between Ships")
            sns.lineplot(x='tcpa', y='distance_12', data=group, color="blue", ax=ax5, legend=False)
            
            

            for ax in fig.get_axes():
                ax.label_outer()
                ax.set_xlabel("TCPA")
            labels = ["LAT", "LON", "Meters/Second_squared", "Degrees", "Meters"]
            for label, ax in zip(labels, fig.get_axes()):
                ax.label_outer()
                ax.set_ylabel(label)

            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')
            save_plot(join(PLOTS_DIR, 'non-encounters', output_name + ".png"), tight=True)

    @time_this
    def plot_alterations(self):
        """Plot heading and sog alterations of true encounters"""

        # plot alteration of give way
        give_way = self._df_give_way()
        stand_on = self._df_stand_on()
        ship_1 = self._df_ship_1()
        ship_2 = self._df_ship_2()
        ship_1["Type"] = "Give Way"
        ship_2["Type"] = "Give Way"      

         # ACCELERATION
        fig, ax = plt.subplots(figsize=(10,10))
        group = pd.concat([give_way, stand_on])
        group = group[group["cpa_distance"]< .25*1852]
        sns.boxplot(y="acceleration", x="Type", hue="ownship", data=group,linewidth=0.5)      #,showfliers=False
        plt.legend()
        save_plot(join(PLOTS_DIR, f"Encounter Acceleration by Responsibility and Type (No Head-On)"), tight=True)
        plt.close()

        df = pd.concat([give_way, stand_on, ship_1, ship_2])
        fig, ax = plt.subplots()
        sns.distplot(df["acceleration"], kde=False)
        ax.set_ylabel("Number of Data Points")
        ax.set_xlabel("Acceleration (meters/second^2)")
        plt.xlim(-0.2, 0.2)
        save_plot(join(PLOTS_DIR, f"Encouter Acceleration"), tight=True)
        plt.close()

        fig, ax = plt.subplots()
        sns.distplot(df["alteration"], kde=False)
        ax.set_ylabel("Number of Data Points")
        ax.set_xlabel("Alteration (Degrees)")
        plt.xlim(-180, 180)
        save_plot(join(PLOTS_DIR, f"Encouter Alteration"), tight=True)
        plt.close()

        fig, ax = plt.subplots()
        gw = pd.concat([give_way, ship_1, ship_2])
        sns.distplot(stand_on["alteration"], label="Stand On", kde=False)
        sns.distplot(gw["alteration"], label="Give Way", kde=False)
        plt.legend()
        ax.set_ylabel("Number of Encounters")
        ax.set_xlabel("Alteration (Degrees)")
        save_plot(join(PLOTS_DIR, f"Encounter Alteration by Responsibility"), tight=True)
        plt.close()

        # BY CPA 
        # TODO: add distance titles
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10,10))
        group = pd.concat([give_way, stand_on])
        bins = np.linspace(group.alteration.min(), group.alteration.max(), 60)
        sns.distplot(stand_on[stand_on["cpa_distance"]< 1*1852]["alteration"], label="Stand On", kde=True, hist=False, bins=bins, ax=ax1)
        sns.distplot(give_way[give_way["cpa_distance"]< 1*1852]["alteration"], label="Give Way", kde=True, hist=False,bins=bins, ax=ax1)

        sns.distplot(stand_on[stand_on["cpa_distance"]< 0.5*1852]["alteration"], label="Stand On", kde=True, hist=False,bins=bins, ax=ax2)
        sns.distplot(give_way[give_way["cpa_distance"]< 0.5*1852]["alteration"], label="Give Way", kde=True, hist=False,bins=bins, ax=ax2)

        sns.distplot(stand_on[stand_on["cpa_distance"]< 0.25*1852]["alteration"], label="Stand On", kde=True, hist=False,bins=bins, ax=ax3)
        sns.distplot(give_way[give_way["cpa_distance"]< 0.25*1852]["alteration"], label="Give Way", kde=True, hist=False,bins=bins, ax=ax3)

        sns.distplot(stand_on[stand_on["cpa_distance"]< 0.1*1852]["alteration"], label="Stand On", kde=True, hist=False,bins=bins, ax=ax4)
        sns.distplot(give_way[give_way["cpa_distance"]< 0.1*1852]["alteration"], label="Give Way", kde=True, hist=False,bins=bins, ax=ax4)
        
        plt.legend()
        ax.set_ylabel("Number of Encounters")
        ax.set_xlabel("Alteration (Degrees)")
        save_plot(join(PLOTS_DIR, f"Encounter Alteration by Responsibility (No Head-On)"), tight=True)
        plt.close()

        # BY VESSEL TYPE
        # fig, axs = plt.subplots()
        # group = pd.concat([give_way, stand_on])
        # bins = np.linspace(group.alteration.min(), group.alteration.max(), 60)
        # give_way = self._df_give_way(vtype="ferry")
        # stand_on = self._df_stand_on(vtype="cargo")
        # sns.boxplot(stand_on[stand_on["cpa_distance"]< 2*1852]["alteration"], label="Stand On Cargo", kde=True, hist=False, ax=ax1)
        # sns.boxplot(give_way[give_way["cpa_distance"]< 2*1852]["alteration"], label="Give Way Ferry", kde=True, hist=False, ax=ax1)

        # give_way = self._df_give_way(vtype="cargo")
        # stand_on = self._df_stand_on(vtype="ferry")
        # sns.boxplot(stand_on[stand_on["cpa_distance"]< 2*1852]["alteration"], label="Stand On Ferry", kde=True, hist=False, ax=ax2)
        # sns.boxplot(give_way[give_way["cpa_distance"]< 2*1852]["alteration"], label="Give Way Cargo", kde=True, hist=False, ax=ax2)

        # plt.legend()
        # ax.set_ylabel("Number of Encounters")
        # ax.set_xlabel("Alteration (Degrees)")
        # save_plot(join(PLOTS_DIR, f"Encounter Alteration by Responsibility and Type (No Head-On)"), tight=True)
        # plt.close()
    
    @time_this
    def plot_colreg_gw_manuevers(self):
        """Plot alteration of gw vessels when so doesnt maneuver"""

        df = self.df_no_so_maneuver
        fig, ax = plt.subplots()
        sns.scatterplot(x=df.alteration, y=df.distance_12, hue=df.TCPA)
        ax.set_ylabel("Distance between Ships (m)")
        ax.set_xlabel("Alteration (degrees)")
        ax.set_title(f"Alterations of Give Way Vessels in COLREGS-compliant Encounters v Distance")
        save_plot(join(PLOTS_DIR, f"Alterations of Give Way Vessels in COLREGS-compliant Encounters v Distance.png"), tight=False)
        plt.close()

        df0 = df[(df['alteration']>10) | (df['alteration']<-10)]
        fig, ax = plt.subplots()
        sns.scatterplot(x=df0.lon_2, y=df0.lat_2, hue=df0.alteration, size=df0.alteration)
        ax.set_ylabel("Latitude")
        ax.set_xlabel("Longitude")
        ax.set_title(f"Location of Alterations of Give Way Vessels in COLREGS-compliant Encounters")
        save_plot(join(PLOTS_DIR, f"Location of Alterations of Give Way Vessels in COLREGS-compliant Encounters.png"), tight=False)
        plt.close()

        fig, ax = plt.subplots()
        sns.scatterplot(x=df.acceleration, y=df.distance_12, hue=df.TCPA)
        ax.set_ylabel("Distance between Ships (m)")
        ax.set_xlabel("Acceleration (degrees)")
        ax.set_title(f"Acceleration of Give Way Vessels in COLREGS-compliant Encounters v Distance")
        save_plot(join(PLOTS_DIR, f"Acceleration of Give Way Vessels in COLREGS-compliant Encounters v Distance.png"), tight=False)
        plt.close()

        # fig, ax = plt.subplots()
        # sns.distplot(self.df_no_so_maneuver["alteration_2"], kde=False)
        # ax.set_ylabel("Number of Encounters")
        # ax.set_xlabel("Alteration (Degrees)")
        # save_plot(join(PLOTS_DIR, f"Alterations of Give Way Vessels in COLREGS-compliant Encounters.png"), tight=False)
        # plt.close()

        # target ship is the give way ship -- TO DO: rename
        # fig, ax = plt.subplots()
        # prior = self.df_no_so_maneuver[(self.df_no_so_maneuver["tcpa"]<1)]
        # sns.boxplot(x="tcpa", y="alteration_2", data=prior)
        # ax.set_ylabel("Number of Encounters")
        # ax.set_xlabel("Alter-course-time (TCPA)")
        # ax.set_title(f"Alterations of Give Way Vessels in COLREGS-compliant Encounters")
        # save_plot(join(PLOTS_DIR, f"Alterations of Give Way Vessels in COLREGS-compliant Encounters.png"), tight=False)
        # plt.close()

    @time_this
    def plot_first_move(self):
        """Plot distribution of maneuvers for when GW moves first and SO moves first"""
        # just encounters with a stand on vessel (first ship)
        limit = 2*1852
        alt_limit = 30
        df = self.df[self.df['give_way_1'] == 0]
        df = df[df["cpa_distance"] < limit]
        df['SO_alter'] = np.where(abs(df['alteration_1']) > alt_limit, 1 , 0)
        df['GW_alter'] = np.where(abs(df['alteration_2']) > alt_limit, 1 , 0)
        df['Both_zero'] = np.where(df['SO_alter'] + df['GW_alter']==0,1,0)
        df = df[df['Both_zero'] != 1]

        first = df.sort_values(["mmsi_1", "trip_1", "mmsi_2", "trip_2", "datetime"]).groupby(["mmsi_1", "trip_1", "mmsi_2", "trip_2"]).first() 
        first['Diff']  = first['GW_alter'] - first['SO_alter']
        first['Moved First'] = np.where(first['Diff'] == 1, "Give Way", "Stand On")

        give_way = first[first["Moved First"] == "Give Way"]
        stand_on = first[first["Moved First"] == "Stand On"]
        
        fig, ax = plt.subplots()
        sns.distplot(give_way["distance_12"]/1852, kde=False, hist=True)
        ax.set_ylabel("Number of Encounters")
        ax.set_xlabel("Alter-course-distance (meters)")
        ax.set_title(f"Give-Way Moved First: ACD during Encounters Alteration > {alt_limit} degrees")
        save_plot(join(PLOTS_DIR, f"Give-Way Moved First - ACD Alteration over {alt_limit} degrees.png"), tight=False)
        plt.close()

        fig, ax = plt.subplots()
        sns.distplot(stand_on["distance_12"]/1852, kde=False, hist=True)
        ax.set_ylabel("Number of Encounters")
        ax.set_xlabel("Alter-course-distance (meters)")
        ax.set_title(f"Stand On Moved First: ACD Alteration > {alt_limit} degrees")
        save_plot(join(PLOTS_DIR, f"Stand-On Moved First - ACD Alteration over {alt_limit} degrees.png"), tight=False)
        plt.close()

        fig, ax = plt.subplots()
        sns.distplot(give_way["alteration_2"], kde=False, hist=True)
        ax.set_ylabel("Number of Encounters")
        ax.set_xlabel("Alteration (Degrees)")
        ax.set_title(f"Give-Way Moved First: ACD Alteration > {alt_limit} degrees")
        save_plot(join(PLOTS_DIR, f"Give-Way Moved First - ACD Alteration over {alt_limit} degrees.png"), tight=False)
        plt.close()

        fig, ax = plt.subplots()
        sns.distplot(stand_on["alteration_1"], kde=False, hist=True)
        ax.set_ylabel("Number of Encounters")
        ax.set_xlabel("Alteration (degrees)")
        ax.set_title(f"Stand On Moved First: ACD Alteration > {alt_limit} degrees")
        save_plot(join(PLOTS_DIR, f"Stand-On Moved First - ACD Alteration over {alt_limit} degrees.png"), tight=False)
        plt.close()
       
    # REGRESSION        
    def regression(self):
        """Run regression"""
        df = self.df
        df.rename(
            columns = {
                "ownship": "type_1" , 
                "target ship": "type_2",
                "ownship in TSS": "tss_1",
                "target ship in TSS": "tss_2"
            }, inplace=True
        )
        df = df[df['tcpa'] == 0]

        df['tss_both'] = df['tss_1']*df['tss_2']
        dummies = pd.get_dummies(data=df, columns=['type_1', 'type_2', 'tss_1', 'tss_2', 'tss_both'])
        dummies['bearing_12_sin'] = np.sin(np.radians(dummies['bearing_12']))
        dummies['bearing_12_cos'] = np.cos(np.radians(dummies['bearing_12']))   


        dummies['type_1_ferry_sin'] = dummies['bearing_12_sin']*dummies['type_1_ferry']
        dummies['type_1_tanker_sin'] = dummies['bearing_12_sin']*dummies['type_1_tanker']
        dummies['type_2_ferry_sin'] = dummies['bearing_12_sin']*dummies['type_2_ferry']
        dummies['type_2_tanker_sin'] = dummies['bearing_12_sin']*dummies['type_2_tanker']
        dummies['tss_1_True_sin'] = dummies['bearing_12_sin']*dummies['tss_1_True']
        dummies['tss_2_True_sin'] = dummies['bearing_12_sin']*dummies['tss_2_True']
        dummies['tss_both_True_sin'] = dummies['bearing_12_sin']*dummies['tss_both_True']


        dummies['type_1_ferry_cos'] = dummies['bearing_12_cos']*dummies['type_1_ferry']
        dummies['type_1_tanker_cos'] = dummies['bearing_12_cos']*dummies['type_1_tanker']
        dummies['type_2_ferry_cos'] = dummies['bearing_12_cos']*dummies['type_2_ferry']
        dummies['type_2_tanker_cos'] = dummies['bearing_12_cos']*dummies['type_2_tanker']
        dummies['tss_1_True_cos'] = dummies['bearing_12_cos']*dummies['tss_1_True']
        dummies['tss_2_True_cos'] = dummies['bearing_12_cos']*dummies['tss_2_True']
        dummies['tss_both_True_cos'] = dummies['bearing_12_cos']*dummies['tss_both_True']

        dummies['group'] = dummies['mmsi_1'].astype(str) + dummies['trip_1'].astype(str)
        X = dummies[[
            'type_1_ferry', 
            'type_1_tanker',
            'type_2_ferry', 
            'type_2_tanker', 
            'tss_1_True', 
            'tss_2_True',  
            'tss_both_True',  
            'bearing_12_sin', 
            'bearing_12_cos',
            'sog_1',
            'sog_2',
            'type_1_ferry_sin',
            'type_1_tanker_sin',
            'type_2_ferry_sin',
            'type_2_tanker_sin',
            'tss_1_True_sin',
            'tss_2_True_sin',
            'tss_both_True_sin',
            'type_1_ferry_cos',
            'type_1_tanker_cos',
            'type_2_ferry_cos',
            'type_2_tanker_cos',
            'tss_1_True_cos',
            'tss_2_True_cos',
            'tss_both_True_cos'    
            ]]
        y = dummies['distance_12']

        X = sm.add_constant(X)
        return sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': dummies['group']})

    def hypotheses(self):
        """Significance of 'features' of encounters have associations with distance"""
        model = self.regression()
        hypothesis_type_1 = '(type_1_ferry_sin = type_1_ferry_cos = type_1_tanker_sin = type_1_tanker_cos = type_1_ferry = type_1_tanker = 0)'
        hypothesis_type_2 = '(type_2_ferry_sin = type_2_ferry_cos = type_2_tanker_sin = type_2_tanker_cos = type_2_ferry = type_2_tanker = 0)'
        hypothesis_tss = '(tss_1_True = tss_2_True = tss_both_True = tss_1_True_sin = tss_2_True_sin = tss_both_True_sin = tss_1_True_cos = tss_2_True_cos = tss_both_True_cos = 0)'
        hypothesis_bearing = '(bearing_12_sin = bearing_12_cos = type_1_ferry_sin = type_1_tanker_sin =  type_2_ferry_sin = type_2_tanker_sin = tss_1_True_sin = tss_2_True_sin = tss_both_True_sin = type_1_ferry_cos = type_1_tanker_cos = type_2_ferry_cos = type_2_tanker_cos = tss_1_True_cos = tss_2_True_cos = tss_both_True_cos = 0)'

        names = ['type_1', 'type_2', 'tss', 'bearing']
        hypos = [hypothesis_type_1, hypothesis_type_2, hypothesis_tss, hypothesis_bearing]
        ftests = []
        for name, hypo in zip(names, hypos):
            ftest = model.f_test(hypo)
            ftests.append([name, ftest.fvalue[[0]], ftest.pvalue, ftest.df_denom, ftest.df_num])
        return pd.DataFrame(ftests, columns=['feature', 'F Value', 'P Value', 'DF Denom', 'DF Num'])  


    @time_this
    def plot_distance_type(self):
        """Plot CPA ship domain"""   
        df = self.df
        df.drop_duplicates(subset=["cpa_distance", "cpa_time", "encounter"], inplace=True)
        df['CPA Distance'] = df['cpa_distance']/1852
        df = df[df['encounter'] == 'head-on']
       
        g = sns.FacetGrid(
            df,
            col="target ship",
            row="ownship",
            sharex=False,
            sharey=False,
            despine=False,
            gridspec_kws={"wspace": 1.5, "hspace": 0.7},
        )
        g.map(sns.distplot, "CPA Distance", kde=False)
        save_plot(join(PLOTS_DIR, "CPA Vessel Type"), tight=False)
        plt.close()

        g = sns.FacetGrid(
            df,
            row="head_on_passing",
            sharex=False,
            sharey=False,
            despine=False,
            gridspec_kws={"wspace": 1.5, "hspace": 0.7},
        )
        g.map(sns.distplot, "CPA Distance", kde=False)
        save_plot(join(PLOTS_DIR, "CPA Encounter Type"), tight=False)
        plt.close()

        sns.boxplot(y=df["alteration_1"], x=df["head_on_passing"])
        save_plot(join(PLOTS_DIR, "Head On Alteration"), tight=False)
        plt.close()


        
        
    def plot_crossings(self):
        for tss in [0,1]:
            give_way = self._df_give_way("crossing", tss)
            stand_on = self._df_stand_on("crossing", tss)
            df = pd.concat([give_way, stand_on])
            df['Altered_Course'] = np.where(abs(df['alteration']) > 10, 1, 0)
            altered = df[(df['Altered_Course'] == 1)]
            give_way_groups = altered[altered["Type"] == "Give Way"].sort_values(["mmsi", "trip", "TCPA"]).groupby(["mmsi", "trip"]).first()
            stand_on_groups = altered[altered["Type"] == "Stand On"].sort_values(["mmsi", "trip", "TCPA"]).groupby(["mmsi", "trip"]).first()
            g = sns.FacetGrid(
                give_way_groups,
                col="ownship",
                row="target ship",
                sharex=False,
                sharey=False,
                despine=False,
                gridspec_kws={"wspace": 1.0},
                height=5
            )
            g.map(sns.distplot, "distance_12", kde=True, hist=False)
            # g.map(sns.distplot, "distance_12", kde=True, hist=False)
            # for axes in g.axes.flat:
            #     ax.set_ylabel("Number of Encounters")
            #     ax.set_xlabel("Alteration (Degrees)")
 
            save_plot(join(PLOTS_DIR, f"Alterations - Crossing Encounters By Vessel Type, TSS {str(tss)}"), tight=True)
            plt.close()


class Unimpeded_Table(Postgres_Table):

    """
    Tracks that are not in an encounter
    """

    def __init__(self, table, input_tracks, input_cpa):
        """
        Connect to default database.
        
        Args:
            table (string): Name of table
            input_table (string): Name of tracks table
        """
        super(Unimpeded_Table, self).__init__(table)
        self.cur = self.conn.cursor()
        self.tracks = input_tracks
        self.cpas = input_cpa
        
    def select_non_encounters(self):

        sql = f"""
            CREATE TABLE {self.table} AS
            SELECT  l.*
            FROM    {self.tracks} l
            WHERE NOT EXISTS
                (
                SELECT  NULL
                FROM    {self.cpas} r
                WHERE   r.cpa_distance < 3*1852
                AND     r.mmsi_1 = l.mmsi
                AND     r.trip_1 = l.trip
                )
        """
        self.run_query(sql)

