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

import warnings

warnings.filterwarnings("ignore")

PLOTS_DIR = abspath(join(dirname(__file__), "..", "reports", "figures"))
# plt.style.use("seaborn")
sns.set(color_codes=True)
# sns.palplot(sns.color_palette("RdBu_r", 7))
sns.set_context("poster") 
plt.style.use('seaborn')


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
        """
        Plot count of in versus out of TSS by vessel type
        """
        df = self.table_dataframe(select_col="mmsi, datetime, in_tss, vesseltype")
        LOGGER.info(f"Plotting tss from {self.table}")
        fig, ax = plt.subplots()
        sns.countplot("in_tss", data=df, palette="Paired", hue="vesseltype")
        ax.set_ylabel("Number of Data Points")
        ax.set_xlabel("In TSS")
        save_plot(join(PLOTS_DIR, "TSS"))
        plt.close()
        del df


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
        # g = sns.FacetGrid(
        #     df,
        #     col="tss_direction",
        #     row="gid",
        #     sharex=False,
        #     sharey=False,
        #     despine=False,
        #     gridspec_kws={"wspace": 1.5},
        #     height=5
        # )
        # g.map(sns.distplot, "entrance_angle", kde=False)
        fig, ax = plt.subplots()
        sns.distplot(df["entrance_angle"], kde=False)
        ax.set_ylabel("Number of Entrances")
        ax.set_xlabel("Entrance Angle")
        save_plot("TSS_Entrance_Angle", False)
        plt.close()
        del df
    

class Ship_Table(Postgres_Table):

    """
    Static ship information from points table
    """

    def __init__(self, table, points):
        """
        Connect to default database and set table schema.
        
        Args:
            table (string): Name of table
            points (string): Name of points table
        """
        super(Ship_Table, self).__init__(table)
        self.points = points

    def extract_ship_info(self):
        """Extract static ship information"""
        sql = f"""
            CREATE TABLE {self.table} AS
            SELECT
                MMSI,
                VesselName,
                VesselType,
                Length,
                COUNT(mmsi) AS data_points,
                COUNT(DISTINCT trip) AS trips,
                FOREIGN KEY (MMSI) REFERENCES {self.points} (MMSI)
            FROM {self.points}
            GROUP BY MMSI
        """
        self.run_query(sql)


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
                And p.trip = c.trip_1
            INNER JOIN {self.points} p2
                ON p2.mmsi = c.mmsi_2
                AND p2.trip = c.trip_2
                AND p2.datetime = p.datetime
        """
        self.run_query(sql)

    def drop_sparse_encounters(self):
        """Drop encounters with less than 15 data points"""
        sql = f"""
        DELETE FROM {self.table}
        USING (
            SELECT mmsi_1, trip_1, mmsi_2, trip_2, count(datetime) AS num_rows
            FROM {self.table}
            GROUP BY mmsi_1, trip_1, mmsi_2, trip_2
            HAVING count(datetime) < 15 
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

    def tcpa(self):
        """Add time to CPA"""
        name = "tcpa"
        self.add_column(name, datatype="float(4)")
        sql = f"""
            UPDATE {self.table}
            SET {name} = EXTRACT(MINUTE FROM (datetime - cpa_time))
        """
        self.run_query(sql)

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

    def df_tss(self, limit):
        """Return df of points inside TSS"""
        limit_cond = self._df["cpa_distance"] < limit
        return self._df[
            limit_cond & 
            (self._df["tss_1"] == True) & 
            (self._df["tss_2"] == True)
        ]

    def df_tss_no(self, limit):
        """Return df of points outside TSS"""
        limit_cond = self._df["cpa_distance"] < limit
        return self._df[
            limit_cond & 
            (self._df["tss_1"] == False) & 
            (self._df["tss_2"] == False)
        ]

    def df_tss_mix(self, limit):
        """Return df of points out/in TSS"""
        limit_cond = self._df["cpa_distance"] < limit
        return self._df[
            limit_cond & 
            ((df["tss_1"] == True) & (df["tss_2"] == False)) | 
            ((df["tss_1"] == False) & (df["tss_2"] == True))
        ]

    # PLOTS
    @time_this
    def plot_ship_domain(self, hue):
        """Plot CPA ship domain"""
        limit = 7408  # 4 nm
        df = self.df
        df["target ship in TSS"] = df["target ship in TSS"].astype(int)
        df.rename(columns={"alteration_degrees_1": "target ship alteration"},inplace=True)
        g = sns.FacetGrid(
            df[df["distance_12"] < limit],
            col="ownship",
            subplot_kws=dict(projection="polar"),
            sharex=False,
            sharey=False,
            despine=False,
            gridspec_kws={"wspace": 1.5},
        )
        g.map(
            sns.scatterplot, 
            "r", 
            "distance_12", 
            hue=df[hue],
            marker=".", 
            s=15, 
            alpha=0.8
        )
        
        # Set north to 0 degrees
        for axes in g.axes.flat:
            axes.set_theta_zero_location("N")
            axes.set_theta_direction(-1)
            axes.title.set_position([0.5, 1.5])
            axes.yaxis.labelpad = 40
            axes.legend()

        g.set_axis_labels("Bearing To Target Ship", "Distance to Target Ship")
        # plt.legend(loc='upper left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        save_plot(join(PLOTS_DIR, f"Ship Domain by Vessel Type {hue}"), tight=False)
        plt.close()

    # def plot_modes(elf):
    #     # plot poly
    #     bearings = [0, 45, 90, 135, 180, 225, 270, 315]
    #     rads = [np.radians(b) for b in bearings]
    #     groups = self.df[(self.df["r"].isin(rads)) & (self.df["cpa_distance"] < limit)]        
    #     modes = groups.groupby("r").agg(lambda x:x.value_counts().index[0])
        
    #     p = sns.FacetGrid(
    #         modes,
    #         col="ownship",
    #         row="target ship in TSS",
    #         subplot_kws=dict(projection="polar"),
    #         sharex=False,
    #         sharey=False,
    #         despine=False,
    #         gridspec_kws={"wspace": 1.5},
    #     )
    #     p.map(
    #         sns.lineplot, 
    #         "r", 
    #         "distance_12", 
    #     )
    #     save_plot(join(PLOTS_DIR, "Ship Domain Mode by Vessel Type"), tight=False)
    #     plt.close()
    #     del modes, groups

    @time_this
    def plot_alteration_dcpa(self):
        """Plot give way and stand on alterations as function of DCPA""" 
        cols = [
            "mmsi", 
            "trip",
            "Type", 
            "ownship", 
            "target ship", 
            "cpa_distance", 
            "acceleration", 
            "alteration", 
            "DCPA", 
            "TCPA", 
            "distance_12",
            "encounter"
        ]
        give_way = self.df[(self.df["ship_1_give_way"] == 1) & ("encounter" != "head-on") ]
        give_way["Type"] = "Give Way"
        give_way.rename(
            columns={
                "mmsi_1": "mmsi",
                "trip_1": "trip",
                "acceleration_1": "acceleration", 
                "alteration_degrees_1": "alteration", 
                "dcpa_1": "DCPA", 
                "tcpa": "TCPA"
            }, 
            inplace=True
        )
        give_way = give_way[cols]

        stand_on = self.df[(self.df["ship_1_give_way"] == 0) & ("encounter" != "head-on")]
        stand_on["Type"] = "Stand On"
        stand_on.rename(
            columns={
                "mmsi_1": "mmsi",
                "trip_1": "trip",
                "acceleration_1": "acceleration", 
                "alteration_degrees_1": "alteration", 
                "dcpa_1": "DCPA", 
                "tcpa": "TCPA"
            }, 
            inplace=True
        )
        stand_on = stand_on[cols]

        for limit in [1852, 2*1852, 3*1852]:
            give_way = give_way[give_way["cpa_distance"] < limit]
            stand_on = stand_on[stand_on["cpa_distance"] < limit]
            fig, ax = plt.subplots()
            sns.distplot(stand_on.groupby(["mmsi","trip"])["alteration"].max(), label="Stand On", kde=True, hist=False)
            sns.distplot(give_way.groupby(["mmsi","trip"])["alteration"].max(), label="Give Way", kde=True, hist=False)
            plt.legend()
            ax.set_ylabel("Number of Encounters")
            ax.set_xlabel("Alteration (Degrees)")
            ax.set_title(f"Alterations during Encounters with CPA less than {limit/1852} nm")
            save_plot(join(PLOTS_DIR, f"Alterations during Encounters with CPA less than {limit/1852} nm.png"), tight=False)
            plt.close()

            alt_limit = 15
            df = pd.concat([give_way, stand_on])
            df['Altered_Course'] = np.where(df['alteration'] > alt_limit, 1, 0)
            altered = df[(df['Altered_Course'] == 1)]
            give_way_groups = altered[altered["Type"] == "Give Way"].sort_values("TCPA").groupby(["mmsi", "trip"]).first()
            stand_on_groups = altered[altered["Type"] == "Stand On"].sort_values("TCPA").groupby(["mmsi", "trip"]).first()

            fig, ax = plt.subplots()
            sns.distplot(give_way_groups["distance_12"]/1852, label="Give Way", kde=True, hist=False)
            sns.distplot(stand_on_groups["distance_12"]/1852, label="Stand On", kde=True, hist=False)
            plt.legend()
            ax.set_ylabel("Number of Encounters")
            ax.set_xlabel("Alter-course-distance (meters)")
            ax.set_title(f"Alter-course-distance during Encounters with CPA less than {limit/1852} nm and Alteration > {alt_limit} degrees")
            save_plot(join(PLOTS_DIR, f"Alter-course-distance during Encounters with CPA less than {limit/1852} nm and Alteration > {alt_limit} degrees.png"), tight=False)
            plt.close()

            fig, ax = plt.subplots()
            groups = altered[altered["TCPA"]<1].sort_values("TCPA").groupby(["mmsi", "trip"]).first()
            sns.countplot("TCPA", hue="Type", data=groups)
            ax.set_ylabel("Number of Encounters")
            ax.set_xlabel("Alter-course-time (TCPA)")
            ax.set_title(f"Alter-course-time during Encounters with CPA less than {limit/1852} nm and Alteration > {alt_limit} degrees")
            save_plot(join(PLOTS_DIR, f"Alter-course-time during Encounters with CPA less than {limit/1852} nm and Alteration > {alt_limit} degrees.png"), tight=False)
            plt.close()


# `   def plot_first_move(self):

#         df= self.df[self.df['ship_1_give_way']==0]
#         df['SO_alter'] = np.where(df['alteration_degrees_1'] > 10, 1 , 0)
#         df['GW_alter'] = np.where(df['alteration_degrees_1'] > 10, 1 , 0)
#         df.drop((df['SO_alter']==0) & (df['GW_alter']==0))
#         df['Diff']  = df['GW_alter'] - df['SO_atler']
#         df['Moved First'] = np.where(df['Diff'] = 1, "Give Way", "Stand On")

`
       

        


        # for i in [0, 1, 2]:
        #     g = sns.FacetGrid(
        #         df,
        #         col="target ship",
        #         row="ownship",
        #         sharex=False,
        #         sharey=False,
        #         despine=False,
        #         gridspec_kws={"wspace": 1.5},
        #         height=5
        #     )
        #     g.map(sns.distplot, x="alteration_degrees", y="DCPA", hue="Type", kde=True)

        #     # Limit to lower upper triangle of grid
        #     g.axes[1, 2].set_visible(False)
        #     g.axes[2, 1].set_visible(False)
        #     g.axes[2, 2].set_visible(False)

        #     for axes in g.axes.flat:
        #         axes.title.set_position([0.5, 0.5])
        #         axes.yaxis.labelpad = 40

        

    # @time_this
    # def plot_distance_type(self):
    #     """Plot CPA ship domain""" 
    #     limit = 11112      
    #     limit_cond = self.df["cpa_distance"] < limit 
    #     df_all = self.df[limit_cond]
    #     df_tss = self.df[
    #         (limit_cond) & (self.df["tss_1"] == True) & (self.df["tss_2"] == True)
    #     ]
    #     df_no_tss = self.df[
    #         (limit_cond) & (self.df["tss_1"] == False) & (self.df["tss_2"] == False)
    #     ]

    #     dfs = [df_all, df_tss, df_no_tss]
    #     names = ["CPA", "CPA_In_TSS", "CPA_Out_TSS"]

    #     for i in [0, 1, 2]:
    #         g = sns.FacetGrid(
    #             dfs[i],
    #             col="target ship",
    #             row="ownship",
    #             sharex=False,
    #             sharey=False,
    #             despine=False,
    #             gridspec_kws={"wspace": 1.5},
    #             height=5
    #         )
    #         g.map(sns.distplot, "cpa_distance", kde=False)

    #         # Limit to lower upper triangle of grid
    #         g.axes[1, 2].set_visible(False)
    #         g.axes[2, 1].set_visible(False)
    #         g.axes[2, 2].set_visible(False)

    #         # for axes in g.axes.flat:
    #             # axes.title.set_position([0.5, 0.5])
    #             # axes.yaxis.labelpad = 40

    #         save_plot(join(PLOTS_DIR, names[i]), tight=False)
    #         plt.close()
        
    #     del df_all, df_tss, df_no_tss


    # @time_this
    # def plot_cpa_type(self):
    #     """Plot CPA ship domain"""
    #     limit = 2778  # 1.5 nm
    #     df = self._df[self._df['datetime'] == self._df['cpa_time']]
    #     limit_cond = df["cpa_distance"] < limit
        
    #     df_all = df[limit_cond]
    #     df_tss = df[
    #         limit_cond & (df["tss_1"] == True) & (df["tss_2"] == True)
    #     ]
    #     df_no_tss = df[
    #         limit_cond & (df["tss_1"] == False) & (df["tss_2"] == False)
    #     ]
    #     df_mix_tss = df[
    #         limit_cond & ((df["tss_1"] == True) & (df["tss_2"] == False)) | (df["tss_1"] == False) & (df["tss_2"] == True)
    #     ]

    #     dfs = [df_all, df_tss, df_no_tss]
    #     names = ["Ship_Domain", "Ship_Domain_In_TSS", "Ship_Domain_Out_TSS"]

    #     for i in [0, 1, 2]:
    #         plot_domain_tss(dfs[i], names[i])
    #     del df_all, df_tss, df_no_tss

    # @time_this
    # def plot_domain_encounter(self):
        # """Plot CPA ship domain based on encounter"""
        # limit = 2778
        # limit_cond = self.df["cpa_distance"] < limit

        # for encounter_type in ["head-on", "overtaking", "crossing"]:
        #     df_all = self.df[limit_cond & (self.df["encounter"] == encounter_type)]
        #     df_tss = self.df[
        #         limit_cond
        #         & (self.df["encounter"] == encounter_type)
        #         & (self.df["tss_1"] == True)
        #         & (self.df["tss_2"] == True)
        #     ]
        #     df_no_tss = self.df[
        #         limit_cond
        #         & (self.df["tss_1"] == False)
        #         & (self.df["tss_2"] == False)
        #         & (self.df["encounter"] == encounter_type)
        #     ]

        #     dfs = [df_all, df_tss, df_no_tss]
        #     names = [
        #         f"Ship_Domain_Encounter_{encounter_type}",
        #         f"Ship_Domain_In_TSS_Encounter_{encounter_type}",
        #         f"Ship_Domain_Out_TSS_Encounter_{encounter_type}",
        #     ]

        #     for i in [0, 1, 2]:
        #         plot_domain_tss(dfs[i], names[i])

        #     del df_all, df_tss, df_no_tss

    def encounter_table(self):
        """Return count of each encounter type by vessel type"""
        df = self.df 
        # add subset duplicates
        df.drop_duplicates(inplace=True)
        return pd.pivot_table(
            df, 
            values='mmsi_1', 
            index=['encounter', 'ownship'],
            columns=['target ship'], aggfunc='count'
        )


class Head_On_Table(Postgres_Table):

    """
    Create a table of only head-on encounters
    """

    def __init__(self, table, input_table):
        """
        Connect to default database.
        
        Args:
            table (string): Name of table
            input_table (string): Name of encounters table
        """
        super(Head_On_Table, self).__init__(table)
        self.cur = self.conn.cursor()
        self.input = input_table
        self._df = None

    def select_type(self):
        """Select only the crossing interactions"""
        sql = f"""
            CREATE TABLE {self.table} AS
            SELECT *
            FROM {self.input}
            WHERE encounter = 'head-on'
        """
        LOGGER.info("Selecting head-on interactions...")
        self.run_query(sql)

    def delete_reciprocal(self):
        """Delete duplicate reciprocal encounters"""
        sql = f"""
            DELETE FROM {self.table} t1
            USING {self.table} t2
            WHERE t1.mmsi_1 < t2.mmsi_1
            AND t1.mmsi_1 = t2.mmsi_2
            AND t1.cpa_distance = t2.cpa_distance
            AND t1.cpa_time = t2.cpa_time
        """
        LOGGER.info("Deleting reciprocal encounters...")
        self.run_query(sql)

    @property
    def df(self):
        """Return table dataframe"""
        if self._df is not None:
            return self._df
        cols = 'mmsi_1, trip_1, type_1, mmsi_2, trip_2, type_2, alteration_degrees_1, alteration_degrees_2, acceleration_1, acceleration_2, give_way_1, give_way_2'
        self._df = self.table_dataframe(select_col=cols)
        self._df["id"] = self._df.index
        self._df = pd.wide_to_long(
            self._df, 
            stubnames=[
                "mmsi", 
                "trip", 
                "type", 
                "alteration_degrees", 
                "acceleration",
                "give_way"
            ], 
            i ="id",
            j="ship", 
            sep='_'
        )
        return self._df

    def plot_alteration(self):
        """plot histogram of alterations"""
        # types = self._df.groupby(["type"])
        # g = sns.FacetGrid(self._df, col="type")
        g.map(self._df.kdeplot, "alteration_degrees")
        g.add_legend();
        
        # Limit to lower upper triangle of grid
        g.axes[1, 2].set_visible(False)
        g.axes[2, 1].set_visible(False)
        g.axes[2, 2].set_visible(False)

        for axes in g.axes.flat:
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

        save_plot(join(PLOTS_DIR, "Head-on Alteration"), tight=False)
        plt.close()



class Overtaking_Table(Postgres_Table):

    """
    Create a table of only overtaking encounters
    """

    def __init__(self, table, input_table):
        """
        Connect to default database.
        
        Args:
            table (string): Name of table
            input_table (string): Name of encounters table
        """
        super(Overtaking_Table, self).__init__(table)
        self.cur = self.conn.cursor()
        self.input = input_table
        self._df = None

    def select_type(self):
        """Select only the crossing interactions"""
        sql = f"""
            CREATE TABLE {self.table} AS
            SELECT *
            FROM {self.input}
            WHERE encounter = 'overtaking'
        """
        LOGGER.info("Selecting overtaking interactions...")
        self.run_query(sql)

    def delete_reciprocal(self):
        """
        Delete duplicate reciprocal encounters

        Results in ship1 = give way, ship2 = standon
        """
        sql = f"""
            DELETE FROM {self.table} t1
            USING {self.table} t2
            WHERE t1.ship_1_give_way = 0
            AND t1.cpa_distance = t2.cpa_distance
            AND t1.cpa_time = t2.cpa_time
        """
        LOGGER.info("Deleting reciprocal encounters...")
        self.run_query(sql)

    @property
    def df(self):
        """Return table dataframe"""
        if self._df is not None:
            return self._df
        cols = 'mmsi_1, trip_1, type_1, mmsi_2, trip_2, type_2, alteration_degrees_1, alteration_degrees_2, acceleration_1, acceleration_2, give_way_1, give_way_2'
        self._df = self.table_dataframe(select_col=cols)
        self._df["id"] = self._df.index
        self._df = pd.wide_to_long(
            self._df, 
            stubnames=[
                "mmsi", 
                "trip", 
                "type", 
                "alteration_degrees", 
                "acceleration",
                "give_way"
            ], 
            i ="id",
            j="ship", 
            sep='_'
        )
        return self._df

    def plot_alteration(self):

        fig, ax = plt.subplots()
        sns.distplot(self._df[self._df['give_way']==0]["alteration_degrees"], kde=False, label="give-way")
        sns.distplot(self._df[self._df['give_way']==1]["alteration_degrees"], kde=False, label="stand-on")
        plt.legend()
        save_plot(join(PLOTS_DIR, "Overtaking Alteration"), tight=False)
        plt.close()

    def plot_acceleration(self):

        fig, ax = plt.subplots()
        sns.distplot(self._df[self._df['give_way']==0]["acceleration"], kde=False, label="give-way")
        sns.distplot(self._df[self._df['give_way']==1]["acceleration"], kde=False, label="stand-on")
        plt.legend()
        save_plot(join(PLOTS_DIR, "Overtaking Alteration"), tight=False)
        plt.close()



class Crossing_Table(Postgres_Table):

    """
    Create a table of only crossing encounters
    """

    def __init__(self, table, input_table):
        """
        Connect to default database.
        
        Args:
            table (string): Name of table
            input_table (string): Name of encounters table
        """
        super(Crossing_Table, self).__init__(table)
        self.cur = self.conn.cursor()
        self.input = input_table
        self._df = None

    def select_type(self):
        """Select only the crossing interactions"""
        sql = f"""
            CREATE TABLE {self.table} AS
            SELECT *
            FROM {self.input}
            WHERE encounter = 'crossing'
        """
        LOGGER.info("Selecting crossing interactions...")
        self.run_query(sql)

    def delete_reciprocal(self):
        """
        Delete duplicate reciprocal encounters

        Results in ship1 = give way, ship2 = standon
        """
        sql = f"""
            DELETE FROM {self.table} t1
            USING {self.table} t2
            WHERE t1.ship_1_give_way = 0
            AND t1.cpa_distance = t2.cpa_distance
            AND t1.cpa_time = t2.cpa_time
        """
        LOGGER.info("Deleting reciprocal encounters...")
        self.run_query(sql)

    @property
    def df(self):
        """Return table dataframe"""
        if self._df is not None:
            return self._df
        cols = 'mmsi_1, trip_1, type_1, mmsi_2, trip_2, type_2, alteration_degrees_1, alteration_degrees_2, acceleration_1, acceleration_2'
        self._df = self.table_dataframe(select_col=cols)
        self._df.rename(columns = {"alteration_degrees_1": "give_way_alteration", "alteration_degrees_2": "stand_on_alteration", "acceleration_1":"give_way_acceleration", "acceleration_2": "stand_on_acceleration"}, inplace=True)
        return self._df

    def plot_alteration(self):

        g = sns.FacetGrid(self.df, col="type_1", row = "type_2")
        g.map(sns.distplot, "give_way_alteration", kde=True)
        g.map(sns.distplot, "stand_on_alteration", kde=True)
        g.add_legend();














    # def plot_encounters(self):

    #     df = self.table.table_dataframe()
    #     for name, group in df.groupby(['mmsi_1','track1','mmsi2','track2']):
    #         name = 'Plot_{0}_{1}_{2}_{3}.png'.format()
    #         fig = plt.figure(figsize=(12, 4), dpi=None, facecolor='white')
    #         fig.suptitle('Track Comparison', fontsize=12, fontweight='bold', x=0.5, y =1.01)
    #         plt.title(' => '.join([str(i) for i in name]), fontsize=10, loc='center')
    #         # plt.yticks([])


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
#     def __init__(self, conn, table, input_analysis):
#         """Connect to default database."""
#         super(Crossing_Table, self).__init__(conn, table)
#         self.cur = self.conn.cursor()
#         self.input = input_analysis
#         self.colregs = "{0}_colregs".format(self.table)
#         self.others = "{0}_others".format(self.table)

#     def select_type(self):
#         """Select only the crossing interactions."""
#         LOGGER.info("Selecting crossing interactions...")
#         sql = f"""
#             CREATE TABLE {self.table} AS
#             SELECT
#                 *,
#                 avg(stand_on_cog_cos) OVER (PARTITION BY mmsi1, track1, mmsi2, track2) AS avg_cog_cos,
#                 avg(stand_on_accel) OVER (PARTITION BY mmsi1, track1, mmsi2, track2) AS avg_accel
#             FROM {1}
#             WHERE encounter = 'crossing'
#         """.format(
#             self.table, self.input
#         )
#         self.cur.execute(sql)
#         self.conn.commit()

#     def separate_colregs(self):
#         """Create table of colregs compliant interactions."""
#         LOGGER.info("Selecting colreg compliant interactions...")
#         sql_colregs = """
#             CREATE TABLE {0} AS
#             SELECT *
#             FROM {1}
#             WHERE avg_cog_cos >= 0.999
#             AND avg_accel <= abs(10)
#         """.format(
#             self.colregs, self.table
#         )
#         self.cur.execute(sql_colregs)
#         self.conn.commit()

#         LOGGER.info("Selecting colreg compliant interactions...")
#         sql_others = """
#             CREATE TABLE {0} AS
#             SELECT *
#             FROM {1}
#             WHERE avg_cog_cos < 0.999
#             AND avg_accel > abs(10)
#         """.format(
#             self.others, self.table
#         )
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
#         LOGGER.info('Selecting overtaking interactions...')
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
#         LOGGER.info('Selecting colreg compliant interactions...')
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
#         LOGGER.info('Selecting colreg compliant interactions...')
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
# #         LOGGER.info('Joining nais_points with itself to make near table...')
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


# def track_changes(self):
#     '''Reorganize data into give way and stand on.'''
#     LOGGER.info('Adding course info to {0}'.format(self.table))
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
#  class Encounter_Table(Postgres_Table):


#

#
#     def check_nearby(self):
#         '''Delete encounters that are just nearby and not really encounter.'''
#         LOGGER.info('Deleting near ships with no encounter...')
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
