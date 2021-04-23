#!/usr/bin/env python
"""
.. script::
    :language: Python Version 3.7.4
    :platform: Windows 10
    :synopsis: build basic tables in Postgres

.. moduleauthor:: Maura Rowell <mkrowell@uw.edu>
"""


# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------
from os.path import abspath, dirname, join
import sys
import yaml

sys.path.append(abspath("."))
import src.database
import src.clean


# ------------------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------------------
city = "seattle"
srid = 32610
year = "2017"
months = ["07"]

# Load the parameters for the seattle region
parameters_file = abspath(join(dirname(__file__), "..", "src", "settings.yaml"))
with open(parameters_file, "r") as stream:
    parameters = yaml.safe_load(stream)[city]


# ------------------------------------------------------------------------------
# BUILD SHORELINE TABLE
# ------------------------------------------------------------------------------
# Create a Postgres table from the shoreline shapefile
shapefile_shore = src.download.Shapefile_Download("shore").output
table_shore = src.database.Postgres_Table("shore")
table_shore.create_table(filepath=shapefile_shore)

# Transform to UTM 10 SRID
table_shore.project_column("geom", "MULTILINESTRING", srid)
table_shore.alter_storage("geom")
table_shore.add_index("idx_geom_shore", "geom", gist=True)
table_shore.add_index("idx_region", "regions", gist=False)

# Keep only the relevant shore line
table_shore.reduce_table("regions", "!=", parameters["region"])


# ------------------------------------------------------------------------------
# BUILD TSS TABLE
# ------------------------------------------------------------------------------
# Create a Postgres table from the tss shapefile"
shapefile_tss = src.download.Shapefile_Download("tss").output
table_tss = src.database.Postgres_Table("tss")
table_tss.create_table(filepath=shapefile_tss)

# Transform to UTM 10 SRID
table_tss.project_column("geom", "MULTIPOLYGON", srid)
table_tss.add_index("idx_geom_tss", "geom", gist=True)
table_tss.add_index("idx_objl", "objl", gist=False)

# Keep only the relevant TSS
table_tss.reduce_table("objl", "!=", parameters["tss"])


# ------------------------------------------------------------------------------
# BUILD FERRY TERMINALS TABLE
# ------------------------------------------------------------------------------
# Create a Postgres table from the ferry terminals shapefile"
shapefile_terminals = src.download.Shapefile_Download("ferry_terminals").output
table_terminal = src.database.Postgres_Table("ferry_terminals", srid="ESRI:102749")
table_terminal.create_table(filepath=shapefile_terminals)

# Transform to UTM 10 SRID
table_terminal.project_column("geom", "POINT", srid)
table_terminal.add_index("idx_geom_terminal", "geom", gist=True)


# ------------------------------------------------------------------------------
# BUILD FERRY ROUTES TABLE
# ------------------------------------------------------------------------------
# Create a Postgres table from the ferry routes shapefile"
shapefile_routes = src.download.Shapefile_Download("ferry_routes").output
table_routes = src.database.Postgres_Table("ferry_routes", srid="ESRI:102749")
table_routes.create_table(filepath=shapefile_routes)

# Transform to UTM 10 SRID
table_routes.project_column("geom", "MULTILINESTRING", srid)
table_routes.add_index("idx_geom_route", "geom", gist=True)


# ------------------------------------------------------------------------------
# BUILD POINTS TABLE
# ------------------------------------------------------------------------------
table_points = src.database.Points_Table("points_07")
table_points.drop_table()
table_points.create_table(columns=table_points.columns)

nais = src.clean.NAIS_Cleaner(city, year)
for month in months:
    nais.month = month
    table_points.copy_data(nais.csv_processed)

table_points.add_geometry()

# Indexes (primary key MMSI, Trip, DateTime)
table_points.add_index("idx_mmsi", "mmsi")
table_points.add_index("idx_datetime", "datetime")
table_points.add_index("idx_geom_points", "geom", gist=True)
table_points.add_index("idx_track", "mmsi, trip, vesseltype")

table_points.add_tss("tss")
table_points.plot_tss()
table_points.add_terminal("ferry_terminals")


# ------------------------------------------------------------------------------
# BUILD TRACKS TABLE
# ------------------------------------------------------------------------------
table_tracks = src.database.Tracks_Table("tracks_07")
table_tracks.drop_table()
table_tracks.convert_to_tracks("points_07")
table_tracks.add_length()
table_tracks.add_displacement()
table_tracks.add_index("idx_period", "period", btree=True)
table_tracks.add_od("points_07")
table_tracks.plot_tracks()

# ------------------------------------------------------------------------------
# BUILD CPA TABLE
# ------------------------------------------------------------------------------
table_cpa = src.database.CPA_Table("cpa_07", "tracks_07", "shore")
table_cpa.drop_table()
table_cpa.tracks_tracks()
table_cpa.add_index("idx_cpa_distance", "cpa_distance")
# Limit to 4NM
table_cpa.reduce_table("cpa_distance", ">", 1852*4)
table_cpa.cpa_time()
table_cpa.cpa_points()
table_cpa.cpa_line()
table_cpa.alter_storage("cpa_line")
table_cpa.add_index("idx_cpa_line", "cpa_line", gist=True)
table_cpa.delete_shore_cross()
table_cpa.add_index("idx_cpa_time", "cpa_time")


# ------------------------------------------------------------------------------
# BUILD ENCOUNTERS TABLES
# ------------------------------------------------------------------------------
table_encounters = src.database.Encounters_Table("encounters_07", "points_07", "cpa_07")
table_encounters.drop_table()
table_encounters.cpa_points()
table_encounters.tcpa()
table_encounters.dcpa()
table_encounters.encounter_type()
table_encounters.mark_ho_passing()
table_encounters.give_way_info()
table_encounters.mark_true()
table_encounters.tss_clearance("tss")

table_encounters.plot_distance_type()
table_encounters.plot_ship_domain(hue="target ship in TSS")

table_encounters.plot_tss_clearance()
table_encounters.drop_sparse_encounters()

table_encounters.plot_ferry()

table_encounters.head_on_table()
table = table_encounters.encounter_table()

model = table_encounters.regression()
print(model.summary())
ftests = table_encounters.hypotheses()


# ------------------------------------------------------------------------------
# BUILD TSS INTERACTIONS
# ------------------------------------------------------------------------------
table_tss_int = src.database.TSS_Intersection_Table("intersections_07", "points_07", "tss")
table_tss_int.drop_table()
table_tss_int.select_intersections()
table_tss_int.add_direction()
table_tss_int.get_tss_heading()
table_tss_int.get_entrance_angle()
table_tss_int.plot_angles()
