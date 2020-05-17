#!/usr/bin/env python
"""
.. module::
    :language: Python Version 3.6.8
    :platform: Windows 10
    :synopsis: process raw NAIS broadcast data points

.. moduleauthor:: Maura Rowell <mkrowell@uw.edu>
"""


# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------
import angles
import datetime
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import abspath, dirname, exists, join
import pandas as pd
import seaborn as sns
from shapely.geometry import Point
import yaml

import src
from src import print_reduction, print_reduction_gdf, time_all, LOGGER


# ------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------
METERS_IN_NM = 1852
EARTH_RADIUS_KM = 6371
PLOTS_DIR = abspath(join(dirname(__file__), "..", "reports", "figures"))

# ------------------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------------------
plt.style.use("seaborn")
sns.set(color_codes=True)


# ------------------------------------------------------------------------------
# SPATIAL FUNCTIONS
# ------------------------------------------------------------------------------
def azimuth_utm(point1, point2):
    """Return the azimuth from point1 to point2 in UTM"""
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
def save_plot(filepath, tight=True):
    """Save and close figure"""
    if exists(filepath):
        os.remove(filepath)
    if tight:
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.2)
    plt.savefig(filepath)


def plot_type(df, prepend):
    """Create a chart of the number of unique vessels per type"""
    filename = f"{prepend}_Type"
    fig, ax = plt.subplots()
    unique = df[["MMSI", "VesselType"]].drop_duplicates(keep="first")
    sns.countplot("VesselType", data=unique, palette="Paired")
    ax.set_ylabel("Number of Unique MMSI")
    save_plot(join(PLOTS_DIR, filename))
    plt.close(fig)


def plot_sog(df, prepend):
    """Create a chart of SOGs observed"""
    filename = f"{prepend}_SOG.png"
    fig, ax = plt.subplots()
    sns.distplot(df["SOG"], kde=False).set_title(f"{prepend} Data")
    ax.set_ylabel("Number of Data Points")
    ax.set_xlabel("SOG (nautical miles/hour)")
    if prepend == "Raw":
        plt.xlim(-50, 50)
    else:
        plt.xlim(0, 50)
    filepath = join(PLOTS_DIR, filename)
    save_plot(filepath)
    plt.close(fig)


def plot_acceleration(df):
    """Create a chart of acceleration observed"""
    filename = f"Acceleration.png"
    filepath = join(PLOTS_DIR, filename)
    fig, ax = plt.subplots()
    sns.distplot(df["Acceleration"], kde=False)
    ax.set_ylabel("Number of Data Points")
    ax.set_xlabel("Acceleration (nautical miles/seconds^2)")
    plt.xlim(-300, 300)
    save_plot(filepath)
    plt.close(fig)


def plot_alteration(df):
    """Create a chart of alteration observed"""
    filename = f"Alteration_Degrees.png"
    filepath = join(PLOTS_DIR, filename)
    fig, ax = plt.subplots()
    sns.distplot(df["Alteration_Degrees"], kde=False)
    ax.set_ylabel("Number of Data Points")
    ax.set_xlabel("Alteration (Degrees)")
    plt.xlim(-180, 180)
    save_plot(filepath)
    plt.close(fig)


def plot_trip_length(df):
    """Create a chart of trip length observed"""
    filename = f"Trip_Length.png"
    filepath = join(PLOTS_DIR, filename)
    fig, ax = plt.subplots()
    trips = df.pivot_table(
        index=["MMSI", "Trip", "VesselType"], values="Step_Distance", aggfunc=np.sum
    ).reset_index()
    cargo = trips.loc[trips["VesselType"] == "cargo"]
    ferry = trips.loc[trips["VesselType"] == "ferry"]
    tanker = trips.loc[trips["VesselType"] == "tanker"]
    sns.distplot(cargo[["Step_Distance"]], kde=True, hist=False, label="cargo")
    sns.distplot(ferry[["Step_Distance"]], kde=True, hist=False, label="ferry")
    sns.distplot(tanker[["Step_Distance"]], kde=True, hist=False, label="tanker")
    ax.set_ylabel("Number of Trips")
    ax.set_xlabel("Trip Length (m)")
    plt.legend()
    save_plot(filepath)
    plt.close(fig)


def plot_step_length(df, prepend):
    """Create a chart of step length observed"""
    filename = f"{prepend}_Step_Length.png"
    filepath = join(PLOTS_DIR, filename)
    fig, ax = plt.subplots()
    sns.distplot(60 * df["Step_Distance"] / df["Interval"], kde=True).set_title(
        "Step Length"
    )
    ax.set_ylabel("Number of Steps")
    ax.set_xlabel("Length (m)")
    save_plot(filepath)
    plt.close(fig)


def plot_latlon(df):
    """create a distribution of lat/lon"""
    filename = f"Spatial_Distribution.png"
    filepath = join(PLOTS_DIR, filename)
    fig, ax = plt.subplots()
    sns.jointplot(x="LON", y="LAT", data=df, kind="kde").set_title(
        "Spatial Distribution"
    )
    save_plot(filepath)
    plt.close(fig)


def plot_cog(df, prepend):
    """Create a chart of COGs observed by type"""
    filename = f"{prepend}_COG.png"
    groups = df.groupby("VesselType")["COG"]
    group_list = [(index, group) for index, group in groups if len(group) > 0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    for k, v in group_list:
        radians = np.deg2rad(v)
        sns.distplot(radians, kde=False).set_title("f{prepend} Data")
        radians.hist(label=k, alpha=0.3, ax=ax)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.legend(loc="upper center", bbox_to_anchor=(0, 1.05), ncol=1, fancybox=True)
    save_plot(join(PLOTS_DIR, filename))
    plt.close(fig)


def plot_time(df, prepend):
    """Create a chart of time intervals observed"""
    filename = f"{prepend}_ToD.png"
    filepath = join(PLOTS_DIR, filename)
    fig, ax = plt.subplots()
    sns.distplot(df["BaseDateTime"], kde=False)
    ax.set_ylabel("Number of Data Points")
    ax.set_xlabel("Time of Day (UTC)")
    save_plot(filepath)
    plt.close(fig)


# ------------------------------------------------------------------------------
# CLEAN
# ------------------------------------------------------------------------------
@time_all
class Basic_Clean(object):

    """
    Clean NAIS data
    """

    def __init__(self, csvFile, minPoints, lonMin, lonMax, latMin, latMax):
        """Initialize attributes and dataframe."""
        self.csv = csvFile
        self.cleaned = self.csv.replace("raw", "cleaned")
        self.month = int(self.csv.split("_")[-2])

        # Spatial parameters
        self.minPoints = minPoints
        self.lonMin = lonMin
        self.lonMax = lonMax
        self.latMin = latMin
        self.latMax = latMax

        # Create raw NAIS dataframe
        self.df = pd.read_csv(self.csv)
        self.required = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading"]
        self.df["BaseDateTime"] = pd.to_datetime(self.df["BaseDateTime"])
        self.df.sort_values("MMSI", inplace=True)

        # Standardize missing values
        self.df["Status"].replace(np.nan, "undefined", inplace=True)
        self.df["Heading"].replace(511, np.nan, inplace=True)

        # Standardize vessel type and cog
        self._map_vessel_types()
        self._normalize_angles()

        # Reduce to area of interest
        self._select_spatial()

        # Save raw df for plotting purposes
        self.df_raw = self.df.copy()

    # MAIN FUNCTIONS
    @print_reduction
    def clean_raw(self):
        """
        Select the area of interest, remove duplicate information, remove
        contradictory information, remove invalid IDs, normalize, write output.
        """
        self._drop_null()
        self._drop_duplicate_rows()

        self._drop_duplicate_keys()
        self._drop_inconsistent_info()
        self._drop_columns()

        self._drop_bad_mmsi()
        self._filter_sog(3)
        self._filter_time(3)

        self._drop_sparse_mmsi()

        self._filter_type()
        self._filter_status()

        self._plot()

        self.df.sort_values(["MMSI", "BaseDateTime"], inplace=True)
        self.df.to_csv(self.cleaned, index=False, header=True)

    # DATAFRAME CLEANING
    def _select_spatial(self):
        """Limit data to bounding box of interest."""
        self.df = self.df[
            (self.df["LON"].between(self.lonMin, self.lonMax))
            & (self.df["LAT"].between(self.latMin, self.latMax))
        ].copy()

    @print_reduction
    def _drop_null(self):
        """
        Drop rows with nulls in all the required columns. 
        No loss of information.
        """
        self.df.replace("", np.nan, inplace=True)
        self.df.dropna(how="any", subset=self.required, inplace=True)

    @print_reduction
    def _drop_duplicate_rows(self):
        """
        Remove entirely duplicated rows. No loss of information.
        """
        self.df.drop_duplicates(keep="first", inplace=True)

    @print_reduction
    def _drop_duplicate_keys(self):
        """
        MMSI, BaseDateTime pairs must be unique. Can't calculate step 
        calculations with duplicate timestamps. Drop both duplicate rows.
        """
        key = ["MMSI", "BaseDateTime"]
        self.df.drop_duplicates(subset=key, keep=False, inplace=True)

    @print_reduction
    def _drop_inconsistent_info(self):
        """
        Confirm that a MMSI is associated with only one name, dimension.
        A mix of vessels using the same MMSI will not be included.
        This data is entered only once, so a mistake in entry will appear
        in all data points and not change over time.
        """
        mmsi = self.df.groupby(["MMSI"])
        self.df = mmsi.filter(lambda g: g["VesselName"].nunique() <= 1)
        self.df = mmsi.filter(lambda g: g["Length"].nunique() <= 1)
        self.df = mmsi.filter(lambda g: g["Width"].nunique() <= 1)

    def _drop_columns(self):
        """Remove unneccessary columns."""
        unused = ["CallSign", "IMO", "Cargo", "Width", "Draft"]
        self.df.drop(columns=unused, inplace=True)

    @print_reduction
    def _drop_bad_mmsi(self):
        """
        MMSI numbers should be 9 digits and between a given range.
        Remove any rows with MMSIs outside of that range."""
        condRange = self.df["MMSI"].between(201000000, 775999999)
        self.df = self.df[condRange]

    @print_reduction
    def _filter_sog(self, limit):
        """Limit to points with > limit SOG."""
        self.df["SOG"] = self.df["SOG"].abs()
        self.df = self.df[self.df["SOG"] > limit]

    @print_reduction
    def _filter_time(self, limit):
        """Limit to points less than 3 minutes from prior data point."""
        col = "Interval"
        group = self.df.sort_values(["MMSI", "BaseDateTime"]).groupby("MMSI")

        self.df[col] = group["BaseDateTime"].diff()
        self.df[col].fillna(datetime.timedelta(seconds=60), inplace=True)
        self.df[col] = self.df[col].astype("timedelta64[s]")
        self.df = self.df[self.df["Interval"] < limit * 60]

    @print_reduction
    def _drop_sparse_mmsi(self):
        """Remove MMSIs with few data points."""
        self.df = self.df.groupby(["MMSI"]).filter(lambda g: len(g) > self.minPoints)

    def _normalize_angles(self):
        """Normalize COG to an angle between [0, 360)."""
        self.df["COG"] = self.df["COG"].apply(lambda x: angles.normalize(x, 0, 360))
        self.df["Heading"] = self.df["Heading"].apply(
            lambda x: angles.normalize(x, 0, 360)
        )

    def _map_vessel_types(self):
        """Map codes to categories."""
        type_dict = abspath(join(dirname(__file__), "vessel_types.yaml"))
        with open("src\\vessel_types.yaml", "r") as stream:
            v_map = yaml.safe_load(stream)

        self.df["VesselType"].replace("", np.nan, inplace=True)
        self.df["VesselType"] = self.df["VesselType"].map(v_map)
        self.df["VesselType"] = self.df["VesselType"].replace(np.nan, "unknown")
        self.df["VesselType"] = self.df["VesselType"].astype("category")

    @print_reduction
    def _filter_type(self):
        """Filter non-normal operating vessels"""
        types = ["tanker", "cargo", "ferry"]
        self.df = self.df[self.df["VesselType"].isin(types)]

    @print_reduction
    def _filter_status(self):
        """Filter non-normal stauts"""
        status = [
            "not under command",
            "restricted maneuverability",
            "engaged in fishing",
            "power-driven vessel towing astern",
            "reserved for future use (9)",
            "power-driven vessel pushing ahead or towing alongside",
        ]
        self.df = self.df[~self.df["Status"].isin(status)]

    def _plot(self):
        """Plot change in speed"""
        plot_sog(self.df_raw, "Raw")
        plot_type(self.df_raw, "Raw")
        plot_sog(self.df, "Cleaned")


# ------------------------------------------------------------------------------
# TRIP GENERATION
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
        self.df["BaseDateTime"] = pd.to_datetime(self.df["BaseDateTime"])
        self.df.sort_values(["MMSI", "BaseDateTime"], inplace=True)

        # Create and project geopandas dataframe
        self.gdf = gpd.GeoDataFrame(
            self.df,
            geometry=gpd.points_from_xy(self.df["LON"], self.df["LAT"]),
            crs={"init": "epsg:4326"},
        )
        self.gdf = self.gdf.to_crs(32610)
        self.gdf["LAT_UTM"] = self.gdf["geometry"].y
        self.gdf["LON_UTM"] = self.gdf["geometry"].x

    @property
    def grouped_mmsi(self):
        """Return sorted dataframe grouped by MMSI."""
        return self.gdf.sort_values(["MMSI", "BaseDateTime"]).groupby("MMSI")

    @property
    def grouped_trip(self):
        """Return sorted dataframe grouped by MMSI."""
        return self.gdf.sort_values(["MMSI", "Trip", "BaseDateTime"]).groupby(
            ["MMSI", "Trip"]
        )

    # MAIN METHOD --------------------------------------------------------------
    def preprocess(self):
        """
        Detect suspicious data.
        """
        self._update_interval()
        self._mark_trips()
        self._drop_sparse_trips()

        self._mark_distance(1.25)

        self._update_interval()
        self._mark_trips()
        self._drop_sparse_trips()

        self._step_cog()
        self._acceleration()
        self._alteration()

        self._normalize_time()
        self._write_output()
        self._plot()

    def _plot(self):
        """Plot data before and after its been cleaned"""
        plot_acceleration(self.gdf)
        plot_alteration(self.gdf)
        plot_trip_length(self.gdf)
        # plot_latlon(self.gdf)
        plot_step_length(self.gdf, "Processed")

    # PREPROCESSING ------------------------------------------------------------
    def _update_interval(self):
        """Update time interval after data cleaning."""
        self.gdf["Interval"] = self.grouped_mmsi["BaseDateTime"].diff()
        self.gdf["Interval"].fillna(datetime.timedelta(seconds=0), inplace=True)
        self.gdf["Interval"] = self.gdf["Interval"].astype("timedelta64[s]")

    def _mark_trips(self):
        """Make trips at time breaks"""
        self.gdf["Break"] = np.where(
            (self.gdf["Interval"] == 0) | (self.gdf["Interval"] > 3 * 60), 1, 0
        )
        self.gdf["Trip"] = self.grouped_mmsi["Break"].cumsum()
        self.gdf["Trip"] = self.gdf["Trip"] + 10000 * self.month
        self.gdf.drop(columns=["Break"], inplace=True)

    @print_reduction_gdf
    def _drop_sparse_trips(self):
        """Remove MMSIs with few data points."""
        self.gdf = self.grouped_trip.filter(lambda g: len(g) > self.minPoints)

    def _step_distance(self):
        """Return distance between lat/lon positions."""

        def distance(df):
            df.reset_index(inplace=True)
            df["Step_Distance"] = df["geometry"].distance(df["geometry"].shift())
            return df.set_index("index")

        self.gdf = self.grouped_trip.apply(distance)
        self.gdf["Step_Distance"].replace("", np.nan, inplace=True)
        self.gdf["Step_Distance"].fillna(0, inplace=True)
        self.gdf["Step_Distance"] / METERS_IN_NM

    @print_reduction_gdf
    def _mark_distance(self, limit):
        """Compare step_distance to speed*time. Remove suspicious data."""
        self._step_distance()
        plot_step_length(self.gdf, "Cleaned")
        self.gdf["Expected_Distance"] = (
            self.gdf["SOG"] * self.gdf["Interval"] * METERS_IN_NM
        ) / 3600

        self.gdf["Outlier"] = np.where(
            self.gdf["Step_Distance"] >= self.gdf["Expected_Distance"] * limit, 1, 0
        )
        self.gdf = self.gdf[self.gdf["Outlier"] == 0]
        self.gdf.drop(columns=["Expected_Distance", "Outlier"], inplace=True)

    def _step_cog(self):
        """Calculate the course between two position points."""

        def course_utm(df):
            df.reset_index(inplace=True)
            df["Step_Azimuth"] = azimuth_utm(
                df["geometry"].shift(), df.loc[1:, "geometry"]
            )
            return df.set_index("index")

        # Calculate and normalize course between successive points
        self.gdf = self.grouped_trip.apply(course_utm)
        self.gdf["Step_Azimuth"].fillna(method="bfill", inplace=True)
        self.gdf["Step_Azimuth"] = round(
            self.gdf["Step_Azimuth"].apply(lambda x: angles.normalize(x, 0, 360))
        )

        # Caclulate error
        self.gdf["Error_COG"] = 180 - abs(
            abs(self.gdf["COG"] - self.gdf["Step_Azimuth"]) - 180
        )
        self.gdf["Error_Heading"] = 180 - abs(
            abs(self.gdf["Heading"] - self.gdf["Step_Azimuth"]) - 180
        )

    def _acceleration(self):
        """Add acceleration."""
        self.gdf["DS"] = self.grouped_trip["SOG"].diff()
        self.gdf["Acceleration"] = 3600 * self.gdf["DS"].divide(
            self.df["Interval"], fill_value=0
        )
        self.gdf.drop(columns=["DS"], inplace=True)
        self.gdf["Acceleration"].fillna(method="bfill", inplace=True)

    def _alteration(self):
        """Calculate change in heading."""

        def delta_heading(df):
            df.reset_index(inplace=True)
            df["Alteration"] = angle_difference(
                df["Heading"].shift(), df.loc[1:, "Heading"]
            )
            return df.set_index("index")

        self.gdf = self.grouped_trip.apply(delta_heading)
        self.gdf["Alteration"].fillna(method="bfill", inplace=True)
        self.gdf["Alteration_Degrees"] = np.degrees(self.gdf["Alteration"])
        self.gdf["Alteration_Cosine"] = np.cos(self.gdf["Alteration"])

    @print_reduction_gdf
    def _normalize_time(self):
        """Round time to nearest minute."""
        self.gdf["DateTime"] = self.gdf["BaseDateTime"].dt.round("1min")
        self.gdf.drop_duplicates(
            subset=["MMSI", "Trip", "DateTime"], keep="first", inplace=True
        )

    def _replace_nan(self):
        """Fill undefined values with default float value."""
        for col in ["Length", "VesselType"]:
            self.gdf[col].replace("", np.nan, inplace=True)
            self.gdf[col].replace(np.nan, -1, inplace=True)

    def _write_output(self):
        """Write to one processed CSV file"""
        self._replace_nan()
        columns = [
            "MMSI",
            "Trip",
            "DateTime",
            "LAT",
            "LAT_UTM",
            "LON",
            "LON_UTM",
            "SOG",
            "COG",
            "Heading",
            "Step_Distance",
            "Step_Azimuth",
            "Acceleration",
            "Alteration",
            "Alteration_Degrees",
            "Alteration_Cosine",
            "VesselName",
            "VesselType",
            "Status",
            "Length",
        ]
        self.gdf = self.gdf[columns]
        self.gdf.reindex(columns)
        self.gdf.sort_values(["MMSI", "Trip", "DateTime"], inplace=True)
        self.gdf.to_csv(self.csv_processed, index=False, header=False)


# ------------------------------------------------------------------------------
# DATAFRAMES


def mmsi_plot(self):
    """Plot raw trajectory for each MMSI."""
    for name, group in self.grouped_trip:
        try:
            mmsi = group["MMSI"].unique()[0]
            print("Plotting MMSI %s" % mmsi)
            data = [["LAT"], ["LON"], ["SOG"], ["COG"], ["Time Interval"]]
            legend = [False, False, "full", "full", False]
            linestyles = ["", "-", "--"]

            # Create plots
            fig, axes = plt.subplots(5, 1, sharex="col", figsize=(8, 11))
            fig.suptitle("MMSI: {0}".format(mmsi), fontsize=14)
            for i in range(5):
                for j in range(len(data[i])):
                    sns.lineplot(
                        x="BaseDateTime",
                        y=data[i][j],
                        style=linestyle[j],
                        data=group,
                        palette=sns.color_palette("paired", len(data[i])),
                        ax=axes[i],
                        label=data[i][j],
                        legend=legend[i],
                    )
                    axes[i].set_ylabel(data[i][0])

            # Format plot area
            plt.xticks(rotation=70)
            plt.tight_layout()
            fig.subplots_adjust(top=0.9)

            # Save to directory
            plt.savefig(join(DIRECTORY_PLOTS, "MMSI", "Raw", "{0}.png".format(mmsi)))
            plt.close()
        except:
            continue
