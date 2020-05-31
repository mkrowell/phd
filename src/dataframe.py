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
from contextlib import contextmanager
from os.path import abspath, dirname, exists, join
from shapely.geometry import Point
import angles
import datetime
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import yaml

# import src
from src import print_reduction, print_reduction_gdf, time_all, LOGGER


# ------------------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------------------
sns.set_palette("Set1")
sns.set_style("darkgrid")


# ------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------
METERS_IN_NM = 1852
PLOTS_DIR = abspath(join(dirname(__file__), "..", "reports", "figures"))


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
@contextmanager
def plot(filepath, tight=True):
    """Context manager for opening, saving, and closing figure"""
    fig, ax = plt.subplots()
    yield ax
    save_plot(filepath, tight)
    plt.close(fig)


def save_plot(filepath, tight):
    """Save figure"""
    if exists(filepath):
        os.remove(filepath)
    if tight:
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.2)
    plt.savefig(filepath)


def plot_type(df, prepend, destination):
    """Create a chart of the number of unique vessels per type"""
    with plot(join(destination, f"{prepend}_Type")) as ax:
        unique = df[["MMSI", "VesselType", "Missing_Heading"]].drop_duplicates(
            keep="first"
        )
        sns.countplot(
            "VesselType", data=unique, hue="Missing_Heading"
        )  # palette="Paired",
        ax.set_ylabel("Number of Unique MMSI")


def plot_sog(df, prepend, destination):
    """Create a chart of SOGs observed"""
    with plot(join(destination, f"{prepend}_SOG")) as ax:
        sns.distplot(df["SOG"], kde=False).set_title(f"{prepend} Data")
        ax.set_ylabel("Number of Data Points")
        ax.set_xlabel("SOG (nautical miles/hour)")
        if prepend == "Raw":
            plt.xlim(-50, 50)
        else:
            plt.xlim(0, 50)


def plot_acceleration(df, destination):
    """Create a chart of acceleration observed"""
    with plot(join(destination, f"Acceleration")) as ax:
        sns.distplot(df["Acceleration"], kde=False)
        ax.set_ylabel("Number of Data Points")
        ax.set_xlabel("Acceleration (meters/second^2)")
        plt.xlim(-0.2, 0.2)


def plot_alteration(df, destination):
    """Create a chart of alteration observed"""
    with plot(join(destination, f"Alteration_Degrees")) as ax:
        sns.distplot(df["Alteration_Degrees"], kde=False)
        ax.set_ylabel("Number of Data Points")
        ax.set_xlabel("Alteration (Degrees)")
        plt.xlim(-180, 180)


def plot_trip_length(df, destination):
    """Create a chart of trip length observed"""
    with plot(join(destination, f"Trip_Length")) as ax:
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


def plot_step_length(df, prepend, destination):
    """Create a chart of step length observed"""
    with plot(join(destination, f"{prepend}_Step_Length")) as ax:
        df["Step_Distance"].fillna(0, inplace=True)
        sns.distplot(60 * df["Step_Distance"] / df["Interval"], kde=False).set_title(
            "Step Length"
        )
        ax.set_ylabel("Number of Steps")
        ax.set_xlabel("Length (m)")


def plot_latlon(df, destination):
    """create a distribution of lat/lon"""
    with plot(join(destination, f"Spatial_Distribution")) as ax:
        sns.jointplot(x="LON", y="LAT", data=df, kind="kde").set_title(
            "Spatial Distribution"
        )


def plot_cog(df, prepend, destination):
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
    save_plot(join(destination, filename))
    plt.close(fig)


# ------------------------------------------------------------------------------
# CLEAN
# ------------------------------------------------------------------------------
@time_all
class Basic_Clean(object):

    """
    Clean NAIS data
    """

    def __init__(
        self,
        csvFile,
        minPoints,
        lonMin1,
        lonMax1,
        latMin1,
        latMax1,
        lonMin2,
        lonMax2,
        latMin2,
        latMax2,
    ):
        """Initialize attributes and dataframe."""
        self.csv = csvFile
        self.csv_study_area = self.csv.replace("raw", "study_area")
        self.cleaned = self.csv.replace("raw", "cleaned")
        self.month = int(self.csv.split("_")[-2])
        self.plot_dir = abspath(join("reports", "figures", str(self.month)))
        os.makedirs(self.plot_dir, exist_ok=True)

        # Spatial parameters
        self.minPoints = minPoints
        self.lonMin1 = lonMin1
        self.lonMax1 = lonMax1
        self.latMin1 = latMin1
        self.latMax1 = latMax1

        self.lonMin2 = lonMin2
        self.lonMax2 = lonMax2
        self.latMin2 = latMin2
        self.latMax2 = latMax2

        # Create raw NAIS dataframe
        self.required = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading"]
        if exists(self.csv_study_area):
            self.df = pd.read_csv(self.csv_study_area)
        else:
            self.df = pd.read_csv(self.csv)

            # Standardize missing values
            self.df["Status"].replace(np.nan, "undefined", inplace=True)
            self.df["Heading"].replace(511, np.nan, inplace=True)

            # Standardize vessel type and cog
            self._map_vessel_types()
            self._normalize_angles()

            # Reduce to area of interest
            self._select_spatial()

            # save to study area
            self.df.to_csv(self.csv_study_area, index=False, header=True)

        self.df["BaseDateTime"] = pd.to_datetime(self.df["BaseDateTime"])
        self.df.sort_values("MMSI", inplace=True)

        # Save raw df for plotting purposes - use for traffic as well
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

        self._filter_status()
        self._filter_type()

        self._plot()

        self.df.sort_values(["MMSI", "BaseDateTime"], inplace=True)
        self.df.to_csv(self.cleaned, index=False, header=True)

    # DATAFRAME CLEANING
    def _select_spatial(self):
        """Limit data to bounding box of interest."""
        self.df = self.df[
            (
                (self.df["LON"].between(self.lonMin1, self.lonMax1))
                & (self.df["LAT"].between(self.latMin1, self.latMax1))
            )
            | (
                (self.df["LON"].between(self.lonMin2, self.lonMax2))
                & (self.df["LAT"].between(self.latMin2, self.latMax2))
            )
        ].copy()

    @print_reduction
    def _drop_null(self):
        """
        Drop rows with nulls in all the required columns. 
        No loss of information.
        """
        self.df.replace("", np.nan, inplace=True)

        # Log information about dropped data
        cols = self.required + ["VesselType"]
        subset = self.df[cols].copy()
        subset["Missing_Heading"] = subset["Heading"].isnull()
        plot_type(subset, "Comparison", self.plot_dir)

        # drop data
        self.df.dropna(how="any", subset=self.required, inplace=True)
        self.mmsi = self.df["MMSI"].unique()

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
        plot_sog(self.df_raw, "Raw", self.plot_dir)
        plot_sog(self.df, "Cleaned", self.plot_dir)


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
        self.plot_dir = abspath(join("reports", "figures", str(self.month)))

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
        self._plot()
        self._log_quantiles()
        self._write_output()

    def _plot(self):
        """Plot data before and after its been cleaned"""
        plot_acceleration(self.gdf, self.plot_dir)
        plot_alteration(self.gdf, self.plot_dir)
        plot_trip_length(self.gdf, self.plot_dir)
        # plot_latlon(self.gdf, self.plot_dir)
        plot_step_length(self.gdf, "Processed", self.plot_dir)

    def _log_quantiles(self):
        accel = self.gdf["Acceleration"].quantile(np.linspace(0.05, 1, 9, 0), "lower")
        alter = self.gdf["Alteration_Degrees"].quantile(
            np.linspace(0.05, 1, 9, 0), "lower"
        )
        LOGGER.info(f"Quantiles: {accel}, {alter}")


    # PREPROCESSING ------------------------------------------------------------
    def _update_interval(self):
        """Update time interval after data cleaning"""
        self.gdf["Interval"] = self.grouped_mmsi["BaseDateTime"].diff()
        self.gdf["Interval"].fillna(datetime.timedelta(seconds=60), inplace=True)
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
        """Remove MMSIs with few data points"""
        self.gdf = self.grouped_trip.filter(lambda g: len(g) > self.minPoints)

    def _step_distance(self):
        """Return distance between lat/lon positions"""

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
        """Compare step_distance to speed*time. Remove suspicious data"""
        self._step_distance()
        self.gdf["Expected_Distance"] = (
            self.gdf["SOG"] * self.gdf["Interval"] * METERS_IN_NM
        ) / 3600

        self.gdf["Outlier"] = np.where(
            self.gdf["Step_Distance"] >= self.gdf["Expected_Distance"] * limit, 1, 0
        )
        self.gdf = self.gdf[self.gdf["Outlier"] == 0]
        self.gdf.drop(columns=["Expected_Distance", "Outlier"], inplace=True)

    def _step_cog(self):
        """Calculate the course between two position points"""

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
        self.gdf["Acceleration"] = self.gdf["DS"].divide(
            self.df["Interval"], fill_value=0
        ) * (METERS_IN_NM / 3600)
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
