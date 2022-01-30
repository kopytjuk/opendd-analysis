from enum import IntEnum

import pandas as pd
import geopandas as gpd
import numpy as np

from shapely.geometry import LineString

from .utils import point_to_bbox

class VehicleState(IntEnum):
    STANDING = 0
    DRIVING = 1


def identify_driving_state(trace: pd.Series, standing_vel_threshold: float = 0.2, 
    standing_acc_threshold: float = 0.5) -> np.ndarray:
    """Identify driving state (driving vs. standing) based on object's velocity and acceleration.

    Args:
        trace (pd.Series): trajectory data
        standing_vel_threshold (float, optional): Maximal velocity while standing. Defaults to 0.2.
        standing_acc_threshold (float, optional): Maximal velocity while standing. Defaults to 0.5.

    Returns:
        np.ndarray: integer array of size `len(trace["V"])` with vehicle states,
            see `VehicleState`
    """
    
    velocity_array = trace["V"]
    acc_array = trace["ACC"]

    states = np.zeros(len(velocity_array), dtype=int)

    states[:] = VehicleState.DRIVING.value

    # identify standing if velocity and acceleration are below the thresholds
    standing_indices = (np.abs(velocity_array) < standing_vel_threshold) & \
        (np.abs(acc_array) < standing_acc_threshold)
    states[standing_indices] = VehicleState.STANDING.value

    return states


def discrete_timestep(df: pd.DataFrame, sampling_frequency: float,
                      column: str = "TIMESTAMP") -> pd.Series:
    sampling_time_ms = 1000/sampling_frequency
    return (df[column]*1000 / sampling_time_ms).round().astype(int)


def convert_to_geopandas(df, x_col="UTM_X", y_col="UTM_Y",
                         crs='EPSG:25832') -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["UTM_X"], df["UTM_Y"]), crs=crs)
    return gdf


def add_bbox(gdf, width_col="WIDTH", length_col="LENGTH", angle_col="UTM_ANGLE"):

    return gdf.apply(lambda row:
                     point_to_bbox(row.geometry, row[width_col],
                                   row[length_col],
                                   row[angle_col],
                                   as_linestring=True), axis=1)


def preprocess(df: pd.DataFrame, sampling_frequency: float) -> gpd.GeoDataFrame:

    gdf = convert_to_geopandas(df)

    gdf["bbox"] = add_bbox(gdf)

    # generate sampling time 'nearest' interpolation strategy
    gdf["k"] = discrete_timestep(gdf, sampling_frequency, "TIMESTAMP")

    return gdf


def _generate_trace(group_records: pd.DataFrame) -> pd.Series:
    
    group_records = group_records.sort_values("TIMESTAMP")
    x_arr = group_records["UTM_X"]
    y_arr = group_records["UTM_Y"]
    ls = LineString([(x, y) for x, y in zip(x_arr, y_arr)])

    velocity = group_records["V"].values
    acceleration = group_records["ACC"].values

    first_sample = group_records.iloc[0]
    
    # helpful metadata
    obj_class = first_sample["CLASS"]
    w = first_sample["WIDTH"]
    l = first_sample["LENGTH"]
    t0 = first_sample["TIMESTAMP"]
    
    # whole trajectory in one record
    s = pd.Series({
        "CLASS": obj_class,
        "WIDTH": w,
        "LENGTH": l,
        "START_TIME": t0,
        "NUM_SAMPLES": len(x_arr),
        "V": velocity,
        "ACC": acceleration,
        "geometry": ls
    })
    return s


def transform_df_to_trajectory_gdf(df: pd.DataFrame, id_column: str = "OBJID",
                                   crs: str = 'EPSG:32632') -> gpd.GeoDataFrame:
    df_traces = df.groupby(id_column).apply(_generate_trace)
    gdf_traces = gpd.GeoDataFrame(df_traces)
    gdf_traces = gdf_traces.set_crs(crs)
    return gdf_traces
