import pandas as pd
import geopandas as gpd

from .utils import point_to_bbox


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
