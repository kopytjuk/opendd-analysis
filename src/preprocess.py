import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString

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


def _generate_trace(group: pd.DataFrame) -> pd.Series:
    
    group = group.sort_values("TIMESTAMP")
    x_arr = group["UTM_X"]
    y_arr = group["UTM_Y"]
    ls = LineString([(x, y) for x, y in zip(x_arr, y_arr)])
    
    first_row = group.iloc[0]
    
    objid = first_row["OBJID"]
    obj_class = first_row["CLASS"]
    w = first_row["WIDTH"]
    l = first_row["LENGTH"]
    
    s = pd.Series({
        "OBJID": objid,
        "CLASS": obj_class,
        "WIDTH": w,
        "LENGTH": l,
        "geometry": ls
    })
    return s


def transform_df_to_trajectory_gdf(df: pd.DataFrame, id_column: str = "OBJID",
                                   crs: str = 'EPSG:32632') -> gpd.GeoDataFrame:
    df_traces = df.groupby(id_column).apply(_generate_trace)
    gdf_traces = gpd.GeoDataFrame(df_traces)
    gdf_traces = gdf_traces.set_crs(crs)
    return gdf_traces
