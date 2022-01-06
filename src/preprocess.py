import pandas as pd
import geopandas as gpd

from .utils import point_to_bbox


def preprocess(df: pd.DataFrame, sampling_frequency: float) -> gpd.GeoDataFrame:

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["UTM_X"], df["UTM_Y"]), crs='EPSG:25832')

    gdf["bbox"] = gdf.apply(lambda row:
                            point_to_bbox(row.geometry, row["WIDTH"],
                                          row["LENGTH"],
                                          row["UTM_ANGLE"],
                                          as_linestring=True), axis=1)

    # generate sampling time 'nearest' interpolation strategy
    sampling_time_ms = 1000/sampling_frequency
    gdf["k"] = (gdf["TIMESTAMP"]*1000 / sampling_time_ms).round().astype(int)

    return gdf                                
