from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd


@dataclass
class Trajectory:

    t: np.ndarray

    # metric coordinates
    xs: np.ndarray
    ys: np.ndarray

    width: Optional[float] = None
    length: Optional[float] = None
    
    object_class: Literal["Car", "Bicycle"] = "Car"
    projection: Literal["EPSG:25832"] = "EPSG:25832"


    def plot_on_map(self, ax: Axes):

        gdf = self.to_geopandas()
        gdf_wgs84 = gdf.to_crs("EPSG:4326")
        gdf_wgs84.plot(ax=ax)

    
    def to_geopandas(self):

        df = pd.DataFrame({
            "t": self.t,
            "x": self.xs,
            "y": self.ys,
        })

        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["x"], df["y"]))
        
        gdf = gdf.set_crs(self.projection)

        return gdf