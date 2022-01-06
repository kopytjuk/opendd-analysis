from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, LinearRing
from shapely import affinity

from .utils import point_to_bbox

@dataclass
class Trajectory:

    t: np.ndarray

    # metric coordinates
    xs: np.ndarray
    ys: np.ndarray
    angles: np.ndarray

    width: Optional[float] = None
    length: Optional[float] = None
    
    object_class: Literal["Car", "Bicycle"] = "Car"
    projection: Literal["EPSG:25832"] = "EPSG:25832"


    def to_geopandas(self, as_bbox: bool = False) -> gpd.GeoDataFrame:

        df = pd.DataFrame({
            "t": self.t,
            "x": self.xs,
            "y": self.ys,
            "theta": self.angles
        })

        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["x"], df["y"]))

        gdf = gdf.set_crs(self.projection)

        if as_bbox:
            gdf.geometry = gdf.apply(lambda row:
                                     point_to_bbox(row.geometry, self.width, self.length, row["theta"]), axis=1)

        return gdf
