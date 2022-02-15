from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple

import matplotlib as mpl
import pandas as pd
import numpy as np
import geopandas as gpd

from .utils import common_time, point_to_bbox


@dataclass
class Trajectory:
    """Helper class to work with generic trajectories
    """
    
    id: int
    X: pd.DataFrame  # time as index, columns as state features

    def plot(self, axs: List[mpl.axes.Axes], columns: Optional[List[str]] = None,
             plot_kwargs: dict = dict()):

        X = self.X

        if columns is None:
            signals_to_plot = self.names
        else:
            signals_to_plot = columns

        assert len(axs) == len(signals_to_plot), \
            "Number of state variables and provided axes is not equal."
        
        for ax, col in zip(axs, signals_to_plot):
            ax.plot(X.index, X[col], **plot_kwargs)
    
    @property
    def dim(self) -> int:
        return len(self.X.columns)
    
    @property
    def name(self) -> int:
        return self.id
    
    @property
    def time(self):
        return self.X.index
    
    @property
    def names(self) -> List[str]:
        return list(self.X.columns)
    
    @classmethod
    def from_trace(cls, trace: pd.Series, columns: List[str], sample_rate: float = 30):
        name = trace.name
        start_time = trace["START_TIME"]
        num_samples = trace["NUM_SAMPLES"]
        time = np.arange(num_samples)/sample_rate + start_time
        df = pd.DataFrame(
            data = {c: trace[c] for c in columns},
            index = time
        )
        return cls(name, df)
    
    def temporal_intersection(self, traj: "Trajectory") -> Tuple[float, float]:
        tp1 = self.time[0], self.time[-1]
        tp2 = traj.time[0], traj.time[-1]
        return common_time(tp1, tp2)
    
    def get(self, variable: str) -> pd.Series:
        return self.X[variable]
    
    def sample(self, t) -> pd.Series:
        i_prev = np.argmax(self.X.index >= t )
        return self.X.iloc[i_prev]


@dataclass
class Trajectory2D:

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
