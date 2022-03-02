import sys
import pathlib
from dataclasses import dataclass
from typing import Literal, List
import random

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from shapely.geometry import LinearRing


def random_colors(N: int, seed: int = 42) -> List[str]:
    random.seed(seed)
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(N)]
    return colors


@dataclass
class ObjectVisualization:

    id: int
    geometry: LinearRing
    color: str = "k"

    def plot(self, ax: Axes):
        x, y = self.geometry.xy
        ax.plot(x, y, color=self.color)

        P = self.geometry.centroid
        ax.text(P.x, P.y, f"{self.id}")


def visualize_objects(ax: Axes, objects: List[ObjectVisualization], trafficlanes: gpd.GeoDataFrame):
    trafficlanes.plot(ax=ax, color="k")
    
    for obj in objects:
        obj.plot(ax=ax)

def visualize_timestep(ax: Axes, objects: gpd.GeoDataFrame, trafficlanes: gpd.GeoDataFrame):
    trafficlanes.plot(ax=ax, color="k")
    objects.plot(ax=ax)
