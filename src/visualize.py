import sys
import pathlib
from dataclasses import dataclass

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def visualize_timestep(ax: Axes, objects: gpd.GeoDataFrame, trafficlanes: gpd.GeoDataFrame):
    trafficlanes.plot(ax=ax, color="k")
    objects.plot(ax=ax, column="CLASS")
