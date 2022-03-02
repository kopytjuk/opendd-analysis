"""Reference path extraction
"""

import math
from typing import Optional, List, Tuple
from dataclasses import dataclass
from itertools import product

import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from shapely import geometry, ops
from shapely.geometry import LineString
import networkx as nx
from networkx import DiGraph
import momepy

from .visualize import random_colors
from .utils import pairwise


def visualize_traffic_lanes(trafficlanes: gpd.GeoDataFrame, ax: Optional[Axes] = None):

    if ax is None:
        ax = plt.gca()
    
    trafficlanes.plot(ax=ax, color=random_colors(len(trafficlanes)), label="traffic lanes")
    ax.set_axis_off()


def traffic_lanes_to_graph(trafficlanes: gpd.GeoDataFrame) -> DiGraph:
    return momepy.gdf_to_nx(trafficlanes, approach='primal', directed=True, multigraph=False)


def graph_path_to_linestring(G: DiGraph, path: List[int]) -> LineString:
    lines = list()
    for n0, n1 in pairwise(path):
        geometry = G.edges[n0, n1]['geometry']
        lines.append(geometry)
    ls = ops.linemerge(lines)
    return ls


@dataclass
class DrivablePath:
    """Data structure representing a drivable path (extracted from the map graph)
    """
    id: int
    path: LineString

    def as_linestring(self) -> LineString:
        return self.path
    
    def plot(self, ax: mpl.axes.Axes, color=None):
        ls = self.path
        x, y = ls.xy
        ax.plot(x, y, label=str(self.id), color=color, lw=3)
        ax.scatter([x[0]], [y[0]], color=color, marker="^")
        ax.scatter([x[-1]], [y[-1]], color=color, marker="s")
    
    @property
    def length(self) -> float:
        ls = self.path
        return ls.length


def graph_path_to_linestring(G: DiGraph, path: List[int]) -> LineString:
    lines = list()
    for n0, n1 in pairwise(path):
        geometry = G.edges[n0, n1]['geometry']
        lines.append(geometry)
    ls = ops.linemerge(lines)
    return ls


def extract_paths_from_graph(G: DiGraph) -> List[DrivablePath]:

    start_nodes = [node for node, degree in G.in_degree if degree == 0]
    end_nodes = [node for node, degree in G.out_degree if degree == 0]

    paths = list()
    for i, (start, end) in enumerate(product(start_nodes, end_nodes)):
        
        # find path
        try:
            path = nx.shortest_path(G, start, end)
        except nx.NetworkXNoPath:
            continue
        
        # convert to linestring
        path_as_ls = graph_path_to_linestring(G, path)
        
        paths.append(DrivablePath(i, path_as_ls))
    
    return paths


def driving_path_overview_plot(trafficlanes: gpd.GeoDataFrame,
    paths: List[DrivablePath], print_length: bool = False) -> mpl.figure.Figure:

    num_paths = len(paths)
    ncols = 5
    nrows = math.ceil(num_paths/ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 20))
    axs = axs.flatten()

    for i, dp in enumerate(paths):
        axi = axs[i]
        trafficlanes.plot(ax=axi, color="k", label="traffic lanes")
        dp.plot(axi, color="red")

        axi.set_title(f"Path {dp.id}" + (f": {dp.length:.2f}m" if print_length else ""))
    # plt.tight_layout()

    for i in range(nrows*ncols):
        axi = axs[i]
        axi.set_axis_off()

    return fig

def extract_drivable_paths(trafficlanes: gpd.GeoDataFrame) -> List[DrivablePath]:
    G = traffic_lanes_to_graph(trafficlanes)
    paths = extract_paths_from_graph(G)
    return paths
