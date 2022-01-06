from dataclasses import dataclass
import pathlib
from typing import Literal, Optional, Tuple, Union

import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, LinearRing
from shapely import affinity

from .visualize import random_colors


def point_to_bbox(pt: Point, width: float, length: float, angle: float,
    as_linestring: bool = False) -> Union[Polygon, LinearRing]:
    bbox = affinity.scale(pt.buffer(0.5, cap_style=3), xfact=length, yfact=width)
    bbox = affinity.rotate(bbox, angle, use_radians=True)
    if as_linestring:
        return bbox.exterior
    else:
        return bbox


def assign_color(df: pd.DataFrame, column: str, seed: int = 42) -> pd.Series:

    unique_values = df[column].unique()
    N = len(unique_values)

    color_map = {val: color for val, color in zip(unique_values, random_colors(N, seed))}

    return df[column].map(color_map)



@dataclass
class WorldDefinition:

    x_scale: float  # pixel with in meters
    y_scale: float

    x_skew: float  # pixel with in meters
    y_skew: float

    x_upper_left: float
    y_upper_left: float

    @classmethod
    def from_pgw_file(cls, path: pathlib.Path):

        content = path.read_text()
        lines = content.splitlines()

        # from https://en.wikipedia.org/wiki/World_file

        x_scale = float(lines[0])  # A
        y_scale = float(lines[3])  # E
        y_skew = float(lines[1])  # B
        x_skew = float(lines[2])  # D
        x_upper_left = float(lines[4])  # C
        y_upper_left = float(lines[5])  # F
        return cls(x_scale, y_scale, x_skew, y_skew, x_upper_left, y_upper_left)

    def _get_affine_matrix(self) -> np.ndarray:

        A, B, C = self.x_scale, self.x_skew, self.x_upper_left
        D, E, F = self.y_skew, self.y_scale, self.y_upper_left

        return np.array([
            [A, B, C],
            [D, E, F]
        ], dtype=np.float64)

    def get_image_extent(self, width: int, height: int) -> Tuple[float, float, float, float]:
        # returns image extent as left, right, bottom, top

        T = self._get_affine_matrix()

        # affine transformation
        upper_left = T @ np.array([0, 0, 1.0])
        lower_right = T @ np.array([width, height, 1.0])

        return (upper_left[0], lower_right[0], lower_right[1], upper_left[1])
