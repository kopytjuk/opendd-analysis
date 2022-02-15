from dataclasses import dataclass
import pathlib
from itertools import tee
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


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def point_to_bbox(pt: Point, width: float, length: float, angle: float,
                  as_linestring: bool = False) -> Union[Polygon, LinearRing]:
    """Create a (center-rotated) bounding box from a single point.

    Args:
        pt (Point): point
        width (float): width
        length (float): length
        angle (float): rotation angle of the object
        as_linestring (bool, optional): Whether to return a LinearRing instead of a Polygon. 
            Defaults to False.

    Returns:
        Union[Polygon, LinearRing]: object representation
    """
    bbox = affinity.scale(pt.buffer(0.5, cap_style=3),
                          xfact=length, yfact=width)
    bbox = affinity.rotate(bbox, angle, use_radians=True)
    if as_linestring:
        return bbox.exterior
    else:
        return bbox


def assign_color(s: pd.Series, seed: int = 42) -> pd.Series:
    """Assign a random color to each row based on unique values in a specific column. E.g. assign
    a distinct color to the object type:

    ```
    colors = assign_color(s)
    assert colors.nunique() == df["type"].nunique()
    ```

    Args:
        df (pd.Series): series  to derive colors from
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        pd.Series: colors
    """

    unique_values = s.unique()
    N = len(unique_values)

    color_map = {val: color for val, color in
                 zip(unique_values, random_colors(N, seed))}

    return s.map(color_map)


TimePeriod = Tuple[float, float]  # seconds

def common_time(tp1: TimePeriod, tp2: TimePeriod) -> TimePeriod:
    t0 = max(tp1[0], tp2[0])
    t1 = min(tp1[1], tp2[1])
    return (t0, t1)


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
