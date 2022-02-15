from typing import List

from scipy.spatial.distance import directed_hausdorff
import numpy as np
from shapely.geometry import LineString

from .path_extraction import DrivablePath


def compute_path_distance(ls1: LineString, ls2: LineString, N: int = 100) -> float:
    
    # linestrings have to be distretized to call `scipy.spatial.distance.directed_hausdorff`
    ls1_pts = LineString([ls1.interpolate(s, normalized=True) for s in np.linspace(0, 1, N)])
    ls1_pts = np.array(ls1_pts.xy).T  # N x 2
    
    ls2_pts = LineString([ls2.interpolate(s, normalized=True) for s in np.linspace(0, 1, N)])
    ls2_pts = np.array(ls2.xy).T
    
    d = directed_hausdorff(ls1_pts, ls2_pts)[0]
    return d


def find_nearest_path(ls: LineString, paths: List[DrivablePath], N: int = 100) -> int:
    """Returns the index of the path list `paths` with minimal distance to `ls` wrt. to
    `directed_hausdorff` metric.
    """
    
    distances = np.zeros(len(paths), dtype=float)
    
    for i, path in enumerate(paths):
    
        d = compute_path_distance(ls, path.as_linestring(), N)
        distances[i] = d

    return distances.argmin()
