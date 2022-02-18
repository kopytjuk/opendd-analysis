import math
from typing import Tuple
from dataclasses import dataclass

from shapely.geometry import LineString
import numpy as np

@dataclass
class DiscreteReferencePath:
    """2D Path represented as an ordered list of points.
    """
    
    points: np.ndarray  # in path length s and cartesian (metric) coordinates, as Nx3 (s, x, y)
    spatial_resolution: float  # approximate
    
    def to_frenet(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """Transform a cartesian coordinate to a Frenet coordinate.

        Args:
            pt (Tuple[float, float]): cartesian coordinate (x, y)

        Returns:
            Tuple[float, float]: frenet coordinate (s, d)
        """
        
        pt = np.array(pt)
        
        # compute distances to each sample of the ref.-path
        deltas = np.linalg.norm(self.points[:, 1:] - pt, axis=1)
        
        # take the index of the minium
        idx_min = np.argmin(deltas)
        
        # arc length
        s = self.points[idx_min, 0]
        # perpendicular distance
        d = deltas[idx_min]
        return s,  d
    
    def to_cartesian(self, s: float, d: 0):
        # needs to implement normal unit vector
        pass
    
    @property
    def length(self) -> float:
        return self.points[-1, 0] + self.spatial_resolution
    
    @classmethod
    def from_linestring(cls, ls: LineString, resolution=1.0, cut_length: float = 1.0):
        
        path_samples = np.arange(cut_length, ls.length-cut_length, resolution)
        ls_pts = LineString([ls.interpolate(s, normalized=False) for s in path_samples])
        ls_pts = np.array(ls_pts.xy).T  # N x 2
        
        return cls(np.c_[path_samples, ls_pts], resolution)
    
    def to_frenet_vectorized(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        F = np.zeros((len(x), 2))
        
        for i, (x_pt, y_pt) in enumerate(zip(x, y)):
             F[i, :] = self.to_frenet((x_pt, y_pt))
        
        return F[:, 0], F[:, 1]
    
    def linestring_to_frenet(self, ls: LineString) -> LineString:
        s_arr, d_arr = self.to_frenet_vectorized(*ls.xy)
        return LineString([(s, d) for s, d in zip(s_arr, d_arr)])
    
    def as_linestring(self) -> LineString:
        return LineString([(x, y) for x, y in zip(self.points[:, 1], self.points[:, 2])])
    
    def __str__(self):
        return f"<DiscreteReferencePath with {self.points.shape[0]} samples>"
