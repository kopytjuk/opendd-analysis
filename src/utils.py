from dataclasses import dataclass
import pathlib
from typing import Tuple

import numpy as np

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

        return (upper_left[0], lower_right[0], lower_right[1], upper_left[1] )
