"""
Created on Sat Feb 07 22:57:14 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path

import shapely

line = shapely.LineString([(0, 0), (1, 1)])
print(
    f"area: {line.area}\n"
    f"bounds: {line.bounds}\n"
    f"length: {line.length}\n"
    f"coords: {list(line.coords)}\n"
)

ring = shapely.LinearRing([(0, 0), (1, 1), (1, 0)])
print(
    f"area: {ring.area}\n"
    f"bounds: {ring.bounds}\n"
    f"length: {ring.length}\n"
    f"coords: {list(ring.coords)}\n"
)


