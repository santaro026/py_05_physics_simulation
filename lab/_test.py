"""
Created on Tue Feb 10 01:26:05 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path

from sympy import Line3D, Point3D


a = Point3D(10, 10, 0)
print("a: ", a)

x_axis = Line3D(Point3D(0, 0, 0), direction_ratio=[1, 0, 0])

b = a.rotate(np.pi/4, x_axis, pt=Point3D(0, 0, 0))
# b = a.rotate(np.pi/4)
print(b)




