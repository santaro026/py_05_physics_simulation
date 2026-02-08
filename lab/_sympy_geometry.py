"""
Created on Sun Feb 08 09:56:26 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path

from sympy import Point, Circle, Line, sqrt

c = Circle(Point(0, 0), 5)
l = Line(Point(0, -5), Point(5, 0))


intersections = c.intersection(l)

print(intersections)
print(c.tangent_lines(Point(4, 4))[0])
print(list(l.p1))

