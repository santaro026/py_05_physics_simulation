"""
Created on Wed Feb 04 01:35:36 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path

import myplotter

    # if (abc0[0] == 0 and abc1[1] == 0) or (abc0[1] == 0 and abc1[0] == 0):
        # raise ValueError(f"two lines must be prallel.")

def distance_p2l(p, abc):
    d = abs(abc[0]*p[0] + abc[1]*p[1] + abc[2]) / np.sqrt(abc[0]**2 + abc[1]**2)
    return d

def direction_p2l(p, abc):
    n = np.array([abc[0], abc[1]], dtype=float)
    n_unit = n / np.linalg.norm(n)
    if (abc[0]*p[0] + abc[1]*p[1] + abc[2]) > 0:
        return n_unit
    else:
        return -n_unit

def distance_l2l(abc0, abc1):
    if not np.isclose(abc0[0] * abc1[1] , abc1[0] * abc0[1]):
        raise ValueError(f"two lines must be prallel.")
    if not np.isclose(abc1[0], 0):
        scale = abc0[0] / abc1[0]
    else:
        scale = abc0[1] / abc1[1]
    d = abs(abc0[2] - abc1[2]*scale) / np.sqrt(abc0[0]**2 + abc0[1]**2)
    return d

def direction_l2l(abc0, abc1):
    if not np.isclose(abc0[0] * abc1[1] , abc1[0] * abc0[1]):
        raise ValueError(f"two lines must be prallel.")
    n = np.array([abc0[0], abc0[1]], dtype=float)
    n_unit = n / np.linalg.norm(n)
    if not np.isclose(abc1[0], 0):
        scale = abc0[0] / abc1[0]
    else:
        scale = abc0[1] / abc1[1]
    if (abc0[2] - abc1[2] * scale) > 0:
        return n_unit
    else:
        return -n_unit

class Point:
    def __init__(self, name="point", p=np.zeros(2), m=1):
        self.name = name
        self.p = np.asarray(p)
        self.m = m

class Line:
    def __init__(self, name="line", a=1, b=1, c=0, m=1):
        self.name = name
        self.a = a
        self.b = b
        self.c = c
        self.m = m
    def generate_node(self, node=100, xrange=(-1, 1), yrange=(-1, 1)):
        if self.b != 0:
            x = np.linspace(xrange[0], xrange[1], node, endpoint=True)
            y = -self.a / self.b * x - self.c / self.b
        elif self.b == 0:
            x = np.full(node, -self.c/self.a)
            y= np.linspace(yrange[0], yrange[1], node, endpoint=True)
        self.node = np.column_stack([x, y])

class Circle:
    def __init__(self, name="circle", center=np.zeros(2), r=1, m=1):
        self.name = name
        self.center = np.asarray(center)
        self.r = r
        self.m = m
    def generate_node(self, node=100):
        n = np.linspace(0, 1, node, endpoint=True)
        x = self.r * np.cos(2*np.pi*n) + self.center[0]
        y = self.r * np.sin(2*np.pi*n) + self.center[1]
        self.node = np.column_stack([x, y])

class GeomContact:
    def __init__(self, object0, object1, name=""):
        self.name = name
        self.object0 = object0
        self.object1 = object1
        self.d = None
        self.types = {type(object0), type(object1)}
    def _get_objects(self, typeA, typeB):
        if isinstance(self.object0, typeA) and isinstance(self.object1, typeB):
            return self.object0, self.object1
        elif isinstance(self.object1, typeA) and isinstance(self.object0, typeB):
            return self.object1, self.object0
        else:
            raise ValueError(f"invalid type object")
    def calc_distance(self):
        if self.types == {Point}:
            p0, p1 = self.object0, self.object1
            d_vct = p0.p - p1.p
            self.d = np.linalg.norm(d_vct)
            self.n_vct = d_vct / self.d
        elif self.types == {Line}:
            l0, l1 = self.object0, self.object1
            self.d = distance_l2l([l0.a, l0.b, l0.c], [l1.a, l1.b, l1.c])
            self.n_vct = direction_l2l([l0.a, l0.b, l0.c], [l1.a, l1.b, l1.c])
        elif self.types == {Point, Line}:
            p, l = self._get_objects(Point, Line)
            if not p or not l:
                raise ValueError(f"distance_c2l requires one Circle and One Line, object1: {type(self.object0)}, object2: {type(self.object1)}")
            self.d = distance_p2l([p.p[0], p.p[1]], [l.a, l.b, l.c])
            self.n_vct = direction_p2l([p.p[0], p.p[1]], [l.a, l.b, l.c])
        elif self.types == {Point, Circle}:
            p, c = self._get_objects(Point, Circle)
            if not p or not c:
                raise ValueError(f"distance_c2l requires one Circle and One Line, object1: {type(self.object0)}, object2: {type(self.object1)}")
            d_vct = p.p - c.center
            self.d = np.linalg.norm(d_vct) - c.r
            self.n_vct = d_vct / np.linalg.norm(d_vct)
        elif self.types == {Line, Circle}:
            l, c = self._get_objects(Line, Circle)
            if not c or not l:
                raise ValueError(f"distance_c2l requires one Circle and One Line, object1: {type(self.object0)}, object2: {type(self.object1)}")
            self.d = distance_p2l([c.center[0], c.center[1]], [l.a, l.b, l.c]) - c.r
            self.n_vct = direction_p2l([c.center[0], c.center[1]], [l.a, l.b, l.c])
        return self.d, self.n_vct

class ContactForce:
    def __init__(self, contact, k=1e4):
        self.contact = contact
        self.k = k
        self.f = np.zeros(2)
    def calc_force(self):
        d, n = self.contact.calc_distance()
        if d >= 0:
            self.f = np.zeros(2)
        elif d < 0:
            self.f = -self.k * d * n
        return self.f



if __name__ == "__main__":
    print("---- run ----")
    plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
    fig, ax = plotter.myfig(xrange=(-10, 10), yrange=(-10, 10), xtick=1, ytick=1, xsigf=1, ysigf=1)
    ax = ax[0]
    ax.grid()

    p1 = Point(p=[-4, 7])
    p2 = Point(p=[2, -4])
    l1 = Line(a=1, b=1, c=1)
    l2 = Line(a=-2, b=2, c=8)
    c1 = Circle()
    c2 = Circle(center=[4, -2], r=4)
    l1.generate_node(xrange=(-5, 5))
    l2.generate_node(xrange=(-5, 5))
    c1.generate_node()
    c2.generate_node()

    contact = GeomContact(c1, l1)
    force = ContactForce(contact)
    force.calc_force()
    contact.calc_distance()
    print(contact.d)
    print(force.f)

    ax.scatter(p1.p[0], p1.p[1], lw=4, c='k')
    ax.scatter(p2.p[0], p2.p[1], lw=4, c='k')
    ax.plot(l1.node[:, 0], l1.node[:, 1], lw=1, c='k')
    ax.plot(l2.node[:, 0], l2.node[:, 1], lw=1, c='g')
    ax.plot(c1.node[:, 0], c1.node[:, 1], lw=1, c='r')
    ax.plot(c2.node[:, 0], c2.node[:, 1], lw=1, c='b')
    ax.quiver([0], [0], force.f[0], force.f[1], scale_units="xy", angles="xy", width=0.004)
    plt.show()

