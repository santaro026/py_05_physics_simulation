"""
Created on Mon Feb 09 18:08:17 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.lines as mlines

from pathlib import Path

from sympy import Point, Line, Circle, sqrt
from sympy import Point3D, Line3D, Plane
from sympy import Matrix, sin, cos, tan

from santamods import mycoord, myplotter

x_axis = Line3D(Point3D(0, 0, 0), direction_ratio=[1, 0, 0])
y_axis = Line3D(Point3D(0, 0, 0), direction_ratio=[0, 1, 0])
z_axis = Line3D(Point3D(0, 0, 0), direction_ratio=[0, 0, 1])

def rotatex(point, theta, center=np.zeros(3)):
    px = point.x - center[0]
    py = point.y - center[1]
    pz = point.z - center[2]
    v = Matrix([px, py, pz])
    R = Matrix([
        [1, 0, 0],
        [0, cos(theta), -sin(theta)],
        [0, sin(theta), cos(theta)],
    ])
    v_rot = R * v
    x = v_rot[0] + center.x
    y = v_rot[1] + center.y
    z = v_rot[2] + center.z
    return Point3D(x, y, z)

def rotatey(point, theta, center=np.zeros(3)):
    px = point.x - center[0]
    py = point.y - center[1]
    pz = point.z - center[2]
    v = Matrix([px, py, pz])
    R = Matrix([
        [cos(theta), 0, sin(theta)],
        [0, 1, 0],
        [-sin(theta), 0, cos(theta)],
    ])
    v_rot = R * v
    x = v_rot[0] + center.x
    y = v_rot[1] + center.y
    z = v_rot[2] + center.z
    return Point3D(x, y, z)

def rotatez(point, theta, center=np.zeros(3)):
    px = point.x - center[0]
    py = point.y - center[1]
    pz = point.z - center[2]
    v = Matrix([px, py, pz])
    R = Matrix([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0, 0, 1],
    ])
    v_rot = R * v
    x = v_rot[0] + center.x
    y = v_rot[1] + center.y
    z = v_rot[2] + center.z
    return Point3D(x, y, z)

def get_rigid_rotation_matrix(cage, theta):
    x = cage[0]
    y = cage[1]
    z = cage[2]
    yz_len = sqrt(y**2 + z**2)
    if yz_len == 0:
        return Matrix.eye(3)
    ux, uy, uz = 0, -z / yz_len, y / yz_len
    c = cos(theta)
    s = sin(theta)
    C = 1 - c
    R = Matrix([
        [c + ux**2*C,    ux*uy*C - uz*s,  ux*uz*C + uy*s],
        [uy*ux*C + uz*s, c + uy**2*C,    uy*uz*C - ux*s],
        [uz*ux*C - uy*s, uz*uy*C + ux*s, c + uz**2*C   ]
    ])
    return R

def rotate_rigid(point, R, center=np.zeros(3)):
    v = Matrix([point.x - center[0], point.y - center[1], point.z - center[2]])
    v_rot = R * v
    return Point3D([v_rot[0] + center.x, v_rot[1] + center.y, v_rot[2] + center.z])


class ARing:
    def __init__(self, name="", PCD=50, ID=48, OD=52,  R=3.2):
        self.name = name
        self.PCD = PCD
        self.ID = ID
        self.OD = OD
        self.R = R
        self.groove = Circle((0, 0), self.PCD/2 + self.R)
    def visualize(self):
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig()
        groove = patches.Circle((0, 0), self.groove.radius, color='k', fill=False)
        axs[0].add_patch(groove)
        axs[0].set(xlim=(-30, 30), ylim=(-30, 30))
        plt.show()

class BRing:
    def __init__(self, name="", PCD=50, ID=48, OD=52,  R=3.2):
        self.name = name
        self.PCD = PCD
        self.ID = ID
        self.OD = OD
        self.R = R
        self.groove = Circle((0, 0), self.PCD/2 - self.R)
    def visualize(self):
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig()
        groove = patches.Circle((0, 0), self.groove.radius, color='k', fill=False)
        axs[0].add_patch(groove)
        axs[0].set(xlim=(-30, 30), ylim=(-30, 30))
        plt.show()

class Ball:
    def __init__(self, name="", PCD=50, num_balls=8, Dw=5.953):
        self.name = name
        self.PCD = PCD
        self.num_balls = num_balls
        self.Dw = Dw
        self.pos_balls = np.linspace(0, 2*np.pi, self.num_balls, endpoint=False) + np.pi/2
        self.ball_centers = []
        for i in range(self.num_balls):
            _c = rotatex(Point3D(0, 0, 0).translate(0, self.PCD/2, 0), self.pos_balls[i], Point3D(0, 0, 0))
            self.ball_centers.append(_c)
    def translate(self, xyz):
        for i in range(self.num_balls):
            self.ball_centers[i] = self.ball_centers[i].translate(xyz[0], xyz[1], xyz[2])
    def rotatex(self, theta, center=np.zeros(3)):
        for i in range(self.num_balls):
            self.ball_centers[i] = rotatex(self.ball_centers[i], theta, center)
    def rotatey(self, theta, center=np.zeros(3)):
        for i in range(self.num_balls):
            self.ball_centers[i] = rotatey(self.ball_centers[i], theta, center)
    def rotatez(self, theta, center=np.zeros(3)):
        for i in range(self.num_balls):
            self.ball_centers[i] = rotatez(self.ball_centers[i], theta, center)
    def tilt(self, xyz, theta, center=np.zeros(3)):
        R = get_rigid_rotation_matrix(Point3D(xyz[0], xyz[1], xyz[2]), theta=theta)
        for i in range(self.num_balls):
            self.ball_centers[i] = rotate_rigid(self.ball_centers[i], R, center)
    def visualize(self):
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig()
        for i in range(self.num_balls):
            _c = np.array([self.ball_centers[i].x, self.ball_centers[i].y, self.ball_centers[i].z])
            _b = patches.Circle((_c[1], _c[2]), self.Dw/2, color='k', fill=False)
            axs[0].add_patch(_b)
        axs[0].set(xlim=(-30, 30), ylim=(-30, 30))
        plt.show()

class SimpleCage:
    def __init__(self, name="", PCD=50, ID=48, OD=52, width=10, num_pockets=8, Dp=6.25):
        self.name = name
        self.cage = Point3D(0, 0, 0)
        self.PCD = PCD
        self.ID = ID
        self.OD = OD
        # self.ring_outlines = [Circle((0, 0), self.ID/2), Circle((0, 0), self.OD/2)]
        self.width = width
        self.num_pockets = num_pockets
        self.Dp = Dp
        self.pos_pockets = np.linspace(0, 2*np.pi, self.num_pockets, endpoint=False) + np.pi/2
        self.pocket_centers = []
        for i in range(self.num_pockets):
            # _c = Point3D(0, 0, 0).translate(0, self.PCD/2, 0).rotate(self.pos_pockets[i], x_axis)
            _c = Point3D(20, 20, 20).rotate(self.pos_pockets[i], x_axis, Point3D(0, 0, 0))
            print(np.degrees(self.pos_pockets[i]))
            print(_c.evalf())
            self.pocket_centers.append(_c)
        self.history = []
    def _generate_pocket_line(self):
        for k, v1, v2 in self.history:
            translate_history = np.zeros(3)
            rotate_history = []
            if k == "translate":
                translate_history += np.asarray(v1)
            elif k == "rotate":
                rotate_history.append([v1, v2])
        _theta_id = np.arcsin(self.Dp / self.ID)
        _theta_od = np.arcsin(self.Dp / self.OD)
        l0x = -(self.PCD - self.ID * np.cos(_theta_id))/2
        l1x = (self.OD * np.cos(_theta_od) - self.PCD)/2
        l0y = self.Dp/2
        l1y = -self.Dp/2
        l00 = Point3D(0, l0x, l0y)
        l01 = Point3D(0, l1x, l0y)
        l10 = Point3D(0, l0x, l1y)
        l11 = Point3D(0, l1x, l1y)
        self.pocket_lines = []
        tx, ty, tz = translate_history
        for i in range(self.num_pockets):
            _l00 = l00.translate(0, self.PCD/2, 0).rotate(self.pos_pockets[i], x_axis)
            _l01 = l01.translate(0, self.PCD/2, 0).rotate(self.pos_pockets[i], x_axis)
            _l10 = l10.translate(0, self.PCD/2, 0).rotate(self.pos_pockets[i], x_axis)
            _l11 = l10.translate(0, self.PCD/2, 0).rotate(self.pos_pockets[i], x_axis)
            for _theta, _axis in rotate_history:
                _l00 = _l00.rotate(_theta, _axis)
                _l01 = _l01.rotate(_theta, _axis)
                _l10 = _l10.rotate(_theta, _axis)
                _l11 = _l11.rotate(_theta, _axis)
            _l00 = _l00.translate(tx, ty, tz)
            _l01 = _l01.translate(tx, ty, tz)
            _l10 = _l10.translate(tx, ty, tz)
            _l11 = _l11.translate(tx, ty, tz)
            _l0 = Line3D(_l00, l01)
            _l1 = Line3D(_l10, l11)
            _ls = [_l0, _l1]
            self.pocket_lines.append(_ls)
    # def _generate_outline(self):
    #     for k, v1, v2 in self.history:
    #         translate_history = np.zeros(3)
    #         rotate_history = []
    #         if k == "translate":
    #             translate_history += np.asarray(v1)
    #         elif k == "rotate":
    #             rotate_history.append([v1, v2])
    def translate(self, xyz):
        for i in range(self.num_pockets):
            self.pocket_centers[i] = self.pocket_centers[i].translate(xyz[0], xyz[1], xyz[2])
        self.cage = self.cage.translate(xyz[0], xyz[1], xyz[2])
        self.history.append(["translate", xyz, None])
    def rotate(self, theta, axis):
        for i in range(self.num_pockets):
            self.pocket_centers[i] = self.pocket_centers[i].rotate(theta, axis)
        self.cage = self.cage.rotate(theta, axis)
        self.history.append(["rotate", theta, axis])
    def visualize(self):
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig()
        self._generate_pocket_line()
        for i in range(self.num_pockets):
            _p11 = np.array(self.pocket_lines[i][0].p1)
            _p12 = np.array(self.pocket_lines[i][0].p2)
            _p1 = np.column_stack([_p11, _p12])
            _p21 = np.array(self.pocket_lines[i][1].p1)
            _p22 = np.array(self.pocket_lines[i][1].p2)
            _p2 = np.column_stack([_p21, _p22])
            _l1 = mlines.Line2D(_p1[1], _p1[2], color='b')
            _l2 = mlines.Line2D(_p2[1], _p2[2], color='b')
            axs[0].add_line(_l1)
            axs[0].add_line(_l2)
            axs[0].scatter(self.pocket_centers[i].y, self.pocket_centers[i].z, s=10, c='b' )
        _id = patches.Circle(np.zeros(3), self.ID/2, color='k', fill=False)
        _od = patches.Circle(np.zeros(3), self.OD/2, color='k', fill=False)
        axs[0].add_patch(_id)
        axs[0].add_patch(_od)
        axs[0].set(xlim=(-30, 30), ylim=(-30, 30))
        plt.show()

class Bearing:
    def __init__(self, ARing, BRing, Ball, Cage, Ball3d=None):
        self.aring = ARing
        self.bring = BRing
        self.ball = Ball
        self.ball3d = Ball3d
        self.cage = Cage
    def calc_contact(self):
        self.distances = np.zeros((self.ball.num_balls, 2))
        self.n_vct = np.zeros((self.ball.num_balls, 2, 2))
        self.p_contact = np.zeros((self.ball.num_balls, 2, 2))
        for i in range(self.ball.num_balls):
            b = self.ball.balls_circles[i]
            l1 = self.cage.pockets_lines[i][0]
            l2 = self.cage.pockets_lines[i][1]
            self.distances[i] = np.array([l1.distance(b.center) - b.radius, l2.distance(b.center) - b.radius])
            self.n_vct[i] = np.array([l1.projection(b.center) - b.center, l2.projection(b.center) -b.center])
            self.n_vct[i] = self.n_vct[i] / np.linalg.norm(self.n_vct[i])
            self.p_contact[i] = np.array([l1.projection(b.center), l2.projection(b.center)])
    def calc_contact_force(self, k=1, d=1):
        self.forces = np.zeros((self.ball.num_balls, 2, 2))
        for i in range(self.ball.num_balls):
            if self.distances[i][0] <= 0:
                _f = -k * self.distances[i, 0] ** d
                self.forces[i, 0] = _f * self.n_vct[i, 0]
            if self.distances[i][1] <= 0:
                _f = -k * self.distances[i, 1] ** d
                self.forces[i, 1] = _f * self.n_vct[i, 1]
    def visualize(self):
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig()
        if ARing:
            groove_a = patches.Circle((0, 0), self.aring.groove.radius, color='k', fill=False)
        if BRing:
            groove_b = patches.Circle((0, 0), self.bring.groove.radius, color='k', fill=False)
        for i in range(self.ball.num_balls):
            _b = patches.Circle(self.ball.balls_circles[i].center, self.ball.balls_circles[i].radius, color='k', fill=False)
            axs[0].add_patch(_b)
        for i in range(self.cage.num_pockets):
            _p11 = np.array(self.cage.pockets_lines[i][0].p1)
            _p12 = np.array(self.cage.pockets_lines[i][0].p2)
            _p1 = np.column_stack([_p11, _p12])
            _p21 = np.array(self.cage.pockets_lines[i][1].p1)
            _p22 = np.array(self.cage.pockets_lines[i][1].p2)
            _p2 = np.column_stack([_p21, _p22])
            _l1 = mlines.Line2D(_p1[0], _p1[1], color='b')
            _l2 = mlines.Line2D(_p2[0], _p2[1], color='b')
            axs[0].add_line(_l1)
            axs[0].add_line(_l2)
            axs[0].scatter(self.cage.pocket_centers[i].x, self.cage.pocket_centers[i].y, s=10, c='b' )
            if self.distances[i, 0] <= 0:
                axs[0].quiver(self.p_contact[i, 0, 0], self.p_contact[i, 0, 1], self.forces[i, 0, 0], self.forces[i, 0, 1], color='k', width=0.002)
            if self.distances[i, 1] <= 0:
                axs[0].quiver(self.p_contact[i, 1, 0], self.p_contact[i, 1, 1], self.forces[i, 1, 0], self.forces[i, 1, 1], color='k', width=0.002)
        axs[0].add_patch(groove_a)
        axs[0].add_patch(groove_b)
        _id = patches.Circle(self.cage.ring_outlines[0].center, self.cage.ring_outlines[0].radius, color='b', fill=False)
        _od = patches.Circle(self.cage.ring_outlines[1].center, self.cage.ring_outlines[1].radius, color='b', fill=False)
        axs[0].add_patch(_id)
        axs[0].add_patch(_od)
        axs[0].set(xlim=(-30, 30), ylim=(-30, 30))
        plt.show()
    def calc_contact3D(self):
        self.distances = np.zeros(self.ball3d.num_balls)
        self.n_vct = np.zeros((self.ball3d.num_balls, 3))
        self.n_vct_local = np.zeros((self.ball3d.num_balls, 3))
        self.p_contact = np.zeros((self.ball3d.num_balls, 3))
        for i in range(self.ball3d.num_balls):
            _b = self.ball3d.ball_centers[i]
            b = Point3D(_b.x, _b.y, _b.z)
            _p = self.cage.pocket_centers[i]
            p = Point3D(0, _p.x, _p.y)
            b = rotatex(b, -self.cage.pos_pockets[i]).translate(0, -self.cage.PCD/2, 0)
            p = rotatex(p, -self.cage.pos_pockets[i]).translate(0, -self.cage.PCD/2, 0)
            b = Point3D(b.x, 0, b.z)
            p = Point3D(p.x, 0, p.z)
            self.distances[i] = self.cage.Dp/2 - p.distance(b) - self.ball.Dw/2
            n_vct = rotatex(Point3D(b.x, 0, b.z), self.cage.pos_pockets[i])
            self.n_vct_local[i] = np.array([b.x, 0, b.z])
            print("\n", i)
            print("v: ", float(b.x), float(b.y), float(b.z))
            print("d: ", self.distances[i])
            self.n_vct[i] = np.array([n_vct.x, n_vct.y, n_vct.z])
            self.n_vct[i] = self.n_vct[i] / np.linalg.norm(self.n_vct[i])
            self.p_contact[i] = np.array([0, _p.x, _p.y])
    def calc_contact_force3D(self, k=1, d=1, eta=0.2):
        self.normal = np.zeros((self.ball3d.num_balls, 3))
        self.fric = np.zeros((self.ball3d.num_balls, 3))
        self.force = np.zeros((self.ball3d.num_balls, 3))
        for i in range(self.ball3d.num_balls):
            if self.distances[i] <= 0:
                _f = -k * self.distances[i] ** d
                self.normal[i] = _f * self.n_vct[i]
                if self.n_vct_local[i, 2] >= 0:
                    dir_fric = rotatex(Point3D(0, 1, 0), self.cage.pos_pockets[i])
                elif self.n_vct_local[i, 2] <= 0:
                    dir_fric = rotatex(Point3D(0, -1, 0), self.cage.pos_pockets[i])
                self.fric[i] = eta * np.linalg.norm(self.normal[i]) * np.array([dir_fric.x, dir_fric.y, dir_fric.z])
                self.force[i] = self.normal[i] + self.fric[i]
    def visualize3D(self):
        plotter = myplotter.MyPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig()
        if ARing:
            groove_a = patches.Circle((0, 0), self.aring.groove.radius, color='k', fill=False)
        if BRing:
            groove_b = patches.Circle((0, 0), self.bring.groove.radius, color='k', fill=False)
        for i in range(self.ball.num_balls):
            _b = patches.Circle((self.ball3d.ball_centers[i].y, self.ball3d.ball_centers[i].z), self.ball.Dw/2, color='k', fill=False)
            axs[0].add_patch(_b)
        for i in range(self.cage.num_pockets):
            _p11 = np.array(self.cage.pockets_lines[i][0].p1)
            _p12 = np.array(self.cage.pockets_lines[i][0].p2)
            _p1 = np.column_stack([_p11, _p12])
            _p21 = np.array(self.cage.pockets_lines[i][1].p1)
            _p22 = np.array(self.cage.pockets_lines[i][1].p2)
            _p2 = np.column_stack([_p21, _p22])
            _l1 = mlines.Line2D(_p1[0], _p1[1], color='b')
            _l2 = mlines.Line2D(_p2[0], _p2[1], color='b')
            axs[0].add_line(_l1)
            axs[0].add_line(_l2)
            axs[0].scatter(self.cage.pocket_centers[i].x, self.cage.pocket_centers[i].y, s=10, c='b' )
            if self.distances[i] <= 0:
                scale = 0.04
                axs[0].quiver(self.p_contact[i, 1], self.p_contact[i, 2], self.normal[i, 1], self.normal[i, 2], color='b', width=0.002, scale_units="xy", angles="xy", scale=scale)
                axs[0].quiver(self.p_contact[i, 1], self.p_contact[i, 2], self.fric[i, 1], self.fric[i, 2], color='r', width=0.002, scale_units="xy", angles="xy", scale=scale)
                axs[0].quiver(self.p_contact[i, 1], self.p_contact[i, 2], self.force[i, 1], self.force[i, 2], color='k', width=0.002, scale_units="xy", angles="xy", scale=scale)
        axs[0].add_patch(groove_a)
        axs[0].add_patch(groove_b)
        _id = patches.Circle(self.cage.ring_outlines[0].center, self.cage.ring_outlines[0].radius, color='b', fill=False)
        _od = patches.Circle(self.cage.ring_outlines[1].center, self.cage.ring_outlines[1].radius, color='b', fill=False)
        axs[0].add_patch(_id)
        axs[0].add_patch(_od)
        axs[0].set(xlim=(-30, 30), ylim=(-30, 30))
        plt.show()


if __name__ == "__main__":
    print("---- run ----")

    cage = SimpleCage()
    cage.translate((0, 0, 0))
    cage.visualize()
    ball = Ball()
    xyz = (0, 0, 0.4)
    # ball.tilt(xyz, np.radians(0))
    # ball.visualize()
    aring = ARing()
    # aring.visualize()
    bring = BRing()
    # bring.visualize()


    # bearing = Bearing(aring, bring, ball, cage)
    # bearing.calc_contact3D()
    # bearing.calc_contact_force3D()
    # bearing.visualize3D()

