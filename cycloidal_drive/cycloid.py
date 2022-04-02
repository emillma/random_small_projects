from matplotlib import pyplot as plt
import numpy as np
from numpy import sin, cos


def circle_points(angle, radius=1):
    return radius*np.array([cos(angle), sin(angle)])


points = []
r_inner = 10
r_outer = 1
r_peg = 1
for angle in np.linspace(0, np.pi*2, 200, endpoint=False):
    inner = circle_points(angle)
    outer_angle = angle + (angle * (r_inner / r_outer))
    point = (circle_points(angle, r_inner+r_outer)
             + circle_points(outer_angle + np.pi, r_outer))

    norm = np.linalg.norm(point)
    points.append(point)


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1, adjustable='box', aspect=1)
ax1.scatter(*zip(*points))
plt.show()
