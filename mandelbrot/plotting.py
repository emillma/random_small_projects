import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mandel import get_mandelbrot
from numba import cuda
from PyQt5.QtCore import pyqtSignal


class Visualizer:

    def __init__(self):
        self.device_array = cuda.device_array((1024, 2048), np.float32)
        self.host_array = cuda.pinned_array((1024, 2048), np.float32)
        self.stream = cuda.stream()
        self.zoom = 0.3
        self.pos = np.array([-1, 0], np.float32)

        dpi = mpl.rcParams['figure.dpi']
        figsize = 2048 / float(dpi), 1024 / float(dpi)

        self.fig = plt.figure(figsize=figsize)
        self.fig.canvas.mpl_connect('scroll_event', self.zoom_cb)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis('off')
        self.calculate()
        self.img = self.ax.imshow(self.host_array)

    def zoom_cb(self, event):
        print(event)
        if event.button == 'up':
            self.zoom *= 1.1
        if event.button == 'down':
            self.zoom *= 0.9
        self.pos -= (0.5-np.array([event.xdata/2048,
                                   event.ydata/1024], np.float32))/self.zoom
        self.calculate()
        self.draw()

    def draw(self, *args):
        self.img.set_data(self.host_array)
        print(np.amax(self.host_array))
        plt.draw()

    def calculate(self):
        get_mandelbrot[2048, (1, 1024), self.stream](
            self.device_array,
            np.array((*self.pos, self.zoom), np.float32))
        self.device_array.copy_to_host(self.host_array)
        self.stream.synchronize()


vis = Visualizer()
plt.show()
