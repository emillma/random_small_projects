import numpy as np
from numba import cuda
import numba as nb
import colorama
from matplotlib import pyplot as plt
import matplotlib as mpl
from numba.cuda import libdevice as libdv
import math
colorama.init(strip=True)
TPB = 1024
BPG = 2048
LIMIT = 1e12


@cuda.jit
def get_mandelbrot(output, area):
    s_value = cuda.shared.array(TPB, np.complex128)
    s_count = cuda.shared.array(TPB, np.int32)
    s_const = cuda.shared.array(TPB, np.complex128)
    tidx = cuda.threadIdx.x
    gidx, gidy = cuda.grid(2)

    x_offset, y_offset, scale = area
    s_const[tidx] = np.complex64(
        x_offset + (gidx - BPG/2)/(BPG * scale)
        + 1j * (y_offset + (gidy - TPB/2)/(BPG * scale)))

    s_value[tidx] = 0
    s_count[tidx] = 0
    for _ in range(1000):
        s_value[tidx] = s_value[tidx] ** 2 + s_const[tidx]
        if s_value[tidx].real**2 + s_value[tidx].imag**2 <= LIMIT:
            s_count[tidx] += 1
        else:
            break

    # output[gidy, gidx] = -math.log(
    #     s_value[tidx].real**2 + s_value[tidx].imag**2)
    # output[gidy, gidx] = math.log(0.24)
    output[gidy, gidx] = s_const[tidx].imag
    # cuda.syncthreads()

    curent_value = 1

    # if x < an_array.shape[0] and y < an_array.shape[1]:
    #     an_array[x, y] += 1


if __name__ == '__main__':
    img = np.zeros((TPB, BPG), np.float32)

    # d_img = cuda.to_device(img)
    iterations = np.int32(100000)
    # increment_a_2D_array[blockspergrid, threadsperblock](img, 10)
    area = np.array([-.7438, .13188204, 45169], np.float32)
    # area = np.array([-0, 0, 0.2], np.float32)
    get_mandelbrot[BPG, (1, TPB)](img, area)

    dpi = mpl.rcParams['figure.dpi']
    figsize = BPG / float(dpi), TPB / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    plt.imshow(img)
    plt.show()
