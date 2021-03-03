from numba import cuda
import numpy as np
import time


N = 1024*1024*1024

d_data = cuda.device_array(N, np.int32)

TPB = 32
BPG = N // TPB


@cuda.jit
def init(arr):
    bx = cuda.grid(1)
    arr[bx] = bx


@cuda.jit
def foo(arr):
    s_data = cuda.shared.array(TPB, np.int32)
    tx = cuda.threadIdx.x
    gx = cuda.blockIdx
    bx = cuda.grid(1)
    grid = cuda.cg.this_grid()

    s_data[tx] = arr[bx]
    # grid.sync()
    arr[bx+1] -= s_data[tx] + 1


@cuda.jit
def count(arr):
    bx = cuda.grid(1)
    val = arr[bx]
    if val:
        cuda.atomic.add(arr, 0, val)


t0 = time.time()
init[BPG, TPB](d_data)
foo[BPG, TPB](d_data)
count[BPG, TPB](d_data)
a = d_data[0]
print(time.time()-t0)
print(a)
# print(np.count_nonzero(h_data))
