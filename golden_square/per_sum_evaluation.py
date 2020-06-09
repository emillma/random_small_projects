# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:00:36 2020

@author: emilm
"""

from numba import cuda
import numba as nb
import numpy as np


def get_device_function(singnature, fn):
    nb_fn = nb.njit(singnature)(fn)
    return cuda.jit(singnature, device=True)(nb_fn)

@nb.jit
def sqrt(n):
    x = n
    y = 1
    while(x > y):
        x = (x+y)//2
        y = n//x
    return np.int32(x)
d_sqrt = cuda.jit('i4(i4)', device=True)(sqrt)


def get_a_min(n):
    """Solve a**2 + (a-1)**2 + (a-2)**2 = n"""
    return 1 + sqrt(3 * n - 6) // 3
d_get_a_min = cuda.jit('i4(i4)', device=True)(get_a_min)

@nb.jit
def get_b_max(n):
    """Solve b**2 + (b-1)**2 = n"""
    return (1 + sqrt(2 * n - 1)) // 2
d_get_b_max = cuda.jit('i4(i4)', device=True)(get_b_max)



@cuda.jit('void(i4, i4, i4, i4[:,:])')
def get_combinations(k_max, k_min, a_min, out):
    shared_b = cuda.shared.array((1), nb.i4)
    tx = cuda.threadIdx.x
    a = cuda.blockIdx.x + a_min

    a_2 = a**2
    b = tx + 1
    b_2 = b**2
    a_rest_max = k_max - a**2
    a_rest_min = k_min - a**2
    b_rest_min = max(0, a_rest_min - b**2)
    c = d_sqrt(b_rest_min)

    b_max = d_sqrt(a_rest_max)

    if tx == 0:
        shared_b[0] = 1025



    while b <= b_max and b < a:
        k = a_2 + b_2 + c**2
        if k_min <= k <= k_max and c < b:
            k_relative = k - k_min
            index = cuda.atomic.add(out, (k_relative, 0), 3)
            # index = 1
            out[k_relative, index:index+3] = a, b, c
            c += 1

        else:
            b = cuda.atomic.add(shared_b, 0, 1)
            b_2 = b**2
            b_rest_min = max(0, a_rest_min - b_2)
            c = d_sqrt(b_rest_min)


if __name__ == '__main__':
    k_max = 1000001
    k_min = k_max -0
    a_max = sqrt(k_max - 1)
    a_min = get_a_min(k_min)
    b_max = get_b_max(k_max)

    k_range = k_max - k_min + 1
    blocks = a_max - a_min + 1
    d_out = cuda.device_array((k_range, 1 + 1000*3), np.int32)
    d_out[:, 0] = 1

    get_combinations[blocks,1014](k_max, k_min, a_min,  d_out)

    h_out = d_out.copy_to_host()
    g = h_out[:,1:].reshape(h_out.shape[0],-1,3)
    # g = g**2
    # g = np.sum(g, axis=-1)