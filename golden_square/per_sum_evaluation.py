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

def sqrt(n):
    x = n
    y = 1
    while(x > y):
        x = (x+y)//2
        y = n//x
    return np.int32(x)
d_sqrt = get_device_function('i4(i4)', sqrt)


def get_a_min(n):
    """Solve a**2 + (a-1)**2 + (a-2)**2 = n"""
    n = (n-5) // 3
    x = n
    y = 1
    step = n
    while(abs(x - y) > 1):
        x = (x + y) // 2
        y = n // (x - 2)
    return np.int32(min(x, y))
d_get_a_min = get_device_function('i4(i4)', get_a_min)


def get_b_min(n):
    """Solve b**2 + (b-1)**2 = n"""
    n = (n-1) // 2
    x = n
    y = 1
    step = n
    while(abs(x - y) > 1):
        x = (x + y) // 2
        y = n // (x - 1)
    return min(x, y)
d_get_b_min = get_device_function('i4(i4)', get_b_min)


def tri_numb(n):
    return ((n*(n+1))//2)
d_tri_numb = get_device_function('i4(i4)', tri_numb)

@cuda.jit('i4(i4)', device = True)
def d_tri_numb(n):
    return ((n*(n+1))//2)

@cuda.jit('i4(i4)', device = True)
def d_tri_inv(n):
    n = n * 2
    x = n
    y = 1
    while(abs(x - y) > 1):
        x = (x+y)//2
        y = n//(x+1)
    return min(x, y)

@cuda.jit('void(i4, i4[:,:])')
def get_combinations(n, out):
    a_max = d_sqrt(n - 1)
    a_min = d_get_a_min(n)

    x = cuda.grid(1)
    tx = cuda.threadIdx.x

@cuda.jit('void(i4[:,:])')
def get_bc_combinations(out):
    x = cuda.grid(1)
    if x >+ out.shape[0]:
        return
    b = d_tri_inv(x)
    c = x - d_tri_numb(b)
    b += 1
    out[x, 0] = b**2 + c**2
    out[x, 1] = b
    out[x, 2] = c


if __name__ == '__main__':
    k_max = int(1e9 + 1)
    a_max = sqrt(k_max - 1)
    a_min = get_a_min(k_max)
    a_rest_max = k_max - a_min**2
    b_max = get_b_min(k_max) - 1

    B = b_max
    L = tri_numb(B-1) + 1
    # d_out = cuda.device_array((L, 3), dtype=np.int32)
    # grid_n = (L - 1) // 1024 + 1
    # get_bc_combinations[grid_n, 1024](d_out)
    # out = d_out.copy_to_host()
    # bc = np.bincount(out[:,0])
    # print(np.count_nonzero(bc[:666645437.0
