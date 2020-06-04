# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:35:28 2020

@author: emilm
"""
import numpy as np
import numba as nb
from numba import cuda
import cupy as cp
def sqrt(n):
        x = n
        y = 1
        while(x > y):
            x = (x+y)//2
            y = n//x
        return x

def get_a_min(n):
        return int(3 + sqrt(3 * n - 4)) // 3


def tet_numb(n):
    return (n * (n + 1) * (n + 2)) // 6

def tet_inv(n):
    n = np.int32(n * 6)
    x = n
    y = np.int32(1)
    step = n
    while(abs(x - y) > 1):
        x = (x+y)//2
        y = n//((x + 1) * (x + 2))
    return min(x, y)

@cuda.jit('i4(i4)', device = True)
def d_tri_numb(n):
    return ((n*(n+1))//2)

@cuda.jit('i4(i4)', device = True)
def d_tri_inv(n):
    n = n * 2
    x = n
    y = 1
    while(x > y):
        x = (x+y)//2
        y = n//(x+1)
    return x

@cuda.jit('i4(i4)', device = True)
def d_tet_numb(n):
    return (n * (n + 1) * (n + 2)) // 6

@cuda.jit('i4(i4)', device = True)
def d_tet_inv(n):
    n = np.int32(n * 6)
    x = n
    y = np.int32(1)
    step = n
    while(abs(x - y) > 1):
        x = (x+y)//2
        y = n//((x + 1) * (x + 2))
    return min(x, y)

@cuda.jit('void(i4[:,:], i4[:])', cache = True)
def get_square_sum(data, count):

    x = cuda.grid(1)
    tx = cuda.threadIdx.x
    if x >= data.shape[0]:
        return
    a = d_tet_inv(x)
    a_rest = x - d_tet_numb(a)
    b = d_tri_inv(a_rest)
    b_rest = a_rest - d_tri_numb(b)
    c = b_rest
    a += 2
    b += 1
    c += 0
    value = a**2 + b**2 + c**2
    data[x] = value, a, b, c
    if value < count.shape[0]:
        cuda.atomic.add(count, value, 1)

a_max = 1023
N = tet_numb(a_max-2)+1
d_data = cuda.device_array((N,4), np.int32)
d_count = cuda.device_array((a_max**2+1), np.int32)
gridsize = (N-1)//1024 + 1
blocksize = 1024
get_square_sum[gridsize, blocksize](d_data, d_count)


# np.take(np.arange(10), 6)

data  = d_data.copy_to_host()
count = d_count.copy_to_host()
# for i in range(100):
#     get_square_sum(i)