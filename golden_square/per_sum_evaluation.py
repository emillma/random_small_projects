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


@nb.jit
def get_a_min(n):
    """Solve a**2 + (a-1)**2 + (a-2)**2 = n"""
    return 1 + sqrt(3 * n - 6) // 3


d_get_a_min = cuda.jit('i4(i4)', device=True)(get_a_min)


@nb.jit
def get_b_max(n):
    """Solve b**2 + (b-1)**2 = n"""
    return (1 + sqrt(2 * n - 1)) // 2


d_get_b_max = cuda.jit('i4(i4)', device=True)(get_b_max)


@cuda.jit('void(i4, i4, i4, i4[:,:])', cache = True)
def get_combinations(k_max, k_min, a_min, combinations):
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
        if k_min <= k < k_max and c < b:
            k_relative = k - k_min
            index = cuda.atomic.add(combinations, (k_relative, 0), 3)
            # index = 1
            combinations[k_relative, index:index+3] = a, b, c
            c += 1

        else:
            b = cuda.atomic.add(shared_b, 0, 1)
            b_2 = b**2
            b_rest_min = max(0, a_rest_min - b_2)
            c = d_sqrt(b_rest_min)


@cuda.jit('void(i4[:,:], i4[:,:])')
def eleminate_by_missing_relations(combinations, out):
    bx = cuda.blockIdx.x
    tx = cuda.threadIdx.x

    s_comb = cuda.shared.array((1023*2, 3), nb.i4)
    s_comb_tmp = cuda.shared.array((341, 3), nb.i4)

    for i in range(3):
        s_comb_tmp[tx, i] = 0
    s_relations = cuda.shared.array((1023, 3), nb.i4)
    for i in range(3):
        s_relations[tx, i] = 0
    s_count = cuda.shared.array((1), nb.i4)

    n_3 = combinations[bx, 0]
    n = (n_3 - 1) // 3

    rotations = n // 341 + 1

    if tx >= n * 3:
        return

    for r in range(rotations): #copy data
        tx_r = tx // 3 + r * 341
        if tx_r < n:
            s_comb[tx_r, tx % 3] = combinations[bx, tx + 1023 * r + 1]
            # s_comb[tx_r, tx % 3] = 1

    cuda.syncthreads()
    for r in range(rotations): #count connections
        tx_r = tx // 3 + r * 341
        if tx_r >= n:
            break

        for comb in range(n):
            for s in range(3):
                if s_comb[tx_r, tx % 3] == s_comb[comb, s]:
                    s_relations[tx_r, tx % 3] += 1

    if tx == 0:
        s_count[0] = 0
    cuda.syncthreads() #remove invalid groups
    copy = True
    sum_ = 0
    for i in range(3):
        copy *= s_relations[tx, i] >= 2
        s += s_relations[tx, i]
    copy *= sum_ >= 7

    cuda.syncthreads()
    if copy:
        increase_ix = cuda.atomic.add(s_count, 0, 1)
        for i in range(3):
            s_comb_tmp[increase_ix, i] = s_comb[tx, i]


    for r in range(rotations):
        tx_r = tx // 3 + r * 341
        if tx_r < n:
            out[bx, tx + 1023 * r] = s_comb_tmp[tx_r, tx % 3]
    return
if __name__ == '__main__':
    k_max = 1024*64 * 6
    k_min = k_max - 1024*64
    a_max = sqrt(k_max - 1)
    a_min = get_a_min(k_min)

    k_range = k_max - k_min
    blocks = a_max - a_min
    d_combinations = cuda.device_array((k_range, 1 + 1023*6), np.int32)
    d_combinations[:, 0] = 1

    get_combinations[blocks, 1024](k_max, k_min, a_min,  d_combinations)

    d_out = cuda.device_array((k_range, 1 + 1023*6), np.int32)
    d_out[:, :] = 0
    eleminate_by_missing_relations[k_max - k_min, 1023](
        d_combinations, d_out)

    h_combinations = d_combinations.copy_to_host()
    h_out = d_out.copy_to_host()
    print(np.any(h_out))
    # g = h_out[:, 1:].reshape(h_out.shape[0], -1, 3)
    # g = g**2
