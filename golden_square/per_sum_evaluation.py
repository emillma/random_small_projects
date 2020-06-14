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

            combinations[k_relative, index:index+3] = a, b, c
            c += 1

        else:
            b = cuda.atomic.add(shared_b, 0, 1)
            b_2 = b**2
            b_rest_min = max(0, a_rest_min - b_2)
            c = d_sqrt(b_rest_min)


@cuda.jit('void(i4[:,:], i4[:,:])', cache = True)
def eleminate_by_missing_relations(combinations, out):
    bx = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    s_comb = cuda.shared.array((1023*3, 3), nb.i4)
    s_relations = cuda.shared.array((341, 4), nb.i4)
    s_valid_pointers = cuda.shared.array((1023*3), nb.u2)
    s_cout = cuda.shared.array((1), nb.i4)
    s_n = cuda.shared.array((1), nb.i4)


    if tx ==0 and ty ==0:
        s_n[0] = (combinations[bx, 0] - 1) // 3
        s_cout[0] = 1
    cuda.syncthreads()

    # Set s_valid
    ix, iy = tx, ty
    while 3 * ix + iy < s_valid_pointers.size:
        s_valid_pointers[3 * ix + iy] = 3 * ix + iy
        ix += 341
    cuda.syncthreads()

    # Set s_comb
    ix, iy = tx, ty
    while ix < s_n[0]:
        s_comb[ix, iy] = combinations[bx, 1 + 3*ix + iy]
        ix += 341

    # Do count
    cuda.syncthreads()
    for i in range((s_n[0]-1)//314 + 1):
        ix, iy = tx + 314 * i, ty
        s_relations[tx, ty] = 0
        if ix < s_n[0]:
            for other_ptr in range(s_n[0]):
                other = s_valid_pointers[other_ptr]
                if (s_comb[ix, iy] == s_comb[other, 0]
                        or s_comb[ix, iy] == s_comb[other, 1]
                        or s_comb[ix, iy] == s_comb[other, 2]):
                    s_relations[ix, iy] += 1

        cuda.atomic.add(s_relations, (ix, 3), s_relations[ix, iy])
        s_relations[ix, iy] = s_relations[ix, iy] >= 2
        cuda.syncthreads()
        if ix < s_n[0]:
            s_relations[ix, 3] = s_relations[ix, 3] >= 7
        cuda.syncthreads()
        for step in (2, 4):
            if iy % step == 0:
                s_relations[ix, iy] *= s_relations[ix, iy + step//2]
            cuda.syncthreads()
        if iy == 0 and s_relations[ix, 0]:
            pointer_index = cuda.atomic.add(s_cout, 0, 1)
            s_valid_pointers[pointer_index] = ix


    cuda.syncthreads()
    ix, iy = tx, ty
    while ix < s_n[0]:
        out[bx, 3 * ix + iy] = s_relations[ix, iy]
        ix += 341

if __name__ == '__main__':
    k_max = 1024*64 * 6
    k_min = k_max - 10
    a_max = sqrt(k_max - 1)
    a_min = get_a_min(k_min)

    k_range = k_max - k_min
    blocks = a_max - a_min
    d_combinations = cuda.device_array((k_range, 1 + 1023*3*3), np.int32)
    d_combinations[:, 0] = 1

    get_combinations[blocks, 1024](k_max, k_min, a_min,  d_combinations)

    d_out = cuda.device_array((k_range, 1 + 1023*3*3), np.int32)
    d_out[:, :] = 0
    eleminate_by_missing_relations[k_max - k_min, (341,3)](
        d_combinations, d_out)

    h_combinations = d_combinations.copy_to_host()
    h_out = d_out.copy_to_host()


    print(np.any(h_out))
    # g = h_out[:, 1:].reshape(h_out.shape[0], -1, 3)
    # g = g**2
