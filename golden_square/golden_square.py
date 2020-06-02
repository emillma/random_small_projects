# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:57:07 2020

@author: emilm
"""


import numba as nb
from numba import cuda
import numpy as np

@nb.cuda.jit('void(u4[:])', cache=True)
def generation(data):
    local_data = cuda.shared.array(shape=(10), dtype=nb.uint8)
    x = cuda.grid(1)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

def solve(a,b,c):
    return int(-b + np.sqrt(b**2 - 4*a*c) )// (2*a)
def find_k(diff, target):
    return int(-2 * target + 2 * np.sqrt(target ** 2 + diff) )//2


solutions = []
def get_combinations(number):
    """ Uded to get number of combinations.
    s_data =  a, b, c, N, d, found_n.
    Parameters
    ----------
    number : int

    Returns
    -------
    None.

    """
    s_data = [0 for i in range(6)]
    a, b, c, N, d, found_n = range(6)
    s_data[N] = number
    s_data[a], s_data[b], s_data[c] = 1, 2, int(np.sqrt(s_data[N] - 1 - 4))
    s_data[d] = s_data[N] - (s_data[a]**2 + s_data[b]**2 + s_data[c]**2)

    while 1:
        print(s_data)
        if s_data[d] >= 2 * s_data[a] + 1: # if not possible a, not possible b
            if s_data[d] >= 2 * s_data[b] + 1:
                target_increase = b
            else:
                target_increase = a

            k = find_k(s_data[d], s_data[target_increase])
            s_data[d] -= 2 * k * s_data[target_increase] + k**2
            s_data[target_increase] += k

        else:
            if not s_data[d]:
                s_data[found_n] += 1
                solutions.append([s_data[a], s_data[b], s_data[c]])

            if s_data[c] - s_data[a] <= 2:
                break

            else:
                if s_data[c] - s_data[b] > 1:
                    target_decrease = c
                else:
                    target_decrease = b

                s_data[d] += 2 * s_data[target_decrease] - 1
                s_data[target_decrease] -= 1




        # else:
        #     if s_data[c] >= s_data[b] + 2:
        #         s_data[c] -= 1


    return s_data


print(get_combinations(1424))
print(solutions)