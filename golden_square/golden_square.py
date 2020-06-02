# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:57:07 2020

@author: emilm
"""


import numba as nb
from numba import cuda
import numpy as np
import time

N, a, b, c, a_min, b_min, a_rest, b_rest = range(8)

solutions = []
def get_combinations(number):
    """ Uded to get number of combinations.
    s_data =  N, a, b, c, a_min, b_min, a_rest, b_rest
    Parameters
    ----------
    number : int

    Returns
    -------
    None.

    """
    s_data = [0 for i in range(8)]
    N, a, b, c, a_min, b_min, a_rest, b_rest = range(8)

    def sqrt(n):
        x = n
        y = 1
        while(x > y):
            x = (x+y)//2
            y = n//x
        return x
    def get_a_min():
        return int(3 + sqrt(3 * s_data[N] - 4)) // 3

    def get_b_min():
        if s_data[a_rest] == 0:
            return 1
        return int(1 + sqrt(2 * s_data[a_rest] - 1)) // 2


    s_data[N] = number
    s_data[a] = sqrt(s_data[N] - 1)

    s_data[a_rest] = s_data[N] - s_data[a] ** 2
    s_data[a_min] = get_a_min()

    s_data[b] = sqrt(s_data[a_rest])
    s_data[b_rest] = s_data[a_rest] - s_data[b] ** 2
    s_data[b_min] = get_b_min()

    s_data[c] = sqrt(s_data[b_rest])
    while s_data[a] >= s_data[a_min]:
        if s_data[b_rest] == s_data[c] * s_data[c]:
            # print(s_data[a]**2 + s_data[b]**2 + s_data[c]**2, s_data[N])
            print(s_data[1:4])
        if s_data[b] > s_data[b_min]:
            s_data[b_rest] += 2 * s_data[b] - 1
            s_data[b] -= 1
            s_data[c] = min(s_data[b] - 1, sqrt(s_data[b_rest]))

        else:
            s_data[a_rest] += 2 * s_data[a] - 1
            s_data[a] -= 1
            s_data[b] = min(s_data[a] - 1, sqrt(s_data[a_rest]))

            s_data[b_rest] = s_data[a_rest] - s_data[b] ** 2
            s_data[b_min] = get_b_min()

            s_data[c] = min(s_data[b] - 1, sqrt(s_data[b_rest]))


    return s_data


get_combinations(6543456)