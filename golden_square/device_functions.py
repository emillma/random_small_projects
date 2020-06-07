# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 23:11:53 2020

@author: emilm
"""

from numba import cuda


@cuda.jit('i4(i4)', device = True)
def d_tri_numb(n):
    return ((n*(n+1))//2)

@cuda.jit('i4(i4)', device = True)
def d_tri_inv(n):
    """Solve (a(a-1))/2 = n"""
    n = n * 2
    x = n
    y = 1
    while(abs(x - y) > 1):
        x = (x+y)//2
        y = n//(x+1)
    return min(x, y)

@cuda.jit('i4(i4)', device = True)
def d_tet_numb(n):
    return (n * (n + 1) * (n + 2)) // 6

@cuda.jit('i4(i4)', device = True)
def d_tet_inv(n):
    """Solve (a(a-1)(a-2))/6 = n"""
    n = n * 6
    x = n
    y = 1
    step = n
    while(abs(x - y) > 1):
        x = (x + y) // 2
        y = n//((x + 1) * (x + 2))
    return min(x, y)

@cuda.jit('i4(i4)', device = True)
def get_a_min(n):
    return n**2 + (n-1)**2 + (n-2)**2


@cuda.jit('i4(i4)', device = True)
def get_a_min_inv(n):
    """Solve a**2 + (a-1)**2 + (a-2)**2 = n"""
    n = (n-5) // 3
    x = n
    y = 1
    step = n
    while(abs(x - y) > 1):
        x = (x + y) // 2
        y = n // (x - 2)
    return min(x, y)

@cuda.jit('i4(i4)', device = True)
def get_a_max_inv(n):
    """Solve a(a-1) = n"""
    n = (n-1) // 2
    x = n
    y = 1
    step = n
    while(abs(x - y) > 1):
        x = (x + y) // 2
        y = n // (x - 1)
    return min(x, y)

@cuda.jit('i4(i4)', device = True)
def get_a_min(n):
    return n**2 + 1



