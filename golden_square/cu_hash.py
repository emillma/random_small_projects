# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:28:00 2020

@author: emilm
"""


import numpy as np
from numba import cuda
import numba as nb


@cuda.jit('void(i4[:])', device=True)
def wait(lock):
    while True:
        old = cuda.atomic.compare_and_swap(lock, 1, 0)
        if old == 1:
            return


@cuda.jit('void(i4[:])', device=True)
def signal(lock):
    cuda.atomic.compare_and_swap(lock, 0, 1)


@cuda.jit('i4(i4[:])', device=True)
def malloc(queue):
    pop_lock = queue[3:4]
    wait(pop_lock)
    first = queue[1]
    if first == -1:
        signal(pop_lock)
        return -1
    second = queue[first]
    queue[1] = second
    signal(pop_lock)
    return first


@cuda.jit('void(i4[:], i4)', device=True)
def free(queue, item):
    push_lock = queue[4:5]
    queue[item] = -1
    wait(push_lock)
    last = queue[2]
    queue[last] = item
    queue[2] = item
    signal(push_lock)


@cuda.jit('void(i4[:], i4[:,:])')
def take_from_queue(queue, out):
    x = cuda.grid(1)
    out[x, 0] = malloc(queue)


@cuda.jit('void(i4[:], i4)')
def fill_queue(queue, s):
    x = cuda.grid(1)
    index = x * s + 5
    if index >= queue.size - s:
        return
    if x == 0:
        queue[0] = s  # Size og every block allocated
        queue[1] = 5  # Start
        queue[2] = queue.size - s  # End
        queue[3:5] = 1  # Locks
        queue[queue.size - s] = -1
    queue[index] = (x + 1) * s + 5


def create_queue(n, s):
    queue = cuda.device_array(5+n*(s), dtype=np.int32)
    blocks = (n - 1)//1024 + 1
    fill_queue[blocks, 1024](queue, s)
    return queue


queue = create_queue(4, 2).copy_to_host()
out = np.zeros((1024,2), dtype=np.int32)
take_from_queue[1,7](queue, out)


