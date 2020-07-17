# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:03:17 2020

@author: Damon
"""

import numpy as np

list1 = [1,2,3]

arr1 = np.array(list1)

list2 = [[1,2,3],[4,5,6],[7,8,9]]

matrix1 = np.array(list2)

print(matrix1.shape)

print(np.zeros((3,5)) + 3)

print(np.zeros(5))

print(np.ones((2,6)) * 4)

print(np.linspace(0,8,17))

print(np.eye(5))

print(np.random.rand(1))

print(np.random.rand(4))

print(np.random.rand(3,6))

np.random.randn(10)

np.random.normal(12.0,3.5,(3,5))

np.random.randint(0,11,50)  # 50 NUMBERS FROM 0 (INCLUSIVE) TO 11 (EXCLUSIVE - so really "10")

np.random.seed(101)
np.random.rand(4)

arr = np.arange(25)

ranarr = np.random.randint(0,50,10)

arr.shape

arr.reshape(5,5)

arr2 = np.arange(25).reshape(5,5)

ranarr.max()
ranarr.argmax()
ranarr.min()
ranarr.argmin()

np.random.randint(0,10,10)
np.random.choice(10,10,replace=False)

np.random.choice(5,3)       # THESE ARE THE SAME
np.random.randint(0,5,3)    # THESE ARE THE SAME



arr3 = np.arange(0,11)

arr3 + 2

# NOTE: REASSIGN OF ARRAY SLICES CHANGES ORIGINAL ARRAY

print(arr3)
print('\n')
arr3_slice = arr3[0:5]
print(arr3_slice)
print('\n')
arr3_slice[:] = np.arange(15,20)
#arr3_slice[:] = 99
print(arr3_slice)
print('\n')
print(arr3)
print('\n')

# SO USE arr.copy() TO AVOID THAT POTENTIAL FOR PROBLEMS

# INDEXING 2D ARRAYS

arr_2d = np.array([[5,10,15],[20,25,30],[35,40,45]])

arr_2d.shape

arr_2d[1]  # ROW INDEX 1

arr_2d[1,2]  # VALUE At ROW INDEX 1 / COLUMN INDEX 2
# OR
arr_2d[1,2]

arr_2d[2,2]

arr_2d[:2]

arr_2d[:2,1:]

arr_2d[:,1] # COLUMN 1 INDEX

arr3 = np.arange(0,15)

arr3 = arr3.reshape(3,5)

arr3 > 4

bool_arr3 = arr3 > 4

arr3[bool_arr3]

arr4 = np.arange(0,15).reshape(3,5)

arr4[arr4 > 7]

arr4[arr4 % 3 == 0]

# SUM ACROSS ROWS (AXIS = 0) -> MEANING SUM OF EACH COLUMN
arr4.sum(axis=0)

# SUM ACROSS COLUMNS (AXIS = 1) -> MEANING SUM OF EACH ROW
arr4.sum(axis=1)








