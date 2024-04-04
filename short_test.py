import numpy as np
from numpy import float32, arange, repeat, newaxis
from cv2 import imread, IMREAD_GRAYSCALE
from numpy import hsplit, vsplit, array
import cv2
import numpy

test = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])
# print(hsplit(test, 2))
# print(vsplit(test, 2))
# print(array([hsplit(row, 2) for row in vsplit(test, 2)]))

#print(test.reshape(-1, 8))
print(arange(10))

my_array = [1, 2, 3]

print(repeat(my_array, 4))

