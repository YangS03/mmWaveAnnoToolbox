import numpy as np

# a = np.array([
#     [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 0]], 
#     [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 0]], 
# ])
a = np.array([1,2,3, 1,2,3, 1,2,3, 1,2,3])

a = a.reshape(4, 3)
# a = a.reshape(2, 3, -1, order='F')

print(a)