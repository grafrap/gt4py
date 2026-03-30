import numpy as np

from gt4py.next.modules.translator import transform_to_unstructured

N = 53
nx = 3
str = "Edge"
b_level = 4
array = np.linspace(0, N-1, N)
print(transform_to_unstructured(array, nx, str, b_level))