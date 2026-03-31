import numpy as np

from gt4py.next.modules.translator import transform_to_unstructured

N = 53
nx = 3
str = "Edge"
b_level = 4
array = np.linspace(0, N-1, N)
print(transform_to_unstructured(array, nx, str, b_level))

N = 30
nx = 5
str = "Cell"
b_level = 3
array = np.linspace(0.1, N*0.1, N)
print(transform_to_unstructured(array, nx, str, b_level))

N = 24
nx = 5
str = "Vertex"
b_level = 2
array = np.linspace(0.1, N*0.1, N)
print(transform_to_unstructured(array, nx, str, b_level))

N = 3 * 6 * 6 + 6 + 6
nx = 6
str = "Edge"
b_level = 3
array = np.linspace(0.1, N*0.1, N)
print(transform_to_unstructured(array, nx, str, b_level))

N = 2 * 6 * 6
nx = 6
str = "Cell"
b_level = 2
array = np.linspace(0.1, N*0.1, N)
print(transform_to_unstructured(array, nx, str, b_level))

N = 7 * 7
nx = 6
str = "Vertex"
b_level = 4
array = np.linspace(0.1, N*0.1, N)
print(transform_to_unstructured(array, nx, str, b_level))

N = 12 * 12
nx = 11
str = "Vertex"
b_level = 5
array = np.linspace(0.1, N*0.1, N)
print(transform_to_unstructured(array, nx, str, b_level))

N = 2 * 11 * 11
nx = 11
str = "Cell"
b_level = 4
array = np.linspace(0.1, N*0.1, N)
print(transform_to_unstructured(array, nx, str, b_level))

N = 3 * 11 * 11 + 11 + 11
nx = 11
str = "Edge"
b_level = 11
array = np.linspace(0.1, N*0.1, N)
print(transform_to_unstructured(array, nx, str, b_level))
