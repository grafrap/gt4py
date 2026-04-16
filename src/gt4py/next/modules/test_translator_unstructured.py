# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from gt4py.next.modules.translator import transform_to_unstructured


N = 53
nx = 3
object_str = "Edge"
b_level = 4
array = np.linspace(0, N - 1, N)
print(transform_to_unstructured(array, nx, object_str, b_level))

N = 30
nx = 5
object_str = "Cell"
b_level = 3
array = np.linspace(0.1, N * 0.1, N)
print(transform_to_unstructured(array, nx, object_str, b_level))

N = 24
nx = 5
object_str = "Vertex"
b_level = 2
array = np.linspace(0.1, N * 0.1, N)
print(transform_to_unstructured(array, nx, object_str, b_level))

N = 3 * 6 * 6 + 6 + 6
nx = 6
object_str = "Edge"
b_level = 3
array = np.linspace(0.1, N * 0.1, N)
print(transform_to_unstructured(array, nx, object_str, b_level))

N = 2 * 6 * 6
nx = 6
object_str = "Cell"
b_level = 2
array = np.linspace(0.1, N * 0.1, N)
print(transform_to_unstructured(array, nx, object_str, b_level))

N = 7 * 7
nx = 6
object_str = "Vertex"
b_level = 4
array = np.linspace(0.1, N * 0.1, N)
print(transform_to_unstructured(array, nx, object_str, b_level))

N = 12 * 12
nx = 11
object_str = "Vertex"
b_level = 5
array = np.linspace(0.1, N * 0.1, N)
print(transform_to_unstructured(array, nx, object_str, b_level))

N = 2 * 11 * 11
nx = 11
object_str = "Cell"
b_level = 4
array = np.linspace(0.1, N * 0.1, N)
print(transform_to_unstructured(array, nx, object_str, b_level))

N = 3 * 11 * 11 + 11 + 11
nx = 11
object_str = "Edge"
b_level = 11
array = np.linspace(0.1, N * 0.1, N)
print(transform_to_unstructured(array, nx, object_str, b_level))

nx = 13
ny = 16
N = (nx + 1) * (ny + 1)
object_str = "Vertex"
b_level = 5
array = np.linspace(0.1, N * 0.1, N)
print(transform_to_unstructured(array, nx, object_str, b_level))
