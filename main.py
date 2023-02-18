import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

size = 40000
dtype = np.dtype([('x', np.int32), ('y', np.int32)])
empty_gpu = drv.mem_alloc(size * size * dtype.itemsize)

for i in range(size):
    for j in range(size):
        offset = i * size + j
        data = np.array((i, j), dtype=dtype)
        drv.memcpy_htod(empty_gpu + offset * dtype.itemsize, data)

empty_cpu = np.empty((size, size), dtype=dtype)
drv.memcpy_dtoh(empty_cpu, empty_gpu)

print(empty_cpu[-1, -1])
