from numba import jit, cuda

@cuda.jit
def insane():
    empty = []
    size = 40_000
    for i in range(size):
        for j in range(size):
            empty.append((i,j))

    return empty[-1]


insane()
numba.cuda.profile_stop()
