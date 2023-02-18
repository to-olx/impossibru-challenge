#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(1); \
    } \
} while (0)

__global__ void fill_array(short2 *arr, int size, int tile_size)
{
    __shared__ short2 tile[16][16];

    int tile_x = blockIdx.x * tile_size;
    int tile_y = blockIdx.y * tile_size;
    int i_start = tile_x + threadIdx.x;
    int j_start = tile_y + threadIdx.y;

    for (int i = i_start; i < tile_x + tile_size && i < size; i += blockDim.x)
    {
        for (int j = j_start; j < tile_y + tile_size && j < size; j += blockDim.y)
        {
            tile[threadIdx.x][threadIdx.y] = make_short2(i,j);
            arr[i*size + j] = tile[threadIdx.x][threadIdx.y];
        }
    }
}

int main()
{
    int size = 40000;
    int tile_size = 64;
    short2 *d_arr, *h_arr;
    size_t arr_size = size * size * sizeof(short2);

    // Allocate pinned memory on the CPU
    CUDA_CHECK(cudaMallocHost((void**)&h_arr, arr_size));

    // Allocate memory on the GPU
    CUDA_CHECK(cudaMalloc((void**)&d_arr, arr_size));

    // Launch the kernel to fill the array
    dim3 block_size(16, 16);
    dim3 grid_size((size + tile_size - 1) / tile_size, (size + tile_size - 1) / tile_size);
    for (int tile_x = 0; tile_x < size; tile_x += tile_size)
    {
        for (int tile_y = 0; tile_y < size; tile_y += tile_size)
        {
            dim3 tile_grid_size((tile_size + block_size.x - 1) / block_size.x, (tile_size + block_size.y - 1) / block_size.y);
            fill_array<<<tile_grid_size, block_size>>>(d_arr + tile_x * size + tile_y, size, tile_size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Copy the result back to the CPU
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, arr_size, cudaMemcpyDeviceToHost));

    // Print the last element of the array
    printf("%d %d\n", h_arr[size * size - 1].x, h_arr[size * size - 1].y);

    // Free the memory
    CUDA_CHECK(cudaFree(d_arr));
    free(h_arr);
    return 0;
}
