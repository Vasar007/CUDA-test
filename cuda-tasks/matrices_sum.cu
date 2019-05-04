#include <iostream>

#include "utils.hpp"


__global__ void kernel(int* const c, const int* const a, const int* const b, const int size)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < size && iy < size)
    {
        const int offset = iy * size + ix;
        c[offset] = a[offset] + b[offset];
    }
}

void print_short(const int* const * const a, const int size, const int border)
{
    for (int i = 0; i < border; ++i)
    {
        for (int j = 0; j < border; ++j)
        {
            std::cout << a[i][j] << '\t';
        }
        std::cout << "..\t";
        for (int j = size - border; j < size; ++j)
        {
            std::cout << a[i][j] << '\t';
        }
        std::cout << '\n';
    }

    std::cout << "..\t\n";

    for (int i = size - border; i < size; ++i)
    {
        for (int j = 0; j < border; ++j)
        {
            std::cout << a[i][j] << '\t';
        }
        std::cout << "..\t";
        for (int j = size - border; j < size; ++j)
        {
            std::cout << a[i][j] << '\t';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

int main()
{
    constexpr int lower_bound = 1;
    constexpr int top_bound = 10;

    constexpr int size = 4000;
    int **a = new int*[size];
    int **b = new int*[size];
    int **c = new int*[size];
    a[0] = new int[size * size];
    b[0] = new int[size * size];
    c[0] = new int[size * size];

    for (int i = 1; i < size; ++i)
    {
        a[i] = a[i - 1] + size;
        b[i] = b[i - 1] + size;
        c[i] = c[i - 1] + size;
    }
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            a[i][j] = utils::random_number(lower_bound, top_bound);
            b[i][j] = utils::random_number(lower_bound, top_bound);
        }
    }

    std::cout << "===== A: =====\n";
    print_short(a, size, 3);
    std::cout << "===== B: =====\n";
    print_short(b, size, 3);

    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;
    cudaMalloc((void**) &dev_a, size * size * sizeof(int));
    cudaMalloc((void**) &dev_b, size * size * sizeof(int));
    cudaMalloc((void**) &dev_c, size * size * sizeof(int));

    cudaMemcpy(dev_a, a[0], size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b[0], size * size * sizeof(int), cudaMemcpyHostToDevice);

    constexpr int threads_per_block_dimension = 16;
    dim3 blocks(size / threads_per_block_dimension, size / threads_per_block_dimension);
    dim3 threads(threads_per_block_dimension, threads_per_block_dimension);
    kernel<<<blocks, threads>>>(dev_c, dev_a, dev_b, size);

    cudaDeviceSynchronize();

    cudaMemcpy(c[0], dev_c, size * size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    std::cout << "===== C: =====\n";
    print_short(c, size, 3);

    delete[] a[0];
    delete[] b[0];
    delete[] c[0];
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}

