#include <iostream>

#include "utils.hpp"


__global__ void kernel(int* const c, const int* const a, const int* const b, const int size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        c[i] = a[i] + b[i];
    }
}

void print_short(const int* const a, const int size, const int border)
{
    for (int i = 0; i < border; ++i)
    {
        std::cout << a[i] << '\t';
    }
    std::cout << "..\t";
    for (int i = size - border; i < size; ++i)
    {
        std::cout << a[i] << '\t';
    }
    std::cout << '\n';
}

int main()
{
    constexpr int lower_bound = 1;
    constexpr int top_bound = 10;

    constexpr int size = 512;
    int a[size];
    int b[size];
    int c[size];
    for (int i = 0; i < size; ++i)
    {
        a[i] = utils::random_number(lower_bound, top_bound);
        b[i] = utils::random_number(lower_bound, top_bound);
    }
    print_short(a, size, 5);
    print_short(b, size, 5);

    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;
    cudaMalloc((void**) &dev_a, size * sizeof(int));
    cudaMalloc((void**) &dev_b, size * sizeof(int));
    cudaMalloc((void**) &dev_c, size * sizeof(int));

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    constexpr int blocks = 16;
    constexpr int threads = size / blocks;
    kernel<<<blocks, threads>>>(dev_c, dev_a, dev_b, size);

    cudaDeviceSynchronize();

    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    print_short(c, size, 5);

    return 0;
}
