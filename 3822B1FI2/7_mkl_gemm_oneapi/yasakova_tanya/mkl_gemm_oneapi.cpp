#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl/blas.hpp>
#include <vector>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    
    sycl::queue queue(device);
    
    std::vector<float> c(size * size, 0.0f);
    
    float* a_dev = sycl::malloc_device<float>(a.size(), queue);
    float* b_dev = sycl::malloc_device<float>(b.size(), queue);
    float* c_dev = sycl::malloc_device<float>(c.size(), queue);
    
    queue.memcpy(a_dev, a.data(), a.size() * sizeof(float)).wait();
    queue.memcpy(b_dev, b.data(), b.size() * sizeof(float)).wait();
    queue.memset(c_dev, 0, c.size() * sizeof(float)).wait();
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    oneapi::mkl::blas::column_major::gemm(
        queue,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        size, size, size,
        alpha,
        a_dev, size,
        b_dev, size,
        beta,
        c_dev, size);
    
    queue.wait();
    
    queue.memcpy(c.data(), c_dev, c.size() * sizeof(float)).wait();
    
    sycl::free(a_dev, queue);
    sycl::free(b_dev, queue);
    sycl::free(c_dev, queue);
    
    return c;
}