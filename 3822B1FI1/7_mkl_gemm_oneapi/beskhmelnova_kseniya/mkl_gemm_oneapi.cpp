#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl/blas.hpp>
#include <cstring>  // Для std::memcpy

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        size_t size,
        sycl::device device) {
    
    sycl::queue q(device, sycl::property::queue::in_order{});

    float* d_a = sycl::malloc_device<float>(size * size, q);
    float* d_b = sycl::malloc_device<float>(size * size, q);
    float* d_c = sycl::malloc_device<float>(size * size, q);

    q.memcpy(d_a, a.data(), size * size * sizeof(float));
    q.memcpy(d_b, b.data(), size * size * sizeof(float));

    oneapi::mkl::blas::row_major::gemm(
        q,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        size,
        size,
        size,
        1.0f,
        d_a, size,
        d_b, size,
        0.0f,
        d_c, size
    );

    std::vector<float> result(size * size);
    q.memcpy(result.data(), d_c, size * size * sizeof(float)).wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);

    return result;
}
