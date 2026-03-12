#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(const std::vector<float>& matrix_a,const std::vector<float>& matrix_b,size_t size,sycl::device device) {
    sycl::queue queue(device);

    std::vector<float> result(size * size, 0.0f);

    {
        sycl::buffer<float> buffer_a(matrix_a.data(), sycl::range<1>(size * size));
        sycl::buffer<float> buffer_b(matrix_b.data(), sycl::range<1>(size * size));
        sycl::buffer<float> buffer_c(result.data(), sycl::range<1>(size * size));

        using oneapi::mkl::transpose;
        using oneapi::mkl::blas::row_major::gemm;

        const float alpha = 1.0f;
        const float beta  = 0.0f;

        gemm(queue,transpose::nontrans,transpose::nontrans,size,size,size,alpha,
             buffer_a,size,buffer_b,size,beta,buffer_c,size);
    }

    queue.wait();

    return result;
}