#include <mkl_gemm_oneapi.h>
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(const std::vector<float> &a, const std::vector<float> &b, size_t size,
                                 sycl::device device)
{
    sycl::queue queue(device);

    std::vector<float> c(size * size, 0.0f);

    {
        sycl::buffer<float> a_buffer(a.data(), sycl::range<1>(a.size()));
        sycl::buffer<float> b_buffer(b.data(), sycl::range<1>(b.size()));
        sycl::buffer<float> c_buffer(c.data(), sycl::range<1>(c.size()));

        const float alpha = 1.0f;
        const float beta = 0.0f;

        oneapi::mkl::blas::column_major::gemm(queue, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                                              size, size, size, alpha, b_buffer, size, a_buffer, size, beta, c_buffer,
                                              size);

        queue.wait();
    }

    return c;
}
