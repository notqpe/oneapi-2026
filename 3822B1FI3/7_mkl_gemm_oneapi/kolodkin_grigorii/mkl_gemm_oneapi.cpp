#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(const std::vector<float>& a, const std::vector<float>& b,
                               size_t size, sycl::device device) {
    size_t n = size;
    std::vector<float> c(n * n, 0.0f);
    
    sycl::queue q(device);
    
    sycl::buffer<float> bufA(a.data(), a.size());
    sycl::buffer<float> bufB(b.data(), b.size());
    sycl::buffer<float> bufC(c.data(), c.size());
    
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    
    oneapi::mkl::blas::row_major::gemm(
        q,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        n,                
        n,                
        n,                
        alpha,
        bufA, n,        
        bufB, n,        
        beta,
        bufC, n         
    );
    q.wait();
    
    return c;
}