#include "block_gemm_oneapi.h"
#include <cmath>

std::vector<float> GemmBlockONEAPI(const std::vector<float>& a, const std::vector<float>& b,
                                    size_t size, sycl::device device) {
    
    size_t n = size;

    std::vector<float> c(n * n, 0.0f);
    
    sycl::queue q(device);
    
    sycl::buffer<float> bufA(a.data(), a.size());
    sycl::buffer<float> bufB(b.data(), b.size());
    sycl::buffer<float> bufC(c.data(), c.size());

    size_t block_size = 16;
    
    q.submit([&](sycl::handler& cgh) {
        auto a_acc = bufA.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = bufB.get_access<sycl::access::mode::read>(cgh);
        auto c_acc = bufC.get_access<sycl::access::mode::write>(cgh);
        
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((n + block_size - 1) / block_size * block_size,
                                             (n + block_size - 1) / block_size * block_size),
                          sycl::range<2>(block_size, block_size)),
            [=](sycl::nd_item<2> item) {
                int row = item.get_global_id(0);
                int col = item.get_global_id(1);
                
                if (row < n && col < n) {
                    float sum = 0.0f;
                    for (int k = 0; k < n; ++k) {
                        sum += a_acc[row * n + k] * b_acc[k * n + col];
                    }
                    c_acc[row * n + col] = sum;
                }
            }
        );
    }).wait();

    return c;
}