#include "shared_jacobi_oneapi.h"
#include <cmath>
#include <vector>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {
    
    const size_t n = b.size();
    const float accuracy_sq = accuracy * accuracy;
    
    sycl::queue q(device, sycl::property::queue::in_order{});
    
    float* A = sycl::malloc_shared<float>(a.size(), q);
    float* B = sycl::malloc_shared<float>(b.size(), q);
    float* X = sycl::malloc_shared<float>(n, q);
    float* Xnew = sycl::malloc_shared<float>(n, q);
    float* norm = sycl::malloc_shared<float>(1, q);
    
    for (size_t i = 0; i < a.size(); ++i) A[i] = a[i];
    for (size_t i = 0; i < b.size(); ++i) B[i] = b[i];
    for (size_t i = 0; i < n; ++i) X[i] = 0.0f;
    
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            size_t i = id[0];
            float sum = 0.0f;
            size_t row = i * n;
            
            for (size_t j = 0; j < n; ++j) {
                if (j != i) {
                    sum += A[row + j] * X[j];
                }
            }
            
            Xnew[i] = (B[i] - sum) / A[row + i];
        });
        
        *norm = 0.0f;
        
        q.submit([&](sycl::handler& h) {
            auto red = sycl::reduction(norm, sycl::plus<float>());
            
            h.parallel_for(
                sycl::range<1>(n),
                red,
                [=](sycl::id<1> id, auto& sum) {
                    float diff = Xnew[id] - X[id];
                    sum += diff * diff;
                });
        }).wait();
        
        if (*norm < accuracy_sq) {
            break;
        }
        
        std::swap(X, Xnew);
    }
    
    std::vector<float> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = X[i];
    }
    
    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(X, q);
    sycl::free(Xnew, q);
    sycl::free(norm, q);
    
    return result;
}