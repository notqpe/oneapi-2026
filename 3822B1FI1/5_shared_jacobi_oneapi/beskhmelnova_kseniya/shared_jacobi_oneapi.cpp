#include "shared_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>
#include <vector>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {

    const size_t n = b.size();
    sycl::queue q(device, sycl::property::queue::in_order{});

    float* s_a = sycl::malloc_shared<float>(n * n, q);
    float* s_b = sycl::malloc_shared<float>(n, q);
    float* s_inv = sycl::malloc_shared<float>(n, q);
    float* s_x_curr = sycl::malloc_shared<float>(n, q);
    float* s_x_next = sycl::malloc_shared<float>(n, q);

    q.memcpy(s_a, a.data(), n * n * sizeof(float));
    q.memcpy(s_b, b.data(), n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        s_inv[i] = 1.0f / a[i * n + i];
    }
    q.fill(s_x_curr, 0.0f, n);
    q.wait();

    const size_t wg_size = 64;
    const size_t global_size = ((n + wg_size - 1) / wg_size) * wg_size;

    std::vector<float> x_host(n);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        q.parallel_for(sycl::nd_range<1>(global_size, wg_size),
            [=](sycl::nd_item<1> item) {
                size_t i = item.get_global_id(0);
                if (i >= n) return;

                float sum = 0.0f;
                size_t row = i * n;
                
                #pragma unroll(4)
                for (size_t j = 0; j < n; j++) {
                    if (j != i) {
                        sum += s_a[row + j] * s_x_curr[j];
                    }
                }
                s_x_next[i] = s_inv[i] * (s_b[i] - sum);
            });

        q.memcpy(x_host.data(), s_x_next, n * sizeof(float)).wait();
        
        float max_diff = 0.0f;
        for (size_t i = 0; i < n; i++) {
            float diff = sycl::fabs(x_host[i] - s_x_curr[i]);
            if (diff > max_diff) max_diff = diff;
        }
        
        if (max_diff < accuracy) {
            std::swap(s_x_curr, s_x_next);
            break;
        }

        std::swap(s_x_curr, s_x_next);
    }

    std::vector<float> result(n);
    q.memcpy(result.data(), s_x_curr, n * sizeof(float)).wait();

    sycl::free(s_a, q);
    sycl::free(s_b, q);
    sycl::free(s_inv, q);
    sycl::free(s_x_curr, q);
    sycl::free(s_x_next, q);

    return result;
}
