#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {
    
    const int n = static_cast<int>(b.size());
    const float accuracy_sq = accuracy * accuracy;
    sycl::queue q(device, sycl::property::queue::in_order{});

    std::vector<float> inv_diag(n);
    for (int i = 0; i < n; i++) {
        inv_diag[i] = 1.0f / a[i * n + i];
    }

    float* d_a = sycl::malloc_device<float>(n * n, q);
    float* d_b = sycl::malloc_device<float>(n, q);
    float* d_inv = sycl::malloc_device<float>(n, q);
    float* d_x_curr = sycl::malloc_device<float>(n, q);
    float* d_x_next = sycl::malloc_device<float>(n, q);

    q.memcpy(d_a, a.data(), sizeof(float) * n * n);
    q.memcpy(d_b, b.data(), sizeof(float) * n);
    q.memcpy(d_inv, inv_diag.data(), sizeof(float) * n);
    q.fill(d_x_curr, 0.0f, n);

    const size_t wg_size = 64;
    const size_t global_size = ((n + wg_size - 1) / wg_size) * wg_size;
    
    bool converged = false;
    const int CHECK_INTERVAL = 8;
    
    std::vector<float> x_host(n);
    std::vector<float> x_prev(n, 0.0f);

    for (int iter = 0; iter < ITERATIONS && !converged; iter++) {
        q.parallel_for(sycl::nd_range<1>(global_size, wg_size),
            [=](sycl::nd_item<1> item) {
                size_t i = item.get_global_id(0);
                if (i >= static_cast<size_t>(n)) return;

                float sum = 0.0f;
                const size_t row_start = i * n;

                #pragma unroll(4)
                for (int j = 0; j < n; j++) {
                    if (j != static_cast<int>(i)) {
                        sum += d_a[row_start + j] * d_x_curr[j];
                    }
                }
                d_x_next[i] = d_inv[i] * (d_b[i] - sum);
            });

        if ((iter + 1) % CHECK_INTERVAL == 0) {
            q.memcpy(x_host.data(), d_x_next, sizeof(float) * n).wait();
            
            float norm_sq = 0.0f;
            for (int i = 0; i < n; i++) {
                float diff = x_host[i] - x_prev[i];
                norm_sq += diff * diff;
            }
            
            if (norm_sq < accuracy_sq) {
                converged = true;
                break;
            }
            
            x_prev = x_host;
        }

        std::swap(d_x_curr, d_x_next);
    }

    q.memcpy(x_host.data(), d_x_curr, sizeof(float) * n).wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_inv, q);
    sycl::free(d_x_curr, q);
    sycl::free(d_x_next, q);

    return x_host; //
}
