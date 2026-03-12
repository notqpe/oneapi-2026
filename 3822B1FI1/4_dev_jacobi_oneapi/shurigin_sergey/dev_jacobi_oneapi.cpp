#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {

    const int n = static_cast<int>(b.size());
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

    std::vector<float> x_host(n, 0.0f);
    std::vector<float> x_next_host(n, 0.0f);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            int i = idx[0];
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                if (i != j) sum += d_a[i * n + j] * d_x_curr[j];
            }
            d_x_next[i] = d_inv[i] * (d_b[i] - sum);
        });

        q.memcpy(x_next_host.data(), d_x_next, sizeof(float) * n).wait();

        float max_diff = 0.0f;
        for (int i = 0; i < n; i++) {
            max_diff = std::max(max_diff, std::abs(x_next_host[i] - x_host[i]));
            x_host[i] = x_next_host[i];
        }

        std::swap(d_x_curr, d_x_next);

        if (max_diff < accuracy) break;
    }

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_inv, q);
    sycl::free(d_x_curr, q);
    sycl::free(d_x_next, q);

    return x_host;
}