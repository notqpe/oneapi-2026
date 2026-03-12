#include "shared_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {

    const size_t n = b.size();
    sycl::queue q(device, sycl::property::queue::in_order{});

    float* s_a = sycl::malloc_shared<float>(n * n, q);
    float* s_b = sycl::malloc_shared<float>(n, q);
    float* s_inv = sycl::malloc_shared<float>(n, q);
    float* s_x = sycl::malloc_shared<float>(n, q);
    float* s_x_new = sycl::malloc_shared<float>(n, q);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) s_a[i * n + j] = a[i * n + j];
        s_b[i] = b[i];
        s_inv[i] = 1.0f / a[i * n + i];
        s_x[i] = 0.0f;
    }

    for (int iter = 0; iter < ITERATIONS; iter++) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            float sum = 0.0f;
            for (size_t j = 0; j < n; j++) {
                if (i != j) sum += s_a[i * n + j] * s_x[j];
            }
            s_x_new[i] = s_inv[i] * (s_b[i] - sum);
        });

        q.wait();

        float max_diff = 0.0f;
        for (size_t i = 0; i < n; i++) {
            max_diff = std::max(max_diff, std::abs(s_x_new[i] - s_x[i]));
            s_x[i] = s_x_new[i];
        }

        if (max_diff < accuracy) break;
    }

    std::vector<float> result(n);
    for (size_t i = 0; i < n; i++) result[i] = s_x[i];

    sycl::free(s_a, q);
    sycl::free(s_b, q);
    sycl::free(s_inv, q);
    sycl::free(s_x, q);
    sycl::free(s_x_new, q);

    return result;
}