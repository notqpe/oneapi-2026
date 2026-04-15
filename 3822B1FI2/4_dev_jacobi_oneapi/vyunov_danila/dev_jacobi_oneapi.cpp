// Copyright 2026 Vyunov Danila
#include "dev_jacobi_oneapi.h"

#include <cmath>
#include <cstdlib>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device) {
    const size_t n = b.size();
    if (n == 0 || a.size() != n * n) {
        return {};
    }

    std::vector<float> result(n, 0.0f);

    sycl::queue q(device, sycl::property::queue::in_order{});

    // Allocate device memory
    float* d_a    = sycl::malloc_device<float>(n * n, q);
    float* d_b    = sycl::malloc_device<float>(n, q);
    float* d_x    = sycl::malloc_device<float>(n, q);
    float* d_xnew = sycl::malloc_device<float>(n, q);
    float* d_diff = sycl::malloc_device<float>(1, q);

    if (!d_a || !d_b || !d_x || !d_xnew || !d_diff) {
        sycl::free(d_a,    q);
        sycl::free(d_b,    q);
        sycl::free(d_x,    q);
        sycl::free(d_xnew, q);
        sycl::free(d_diff, q);
        return {};
    }

    // Copy input data to device; initialise x with zeros
    q.memcpy(d_a, a.data(), sizeof(float) * n * n);
    q.memcpy(d_b, b.data(), sizeof(float) * n);
    q.fill(d_x,    0.0f, n);
    q.fill(d_xnew, 0.0f, n);

    float max_diff = 0.0f;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        // Compute next iteration of x
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                const size_t i = id[0];
                float sum = 0.0f;
                for (size_t j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += d_a[i * n + j] * d_x[j];
                    }
                }
                d_xnew[i] = (d_b[i] - sum) / d_a[i * n + i];
            });
        });

        // Compute max |x_new - x| via reduction
        q.submit([&](sycl::handler& h) {
            auto red = sycl::reduction(d_diff, h, 0.0f, sycl::maximum<float>());
            h.parallel_for(sycl::range<1>(n), red, [=](sycl::id<1> id, auto& mx) {
                mx.combine(sycl::fabs(d_xnew[id] - d_x[id]));
            });
        });

        // Read convergence value back to host
        q.memcpy(&max_diff, d_diff, sizeof(float)).wait();

        // Swap current and next pointers
        float* tmp = d_x;
        d_x    = d_xnew;
        d_xnew = tmp;

        if (max_diff < accuracy) {
            break;
        }
    }

    // Copy result back
    q.memcpy(result.data(), d_x, sizeof(float) * n).wait();

    sycl::free(d_a,    q);
    sycl::free(d_b,    q);
    sycl::free(d_x,    q);
    sycl::free(d_xnew, q);
    sycl::free(d_diff, q);

    return result;
}
