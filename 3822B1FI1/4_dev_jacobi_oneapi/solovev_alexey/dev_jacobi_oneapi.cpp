#include "dev_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiDevONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float accuracy,
    sycl::device device) {

    const size_t n = b.size();
    const float accuracy_sq = accuracy * accuracy;

    sycl::queue q(device, sycl::property::queue::in_order{});

    float* A = sycl::malloc_device<float>(a.size(), q);
    float* B = sycl::malloc_device<float>(b.size(), q);
    float* X = sycl::malloc_device<float>(n, q);
    float* Xnew = sycl::malloc_device<float>(n, q);
    float* norm = sycl::malloc_device<float>(1, q);

    q.memcpy(A, a.data(), sizeof(float) * a.size());
    q.memcpy(B, b.data(), sizeof(float) * b.size());
    q.fill(X, 0.0f, n);
    q.wait();

    for (int iter = 0; iter < ITERATIONS; ++iter) {

        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            size_t i = id[0];
            float sum = 0.0f;
            size_t row = i * n;

            for (size_t j = 0; j < n; ++j)
                if (j != i)
                    sum += A[row + j] * X[j];

            Xnew[i] = (B[i] - sum) / A[row + i];
            });

        q.fill(norm, 0.0f, 1);

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

        float host_norm;
        q.memcpy(&host_norm, norm, sizeof(float)).wait();

        if (host_norm < accuracy_sq)
            break;

        std::swap(X, Xnew);
    }

    std::vector<float> result(n);
    q.memcpy(result.data(), X, sizeof(float) * n).wait();

    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(X, q);
    sycl::free(Xnew, q);
    sycl::free(norm, q);

    return result;
}