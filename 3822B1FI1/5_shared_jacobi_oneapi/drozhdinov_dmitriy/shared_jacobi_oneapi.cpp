#include "shared_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    
	size_t n = b.size();
    std::vector<float> result(n, 0.0f);

    sycl::queue queue(device);

    float* a_shared = sycl::malloc_shared<float>(n * n, queue);
    float* b_shared = sycl::malloc_shared<float>(n, queue);
    float* x_shared = sycl::malloc_shared<float>(n, queue);
    float* x_new_shared = sycl::malloc_shared<float>(n, queue);

    std::copy(a.begin(), a.end(), a_shared);
    std::copy(b.begin(), b.end(), b_shared);
    std::fill(x_shared, x_shared + n, 0.0f);
    std::fill(x_new_shared, x_new_shared + n, 0.0f);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            float sigma = 0.0f;
            for (size_t j = 0; j < n; ++j) {
                if (j != i) {
                    sigma += a_shared[i * n + j] * x_shared[j];
                }
            }
            x_new_shared[i] = (b_shared[i] - sigma) / a_shared[i * n + i];
            }).wait();

            float max_diff = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                float diff = std::fabs(x_new_shared[i] - x_shared[i]);
                max_diff = std::max(max_diff, diff);
            }

            if (max_diff < accuracy) {
                break;
            }

            std::copy(x_new_shared, x_new_shared + n, x_shared);
    }

    std::copy(x_new_shared, x_new_shared + n, result.begin());

    sycl::free(a_shared, queue);
    sycl::free(b_shared, queue);
    sycl::free(x_shared, queue);
    sycl::free(x_new_shared, queue);

    return result;
}
