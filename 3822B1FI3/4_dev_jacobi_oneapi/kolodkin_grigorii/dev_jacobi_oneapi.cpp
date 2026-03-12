#include "dev_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiDevONEAPI(const std::vector<float>& a,
                                         const std::vector<float>& b,
                                         float accuracy,
                                         sycl::device device) {
    const int n = b.size();


    std::vector<float> x_host(n);
    sycl::queue q(device);
    auto dev = q.get_device();
    auto ctx = q.get_context();

    float* a_dev = sycl::malloc_device<float>(n * n, q);
    float* b_dev = sycl::malloc_device<float>(n, q);
    float* x_dev = sycl::malloc_device<float>(n, q);
    float* x_new_dev = sycl::malloc_device<float>(n, q);

    q.memcpy(a_dev, a.data(), sizeof(float) * n * n).wait();
    q.memcpy(b_dev, b.data(), sizeof(float) * n).wait();
    q.memset(x_dev, 0, sizeof(float) * n).wait();       
    q.memset(x_new_dev, 0, sizeof(float) * n).wait();

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i_id) {
            int i = i_id[0];
            float sum = 0.0f;
            float a_ii = a_dev[i * n + i];
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    sum += a_dev[i * n + j] * x_dev[j];
                }
            }
            x_new_dev[i] = (b_dev[i] - sum) / a_ii;
        }).wait();

        q.memcpy(x_host.data(), x_new_dev, sizeof(float)*n).wait();

        bool converged = true;

        std::vector<float> x_old(n);
        q.memcpy(x_old.data(), x_dev, sizeof(float)*n).wait();

        for (int i = 0; i < n; ++i) {
            float diff = std::fabs(x_host[i] - x_old[i]);
            if (diff >= accuracy) {
                converged = false;
            }
        }

        q.memcpy(x_dev, x_host.data(), sizeof(float)*n).wait();

        if (converged) {
            break;
        }
    }

    sycl::free(a_dev, q);
    sycl::free(b_dev, q);
    sycl::free(x_dev, q);
    sycl::free(x_new_dev, q);

    return x_host;

}
