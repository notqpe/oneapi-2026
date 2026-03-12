#include "acc_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {

    const int n = b.size();

    std::vector<float> x(n, 0.0f);
    std::vector<float> x_new(n, 0.0f);

    sycl::buffer<float, 1> buf_a(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float, 1> buf_b(b.data(), sycl::range<1>(b.size()));
    sycl::buffer<float, 1> buf_x(x.data(), sycl::range<1>(n));
    sycl::buffer<float, 1> buf_x_new(x_new.data(), sycl::range<1>(n));

    sycl::queue q(device);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        q.submit([&](sycl::handler& cgh) {
            auto a_acc = buf_a.get_access<sycl::access::mode::read>(cgh);
            auto b_acc = buf_b.get_access<sycl::access::mode::read>(cgh);
            auto x_acc = buf_x.get_access<sycl::access::mode::read>(cgh);
            auto x_new_acc = buf_x_new.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<class jacobi_step>(sycl::range<1>(n), [=](sycl::id<1> i_id) {
                int i = i_id[0];
                float sum = 0.0f;
                float a_ii = a_acc[i * n + i];
                for (int j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += a_acc[i * n + j] * x_acc[j];
                    }
                }
                x_new_acc[i] = (b_acc[i] - sum) / a_ii;
            });
        }).wait();

        bool converged = true;
        {
            auto x_new_host = buf_x_new.get_access<sycl::access::mode::read>();
            auto x_host = buf_x.get_access<sycl::access::mode::write>();

            for (int i = 0; i < n; ++i) {
                float diff = std::fabs(x_new_host[i] - x_host[i]);
                x_host[i] = x_new_host[i];
                if (diff >= accuracy) {
                    converged = false;
                }
            }
        }

        if (converged) {
            break;
        }
    }

    std::vector<float> result(n);
    {
        auto res_access = buf_x.get_access<sycl::access::mode::read>();
        for (int i = 0; i < n; ++i) {
            result[i] = res_access[i];
        }
    }

    return result;
}