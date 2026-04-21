#include "acc_jacobi_oneapi.h"

#include <cmath>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const int n = static_cast<int>(b.size());

    std::vector<float> x_curr(n, 0.0f);
    std::vector<float> x_next(n, 0.0f);

    sycl::queue q(device);

    sycl::buffer<float> a_buf(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float> b_buf(b.data(), sycl::range<1>(n));
    sycl::buffer<float> curr_buf(x_curr.data(), sycl::range<1>(n));
    sycl::buffer<float> next_buf(x_next.data(), sycl::range<1>(n));

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        q.submit([&](sycl::handler& h) {
            auto a_acc   = a_buf.get_access<sycl::access::mode::read>(h);
            auto b_acc   = b_buf.get_access<sycl::access::mode::read>(h);
            auto x_acc   = curr_buf.get_access<sycl::access::mode::read>(h);
            auto xn_acc  = next_buf.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int i = idx[0];
                float sum = 0.0f;
                for (int j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += a_acc[i * n + j] * x_acc[j];
                    }
                }
                xn_acc[i] = (b_acc[i] - sum) / a_acc[i * n + i];
            });
        }).wait();

        bool converged = true;
        {
            auto curr_acc = curr_buf.get_host_access();
            auto next_acc = next_buf.get_host_access();
            for (int i = 0; i < n; ++i) {
                if (std::fabs(next_acc[i] - curr_acc[i]) >= accuracy) {
                    converged = false;
                }
                curr_acc[i] = next_acc[i];
            }
        }

        if (converged) break;
    }

    auto result = curr_buf.get_host_access();
    return std::vector<float>(result.get_pointer(), result.get_pointer() + n);
}
