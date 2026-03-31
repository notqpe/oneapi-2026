#include "acc_jacobi_oneapi.h"

#include <cmath>

std::vector<float> JacobiAccONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float accuracy,
    sycl::device device) {

    const int dim = b.size();

    std::vector<float> curr(dim, 0.0f);
    std::vector<float> prev(dim, 0.0f);

    sycl::buffer<float> a_buff(a.data(), a.size());
    sycl::buffer<float> b_buff(b.data(), b.size());
    sycl::buffer<float> curr_buff(curr.data(), curr.size());
    sycl::buffer<float> prev_buff(prev.data(), prev.size());

    sycl::queue queue(device);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a_buff.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b_buff.get_access<sycl::access::mode::read>(cgh);
        auto prev_acc = prev_buff.get_access<sycl::access::mode::read>(cgh);
        auto curr_acc = curr_buff.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for(sycl::range<1>(dim), [=](sycl::id<1> id) {
            int i = id[0];
            float value = b_acc[i];

            for (int j = 0; j < dim; ++j) {
                if (i != j) {
                    value -= a_acc[i * dim + j] * prev_acc[j];
                }
            }

            curr_acc[i] = value / a_acc[i * dim + i];
            });
        }).wait();

        bool flag = true;
        {
            auto prev_acc = prev_buff.get_host_access();
            auto curr_acc = curr_buff.get_host_access();

            for (int i = 0; i < dim; ++i) {
                if (std::fabs(curr_acc[i] - prev_acc[i]) >= accuracy) {
                    ok = false;
                }
                prev_acc[i] = curr_acc[i];
            }
        }

        if (flag) {
            break;
        }
    }

    return prev;
}