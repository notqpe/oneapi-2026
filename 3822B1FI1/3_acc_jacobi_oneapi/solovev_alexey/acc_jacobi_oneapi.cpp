#include "acc_jacobi_oneapi.h"
#include <cmath>
#include <utility>

std::vector<float> JacobiAccONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float accuracy,
    sycl::device device) {

    const size_t n = b.size();
    const float accuracy_sq = accuracy * accuracy;

    sycl::queue q(device, sycl::property::queue::in_order{});

    std::vector<float> x0(n, 0.0f);

    sycl::buffer<float> A(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float> B(b.data(), sycl::range<1>(b.size()));
    sycl::buffer<float> Xcurr(x0.data(), sycl::range<1>(n));
    sycl::buffer<float> Xnext{sycl::range<1>(n)};
    sycl::buffer<float> Diff{sycl::range<1>(1)};

    for (int iter = 0; iter < ITERATIONS; ++iter) {

        q.submit([&](sycl::handler& h) {
            auto d = Diff.get_access<sycl::access::mode::discard_write>(h);
            h.fill(d, 0.0f);
        });

        q.submit([&](sycl::handler& h) {
            auto a_acc = A.get_access<sycl::access::mode::read>(h);
            auto b_acc = B.get_access<sycl::access::mode::read>(h);
            auto x_acc = Xcurr.get_access<sycl::access::mode::read>(h);
            auto xn_acc = Xnext.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                size_t i = id[0];
                float sigma = 0.0f;
                size_t row = i * n;

                for (size_t j = 0; j < n; ++j)
                    if (j != i)
                        sigma += a_acc[row + j] * x_acc[j];

                xn_acc[i] = (b_acc[i] - sigma) / a_acc[row + i];
            });
        });

        q.submit([&](sycl::handler& h) {
            auto x_acc = Xcurr.get_access<sycl::access::mode::read>(h);
            auto xn_acc = Xnext.get_access<sycl::access::mode::read>(h);

            auto red = sycl::reduction(Diff, h, sycl::plus<float>());

            h.parallel_for(
                sycl::range<1>(n),
                red,
                [=](sycl::id<1> id, auto& sum) {
                    float diff = xn_acc[id] - x_acc[id];
                    sum += diff * diff;
                });
        }).wait();

        {
            auto host_diff = Diff.get_host_access();
            if (host_diff[0] < accuracy_sq)
                break;
        }

        std::swap(Xcurr, Xnext);
    }

    std::vector<float> result(n);
    {
        auto acc = Xcurr.get_host_access();
        for (size_t i = 0; i < n; ++i)
            result[i] = acc[i];
    }

    return result;
}
