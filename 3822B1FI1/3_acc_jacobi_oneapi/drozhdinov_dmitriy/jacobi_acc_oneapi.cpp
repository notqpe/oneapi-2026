#include "jacobi_acc_oneapi.h"

std::vector<float> JacobiAccONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    float accuracy, sycl::device device) {
    int n = b.size();
    std::vector<float> ans(n, 0);
    std::vector<float> prev_ans = ans;
    auto norm2 = [](const std::vector<float>& a, const std::vector<float>& b)->float {
        float res = 0;
        for (int i = 0; i < a.size(); i++) {
            float d = a[i] - b[i];
            res += d * d;
        }
        return res;
    };
    int iterations = 0;
    sycl::queue q(device);
    do {
        iterations++;
        swap(ans, prev_ans);
        {
            sycl::buffer<float> a_buf(a.data(), a.size());
            sycl::buffer<float> b_buf(b.data(), b.size());
            sycl::buffer<float> ans_buf(ans.data(), ans.size());
            sycl::buffer<float> prev_ans_buf(prev_ans.data(), prev_ans.size());
            sycl::event e = q.submit([&](sycl::handler& h) {
                auto in_a = a_buf.get_access<sycl::access::mode::read>(h);
                auto in_b = b_buf.get_access<sycl::access::mode::read>(h);
                auto in_prev = prev_ans_buf.get_access<sycl::access::mode::read>(h);
                auto out_ans = ans_buf.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                    int i = id.get(0);
                    float res = 0;
                    for (int j = 0; j < n; j++) {
                        if (i == j) {
                            res += in_b[j];
                        }
                        else {
                            res -= in_a[i * n + j] * in_prev[j];
                        }
                    }
                    res /= in_a[i * n + i];
                    out_ans[i] = res;
                    });
                });
            e.wait();
        }
    } while (iterations < ITERATIONS && norm2(ans, prev_ans) >= accuracy * accuracy);
    return ans;
}
