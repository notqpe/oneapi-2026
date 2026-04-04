#include "acc_jacobi_oneapi.h"
#include <algorithm>

std::vector<float> JacobiAccONEAPI(
    const std::vector<float>& matrix_a, const std::vector<float>& vector_b,
    float eps, sycl::device dev) {
    
    const size_t dim = vector_b.size();
    std::vector<float> solution(dim, 0.0f);
    
    try {
        sycl::queue q(dev);
        
        // Создаем буферы. Для векторов x и x_next используем инициализацию нулями.
        sycl::buffer<float, 1> buf_a(matrix_a.data(), sycl::range<1>(matrix_a.size()));
        sycl::buffer<float, 1> buf_b(vector_b.data(), sycl::range<1>(dim));
        sycl::buffer<float, 1> buf_curr(dim);
        sycl::buffer<float, 1> buf_next(dim);

        q.submit([&](sycl::handler& h) {
            auto out = buf_curr.get_access<sycl::access::mode::write>(h);
            h.fill(out, 0.0f);
        });

        float current_error = 0.0f;
        int step_count = 0;

        do {
            q.submit([&](sycl::handler& h) {
                auto a = buf_a.get_access<sycl::access::mode::read>(h);
                auto b = buf_b.get_access<sycl::access::mode::read>(h);
                auto x = buf_curr.get_access<sycl::access::mode::read>(h);
                auto x_new = buf_next.get_access<sycl::access::mode::write>(h);

                h.parallel_for(sycl::range<1>(dim), [=](sycl::id<1> id) {
                    int i = id[0];
                    float sigma = 0.0f;
                    
                    for (int j = 0; j < dim; ++j) {
                        if (i != j) {
                            sigma += a[i * dim + j] * x[j];
                        }
                    }
                    x_new[i] = (b[i] - sigma) / a[i * dim + i];
                });
            });

            sycl::buffer<float, 1> error_buf(&current_error, 1);
            q.submit([&](sycl::handler& h) {
                auto x_old = buf_curr.get_access<sycl::access::mode::read>(h);
                auto x_new = buf_next.get_access<sycl::access::mode::read>(h);
                auto red = sycl::reduction(error_buf, h, sycl::maximum<float>());

                h.parallel_for(sycl::range<1>(dim), red, [=](sycl::id<1> id, auto& max_val) {
                    float diff = sycl::fabs(x_new[id] - x_old[id]);
                    max_val.combine(diff);
                });
            }).wait();

            q.submit([&](sycl::handler& h) {
                auto src = buf_next.get_access<sycl::access::mode::read>(h);
                auto dst = buf_curr.get_access<sycl::access::mode::write>(h);
                h.copy(src, dst);
            });

            step_count++;
            
            current_error = error_buf.get_host_access()[0];

        } while (step_count < ITERATIONS && current_error >= eps);

        auto final_access = buf_curr.get_host_access();
        std::copy(final_access.get_pointer(), final_access.get_pointer() + dim, solution.begin());

    } catch (sycl::exception const& ex) {
        return {}; // Возвращаем пустой вектор в случае ошибки
    }

    return solution;
}