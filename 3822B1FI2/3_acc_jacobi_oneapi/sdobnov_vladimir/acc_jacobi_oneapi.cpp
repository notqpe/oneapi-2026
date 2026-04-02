#include "acc_jacobi_oneapi.h"

#include <cmath>

std::vector<float> JacobiAccONEAPI(
	const std::vector<float>& a,
	const std::vector<float>& b,
	float accuracy,
	sycl::device device
) {
	int n = b.size();

	std::vector<float> x(n, 0.0f);
	std::vector<float> x_new(n, 0.0f);

	sycl::queue q(device);

	for (int iter = 0; iter < ITERATIONS; iter++) {
		float diff = 0.0f;

		{
			sycl::buffer<float> a_buf(a.data(), sycl::range<1>(n * n));
			sycl::buffer<float> b_buf(b.data(), sycl::range<1>(n));
			sycl::buffer<float> x_buf(x.data(), sycl::range<1>(n));
			sycl::buffer<float> x_new_buf(x_new.data(), sycl::range<1>(n));
			sycl::buffer<float> diff_buf(&diff, sycl::range<1>(1));

			q.submit([&](sycl::handler& h) {
				auto a_acc = a_buf.get_access<sycl::access::mode::read>(h);
				auto b_acc = b_buf.get_access<sycl::access::mode::read>(h);
				auto x_acc = x_buf.get_access<sycl::access::mode::read>(h);
				auto x_new_acc = x_new_buf.get_access<sycl::access::mode::write>(h);

				auto reduction = sycl::reduction(
					diff_buf, h, 0.0f, sycl::maximum<>()
				);

				h.parallel_for(
					sycl::range<1>(n),
					reduction,
					[=](sycl::id<1> i, auto& max_diff) {
						int row = i[0];

						float sum = 0.0f;

						for (int j = 0; j < n; j++) {
							if (j != row) {
								sum += a_acc[row * n + j] * x_acc[j];
							}
						}

						float new_val =
							(b_acc[row] - sum) / a_acc[row * n + row];

						x_new_acc[row] = new_val;

						float local_diff = sycl::fabs(new_val - x_acc[row]);
						max_diff.combine(local_diff);
					}
				);
				});
		}

		if (diff < accuracy) break;

		std::swap(x, x_new);
	}

	return x;
}