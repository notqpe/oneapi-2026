#include "shared_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {

    const int matrix_dimension = b.size();
    sycl::queue compute_queue(device);

    float* shared_matrix_a = sycl::malloc_shared<float>(a.size(), compute_queue);
    float* shared_vector_b = sycl::malloc_shared<float>(b.size(), compute_queue);
    float* shared_previous_x = sycl::malloc_shared<float>(matrix_dimension, compute_queue);
    float* shared_current_x = sycl::malloc_shared<float>(matrix_dimension, compute_queue);

    compute_queue.memcpy(shared_matrix_a, a.data(), sizeof(float) * a.size()).wait();
    compute_queue.memcpy(shared_vector_b, b.data(), sizeof(float) * b.size()).wait();
    compute_queue.memset(shared_previous_x, 0, sizeof(float) * matrix_dimension).wait();
    compute_queue.memset(shared_current_x, 0, sizeof(float) * matrix_dimension).wait();

    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        compute_queue.parallel_for(sycl::range<1>(matrix_dimension), [=](sycl::id<1> element_id) {
            int row_index = element_id[0];
            float sum = shared_vector_b[row_index];
            for (int col_index = 0; col_index < matrix_dimension; ++col_index) {
                if (col_index != row_index) {
                    sum -= shared_matrix_a[row_index * matrix_dimension + col_index] * shared_previous_x[col_index];
                }
            }
            float diagonal_element = shared_matrix_a[row_index * matrix_dimension + row_index];
            shared_current_x[row_index] = sum / diagonal_element;
        }).wait();

        bool has_converged = true;
        for (int i = 0; i < matrix_dimension; ++i) {
            float difference = std::fabs(shared_current_x[i] - shared_previous_x[i]);
            if (difference >= accuracy) {
                has_converged = false;
            }
            shared_previous_x[i] = shared_current_x[i];
        }
        if (has_converged) {
            break;
        }
    }
    std::vector<float> solution_vector(matrix_dimension);
    for (int i = 0; i < matrix_dimension; ++i) {
        solution_vector[i] = shared_previous_x[i];
    }

    sycl::free(shared_matrix_a, compute_queue);
    sycl::free(shared_vector_b, compute_queue);
    sycl::free(shared_previous_x, compute_queue);
    sycl::free(shared_current_x, compute_queue);

    return solution_vector;
}