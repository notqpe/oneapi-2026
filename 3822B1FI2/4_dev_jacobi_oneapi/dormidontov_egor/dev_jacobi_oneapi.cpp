#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {
    
    const size_t matrix_size = b.size();
    
    if (matrix_size == 0 || a.size() != matrix_size * matrix_size) {
        return {};
    }
    
    if (accuracy <= 0.0f) {
        accuracy = 1e-6f;
    }
    
    const float squared_accuracy = accuracy * accuracy;

    sycl::queue compute_queue(device, sycl::property::queue::in_order{});

    std::vector<float> inverse_diagonal(matrix_size);
    for (size_t i = 0; i < matrix_size; ++i) {
        float diag = a[i * matrix_size + i];
        inverse_diagonal[i] = (std::fabs(diag) > 1e-12f) ? (1.0f / diag) : 0.0f;
    }

    float* device_matrix = sycl::malloc_device<float>(matrix_size * matrix_size, compute_queue);
    float* device_rhs = sycl::malloc_device<float>(matrix_size, compute_queue);
    float* device_inv_diag = sycl::malloc_device<float>(matrix_size, compute_queue);
    float* device_current_x = sycl::malloc_device<float>(matrix_size, compute_queue);
    float* device_next_x = sycl::malloc_device<float>(matrix_size, compute_queue);
    float* device_diff_buffer = sycl::malloc_device<float>(1, compute_queue);

    if (!device_matrix || !device_rhs || !device_inv_diag || 
        !device_current_x || !device_next_x || !device_diff_buffer) {
        sycl::free(device_matrix, compute_queue);
        sycl::free(device_rhs, compute_queue);
        sycl::free(device_inv_diag, compute_queue);
        sycl::free(device_current_x, compute_queue);
        sycl::free(device_next_x, compute_queue);
        sycl::free(device_diff_buffer, compute_queue);
        return {};
    }

    compute_queue.memcpy(device_matrix, a.data(), sizeof(float) * matrix_size * matrix_size);
    compute_queue.memcpy(device_rhs, b.data(), sizeof(float) * matrix_size);
    compute_queue.memcpy(device_inv_diag, inverse_diagonal.data(), sizeof(float) * matrix_size);
    compute_queue.fill(device_current_x, 0.0f, matrix_size);
    compute_queue.fill(device_next_x, 0.0f, matrix_size);

    const size_t work_group_size = 256;
    const size_t global_size = ((matrix_size + work_group_size - 1) / work_group_size) * work_group_size;
    
    bool solution_converged = false;
    int check_interval = 8;
    
    std::vector<float> host_current_x(matrix_size);
    std::vector<float> host_previous_x(matrix_size, 0.0f);

    for (int iteration = 0; iteration < ITERATIONS && !solution_converged; ++iteration) {
        compute_queue.parallel_for(
            sycl::nd_range<1>(global_size, work_group_size),
            [=](sycl::nd_item<1> work_item) {
                size_t row_index = work_item.get_global_id(0);
                if (row_index >= matrix_size) return;

                float sum_off_diagonal = 0.0f;
                const size_t row_start = row_index * matrix_size;
                
                for (size_t col_index = 0; col_index < matrix_size; ++col_index) {
                    if (col_index != row_index) {
                        sum_off_diagonal += device_matrix[row_start + col_index] * device_current_x[col_index];
                    }
                }
                device_next_x[row_index] = device_inv_diag[row_index] * (device_rhs[row_index] - sum_off_diagonal);
            }
        );
        
        if ((iteration + 1) % check_interval == 0) {
            compute_queue.memcpy(host_current_x.data(), device_next_x, sizeof(float) * matrix_size).wait();
            float norm_squared = 0.0f;
            for (size_t i = 0; i < matrix_size; ++i) {
                float difference = host_current_x[i] - host_previous_x[i];
                norm_squared += difference * difference;
            } 
            if (norm_squared < squared_accuracy) {
                solution_converged = true;
                break;
            }
            host_previous_x = host_current_x;
        }
        std::swap(device_current_x, device_next_x);
    }
    compute_queue.memcpy(host_current_x.data(), device_current_x, sizeof(float) * matrix_size).wait();
    sycl::free(device_matrix, compute_queue);
    sycl::free(device_rhs, compute_queue);
    sycl::free(device_inv_diag, compute_queue);
    sycl::free(device_current_x, compute_queue);
    sycl::free(device_next_x, compute_queue);
    sycl::free(device_diff_buffer, compute_queue);
    
    return host_current_x;
}