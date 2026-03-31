#include "block_gemm_oneapi.h"
#include <cmath>

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    
    constexpr size_t block_dim = 16;
    const size_t total_elements = size * size;
    std::vector<float> result_matrix(total_elements, 0.0f);
    sycl::queue compute_queue(device);

    float* device_matrix_a = sycl::malloc_device<float>(total_elements, compute_queue);
    float* device_matrix_b = sycl::malloc_device<float>(total_elements, compute_queue);
    float* device_matrix_c = sycl::malloc_device<float>(total_elements, compute_queue);

    compute_queue.memcpy(device_matrix_a, a.data(), total_elements * sizeof(float));
    compute_queue.memcpy(device_matrix_b, b.data(), total_elements * sizeof(float));
    compute_queue.memset(device_matrix_c, 0, total_elements * sizeof(float));

    const size_t block_count = size / block_dim;
    sycl::range<2> global_work_size(size, size);
    sycl::range<2> local_work_size(block_dim, block_dim);

    compute_queue.submit([&](sycl::handler& kernel_handler) {
        sycl::local_accessor<float, 2> block_a(local_work_size, kernel_handler);
        sycl::local_accessor<float, 2> block_b(local_work_size, kernel_handler);
        kernel_handler.parallel_for(
            sycl::nd_range<2>(global_work_size, local_work_size),
            [=](sycl::nd_item<2> work_item) {
                const size_t global_row = work_item.get_global_id(0);
                const size_t global_col = work_item.get_global_id(1);
                const size_t local_row = work_item.get_local_id(0);
                const size_t local_col = work_item.get_local_id(1);
                float partial_sum = 0.0f;
                for (size_t block_k = 0; block_k < block_count; ++block_k) {
                    block_a[local_row][local_col] = device_matrix_a[global_row * size + block_k * block_dim + local_col];
                    block_b[local_row][local_col] = device_matrix_b[(block_k * block_dim + local_row) * size + global_col];

                    work_item.barrier(sycl::access::fence_space::local_space);

                    for (size_t k_index = 0; k_index < block_dim; ++k_index) {
                        partial_sum += block_a[local_row][k_index] * block_b[k_index][local_col];
                    }

                    work_item.barrier(sycl::access::fence_space::local_space);
                }

                device_matrix_c[global_row * size + global_col] = partial_sum;
                
            }
        );
    }).whait();
    
    compute_queue.memcpy(result_matrix.data(), device_matrix_c, total_elements * sizeof(float)).wait();

    sycl::free(device_matrix_a, compute_queue);
    sycl::free(device_matrix_b, compute_queue);
    sycl::free(device_matrix_c, compute_queue);
    
    return result_matrix;
}