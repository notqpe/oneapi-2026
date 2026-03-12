#include "shared_jacobi_oneapi.h"
#include <cmath>
#include <vector>
#include <iostream>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    
    size_t n = b.size();
    std::vector<float> x(n, 0.0f);
    
    try {
        sycl::queue queue(device, sycl::property::queue::in_order{});
        
        float* a_shared = sycl::malloc_shared<float>(n * n, queue);
        float* b_shared = sycl::malloc_shared<float>(n, queue);
        float* inv_diag_shared = sycl::malloc_shared<float>(n, queue);
        float* x_old_shared = sycl::malloc_shared<float>(n, queue);
        float* x_new_shared = sycl::malloc_shared<float>(n, queue);
        float* diff_shared = sycl::malloc_shared<float>(n, queue);
        
        if (!a_shared || !b_shared || !inv_diag_shared || !x_old_shared || !x_new_shared || !diff_shared) {
            throw std::runtime_error("Failed to allocate shared memory");
        }
        
        for (size_t i = 0; i < n * n; ++i) {
            a_shared[i] = a[i];
        }
        
        for (size_t i = 0; i < n; ++i) {
            b_shared[i] = b[i];
            x_old_shared[i] = 0.0f;
        }
        
        for (size_t i = 0; i < n; ++i) {
            inv_diag_shared[i] = 1.0f / a_shared[i * n + i];
        }
        
        const size_t wg_size = 64;
        const size_t global_size = ((n + wg_size - 1) / wg_size) * wg_size;
        const int CHECK_INTERVAL = 8;
        
        int iteration = 0;
        float max_diff = 0.0f;
        bool converged = false;
        float accuracy_sq = accuracy * accuracy;
        
        std::vector<float> x_prev(n, 0.0f);
        
        do {
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::nd_range<1>(global_size, wg_size), [=](sycl::nd_item<1> item) {
                    size_t i = item.get_global_id(0);
                    if (i >= n) return;
                    
                    float sum = 0.0f;
                    
                    for (size_t j = 0; j < n; ++j) {
                        if (j != i) {
                            sum += a_shared[i * n + j] * x_old_shared[j];
                        }
                    }
                    
                    x_new_shared[i] = inv_diag_shared[i] * (b_shared[i] - sum);
                });
            });
            
            if ((iteration + 1) % CHECK_INTERVAL == 0) {
                queue.wait();
                
                float norm_sq = 0.0f;
                for (size_t i = 0; i < n; ++i) {
                    float diff = x_new_shared[i] - x_prev[i];
                    norm_sq += diff * diff;
                }
                
                if (norm_sq < accuracy_sq) {
                    converged = true;
                    break;
                }
                
                for (size_t i = 0; i < n; ++i) {
                    x_prev[i] = x_new_shared[i];
                }
            }
            
            std::swap(x_old_shared, x_new_shared);
            
            iteration++;
            
        } while (iteration < ITERATIONS && !converged);
        
        queue.wait();
        
        for (size_t i = 0; i < n; ++i) {
            x[i] = x_old_shared[i];
        }
        
        sycl::free(a_shared, queue);
        sycl::free(b_shared, queue);
        sycl::free(inv_diag_shared, queue);
        sycl::free(x_old_shared, queue);
        sycl::free(x_new_shared, queue);
        sycl::free(diff_shared, queue);
        
    } catch (sycl::exception& e) {
        return std::vector<float>();
    } catch (std::exception& e) {
        return std::vector<float>();
    }
    
    return x;
}