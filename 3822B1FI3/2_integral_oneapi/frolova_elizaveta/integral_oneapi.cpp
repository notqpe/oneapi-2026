#include "integral_oneapi.h"
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {

    float h = (end - start) / count;
    float result = 0.0f;
    
    try {
        sycl::queue queue(device);
        
        sycl::buffer<float> result_buf(&result, 1);
        
        queue.submit([&](sycl::handler& cgh) {
            auto reduction = sycl::reduction(result_buf, cgh, sycl::plus<float>());
            
            cgh.parallel_for(sycl::range<2>(count, count), reduction,
                [=](sycl::id<2> idx, auto& sum) {
                    int i = idx[0];
                    int j = idx[1];
                    
                    float x_center = start + (i + 0.5f) * h;
                    float y_center = start + (j + 0.5f) * h;
                    
                    float f_value = sycl::sin(x_center) * sycl::cos(y_center);
                    
                    float area = h * h;
                    
                    sum += f_value * area;
                });
        });
        
        queue.wait();
        
    } catch (sycl::exception& e) {
        return 0.0f;
    }
    
    return result;
}