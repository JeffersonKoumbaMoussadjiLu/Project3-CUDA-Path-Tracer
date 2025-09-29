#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			// TODO use `thrust::exclusive_scan`
            if (n <= 0) {
                return;
            }

            // Allocate device_vectors and copy input data
			thrust::device_vector<int> dv_in(n); // device_vector manages memory and automatically cleans up
			thrust::device_vector<int> dv_out(n); // device_vector manages memory and automatically cleans up
            cudaMemcpy(thrust::raw_pointer_cast(dv_in.data()), idata, 
				n * sizeof(int), cudaMemcpyHostToDevice); // copy input array to device
            checkCUDAError("cudaMemcpy H2D for thrust scan input");


            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

            // Use Thrust library's exclusive_scan on the device vector
            thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

            timer().endGpuTimer();

            // Copy result back to host
            cudaMemcpy(odata, thrust::raw_pointer_cast(dv_out.data()), 
				n * sizeof(int), cudaMemcpyDeviceToHost); // copy output array to host
            checkCUDAError("cudaMemcpy D2H for thrust scan output"); 
        }
    }
}
