#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

		// Performs one step of the scan, with the given offset
        __global__ void kernScanStep(int n, int offset, const int* idata, int* odata) {

			int k = threadIdx.x + blockIdx.x * blockDim.x; // map from threadIdx to array index

			if (k >= n) return; // check array bounds

            if (k >= offset) {
                // Add the element from index [k - offset] (previous partial sum)
                odata[k] = idata[k] + idata[k - offset];
            }
            else {
                // Copy the element as is (no element to add from behind)
                odata[k] = idata[k];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            //TODO
            // Allocate device memory
            if (n <= 0) {
                return; // no elements to scan
            }

            int* dev_in = nullptr; // input array
            int* dev_out = nullptr; // output array
            cudaMalloc(&dev_in, n * sizeof(int)); // allocate device memory
            cudaMalloc(&dev_out, n * sizeof(int)); // allocate device memory
            checkCUDAError("cudaMalloc failed for scan buffers"); // check for errors

            // Exclusive scan initial state: dev_in[0] = 0, dev_in[1..n-1] = idata[0..n-2]
            cudaMemset(dev_in, 0, sizeof(int)); // set first element to 0
            if (n > 1) {
                cudaMemcpy(dev_in + 1, idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice); // copy rest of elements
            }
            checkCUDAError("cudaMemcpy H2D for input data"); // check for errors


            timer().startGpuTimer();
            // TODO

            // Iterative up-sweep (Hillis-Steele): offset = 1, 2, 4, ... < n
			const int blockSize = 128; //128 // number of threads per block

			// Ping-pong buffers: in each iteration, read from dev_in, write to dev_out, then swap
            for (int offset = 1; offset < n; offset *= 2) {

				// Launch kernel
                int fullBlocks = (n + blockSize - 1) / blockSize;
				kernScanStep << <fullBlocks, blockSize >> > (n, offset, dev_in, dev_out); // launch kernel
				checkCUDAError("kernScanStep kernel"); // check for errors

				cudaDeviceSynchronize(); // wait for kernel to finish, for timing purposes only
                // Swap the ping-pong buffers for next iteration
                int* temp = dev_in;
                dev_in = dev_out;
                dev_out = temp;
            }

            timer().endGpuTimer();

            // At this point, dev_in holds the scanned output (because of final swap)
			cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost); // copy result back to host
			checkCUDAError("cudaMemcpy D2H for scan result"); // check for errors

			cudaFree(dev_in); // free device memory
			cudaFree(dev_out); // free device memory
        }
    }
}
