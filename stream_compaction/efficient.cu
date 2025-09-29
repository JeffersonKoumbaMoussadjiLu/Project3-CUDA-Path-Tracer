#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		// __global__ kernels and device code here

		// Up-Sweep (reduce) kernel
        __global__ void kernUpSweep(int n, int step, int* data) {
			int index = (threadIdx.x + blockIdx.x * blockDim.x) * step + (step - 1); // get the index of the last element in the current segment

			// make sure we don't go out of bounds
            if (index < n) {
				data[index] += data[index - step / 2]; // add the value of the left child to the parent
            }
        }

		// Down-Sweep kernel
        __global__ void kernDownSweep(int n, int step, int* data) {

			int index = (threadIdx.x + blockIdx.x * blockDim.x) * step + (step - 1); // get the index of the last element in the current segment

			// make sure we don't go out of bounds
            if (index < n) {
				int left = index - step / 2; // get the index of the left child
				int t = data[left]; // store the left child's value
				data[left] = data[index]; // set the left child's value to the parent's value
				data[index] += t; // set the parent's value to the sum of the left child's value and the parent's value
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			// TODO
            // Edge case: if n <= 0, return
            if (n <= 0) {
                return;
            }

            // Allocate space rounded up to next power of two
			int pow2 = 1 << ilog2ceil(n); // next power of 2
			int* dev_data = nullptr; // device array
			cudaMalloc(&dev_data, pow2 * sizeof(int)); // allocate device memory
			checkCUDAError("cudaMalloc dev_data for efficient scan"); // check for errors

            // Copy input data to device and pad the rest with 0s
			cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice); // copy input data to device

			// Pad the rest with 0s
            if (pow2 > n) {
				cudaMemset(dev_data + n, 0, (pow2 - n) * sizeof(int)); // set the rest to 0
            }
			checkCUDAError("cudaMemcpy H2D and padding for efficient scan"); // check for errors

            timer().startGpuTimer();
            // TODO

			const int blockSize = 128; //128 // number of threads per block

            // Up-sweep (reduce) phase: build sum in place
            for (int step = 2; step <= pow2; step *= 2) {
                int threads = pow2 / step;        // number of add operations at this level
				int fullBlocks = (threads + blockSize - 1) / blockSize; // number of blocks needed
				kernUpSweep << <fullBlocks, blockSize >> > (pow2, step, dev_data); // launch kernel
				checkCUDAError("kernUpSweep kernel"); // check for errors
            }

            // Set the last element (total sum) to 0 for exclusive scan
			cudaMemset(dev_data + (pow2 - 1), 0, sizeof(int)); // set the last element to 0
			checkCUDAError("cudaMemset root (exclusive scan)"); // check for errors

            // Down-sweep phase: distribute the prefix sums
            for (int step = pow2; step >= 2; step /= 2) {

				// launch kernel
				int threads = pow2 / step; // number of add operations at this level
				int fullBlocks = (threads + blockSize - 1) / blockSize; // number of blocks needed
				kernDownSweep << <fullBlocks, blockSize >> > (pow2, step, dev_data); // launch kernel
				checkCUDAError("kernDownSweep kernel"); // check for errors
            }

            timer().endGpuTimer();

            // Read back the scanned result (first n elements)
			cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost); // copy result back to host
			checkCUDAError("cudaMemcpy D2H for efficient scan result"); // check for errors

			cudaFree(dev_data); // free device memory
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO

			// Edge case: if n <= 0, return 0
            if (n <= 0) {
                timer().endGpuTimer();
                return 0;
            }

            // Device memory allocation
			int* dev_idata = nullptr; // device input array
			int* dev_bools = nullptr; // device boolean array
			int* dev_indices = nullptr; // device indices array
			int* dev_odata = nullptr; // device output array

			cudaMalloc(&dev_idata, n * sizeof(int)); // allocate device memory for input
			cudaMalloc(&dev_bools, n * sizeof(int)); // allocate device memory for boolean array
			cudaMalloc(&dev_odata, n * sizeof(int)); // allocate device memory for output
            checkCUDAError("cudaMalloc failed for compaction arrays");

            // We will allocate dev_indices with padding for scan
			int pow2 = 1 << ilog2ceil(n); // next power of 2
			cudaMalloc(&dev_indices, pow2 * sizeof(int)); // allocate device memory for indices with padding
			checkCUDAError("cudaMalloc failed for indices array"); // check for errors

            // Copy input to device
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice); // copy input data to device
			checkCUDAError("cudaMemcpy H2D for compaction input"); // check for errors

            // Map input to booleans (1 = keep, 0 = discard)
			int blockSize = 128; // number of threads per block
			int fullBlocks = (n + blockSize - 1) / blockSize; // number of blocks needed
			StreamCompaction::Common::kernMapToBoolean <<<fullBlocks, blockSize >>> (n, dev_bools, dev_idata); // launch kernel
			checkCUDAError("kernMapToBoolean kernel"); // check for errors

            // Scan on dev_bools -> dev_indices (inclusive of padding)
            // Copy bools to indices array (and pad remaining space with 0)
			cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice); // copy bools to indices

            if (pow2 > n) {
				cudaMemset(dev_indices + n, 0, (pow2 - n) * sizeof(int)); // pad remaining space with 0
            }
            checkCUDAError("copy + pad boolean array for scan");

            // Up-sweep phase on indices array
            for (int step = 2; step <= pow2; step *= 2) {

				int threads = pow2 / step; // number of add operations at this level
				fullBlocks = (threads + blockSize - 1) / blockSize; // number of blocks needed
				kernUpSweep <<<fullBlocks, blockSize >>> (pow2, step, dev_indices); // launch kernel
                checkCUDAError("kernUpSweep (compaction) kernel");
            }

            // Set last element to 0 (prepare for exclusive scan)
			cudaMemset(dev_indices + (pow2 - 1), 0, sizeof(int)); // set last element to 0
            checkCUDAError("cudaMemset root for compaction scan");

            // Down-sweep phase
            for (int step = pow2; step >= 2; step /= 2) {

				int threads = pow2 / step; // number of add operations at this level
				fullBlocks = (threads + blockSize - 1) / blockSize; // number of blocks needed
				kernDownSweep <<<fullBlocks, blockSize >>> (pow2, step, dev_indices); // launch kernel
                checkCUDAError("kernDownSweep (compaction) kernel");
            }

            // Scatter non-zero elements to output array using computed indices
			fullBlocks = (n + blockSize - 1) / blockSize; // number of blocks needed
            StreamCompaction::Common::kernScatter <<<fullBlocks, blockSize >>> (
				n, dev_odata, dev_idata, dev_bools, dev_indices); // launch kernel
            checkCUDAError("kernScatter kernel");


            timer().endGpuTimer();
            
            // Copy compacted data back to host
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy D2H for compaction output");

            // Compute and return count of non-zero elements
            int count = 0;
            int lastBool, lastIndex;
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy D2H for compaction count");
            if (n > 0) {
                count = lastIndex + lastBool;
            }

            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);
            return count;
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
	//Extra Credit


	// Radix Sort
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer() {
            static PerformanceTimer timer;
            return timer;
        }

        // Kernel to flip the sign bit of each integer (convert to "unsigned" key by XOR with 0x80000000)
        __global__ void kernFlipBits(int n, int* data) {
			int i = threadIdx.x + blockIdx.x * blockDim.x; // global index
            if (i < n) {
                data[i] ^= 0x80000000;  // Flip MSB to handle signed values
            }
        }

        // Kernel to map each element's specific bit to a boolean (0 or 1).
        __global__ void kernMapToBit(int n, int bit, int* bools, const int* idata) {
			int i = threadIdx.x + blockIdx.x * blockDim.x; // global index
            if (i < n) {
                int mask = 1 << bit;
                // If the bit is set, mark 1; otherwise 0
                bools[i] = (idata[i] & mask) ? 1 : 0;
            }
        }

        // Kernel to scatter elements into output array based on computed indices for radix sort.
        __global__ void kernScatterRadix(int n, const int* idata, int* odata,
            const int* bools, const int* indices, int totalFalse) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n) {
                if (bools[i] == 0) {
                    // Element has 0 in current bit, write to index = i - number of 1s before i
                    int pos = i - indices[i];
                    odata[pos] = idata[i];
                }
                else {
                    // Element has 1 in current bit, write to index = totalFalse + (number of 1s before i)
                    int pos = totalFalse + indices[i];
                    odata[pos] = idata[i];
                }
            }
        }

        /**
         * Performs radix sort on idata, storing the sorted result into odata.
         * Handles signed integers by using a bit flip (XOR 0x80000000) so that negative numbers are sorted correctly.
         */
        void sort(int n, int* odata, const int* idata) {
            if (n <= 0) {
                return;
            }
            // Device memory allocation
			int* dev_in = nullptr; // input array
			int* dev_out = nullptr; // output array
			int* dev_bools = nullptr; // boolean array for current bit
			int* dev_indices = nullptr; // scanned indices array

			cudaMalloc(&dev_in, n * sizeof(int));  // allocate device memory for input
			cudaMalloc(&dev_out, n * sizeof(int)); // allocate device memory for output
			cudaMalloc(&dev_bools, n * sizeof(int)); // allocate device memory for boolean array

			int pow2Length = 1 << ilog2ceil(n); // next power of 2 for scan
			cudaMalloc(&dev_indices, pow2Length * sizeof(int)); // allocate device memory for indices with padding
            checkCUDAError("cudaMalloc failed for radix sort arrays");

            // Copy input data to device
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy H2D for radix sort input");

            // Transform all numbers by flipping MSB (to handle signed ints)
            const int blockSize = 128;
			int fullBlocks = (n + blockSize - 1) / blockSize; // number of blocks needed
			kernFlipBits <<<fullBlocks, blockSize >>> (n, dev_in); // flip sign bits
            checkCUDAError("kernFlipBits kernel");

            timer().startGpuTimer();
            // Loop over each bit (0 to 31)
            for (int bit = 0; bit < 32; ++bit) {

                // Map each element to a boolean (is bit == 1?)
                fullBlocks = (n + blockSize - 1) / blockSize;
				kernMapToBit <<<fullBlocks, blockSize >> > (n, bit, dev_bools, dev_in); // map to booleans
                checkCUDAError("kernMapToBit kernel");

                // Copy boolean array to dev_indices (pad to next power of two length)
				cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice); // copy bools to indices
                if (pow2Length > n) {
                    cudaMemset(dev_indices + n, 0, (pow2Length - n) * sizeof(int));
                }
                checkCUDAError("cudaMemcpy + cudaMemset for boolean array");

                // Inclusive scan (prefix sum) on dev_indices (booleans array) using Blelloch scan (upsweep and downsweep)
                // Upsweep phase
                for (int stride = 2; stride <= pow2Length; stride *= 2) {
                    int threads = pow2Length / stride;
                    int blocks = (threads + blockSize - 1) / blockSize;
                    StreamCompaction::Efficient::kernUpSweep <<<blocks, blockSize >>> (pow2Length, stride, dev_indices);
                    checkCUDAError("kernUpSweep (radix) kernel");
                }
                // Set last element to 0 (prepare for exclusive scan)
                cudaMemset(dev_indices + (pow2Length - 1), 0, sizeof(int));
                checkCUDAError("cudaMemset root for radix scan");
                // Downsweep phase
                for (int stride = pow2Length; stride >= 2; stride /= 2) {
                    int threads = pow2Length / stride;
                    int blocks = (threads + blockSize - 1) / blockSize;
                    StreamCompaction::Efficient::kernDownSweep << <blocks, blockSize >> > (pow2Length, stride, dev_indices);
                    checkCUDAError("kernDownSweep (radix) kernel");
                }

                // Calculate total number of 0s (false) for this bit to determine scatter offsets
                int lastBool, lastIndex;
                cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy D2H for radix scan totals");
                int totalOnes = lastIndex + lastBool;
                int totalZeros = n - totalOnes;

                // Scatter: stable partition into dev_out based on bit
                fullBlocks = (n + blockSize - 1) / blockSize;
                kernScatterRadix <<<fullBlocks, blockSize >>> (n, dev_in, dev_out, dev_bools, dev_indices, totalZeros);
                checkCUDAError("kernScatterRadix kernel");

                // Swap dev_in and dev_out for next iteration
                int* temp = dev_in;
                dev_in = dev_out;
                dev_out = temp;
            }
            timer().endGpuTimer();

            // After 32 iterations, dev_in now holds the fully sorted values (with MSB flipped).
            // Flip bits back to restore original sign bits
            fullBlocks = (n + blockSize - 1) / blockSize;
            kernFlipBits <<<fullBlocks, blockSize >>> (n, dev_in);
            checkCUDAError("kernFlipBits kernel (restore)");

            // Copy sorted data back to host
            cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy D2H for radix sort output");

            // Free allocated memory
            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
        }
    }

	///////////////////////////////////////////////////////////////////////////////
    namespace Shared {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer() {
            static PerformanceTimer timer;
            return timer;
        }

        // Macro for conflict-free indexing offset (to avoid shared memory bank conflicts)
        #define LOG_NUM_BANKS 5  // 32 banks
        #define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)

        // Kernel to perform scan (exclusive prefix sum) on each block using shared memory (Blelloch algorithm).
        // Also computes the sum of each block's elements and stores it in blockSums[blockIdx.x].
        __global__ void kernScanBlock(int n, const int* idata, int* odata, int* blockSums) {
            extern __shared__ int temp[];  // shared memory array for scanning (with padding for bank conflicts)
            int index = threadIdx.x;
            int globalIndex = blockIdx.x * blockDim.x + index;
            int blockSize = blockDim.x;
            // Load data into shared memory (with conflict-free padding)
            if (globalIndex < n) {
                temp[index + CONFLICT_FREE_OFFSET(index)] = idata[globalIndex];
            }
            else {
                temp[index + CONFLICT_FREE_OFFSET(index)] = 0;
            }
            __syncthreads();

            int offset = 1;
            int paddedLength = blockSize;  // we treat each block segment length as blockSize (pad with zeros if needed)
            // Up-sweep (reduce) phase
            for (offset = 1; offset < paddedLength; offset *= 2) {
                int idx = (index + 1) * offset * 2 - 1;
                if (idx < paddedLength) {
                    int ai = idx - offset;
                    int bi = idx;
                    // Add value from ai to bi with conflict-free indexing
                    int offAi = ai + CONFLICT_FREE_OFFSET(ai);
                    int offBi = bi + CONFLICT_FREE_OFFSET(bi);
                    temp[offBi] += temp[offAi];
                }
                __syncthreads();
            }

            // Save total sum of this block and clear the last element for down-sweep
            if (index == 0) {
                int lastIdx = paddedLength - 1;
                int offLast = lastIdx + CONFLICT_FREE_OFFSET(lastIdx);
                blockSums[blockIdx.x] = temp[offLast];  // total sum of block
                temp[offLast] = 0;                      // set root to 0 for exclusive scan
            }
            __syncthreads();

            // Down-sweep phase
            for (offset = blockSize; offset >= 1; offset /= 2) {
                int idx = (index + 1) * offset * 2 - 1;
                if (idx < blockSize) {
                    int ai = idx - offset;
                    int bi = idx;
                    int offAi = ai + CONFLICT_FREE_OFFSET(ai);
                    int offBi = bi + CONFLICT_FREE_OFFSET(bi);
                    int t = temp[offAi];
                    temp[offAi] = temp[offBi];
                    temp[offBi] += t;
                }
                __syncthreads();
            }

            // Write the exclusive scan results for this block to global memory
            if (globalIndex < n) {
                odata[globalIndex] = temp[index + CONFLICT_FREE_OFFSET(index)];
            }
        }

        // Kernel to add block offsets to each element of the scanned blocks.
        __global__ void kernAddBlockOffsets(int n, const int* blockOffsets, int* odata) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n) {
                // Determine which block this index belongs to
                int blockId = i / blockDim.x;
                if (blockId > 0) {
                    // Add the sum of all preceding blocks to this element
                    odata[i] += blockOffsets[blockId];
                }
            }
        }

        /**
         * Performs exclusive prefix-sum (scan) on idata, storing the result into odata.
         * Uses a work-efficient approach with shared memory for per-block scans and a second pass to add block sums.
         */
        void scan(int n, int* odata, const int* idata) {
            if (n <= 0) {
                return;
            }
            // Allocate device memory for input and output
            int* dev_in = nullptr;
            int* dev_out = nullptr;
            int* dev_blockSums = nullptr;
            int* dev_blockScan = nullptr;
            cudaMalloc(&dev_in, n * sizeof(int));
            cudaMalloc(&dev_out, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_in/dev_out for shared scan");
            // Determine number of blocks needed
            const int blockSize = 128;
            int numBlocks = (n + blockSize - 1) / blockSize;
            cudaMalloc(&dev_blockSums, numBlocks * sizeof(int));
            // Allocate array for scanned block sums (pad to next power of two for scanning)
            int pow2Blocks = 1 << ilog2ceil(numBlocks);
            cudaMalloc(&dev_blockScan, pow2Blocks * sizeof(int));
            checkCUDAError("cudaMalloc blockSums/blockScan for shared scan");

            // Copy input to device
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy H2D for shared scan input");

            timer().startGpuTimer();
            //  Scan each block's data into dev_out, and write block sums
            //   Use dynamic shared memory: each block gets (blockSize + blockSize/32) * sizeof(int) bytes
            size_t sharedMemSize = (blockSize + (blockSize >> 5)) * sizeof(int);
            kernScanBlock <<<numBlocks, blockSize, sharedMemSize >>> (n, dev_in, dev_out, dev_blockSums);
            checkCUDAError("kernScanBlock kernel");

            // Scan the array of block sums to get the offset for each block
            // Copy block sums to padded array for scanning
            cudaMemcpy(dev_blockScan, dev_blockSums, numBlocks * sizeof(int), cudaMemcpyDeviceToDevice);
            if (pow2Blocks > numBlocks) {
                cudaMemset(dev_blockScan + numBlocks, 0, (pow2Blocks - numBlocks) * sizeof(int));
            }
            checkCUDAError("cudaMemcpy + cudaMemset for block sums");
            // Upsweep on block sums
            for (int stride = 2; stride <= pow2Blocks; stride *= 2) {
                int threads = pow2Blocks / stride;
                int blocks = (threads + blockSize - 1) / blockSize;
                StreamCompaction::Efficient::kernUpSweep <<<blocks, blockSize >>> (pow2Blocks, stride, dev_blockScan);
                checkCUDAError("kernUpSweep (blockSums) kernel");
            }
            // Set last element to 0 for exclusive scan
            cudaMemset(dev_blockScan + (pow2Blocks - 1), 0, sizeof(int));
            checkCUDAError("cudaMemset root for block sums scan");
            // Downsweep on block sums
            for (int stride = pow2Blocks; stride >= 2; stride /= 2) {
                int threads = pow2Blocks / stride;
                int blocks = (threads + blockSize - 1) / blockSize;
                StreamCompaction::Efficient::kernDownSweep <<<blocks, blockSize >>> (pow2Blocks, stride, dev_blockScan);
                checkCUDAError("kernDownSweep (blockSums) kernel");
            }

            // Add block offsets to every element in dev_out to get the final scanned output
            int fullBlocks = (n + blockSize - 1) / blockSize;
            kernAddBlockOffsets <<<fullBlocks, blockSize >>> (n, dev_blockScan, dev_out);
            checkCUDAError("kernAddBlockOffsets kernel");
            timer().endGpuTimer();

            // Copy the result back to host
            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy D2H for shared scan output");

            // Free device memory
            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_blockSums);
            cudaFree(dev_blockScan);
        }
    }



}
