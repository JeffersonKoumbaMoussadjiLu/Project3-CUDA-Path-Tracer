#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            
			// exclusive scan
            if (n <= 0) {

                // Empty input, nothing to do
                timer().endCpuTimer();
                return;
            }

			odata[0] = 0; // first element is always 0 for exclusive scan

			// compute the rest of the elements
            for (int i = 1; i < n; ++i) {
				odata[i] = odata[i - 1] + idata[i - 1]; // exclusive scan
            }

            timer().endCpuTimer();			
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

			int count = 0; // number of non-zero elements

			// iterate through input array
            for (int i = 0; i < n; ++i) {

                if (idata[i] != 0) {           // keep only non-zero elements
					odata[count] = idata[i]; // write to output array
					count++; // increment count
                }
            }
            
            timer().endCpuTimer();
			
			return count; // return number of non-zero elements
            //return -1;
        }



        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

			// Handle edge case of empty input
            if (n <= 0) {
                timer().endCpuTimer();
                return 0;
            }

            // Map to boolean array (1 for non-zero, 0 for zero)
            int* bools = new int[n];
            for (int i = 0; i < n; ++i) {
                bools[i] = (idata[i] != 0) ? 1 : 0;
            }

            // Exclusive prefix sum (scan) on the boolean array
			int* indices = new int[n]; // to hold the scanned indices
			indices[0] = 0; // first element is always 0 for exclusive scan

			// Compute the rest of the elements
            for (int i = 1; i < n; ++i) {
				indices[i] = indices[i - 1] + bools[i - 1]; // exclusive scan
            }

            // Compute total count of non-zero elements
            int count = indices[n - 1] + bools[n - 1];

            // Scatter - Write all non-zero elements to odata at computed indices
            for (int i = 0; i < n; ++i) {

				// If the element is non-zero, write it to the output array at the scanned index
                if (bools[i] == 1) {
					odata[indices[i]] = idata[i]; // scatter
                }
            }

			delete[] bools; // free temporary boolean array
			delete[] indices; // free temporary indices array

            timer().endCpuTimer();
			return count; // return number of non-zero elements
            //return -1;
        }
    }
}
