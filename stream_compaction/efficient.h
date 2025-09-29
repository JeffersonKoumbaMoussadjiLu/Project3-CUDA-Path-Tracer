#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);
        int compact(int n, int *odata, const int *idata);
    }
	// Extra credit

    namespace Radix {
		StreamCompaction::Common::PerformanceTimer& timer(); //noexcept
		void sort(int n, int* odata, const int* idata); // Stable sort from least significant bit to most significant bit
    }
    namespace Shared {
		StreamCompaction::Common::PerformanceTimer& timer(); //noexcept
		void scan(int n, int* odata, const int* idata); // Shared memory scan kernel
    }
}
