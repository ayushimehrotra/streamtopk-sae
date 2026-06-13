#pragma once
#include <limits>

// Per-thread top-k buffer for CUDA kernels.
// Always uses insertion-sort (values in descending order) so the block-wide
// reduction can use values[K-1] as the minimum without heap-order surprises.
// Heap variant intentionally disabled: approx kernel reduction assumes sorted layout.

template<int K, bool USE_HEAP = false>
struct TopKBuffer {
    float values[K];
    int   indices[K];

    __device__ __forceinline__ void init() {
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            values[i]  = -3.402823e+38f; // -FLT_MAX
            indices[i] = -1;
        }
    }

    __device__ __forceinline__ float threshold() const {
        if constexpr (!USE_HEAP) {
            return values[K - 1]; // minimum in descending sorted array
        } else {
            return values[0];     // min-heap root
        }
    }

    __device__ __forceinline__ void try_insert(float v, int idx) {
        if constexpr (!USE_HEAP) {
            if (v <= values[K - 1]) return;
            // Find insertion position from back (descending order)
            int pos = K - 1;
            #pragma unroll
            for (int p = K - 2; p >= 0; --p) {
                if (values[p] >= v) {
                    pos = p + 1;
                    break;
                }
                if (p == 0) pos = 0;
            }
            // Shift elements right
            #pragma unroll
            for (int p = K - 1; p > pos; --p) {
                values[p]  = values[p - 1];
                indices[p] = indices[p - 1];
            }
            values[pos]  = v;
            indices[pos] = idx;
        } else {
            // Min-heap: root is minimum
            if (v <= values[0]) return;
            values[0]  = v;
            indices[0] = idx;
            // Sift down
            int i = 0;
            #pragma unroll
            for (int iter = 0; iter < 8; ++iter) {
                int left  = 2 * i + 1;
                int right = 2 * i + 2;
                int smallest = i;
                if (left  < K && values[left]  < values[smallest]) smallest = left;
                if (right < K && values[right] < values[smallest]) smallest = right;
                if (smallest == i) break;
                float tv = values[i];  values[i]  = values[smallest];  values[smallest]  = tv;
                int   ti = indices[i]; indices[i] = indices[smallest]; indices[smallest] = ti;
                i = smallest;
            }
        }
    }

    __device__ __forceinline__ void merge(const TopKBuffer<K, USE_HEAP>& other) {
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            if (other.values[i] > threshold()) {
                try_insert(other.values[i], other.indices[i]);
            }
        }
    }
};
