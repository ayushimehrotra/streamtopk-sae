#pragma once
#include <cstring>
#include <limits>
#include <algorithm>

// Insertion-sort based TopKBuffer for K <= 64, min-heap for larger.
// Maintained in descending order (values[0] is the largest).

template<int K, bool USE_HEAP = (K > 64)>
struct TopKBuffer {
    float values[K];
    int   indices[K];

    inline void init() {
        for (int i = 0; i < K; ++i) {
            values[i]  = -std::numeric_limits<float>::infinity();
            indices[i] = -1;
        }
    }

    // Returns the current minimum (threshold for early rejection).
    inline float threshold() const {
        if constexpr (!USE_HEAP) {
            return values[K - 1];
        } else {
            return values[0]; // min-heap: root is minimum
        }
    }

    inline void try_insert(float v, int idx) {
        if constexpr (!USE_HEAP) {
            // Insertion sort: buffer kept in descending order.
            if (v <= values[K - 1]) return;
            // Find insertion point via linear scan from the back.
            int pos = K - 1;
            while (pos > 0 && values[pos - 1] < v) {
                values[pos]  = values[pos - 1];
                indices[pos] = indices[pos - 1];
                --pos;
            }
            values[pos]  = v;
            indices[pos] = idx;
        } else {
            // Min-heap: root is minimum.
            if (v <= values[0]) return;
            // Sift down
            values[0]  = v;
            indices[0] = idx;
            int i = 0;
            while (true) {
                int left  = 2 * i + 1;
                int right = 2 * i + 2;
                int smallest = i;
                if (left  < K && values[left]  < values[smallest]) smallest = left;
                if (right < K && values[right] < values[smallest]) smallest = right;
                if (smallest == i) break;
                std::swap(values[i],  values[smallest]);
                std::swap(indices[i], indices[smallest]);
                i = smallest;
            }
        }
    }

    // Merge another buffer into this one (for reduction).
    inline void merge(const TopKBuffer<K, USE_HEAP>& other) {
        for (int i = 0; i < K; ++i) {
            if (other.values[i] > threshold()) {
                try_insert(other.values[i], other.indices[i]);
            }
        }
    }
};
