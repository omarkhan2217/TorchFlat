/*
 * masked_median_kernel.hip
 *
 * GPU kernel for O(n) masked median via quickselect.
 * Replaces torch.sort (O(n log n)) for the biweight detrending pipeline.
 *
 * Native HIP source — compiles directly with hipcc on ROCm.
 * For NVIDIA CUDA, use the .cu version instead.
 */

#include <torch/types.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

// Maximum elements per row that the kernel supports.
// W=361 (biweight) and W=25 (clipping) are the main use cases.
constexpr int MAX_LOCAL_SIZE = 512;

// ---------------------------------------------------------------------------
// Device helper: swap two values
// ---------------------------------------------------------------------------
template <typename scalar_t>
__device__ __forceinline__ void swap_vals(scalar_t& a, scalar_t& b) {
    scalar_t tmp = a;
    a = b;
    b = tmp;
}

// ---------------------------------------------------------------------------
// Device helper: median-of-3 pivot selection
// Orders buf[lo], buf[mid], buf[hi] and returns the middle value as pivot.
// Also partially sorts these three positions.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__device__ __forceinline__ scalar_t median_of_3(
    scalar_t* buf, int lo, int mid, int hi
) {
    if (buf[mid] < buf[lo]) swap_vals(buf[lo], buf[mid]);
    if (buf[hi]  < buf[lo]) swap_vals(buf[lo], buf[hi]);
    if (buf[hi]  < buf[mid]) swap_vals(buf[mid], buf[hi]);
    return buf[mid];
}

// ---------------------------------------------------------------------------
// Device helper: iterative quickselect (nth_element)
// Rearranges buf so that buf[k] contains the k-th smallest element.
// Elements before k are all <= buf[k], elements after are all >= buf[k].
// O(n) average case with median-of-3 pivot.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__device__ void nth_element(scalar_t* buf, int lo, int hi, int k) {
    while (lo < hi) {
        if (hi - lo < 3) {
            // Insertion sort for tiny partitions
            for (int i = lo + 1; i <= hi; i++) {
                scalar_t val = buf[i];
                int j = i;
                while (j > lo && buf[j - 1] > val) {
                    buf[j] = buf[j - 1];
                    j--;
                }
                buf[j] = val;
            }
            return;
        }

        int mid = lo + (hi - lo) / 2;
        scalar_t pivot = median_of_3(buf, lo, mid, hi);

        // Park pivot at hi-1
        swap_vals(buf[mid], buf[hi - 1]);

        // Hoare partition
        int i = lo;
        int j = hi - 1;
        while (true) {
            while (buf[++i] < pivot);
            while (buf[--j] > pivot);
            if (i >= j) break;
            swap_vals(buf[i], buf[j]);
        }
        // Restore pivot
        swap_vals(buf[i], buf[hi - 1]);

        // Narrow to the partition containing k
        if (k < i) {
            hi = i - 1;
        } else if (k > i) {
            lo = i + 1;
        } else {
            return; // k == i, found it
        }
    }
}

// ---------------------------------------------------------------------------
// Main kernel: one thread per row, finds masked median via quickselect
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void masked_median_kernel(
    const scalar_t* __restrict__ x,      // [..., W] flattened
    const bool*     __restrict__ mask,    // [..., W] flattened
    scalar_t*       __restrict__ out,     // [...] flattened
    const int W,                          // last dimension size
    const int64_t n_rows                  // total rows = product of all dims except last
) {
    const int64_t row_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row_idx >= n_rows) return;

    const scalar_t* row_x    = x    + row_idx * W;
    const bool*     row_mask = mask + row_idx * W;

    // Gather valid elements into thread-local buffer
    scalar_t buf[MAX_LOCAL_SIZE];
    int n_valid = 0;
    for (int i = 0; i < W; i++) {
        if (row_mask[i]) {
            buf[n_valid++] = row_x[i];
        }
    }

    // Edge cases
    if (n_valid == 0) {
        out[row_idx] = static_cast<scalar_t>(NAN);
        return;
    }
    if (n_valid == 1) {
        out[row_idx] = buf[0];
        return;
    }
    if (n_valid == 2) {
        out[row_idx] = (buf[0] + buf[1]) / static_cast<scalar_t>(2);
        return;
    }

    // Median indices (numpy convention: average two middle for even count)
    const int k_hi = n_valid / 2;
    const int k_lo = (n_valid - 1) / 2;

    // Find k_hi-th smallest element
    nth_element(buf, 0, n_valid - 1, k_hi);
    scalar_t val_hi = buf[k_hi];

    if (k_lo == k_hi) {
        // Odd count: single middle element
        out[row_idx] = val_hi;
    } else {
        // Even count: average of two middle elements
        // After nth_element(k_hi), all buf[0..k_hi-1] <= val_hi
        // Find the maximum of buf[0..k_hi-1] to get val_lo
        scalar_t val_lo = buf[0];
        for (int i = 1; i < k_hi; i++) {
            if (buf[i] > val_lo) val_lo = buf[i];
        }
        out[row_idx] = (val_lo + val_hi) / static_cast<scalar_t>(2);
    }
}

// ---------------------------------------------------------------------------
// C++ launch wrapper (called from pybind11 binding)
// ---------------------------------------------------------------------------
torch::Tensor masked_median_cuda(torch::Tensor x, torch::Tensor mask) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA/HIP tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA/HIP tensor");
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
    TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be boolean");

    const int W = x.size(-1);
    TORCH_CHECK(W <= MAX_LOCAL_SIZE,
                "Last dimension ", W, " exceeds kernel limit of ", MAX_LOCAL_SIZE);

    // Ensure contiguous layout
    auto x_c = x.contiguous();
    auto mask_c = mask.contiguous();

    // Output shape: all dims except last
    auto out_sizes = x_c.sizes().vec();
    out_sizes.pop_back();
    const int64_t n_rows = x_c.numel() / W;

    auto out = torch::empty(out_sizes, x_c.options());

    if (n_rows == 0) return out;

    // Launch configuration
    const int threads = 256;
    const int blocks = static_cast<int>((n_rows + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(x_c.scalar_type(), "masked_median_cuda", [&] {
        masked_median_kernel<scalar_t><<<blocks, threads, 0,
            at::hip::getCurrentHIPStream()>>>(
            x_c.data_ptr<scalar_t>(),
            mask_c.data_ptr<bool>(),
            out.data_ptr<scalar_t>(),
            W,
            n_rows
        );
    });

    return out;
}
