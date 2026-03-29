/*
 * umi_kernel.cu - UMI (Unified Median Iterative) GPU kernel
 *
 * Two entry points:
 *   1. umi_median_mad: exact median + MAD (for diagnostics/testing)
 *   2. umi_detrend: fused median → MAD → biweight iterations → location
 *      (single kernel call, no Python loop overhead)
 *
 * Compiles on both AMD ROCm (via hipify) and NVIDIA CUDA.
 */

#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>

constexpr int MAX_LOCAL_SIZE = 512;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
template <typename scalar_t>
__device__ __forceinline__ void swap_vals(scalar_t& a, scalar_t& b) {
    scalar_t tmp = a;
    a = b;
    b = tmp;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t median_of_3(
    scalar_t* buf, int lo, int mid, int hi
) {
    if (buf[mid] < buf[lo]) swap_vals(buf[lo], buf[mid]);
    if (buf[hi]  < buf[lo]) swap_vals(buf[lo], buf[hi]);
    if (buf[hi]  < buf[mid]) swap_vals(buf[mid], buf[hi]);
    return buf[mid];
}

template <typename scalar_t>
__device__ void nth_element(scalar_t* buf, int lo, int hi, int k) {
    while (lo < hi) {
        if (hi - lo < 3) {
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

        swap_vals(buf[mid], buf[hi - 1]);

        int i = lo;
        int j = hi - 1;
        while (true) {
            while (buf[++i] < pivot);
            while (buf[--j] > pivot);
            if (i >= j) break;
            swap_vals(buf[i], buf[j]);
        }
        swap_vals(buf[i], buf[hi - 1]);

        if (k < i) {
            hi = i - 1;
        } else if (k > i) {
            lo = i + 1;
        } else {
            return;
        }
    }
}

template <typename scalar_t>
__device__ scalar_t compute_median(scalar_t* buf, int n) {
    if (n == 1) return buf[0];
    if (n == 2) return (buf[0] + buf[1]) / static_cast<scalar_t>(2);

    const int k_hi = n / 2;
    const int k_lo = (n - 1) / 2;

    nth_element(buf, 0, n - 1, k_hi);
    scalar_t val_hi = buf[k_hi];

    if (k_lo == k_hi) {
        return val_hi;
    } else {
        scalar_t val_lo = buf[0];
        for (int i = 1; i < k_hi; i++) {
            if (buf[i] > val_lo) val_lo = buf[i];
        }
        return (val_lo + val_hi) / static_cast<scalar_t>(2);
    }
}

// ---------------------------------------------------------------------------
// Kernel 1: median + MAD only (for diagnostics)
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void umi_median_mad_kernel(
    const scalar_t* __restrict__ x,
    const bool*     __restrict__ mask,
    scalar_t*       __restrict__ out_median,
    scalar_t*       __restrict__ out_mad,
    const int W,
    const int64_t n_rows
) {
    const int64_t row_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row_idx >= n_rows) return;

    const scalar_t* row_x    = x    + row_idx * W;
    const bool*     row_mask = mask + row_idx * W;

    scalar_t buf[MAX_LOCAL_SIZE];
    int n_valid = 0;
    for (int i = 0; i < W; i++) {
        if (row_mask[i]) {
            buf[n_valid++] = row_x[i];
        }
    }

    if (n_valid == 0) {
        out_median[row_idx] = static_cast<scalar_t>(NAN);
        out_mad[row_idx]    = static_cast<scalar_t>(NAN);
        return;
    }

    scalar_t median_val = compute_median(buf, n_valid);
    out_median[row_idx] = median_val;

    for (int i = 0; i < n_valid; i++) {
        scalar_t d = buf[i] - median_val;
        buf[i] = d < 0 ? -d : d;
    }

    scalar_t mad_val = compute_median(buf, n_valid);
    out_mad[row_idx] = mad_val;
}

// ---------------------------------------------------------------------------
// Kernel 2: FUSED median → MAD → UMI weight iterations → location
// ---------------------------------------------------------------------------
// Everything happens per-thread in registers/local memory.
// Zero global memory traffic between iterations - no Python loop overhead.
//
// The UMI weight function is an asymmetric variant of Tukey's bisquare:
//   u_eff = u * asymmetry  (if u < 0, i.e. below the location)
//   u_eff = u              (if u >= 0)
//   w = (1 - u_eff^2)^2    (if |u_eff| < 1, else 0)
//
// With asymmetry=1.0 this is standard biweight.  With asymmetry>1,
// downward deviations (transit dips) are rejected more aggressively,
// preserving transit depth in the detrended output.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void umi_detrend_kernel(
    const scalar_t* __restrict__ x,         // [n_rows, W]
    const bool*     __restrict__ mask,       // [n_rows, W]
    scalar_t*       __restrict__ out_loc,    // [n_rows]
    const int W,
    const int64_t n_rows,
    const scalar_t cval,                     // rejection threshold (default 5.0)
    const scalar_t asymmetry,               // dip penalty factor (default 1.3)
    const int n_iter,                        // iterations (default 5)
    const int min_valid                      // minimum valid points (default 50)
) {
    const int64_t row_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row_idx >= n_rows) return;

    const scalar_t* row_x    = x    + row_idx * W;
    const bool*     row_mask = mask + row_idx * W;

    // Step 1: Gather valid elements
    scalar_t buf[MAX_LOCAL_SIZE];
    int n = 0;
    for (int i = 0; i < W; i++) {
        if (row_mask[i]) {
            buf[n++] = row_x[i];
        }
    }

    if (n < min_valid) {
        out_loc[row_idx] = static_cast<scalar_t>(NAN);
        return;
    }

    // Step 2: Quickselect for median
    scalar_t location = compute_median(buf, n);

    // Step 3: Compute absolute deviations in-place for MAD
    // (buf is already partially reordered by quickselect - that's fine,
    //  we just need the values, not their order)
    for (int i = 0; i < n; i++) {
        scalar_t d = buf[i] - location;
        buf[i] = d < 0 ? -d : d;
    }

    // Step 4: Quickselect for MAD
    scalar_t mad = compute_median(buf, n);
    if (mad < static_cast<scalar_t>(1e-10))
        mad = static_cast<scalar_t>(1e-10);
    scalar_t safe_scale = cval * mad;

    // Step 5: Re-gather original values (buf was overwritten by abs deviations)
    for (int i = 0; i < W; i++) {
        if (row_mask[i]) {
            // Recount is wasteful but we need original values for biweight
            break;
        }
    }
    n = 0;
    for (int i = 0; i < W; i++) {
        if (row_mask[i]) {
            buf[n++] = row_x[i];
        }
    }

    // Step 6: UMI weight iterations (asymmetric bisquare)
    for (int iter = 0; iter < n_iter; iter++) {
        scalar_t w_sum = 0;
        scalar_t wx_sum = 0;
        for (int j = 0; j < n; j++) {
            scalar_t u = (buf[j] - location) / safe_scale;
            // Asymmetric: penalize downward dips (u < 0) more
            scalar_t u_abs = u < 0 ? -u * asymmetry : u;
            if (u_abs < static_cast<scalar_t>(1)) {
                scalar_t u2 = u_abs * u_abs;
                scalar_t w = (static_cast<scalar_t>(1) - u2);
                w = w * w;
                w_sum += w;
                wx_sum += w * buf[j];
            }
        }
        if (w_sum > static_cast<scalar_t>(1e-10)) {
            location = wx_sum / w_sum;
        }
    }

    out_loc[row_idx] = location;
}

// ---------------------------------------------------------------------------
// C++ launch wrappers
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> umi_median_mad_cuda(
    torch::Tensor x,
    torch::Tensor mask
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA/HIP tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA/HIP tensor");
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
    TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be boolean");

    const int W = x.size(-1);
    TORCH_CHECK(W <= MAX_LOCAL_SIZE,
                "Last dimension ", W, " exceeds kernel limit of ", MAX_LOCAL_SIZE);

    auto x_c = x.contiguous();
    auto mask_c = mask.contiguous();

    auto out_sizes = x_c.sizes().vec();
    out_sizes.pop_back();
    const int64_t n_rows = x_c.numel() / W;

    auto out_median = torch::empty(out_sizes, x_c.options());
    auto out_mad    = torch::empty(out_sizes, x_c.options());

    if (n_rows == 0) return {out_median, out_mad};

    const int threads = 256;
    const int blocks = static_cast<int>((n_rows + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(x_c.scalar_type(), "umi_median_mad_cuda", [&] {
        umi_median_mad_kernel<scalar_t><<<blocks, threads, 0,
            c10::cuda::getCurrentCUDAStream()>>>(
            x_c.data_ptr<scalar_t>(),
            mask_c.data_ptr<bool>(),
            out_median.data_ptr<scalar_t>(),
            out_mad.data_ptr<scalar_t>(),
            W,
            n_rows
        );
    });

    return {out_median, out_mad};
}


torch::Tensor umi_detrend_cuda(
    torch::Tensor x,
    torch::Tensor mask,
    double cval,
    double asymmetry,
    int64_t n_iter,
    int64_t min_valid
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA/HIP tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA/HIP tensor");
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
    TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be boolean");

    const int W = x.size(-1);
    TORCH_CHECK(W <= MAX_LOCAL_SIZE,
                "Last dimension ", W, " exceeds kernel limit of ", MAX_LOCAL_SIZE);

    auto x_c = x.contiguous();
    auto mask_c = mask.contiguous();

    auto out_sizes = x_c.sizes().vec();
    out_sizes.pop_back();
    const int64_t n_rows = x_c.numel() / W;

    auto out_loc = torch::empty(out_sizes, x_c.options());

    if (n_rows == 0) return out_loc;

    const int threads = 256;
    const int blocks = static_cast<int>((n_rows + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(x_c.scalar_type(), "umi_detrend_cuda", [&] {
        umi_detrend_kernel<scalar_t><<<blocks, threads, 0,
            c10::cuda::getCurrentCUDAStream()>>>(
            x_c.data_ptr<scalar_t>(),
            mask_c.data_ptr<bool>(),
            out_loc.data_ptr<scalar_t>(),
            W,
            n_rows,
            static_cast<scalar_t>(cval),
            static_cast<scalar_t>(asymmetry),
            static_cast<int>(n_iter),
            static_cast<int>(min_valid)
        );
    });

    return out_loc;
}
