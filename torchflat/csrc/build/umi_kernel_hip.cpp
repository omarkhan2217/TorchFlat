/*
 * umi_kernel_hip.cpp - Manually hipified UMI kernel
 *
 * Uses c10/hip/HIPStream.h (not ATen/hip/HIPContext.h) to avoid
 * cusolver_common.h missing header issue on Windows ROCm.
 */

#include <torch/types.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

constexpr int MAX_LOCAL_SIZE = 512;

template <typename scalar_t>
__device__ __forceinline__ void swap_vals(scalar_t& a, scalar_t& b) {
    scalar_t tmp = a; a = b; b = tmp;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t median_of_3(scalar_t* buf, int lo, int mid, int hi) {
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
                scalar_t val = buf[i]; int j = i;
                while (j > lo && buf[j-1] > val) { buf[j] = buf[j-1]; j--; }
                buf[j] = val;
            }
            return;
        }
        int mid = lo + (hi - lo) / 2;
        scalar_t pivot = median_of_3(buf, lo, mid, hi);
        swap_vals(buf[mid], buf[hi-1]);
        int i = lo, j = hi - 1;
        while (true) {
            while (buf[++i] < pivot);
            while (buf[--j] > pivot);
            if (i >= j) break;
            swap_vals(buf[i], buf[j]);
        }
        swap_vals(buf[i], buf[hi-1]);
        if (k < i) hi = i - 1;
        else if (k > i) lo = i + 1;
        else return;
    }
}

template <typename scalar_t>
__device__ scalar_t compute_median(scalar_t* buf, int n) {
    if (n == 1) return buf[0];
    if (n == 2) return (buf[0] + buf[1]) / static_cast<scalar_t>(2);
    const int k_hi = n / 2, k_lo = (n - 1) / 2;
    nth_element(buf, 0, n - 1, k_hi);
    scalar_t val_hi = buf[k_hi];
    if (k_lo == k_hi) return val_hi;
    scalar_t val_lo = buf[0];
    for (int i = 1; i < k_hi; i++) if (buf[i] > val_lo) val_lo = buf[i];
    return (val_lo + val_hi) / static_cast<scalar_t>(2);
}

// Kernel 1: median + MAD
template <typename scalar_t>
__global__ void umi_median_mad_kernel(
    const scalar_t* __restrict__ x, const bool* __restrict__ mask,
    scalar_t* __restrict__ out_median, scalar_t* __restrict__ out_mad,
    const int W, const int64_t n_rows
) {
    const int64_t row_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row_idx >= n_rows) return;
    const scalar_t* rx = x + row_idx * W;
    const bool* rm = mask + row_idx * W;
    scalar_t buf[MAX_LOCAL_SIZE]; int n = 0;
    for (int i = 0; i < W; i++) if (rm[i]) buf[n++] = rx[i];
    if (n == 0) { out_median[row_idx] = NAN; out_mad[row_idx] = NAN; return; }
    scalar_t med = compute_median(buf, n);
    out_median[row_idx] = med;
    for (int i = 0; i < n; i++) { scalar_t d = buf[i] - med; buf[i] = d < 0 ? -d : d; }
    out_mad[row_idx] = compute_median(buf, n);
}

// Launch wrappers
std::vector<torch::Tensor> umi_median_mad_cuda(torch::Tensor x, torch::Tensor mask) {
    TORCH_CHECK(x.is_cuda() && mask.is_cuda());
    TORCH_CHECK(x.sizes() == mask.sizes() && mask.dtype() == torch::kBool);
    const int W = x.size(-1);
    TORCH_CHECK(W <= MAX_LOCAL_SIZE);
    auto xc = x.contiguous(), mc = mask.contiguous();
    auto os = xc.sizes().vec(); os.pop_back();
    const int64_t nr = xc.numel() / W;
    auto om = torch::empty(os, xc.options()), od = torch::empty(os, xc.options());
    if (nr == 0) return {om, od};
    const int t = 256, b = (nr + t - 1) / t;
    AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "umi_median_mad", [&] {
        umi_median_mad_kernel<scalar_t><<<b, t, 0, c10::hip::getCurrentHIPStream()>>>(
            xc.data_ptr<scalar_t>(), mc.data_ptr<bool>(),
            om.data_ptr<scalar_t>(), od.data_ptr<scalar_t>(), W, nr);
    });
    return {om, od};
}

// Kernel 2: Direct from raw [B, L] arrays
template <typename scalar_t>
__global__ void umi_detrend_direct_kernel(
    const scalar_t* __restrict__ flux, const bool* __restrict__ valid,
    const int* __restrict__ seg, scalar_t* __restrict__ out_loc,
    const int B, const int L, const int W, const int N_pos,
    const scalar_t cval, const scalar_t asymmetry,
    const int n_iter, const int min_valid
) {
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(B) * N_pos;
    if (tid >= total) return;
    const int star = static_cast<int>(tid / N_pos);
    const int pos = static_cast<int>(tid % N_pos);
    const int half_w = W / 2;
    const scalar_t* f = flux + static_cast<int64_t>(star) * L;
    const bool* v = valid + static_cast<int64_t>(star) * L;
    const int* s = seg + static_cast<int64_t>(star) * L;
    const int center_seg = s[pos + half_w];

    scalar_t buf[MAX_LOCAL_SIZE]; int n = 0;
    for (int i = 0; i < W; i++) {
        int idx = pos + i;
        if (v[idx] && s[idx] == center_seg) buf[n++] = f[idx];
    }
    if (n < min_valid) { out_loc[tid] = NAN; return; }

    scalar_t location = compute_median(buf, n);
    // Upper-RMS scale: only points above median
    scalar_t upper_sq_sum = 0; int n_above = 0;
    for (int i = 0; i < n; i++) {
        scalar_t d = buf[i] - location;
        if (d > 0) { upper_sq_sum += d * d; n_above++; }
    }
    scalar_t scale;
    if (n_above > 0) {
        scale = sqrt(upper_sq_sum / static_cast<scalar_t>(n_above)) * static_cast<scalar_t>(0.6745);
    } else { scale = static_cast<scalar_t>(1e-10); }
    if (scale < static_cast<scalar_t>(1e-10)) scale = static_cast<scalar_t>(1e-10);
    scalar_t safe_scale = cval * scale;

    // buf still has original values (upper-RMS doesn't modify it)
    for (int iter = 0; iter < n_iter; iter++) {
        scalar_t w_sum = 0, wx_sum = 0;
        for (int j = 0; j < n; j++) {
            scalar_t u = (buf[j] - location) / safe_scale;
            scalar_t u_abs = u < 0 ? -u * asymmetry : u;
            if (u_abs < static_cast<scalar_t>(1)) {
                scalar_t u2 = u_abs * u_abs;
                scalar_t w = (static_cast<scalar_t>(1) - u2);
                w = w * w;
                w_sum += w; wx_sum += w * buf[j];
            }
        }
        if (w_sum > static_cast<scalar_t>(1e-10)) location = wx_sum / w_sum;
    }
    out_loc[tid] = location;
}

torch::Tensor umi_detrend_direct_cuda(torch::Tensor flux, torch::Tensor valid_mask,
                                       torch::Tensor segment_id, int64_t W,
                                       double cval, double asymmetry,
                                       int64_t n_iter, int64_t min_valid) {
    TORCH_CHECK(flux.is_cuda() && flux.dim() == 2);
    auto f = flux.contiguous(), v = valid_mask.contiguous();
    auto s = segment_id.contiguous().to(torch::kInt32);
    const int B = f.size(0), L = f.size(1), N_pos = L - W + 1;
    if (N_pos <= 0) return torch::empty({B, 0}, f.options());
    auto ol = torch::empty({B, N_pos}, f.options());
    const int64_t total = static_cast<int64_t>(B) * N_pos;
    const int t = 256, b = (total + t - 1) / t;
    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "umi_detrend_direct", [&] {
        umi_detrend_direct_kernel<scalar_t><<<b, t, 0, c10::hip::getCurrentHIPStream()>>>(
            f.data_ptr<scalar_t>(), v.data_ptr<bool>(), s.data_ptr<int>(),
            ol.data_ptr<scalar_t>(), B, L, static_cast<int>(W), N_pos,
            static_cast<scalar_t>(cval), static_cast<scalar_t>(asymmetry),
            static_cast<int>(n_iter), static_cast<int>(min_valid));
    });
    return ol;
}
