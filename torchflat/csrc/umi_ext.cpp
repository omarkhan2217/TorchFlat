/*
 * umi_ext.cpp - Pybind11 binding for UMI kernels.
 */

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> umi_median_mad_cuda(torch::Tensor x, torch::Tensor mask);
torch::Tensor umi_detrend_direct_cuda(torch::Tensor flux, torch::Tensor valid_mask,
                                       torch::Tensor segment_id, int64_t W,
                                       double cval, double asymmetry,
                                       int64_t n_iter, int64_t min_valid);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("umi_median_mad", &umi_median_mad_cuda,
          "Compute masked median and MAD via quickselect (GPU)");
    m.def("umi_detrend_direct", &umi_detrend_direct_cuda,
          "Direct UMI detrending from raw [B,L] arrays (GPU)");
}
