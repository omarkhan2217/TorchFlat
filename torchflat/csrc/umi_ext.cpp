/*
 * umi_ext.cpp - Pybind11 binding for UMI kernels.
 */

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> umi_median_mad_cuda(torch::Tensor x, torch::Tensor mask);
torch::Tensor umi_detrend_cuda(torch::Tensor x, torch::Tensor mask,
                                double cval, double asymmetry,
                                int64_t n_iter, int64_t min_valid);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("umi_median_mad", &umi_median_mad_cuda,
          "Compute masked median and MAD via quickselect (GPU)");
    m.def("umi_detrend", &umi_detrend_cuda,
          "Fused median + MAD + asymmetric UMI weight detrending (GPU)");
}
