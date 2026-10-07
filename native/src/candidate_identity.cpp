#include "mars_titan/candidate.hpp"

#include <ATen/ATen.h>
#include <openssl/evp.h>

#include <array>
#include <stdexcept>
#include <string_view>

namespace mars_titan::candidate {
namespace {
constexpr unsigned int sha256_size = 32;
constexpr unsigned int nibble_bits = 4;
constexpr unsigned int nibble_mask = 15;
void check_device(const at::Device& device) {
    if (!device.is_cpu() && !(device.is_cuda() && device.index() == 0)) {
        throw std::invalid_argument("El dispositivo debe ser cpu o cuda:0");
    }
#ifndef MARS_TITAN_LIBTORCH_CUDA
    if (!device.is_cpu()) {
        throw std::invalid_argument("Este ejecutable se compiló sin backend CUDA");
    }
#endif
}
void check_dtype(at::ScalarType dtype) {
    if (dtype != at::kFloat && dtype != at::kDouble) {
        throw std::invalid_argument("El candidato admite FP32 o FP64");
    }
}
std::string digest(const at::Tensor& tensor) {
    const auto bytes = tensor.detach().to(at::kCPU).contiguous();
    std::array<unsigned char, EVP_MAX_MD_SIZE> output{};
    unsigned int length = 0;
    if (EVP_Digest(bytes.const_data_ptr(), bytes.nbytes(), output.data(), &length, EVP_sha256(),
                   nullptr) != 1 ||
        length != sha256_size) {
        throw std::runtime_error("No se pudo calcular SHA-256 de la representación fija");
    }
    constexpr std::string_view digits = "0123456789abcdef";
    std::string result;
    result.reserve(2 * static_cast<std::size_t>(sha256_size));
    for (unsigned int i = 0; i < length; ++i) {
        result += digits[output.at(i) >> nibble_bits];
        result += digits[output.at(i) & nibble_mask];
    }
    return result;
}
} // namespace
void Candidate::refresh_representation() {
    std::string id = "candidate-fixed-v1:" + config_.normalization_id + ":";
    for (const auto dimension : config_.dimensions) {
        id += std::to_string(dimension) + ":";
    }
    representation_id_ = id + std::to_string(static_cast<int>(feature_projection_.scalar_type())) +
                         ":" + digest(feature_projection_) + ":" + digest(key_projection_);
}
void Candidate::to(torch::Device device, torch::Dtype dtype, bool non_blocking) {
    check_device(device);
    check_dtype(dtype);
    const bool changed = dtype != feature_projection_.scalar_type();
    torch::nn::Module::to(device, dtype, non_blocking);
    feature_projection_ = feature_projection_.to(device, dtype, non_blocking);
    key_projection_ = key_projection_.to(device, dtype, non_blocking);
    if (changed) {
        refresh_representation();
    }
}
void Candidate::to(torch::Dtype dtype, bool non_blocking) {
    to(feature_projection_.device(), dtype, non_blocking);
}
void Candidate::to(torch::Device device, bool non_blocking) {
    to(device, feature_projection_.scalar_type(), non_blocking);
}
} // namespace mars_titan::candidate
