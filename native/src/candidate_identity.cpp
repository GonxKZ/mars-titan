#include "mars_titan/candidate.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
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
std::string digest_bytes(const void* data, std::size_t size) {
    std::array<unsigned char, EVP_MAX_MD_SIZE> output{};
    unsigned int length = 0;
    if (EVP_Digest(data, size, output.data(), &length, EVP_sha256(), nullptr) != 1 ||
        length != sha256_size) {
        throw std::runtime_error("No se pudo calcular SHA-256 de los tensores del candidato");
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
std::string digest(const at::Tensor& tensor) {
    const auto bytes = tensor.detach().to(at::kCPU).contiguous();
    return digest_bytes(bytes.const_data_ptr(), bytes.nbytes());
}
} // namespace
void Candidate::refresh_representation() {
    std::string id = (config_.input_policy == "strict_inputs_v1" ? "candidate-fixed-v1:"
                                                                 : "candidate-masked-fixed-v1:") +
                     config_.normalization_id + ":";
    for (const auto dimension : config_.dimensions) {
        id += std::to_string(dimension) + ":";
    }
    representation_id_ = id + std::to_string(static_cast<int>(feature_projection_.scalar_type())) +
                         ":" + digest(feature_projection_) + ":" + digest(key_projection_);
}
std::string Candidate::parameter_fingerprint() const {
    std::string identity = "candidate-parameters-v1:";
    for (const auto& parameter : named_parameters()) {
        const auto& value = parameter.value();
        if (!at::isfinite(value).all().item<bool>()) {
            throw std::invalid_argument("Un parámetro del candidato contiene NaN o infinito");
        }
        identity += std::to_string(parameter.key().size()) + ":" + parameter.key() + ":" +
                    std::to_string(static_cast<int>(value.scalar_type())) + ":";
        for (const auto dimension : value.sizes()) {
            identity += std::to_string(dimension) + ",";
        }
        identity += ":" + digest(value) + ":";
    }
    return digest_bytes(identity.data(), identity.size());
}
std::vector<std::string> Candidate::transfer_strict_parameters(const Candidate& source) {
    if (config_.input_policy != "historical_masked_2000_v1" ||
        source.config_.input_policy != "strict_inputs_v1" ||
        config_.dimensions != source.config_.dimensions ||
        config_.normalization_id != source.config_.normalization_id ||
        feature_projection_.scalar_type() != source.feature_projection_.scalar_type() ||
        feature_projection_.device() != source.feature_projection_.device()) {
        throw std::invalid_argument(
            "El traslado necesita contratos compatibles y políticas distintas");
    }
    const auto target = named_parameters();
    const auto origin = source.named_parameters();
    if (target.size() != origin.size()) {
        throw std::invalid_argument("El traslado tiene un conjunto de parámetros incompatible");
    }
    std::vector<std::string> copied;
    copied.reserve(target.size());
    // Todas las comprobaciones preceden a la primera copia, incluidas las de almacenamiento.
    for (const auto& item : target) {
        const auto* original = origin.find(item.key());
        const auto destination = item.key() == "fusion_weight"
                                     ? item.value().narrow(1, 0, modality_count * hidden_width)
                                     : item.value();
        if (original == nullptr || original->sizes() != destination.sizes() ||
            original->scalar_type() != destination.scalar_type() ||
            original->device() != destination.device() ||
            !at::isfinite(*original).all().item<bool>()) {
            throw std::invalid_argument("Un parámetro de origen es incompatible o no finito");
        }
        for (const auto& other : target) {
            if (original->is_alias_of(other.value())) {
                throw std::invalid_argument("El traslado no admite almacenamiento compartido");
            }
        }
        copied.push_back(item.key());
    }
    const at::NoGradGuard guard;
    for (const auto& name : copied) {
        if (name == "fusion_weight") {
            target[name].narrow(1, 0, modality_count * hidden_width).copy_(origin[name]);
            target[name].narrow(1, modality_count * hidden_width, modality_count).zero_();
        } else {
            target[name].copy_(origin[name]);
        }
    }
    return copied;
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
