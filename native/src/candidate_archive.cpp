#include "mars_titan/candidate.hpp"

#include <ATen/ATen.h>
#include <torch/serialize/archive.h>
#include <torch/version.h>

#include <array>
#include <cstddef>
#include <istream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace mars_titan::candidate {
namespace {
constexpr int64_t schema_version = 1;
constexpr std::size_t maximum_archive_bytes = 128U << 20;
constexpr std::size_t read_block_bytes = 64U << 10;
constexpr int64_t metadata_width = 3;
constexpr int64_t fp32_code = 32;
constexpr int64_t fp64_code = 64;

void require(bool value, std::string_view reason) {
    if (!value) { throw std::invalid_argument(std::string(reason)); }
}
at::Tensor read_integer_vector(torch::serialize::InputArchive& archive, const std::string& key,
                               int64_t count) {
    at::Tensor result;
    archive.read(key, result, true);
    require(result.device().is_cpu() && result.scalar_type() == at::kLong &&
            result.sizes() == at::IntArrayRef({count}), "Metadatos incompatibles en el archivo candidato");
    return result;
}
std::string read_bounded(std::istream& source) {
    std::string bytes;
    std::array<char, read_block_bytes> block{};
    while (source.read(block.data(), static_cast<std::streamsize>(block.size())) || source.gcount() > 0) {
        const auto count = static_cast<std::size_t>(source.gcount());
        require(bytes.size() <= maximum_archive_bytes - count, "El archivo candidato excede 128 MiB");
        bytes.append(block.data(), count);
    }
    require(source.eof() && !source.bad() && !bytes.empty(), "No se pudo leer el archivo candidato");
    return bytes;
}
struct TensorShape {
    std::string name;
    std::vector<int64_t> dimensions;
};
std::vector<TensorShape> shapes(const torch::OrderedDict<std::string, at::Tensor>& tensors) {
    std::vector<TensorShape> result;
    result.reserve(tensors.size());
    for (const auto& item : tensors) { result.push_back({item.key(), item.value().sizes().vec()}); }
    return result;
}
void check_loaded(const torch::OrderedDict<std::string, at::Tensor>& tensors,
                  const std::vector<TensorShape>& expected, at::ScalarType dtype) {
    require(tensors.size() == expected.size(), "El número de tensores recuperados no coincide");
    for (const auto& shape : expected) {
        const auto& value = tensors[shape.name];
        require(value.sizes() == at::IntArrayRef(shape.dimensions) && value.scalar_type() == dtype &&
                value.device().is_cpu() && at::isfinite(value).all().item<bool>(),
                "Un tensor recuperado tiene forma, tipo o valores incompatibles");
    }
}
}

void Candidate::save_state(std::ostream& destination) const {
    check_fixed_buffers();
    torch::serialize::OutputArchive archive;
    archive.write("schema", at::tensor({schema_version,
        feature_projection_.scalar_type() == at::kDouble ? fp64_code : fp32_code,
        static_cast<int64_t>(is_training())}, at::kLong), true);
    archive.write("dimensions", at::tensor(at::ArrayRef<int64_t>(config_.dimensions), at::kLong), true);
    archive.write("limits", at::tensor({config_.max_batch, config_.max_episodes, config_.neighbors}, at::kLong), true);
    archive.write("seeds", at::tensor({config_.parameter_seed, config_.feature_seed, config_.key_seed}, at::kLong), true);
    archive.write("temperature", at::scalar_tensor(config_.temperature, at::kDouble), true);
    archive.write("normalization_id", c10::IValue(config_.normalization_id));
    archive.write("representation_id", c10::IValue(representation_id_));
    archive.write("torch_version", c10::IValue(TORCH_VERSION));
    torch::serialize::OutputArchive network;
    torch::nn::Module::save(network);
    archive.write("network", network);
    std::ostringstream buffer;
    archive.save_to(buffer);
    const auto bytes = std::move(buffer).str();
    require(bytes.size() <= maximum_archive_bytes, "El archivo candidato excede 128 MiB");
    destination.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    require(destination.good(), "No se pudo guardar el cálculo candidato");
}

std::shared_ptr<Candidate> Candidate::load_state(std::istream& source, const at::Device& device) {
    std::istringstream buffer(read_bounded(source));
    torch::serialize::InputArchive archive;
    archive.load_from(buffer, at::Device(at::kCPU));
    c10::IValue torch_version;
    archive.read("torch_version", torch_version);
    require(torch_version.isString() && torch_version.toStringRef() == TORCH_VERSION,
            "La versión LibTorch no coincide con el archivo candidato");
    const auto schema = read_integer_vector(archive, "schema", metadata_width);
    require(schema[0].item<int64_t>() == schema_version &&
            (schema[1].item<int64_t>() == fp32_code || schema[1].item<int64_t>() == fp64_code) &&
            (schema[2].item<int64_t>() == 0 || schema[2].item<int64_t>() == 1),
            "Versión, precisión o modo del candidato incompatible");
    Config config;
    const auto dimensions = read_integer_vector(archive, "dimensions", modality_count);
    for (std::size_t i = 0; i < config.dimensions.size(); ++i) {
        config.dimensions.at(i) = dimensions[static_cast<int64_t>(i)].item<int64_t>();
    }
    const auto limits = read_integer_vector(archive, "limits", metadata_width);
    config.max_batch = limits[0].item<int64_t>();
    config.max_episodes = limits[1].item<int64_t>();
    config.neighbors = limits[2].item<int64_t>();
    const auto seeds = read_integer_vector(archive, "seeds", metadata_width);
    config.parameter_seed = seeds[0].item<int64_t>();
    config.feature_seed = seeds[1].item<int64_t>();
    config.key_seed = seeds[2].item<int64_t>();
    at::Tensor temperature;
    archive.read("temperature", temperature, true);
    require(temperature.device().is_cpu() && temperature.scalar_type() == at::kDouble && temperature.dim() == 0,
            "La temperatura recuperada no es un escalar FP64");
    config.temperature = temperature.item<double>();
    c10::IValue normalization;
    archive.read("normalization_id", normalization);
    require(normalization.isString(), "La identidad de normalización debe ser texto");
    config.normalization_id = normalization.toStringRef();
    const auto dtype = schema[1].item<int64_t>() == fp64_code ? at::kDouble : at::kFloat;
    auto result = std::make_shared<Candidate>(std::move(config), dtype);
    const auto parameter_shapes = shapes(result->named_parameters());
    const auto buffer_shapes = shapes(result->named_buffers());
    torch::serialize::InputArchive network;
    archive.read("network", network);
    result->torch::nn::Module::load(network);
    check_loaded(result->named_parameters(), parameter_shapes, dtype);
    check_loaded(result->named_buffers(), buffer_shapes, dtype);
    result->refresh_representation();
    c10::IValue representation;
    archive.read("representation_id", representation);
    require(representation.isString() && representation.toStringRef() == result->representation_id_,
            "La huella de las matrices fijas no coincide con el archivo candidato");
    result->train(schema[2].item<int64_t>() == 1);
    require(device.is_cpu() || (device.is_cuda() && device.index() == 0), "El dispositivo debe ser cpu o cuda:0");
#ifndef MARS_TITAN_LIBTORCH_CUDA
    require(device.is_cpu(), "Este ejecutable se compiló sin backend CUDA");
#endif
    result->to(device);
    return result;
}
}
