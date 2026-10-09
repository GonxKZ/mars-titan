#include "mars_titan/candidate.hpp"

#include <ATen/ATen.h>
#include <miniz.h>
#include <torch/serialize/archive.h>
#include <torch/version.h>

#include <array>
#include <cstddef>
#include <istream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mars_titan::candidate {
namespace {
constexpr int64_t schema_version = 2;
constexpr int64_t historical_schema_version = 3;
constexpr std::size_t maximum_archive_bytes = 128U << 20;
constexpr mz_uint maximum_records = 1024;
constexpr mz_uint maximum_record_name = 512;
constexpr std::size_t read_block_bytes = 64U << 10;
constexpr int64_t metadata_width = 3;
constexpr int64_t fp32_code = 32;
constexpr int64_t fp64_code = 64;

void require(bool value, std::string_view reason) {
    if (!value) {
        throw std::invalid_argument(std::string(reason));
    }
}
at::Tensor read_integer_vector(torch::serialize::InputArchive& archive, const std::string& key,
                               int64_t count) {
    at::Tensor result;
    archive.read(key, result, true);
    require(result.device().is_cpu() && result.scalar_type() == at::kLong &&
                result.sizes() == at::IntArrayRef({count}),
            "Metadatos incompatibles en el archivo candidato");
    return result;
}
std::string read_bounded(std::istream& source) {
    std::string bytes;
    std::array<char, read_block_bytes> block{};
    while (source.read(block.data(), static_cast<std::streamsize>(block.size())) ||
           source.gcount() > 0) {
        const auto count = static_cast<std::size_t>(source.gcount());
        require(bytes.size() <= maximum_archive_bytes - count,
                "El archivo candidato excede 128 MiB");
        bytes.append(block.data(), count);
    }
    require(source.eof() && !source.bad() && !bytes.empty(),
            "No se pudo leer el archivo candidato");
    return bytes;
}
struct ZipDirectory {
    ZipDirectory() = default;
    ZipDirectory(const ZipDirectory&) = delete;
    ZipDirectory& operator=(const ZipDirectory&) = delete;
    ZipDirectory(ZipDirectory&&) = delete;
    ZipDirectory& operator=(ZipDirectory&&) = delete;
    mz_zip_archive archive{};
    ~ZipDirectory() {
        if (archive.m_pState != nullptr) {
            (void)mz_zip_reader_end(&archive);
        }
    }
};
void check_archive_directory(const std::string& bytes) {
    ZipDirectory directory;
    require(mz_zip_reader_init_mem(&directory.archive, bytes.data(), bytes.size(), 0) != 0,
            "El directorio ZIP es inválido o está truncado");
    const auto records = mz_zip_reader_get_num_files(&directory.archive);
    require(records > 0 && records <= maximum_records,
            "El número de registros ZIP excede el límite");
    std::size_t expanded = 0;
    std::unordered_set<std::string> names;
    for (mz_uint i = 0; i < records; ++i) {
        mz_zip_archive_file_stat stat{};
        require(mz_zip_reader_file_stat(&directory.archive, i, &stat) != 0,
                "No se pudo leer un registro del directorio ZIP");
        require(stat.m_uncomp_size <= maximum_archive_bytes - expanded,
                "El tamaño descomprimido del archivo excede 128 MiB");
        expanded += static_cast<std::size_t>(stat.m_uncomp_size);
        const auto size = mz_zip_reader_get_filename(&directory.archive, i, nullptr, 0);
        require(size > 1 && size <= maximum_record_name,
                "El nombre de registro ZIP excede el límite");
        std::vector<char> filename(size);
        require(mz_zip_reader_get_filename(&directory.archive, i, filename.data(), size) == size &&
                    filename.back() == '\0',
                "El nombre de registro ZIP es inválido");
        std::string name(filename.data(), size - 1);
        // miniz busca registros sin distinguir mayúsculas por defecto.
        for (auto& character : name) {
            if (character >= 'A' && character <= 'Z') {
                character = static_cast<char>(character + ('a' - 'A'));
            }
        }
        require(names.insert(std::move(name)).second,
                "El archivo ZIP contiene un registro duplicado");
        require(stat.m_is_encrypted == 0 && stat.m_is_supported != 0,
                "El registro ZIP usa cifrado o compresión no admitidos");
    }
}
void read_projection(torch::serialize::InputArchive& archive, const std::string& name,
                     at::Tensor& expected) {
    at::Tensor value;
    archive.read(name, value, true);
    require(value.sizes() == expected.sizes() && value.scalar_type() == expected.scalar_type() &&
                value.device().is_cpu() && at::isfinite(value).all().item<bool>(),
            "La proyección recuperada tiene forma, tipo o valores incompatibles");
    // Un archivo puede compartir storage entre parámetros y proyecciones.
    expected = value.detach().clone();
}
struct TensorShape {
    std::string name;
    std::vector<int64_t> dimensions;
};
std::vector<TensorShape> shapes(const torch::OrderedDict<std::string, at::Tensor>& tensors) {
    std::vector<TensorShape> result;
    result.reserve(tensors.size());
    for (const auto& item : tensors) {
        result.push_back({item.key(), item.value().sizes().vec()});
    }
    return result;
}
void check_loaded(const torch::OrderedDict<std::string, at::Tensor>& tensors,
                  const std::vector<TensorShape>& expected, at::ScalarType dtype) {
    require(tensors.size() == expected.size(), "El número de tensores recuperados no coincide");
    for (const auto& shape : expected) {
        const auto& value = tensors[shape.name];
        require(value.sizes() == at::IntArrayRef(shape.dimensions) &&
                    value.scalar_type() == dtype && value.device().is_cpu() &&
                    at::isfinite(value).all().item<bool>(),
                "Un tensor recuperado tiene forma, tipo o valores incompatibles");
    }
}
} // namespace

void Candidate::save_state(std::ostream& destination) const {
    torch::serialize::OutputArchive archive;
    const bool historical = config_.input_policy == "historical_masked_2000_v1";
    archive.write(
        "schema",
        at::tensor({historical ? historical_schema_version : schema_version,
                    feature_projection_.scalar_type() == at::kDouble ? fp64_code : fp32_code,
                    static_cast<int64_t>(is_training())},
                   at::kLong),
        true);
    archive.write("dimensions", at::tensor(at::ArrayRef<int64_t>(config_.dimensions), at::kLong),
                  true);
    archive.write(
        "limits",
        at::tensor({config_.max_batch, config_.max_episodes, config_.neighbors}, at::kLong), true);
    archive.write(
        "seeds",
        at::tensor({config_.parameter_seed, config_.feature_seed, config_.key_seed}, at::kLong),
        true);
    archive.write("temperature", at::scalar_tensor(config_.temperature, at::kDouble), true);
    archive.write("normalization_id", c10::IValue(config_.normalization_id));
    if (historical) {
        archive.write("input_policy", c10::IValue(config_.input_policy));
    }
    archive.write("representation_id", c10::IValue(representation_id_));
    archive.write("torch_version", c10::IValue(TORCH_VERSION));
    archive.write("feature_projection", feature_projection_, true);
    archive.write("key_projection", key_projection_, true);
    torch::serialize::OutputArchive network;
    torch::nn::Module::save(network);
    archive.write("network", network);
    std::ostringstream buffer;
    archive.save_to(buffer);
    const auto bytes = std::move(buffer).str();
    require(bytes.size() <= maximum_archive_bytes, "El archivo candidato excede 128 MiB");
    check_archive_directory(bytes);
    destination.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    require(destination.good(), "No se pudo guardar el cálculo candidato");
}

std::shared_ptr<Candidate> Candidate::load_state(std::istream& source, const at::Device& device) {
    return load_state(source, device, "strict_inputs_v1");
}

std::shared_ptr<Candidate> Candidate::load_state(std::istream& source, const at::Device& device,
                                                 const std::string& expected_policy) {
    require(expected_policy == "strict_inputs_v1" || expected_policy == "historical_masked_2000_v1",
            "La política esperada del archivo candidato no está admitida");
    auto bytes = read_bounded(source);
    check_archive_directory(bytes);
    std::istringstream buffer(std::move(bytes));
    torch::serialize::InputArchive archive;
    archive.load_from(buffer, at::Device(at::kCPU));
    c10::IValue torch_version;
    archive.read("torch_version", torch_version);
    require(torch_version.isString() && torch_version.toStringRef() == TORCH_VERSION,
            "La versión LibTorch no coincide con el archivo candidato");
    const auto schema = read_integer_vector(archive, "schema", metadata_width);
    require(
        schema[0].item<int64_t>() == (expected_policy == "strict_inputs_v1"
                                          ? schema_version
                                          : historical_schema_version) &&
            (schema[1].item<int64_t>() == fp32_code || schema[1].item<int64_t>() == fp64_code) &&
            (schema[2].item<int64_t>() == 0 || schema[2].item<int64_t>() == 1),
        "Versión, precisión o modo del candidato incompatible");
    Config config;
    config.input_policy = expected_policy;
    if (expected_policy == "historical_masked_2000_v1") {
        c10::IValue policy;
        archive.read("input_policy", policy);
        require(policy.isString() && policy.toStringRef() == expected_policy,
                "La política del archivo candidato no coincide");
    }
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
    require(temperature.device().is_cpu() && temperature.scalar_type() == at::kDouble &&
                temperature.dim() == 0,
            "La temperatura recuperada no es un escalar FP64");
    config.temperature = temperature.item<double>();
    c10::IValue normalization;
    archive.read("normalization_id", normalization);
    require(normalization.isString(), "La identidad de normalización debe ser texto");
    config.normalization_id = normalization.toStringRef();
    const auto dtype = schema[1].item<int64_t>() == fp64_code ? at::kDouble : at::kFloat;
    auto result = std::make_shared<Candidate>(std::move(config), dtype);
    const auto parameter_shapes = shapes(result->named_parameters());
    torch::serialize::InputArchive network;
    archive.read("network", network);
    result->torch::nn::Module::load(network);
    check_loaded(result->named_parameters(), parameter_shapes, dtype);
    read_projection(archive, "feature_projection", result->feature_projection_);
    read_projection(archive, "key_projection", result->key_projection_);
    result->refresh_representation();
    c10::IValue representation;
    archive.read("representation_id", representation);
    require(representation.isString() && representation.toStringRef() == result->representation_id_,
            "La huella de las matrices fijas no coincide con el archivo candidato");
    result->train(schema[2].item<int64_t>() == 1);
    require(device.is_cpu() || (device.is_cuda() && device.index() == 0),
            "El dispositivo debe ser cpu o cuda:0");
#ifndef MARS_TITAN_LIBTORCH_CUDA
    require(device.is_cpu(), "Este ejecutable se compiló sin backend CUDA");
#endif
    result->to(device);
    return result;
}
} // namespace mars_titan::candidate
