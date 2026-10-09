#include "mars_titan/candidate.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>

#include <stdexcept>

namespace mars_titan::candidate {
namespace {
constexpr double key_tolerance = 1e-5;
constexpr std::size_t metadata_reserve = 64U << 10;
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}
} // namespace

at::Tensor Candidate::flatten_inputs(const Inputs& inputs, const Config& config) {
    const at::NoGradGuard guard;
    std::vector<at::Tensor> blocks{inputs.prices.flatten(1), inputs.news, inputs.charts,
                                   inputs.fundamentals, inputs.macro};
    if (config.input_policy == "historical_masked_2000_v1") {
        blocks.push_back(inputs.presence.to(inputs.prices.scalar_type()));
    }
    return at::cat(blocks, -1);
}

EpisodeEncoding Candidate::project_episodes(const at::Tensor& flattened,
                                            const at::Tensor& feature_projection,
                                            const at::Tensor& key_projection) {
    const at::NoGradGuard guard;
    const auto values = at::matmul(flattened, feature_projection);
    const auto keys = normalize(at::matmul(values, key_projection));
    require(at::isfinite(values).all().item<bool>() && at::isfinite(keys).all().item<bool>(),
            "La codificación episódica excede el rango numérico");
    return {keys, values};
}

CpuEpisodeCodec::CpuEpisodeCodec(const Candidate& source, std::size_t max_working_bytes)
    : config_(source.config_), projection_id_(source.representation_id_),
      max_working_bytes_(max_working_bytes) {
    const auto bytes = source.feature_projection_.nbytes() + source.key_projection_.nbytes();
    require(max_working_bytes > 0 && max_working_bytes <= maximum_codec_bytes &&
                bytes + metadata_reserve <= max_working_bytes,
            "Las proyecciones exceden el presupuesto del codec CPU");
    const at::NoGradGuard guard;
    feature_projection_ = source.feature_projection_.detach().to(
        at::Device(at::kCPU), source.feature_projection_.scalar_type(), false, true);
    key_projection_ = source.key_projection_.detach().to(
        at::Device(at::kCPU), source.key_projection_.scalar_type(), false, true);
}

const std::string& CpuEpisodeCodec::projection_id() const noexcept { return projection_id_; }
at::ScalarType CpuEpisodeCodec::dtype() const noexcept { return feature_projection_.scalar_type(); }

std::size_t CpuEpisodeCodec::estimated_bytes(int64_t batch) const {
    require(batch > 0 && batch <= config_.max_batch, "El lote supera el presupuesto del codec");
    const auto rows = static_cast<std::size_t>(batch);
    const auto element_bytes = static_cast<std::size_t>(feature_projection_.element_size());
    const auto inputs =
        rows * static_cast<std::size_t>(feature_projection_.size(0)) * element_bytes;
    const auto outputs =
        rows * static_cast<std::size_t>(hidden_width + feature_width) * element_bytes;
    // Proyecciones propias, input y concatenación, outputs y copias CPU de la frontera Python.
    return feature_projection_.nbytes() + key_projection_.nbytes() + 2 * inputs + 3 * outputs +
           metadata_reserve;
}

EpisodeEncoding CpuEpisodeCodec::encode(const Inputs& inputs) const {
    require(inputs.prices.defined() && inputs.prices.dim() == 3,
            "Los precios necesitan tres dimensiones");
    const auto batch = inputs.prices.size(0);
    require(estimated_bytes(batch) <= max_working_bytes_,
            "El codec supera el presupuesto de trabajo");
    (void)Candidate::validate_inputs(inputs, config_, feature_projection_);
    const at::NoGradGuard guard;
    const auto flattened = Candidate::flatten_inputs(inputs, config_);
    EpisodeEncoding result{at::empty({batch, hidden_width}, feature_projection_.options()),
                           at::empty({batch, feature_width}, feature_projection_.options())};
    for (int64_t row = 0; row < batch; ++row) {
        const auto projected = Candidate::project_episodes(flattened.narrow(0, row, 1),
                                                           feature_projection_, key_projection_);
        const auto norm = projected.keys.norm(2, -1);
        const auto exact_zero = (projected.keys == 0).all(-1);
        require(((norm - 1).abs() <= key_tolerance).logical_or(exact_zero).all().item<bool>(),
                "La clave episódica no es unitaria ni exactamente nula");
        result.keys.narrow(0, row, 1).copy_(projected.keys);
        result.values.narrow(0, row, 1).copy_(projected.values);
    }
    return result;
}

std::shared_ptr<CpuEpisodeCodec> Candidate::cpu_episode_codec(std::size_t max_working_bytes) const {
    return std::make_shared<CpuEpisodeCodec>(*this, max_working_bytes);
}
} // namespace mars_titan::candidate
