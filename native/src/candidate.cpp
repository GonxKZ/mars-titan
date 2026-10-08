#include "mars_titan/candidate.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/grad_mode.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace mars_titan::candidate {
namespace {
constexpr int64_t maximum_dimension = 4096;
constexpr int64_t maximum_input_width = 16384;
constexpr int64_t gate_count = 3;
constexpr int64_t read_input_width = feature_width + 1;
constexpr int64_t update_input_width = hidden_width + feature_width + hidden_width + 1;
constexpr double initial_step = 0.1;
constexpr double key_tolerance = 1e-5;
constexpr int64_t median_index = 2;

void require(bool value, std::string_view reason) {
    if (!value) {
        throw std::invalid_argument(std::string(reason));
    }
}
int64_t input_width(const Config& config) {
    int64_t width = 0;
    for (std::size_t i = 0; i < config.dimensions.size(); ++i) {
        const auto dimension = config.dimensions.at(i);
        require(dimension > 0 && dimension <= maximum_dimension,
                "La dimensión de una modalidad está fuera del presupuesto");
        width += dimension * (i == 0 ? price_window : 1);
    }
    require(width <= maximum_input_width, "Las entradas exceden el presupuesto de proyección");
    return width;
}
void check_config(const Config& config) {
    (void)input_width(config);
    require(config.max_batch > 0 && config.max_batch <= maximum_batch && config.max_episodes > 0 &&
                config.max_episodes <= maximum_episodes && config.neighbors > 0 &&
                config.neighbors <= maximum_neighbors,
            "El lote, la memoria o los vecinos exceden los límites del candidato");
    require(std::isfinite(config.temperature) && config.temperature >= normalization_epsilon,
            "La temperatura debe ser finita y al menos epsilon");
    require(config.parameter_seed >= 0 && config.feature_seed >= 0 && config.key_seed >= 0,
            "Las semillas deben ser no negativas");
    require(!config.normalization_id.empty() && config.normalization_id.size() <= feature_width,
            "Se necesita una identidad de normalización de hasta 256 bytes");
}
void check_tensor(const at::Tensor& value, at::IntArrayRef shape, const at::Tensor& reference,
                  bool finite = false) {
    require(value.defined() && value.layout() == at::kStrided && value.sizes() == shape &&
                value.scalar_type() == reference.scalar_type() &&
                value.device() == reference.device(),
            "Forma, tipo o dispositivo incompatible con el candidato");
    if (finite) {
        require(at::isfinite(value).all().item<bool>(), "El candidato recibió NaN o infinito");
    }
}
at::Tensor normalize(const at::Tensor& value) {
    // Escalar antes de elevar al cuadrado evita convertir una norma desbordada en ceros.
    const auto scale = value.abs().amax(-1, true).clamp_min(normalization_epsilon);
    const auto scaled = value / scale;
    return scaled / scaled.norm(2, -1, true).clamp_min(normalization_epsilon / scale);
}
} // namespace

int64_t MemorySnapshot::size() const { return keys_.size(0); }

Candidate::Candidate(Config config, at::ScalarType dtype, const at::Device& device)
    : config_(std::move(config)) {
    check_config(config_);
    require(dtype == at::kFloat || dtype == at::kDouble, "El candidato admite FP32 o FP64");
    require(device.is_cpu() || (device.is_cuda() && device.index() == 0),
            "El dispositivo debe ser cpu o cuda:0");
#ifndef MARS_TITAN_LIBTORCH_CUDA
    require(device.is_cpu(), "Este ejecutable se compiló sin backend CUDA");
#endif
    const at::NoGradGuard guard;
    auto generator = at::detail::createCPUGenerator(static_cast<uint64_t>(config_.parameter_seed));
    const auto options = at::TensorOptions().dtype(dtype);
    const double bound = 1 / std::sqrt(static_cast<double>(hidden_width));
    const std::array<std::pair<std::string, std::vector<int64_t>>, 4> gru_shapes{
        {{"price_weight_ih", {gate_count * hidden_width, config_.dimensions.front()}},
         {"price_weight_hh", {gate_count * hidden_width, hidden_width}},
         {"price_bias_ih", {gate_count * hidden_width}},
         {"price_bias_hh", {gate_count * hidden_width}}}};
    for (const auto& [name, shape] : gru_shapes) {
        gru_.push_back(
            register_parameter(name, at::empty(shape, options).uniform_(-bound, bound, generator)));
    }
    const std::array<std::string, modality_count - 1> names{"news", "charts", "fundamentals",
                                                            "macro"};
    for (std::size_t i = 0; i < modalities_.size(); ++i) {
        modalities_.at(i) =
            linear(names.at(i), config_.dimensions.at(i + 1), hidden_width, generator, dtype);
    }
    fusion_ = linear("fusion", modality_count * hidden_width, feature_width, generator, dtype);
    initial_ = linear("initial", feature_width, hidden_width, generator, dtype);
    query_ = linear("query", hidden_width, hidden_width, generator, dtype);
    value_ = linear("value", read_input_width, hidden_width, generator, dtype);
    update_ = linear("update", update_input_width, hidden_width, generator, dtype);
    head_ = linear("head", hidden_width, quantile_count, generator, dtype);
    step_logit_ = register_parameter(
        "step_logit", at::full({}, std::log(initial_step / (1 - initial_step)), options));
    auto feature_generator =
        at::detail::createCPUGenerator(static_cast<uint64_t>(config_.feature_seed));
    auto key_generator = at::detail::createCPUGenerator(static_cast<uint64_t>(config_.key_seed));
    const auto width = input_width(config_);
    feature_projection_ =
        at::empty({width, feature_width}, options)
            .normal_(0, 1 / std::sqrt(static_cast<double>(width)), feature_generator);
    key_projection_ =
        at::empty({feature_width, hidden_width}, options)
            .normal_(0, 1 / std::sqrt(static_cast<double>(feature_width)), key_generator);
    refresh_representation();
    to(device);
}

Candidate::Linear Candidate::linear(const std::string& name, int64_t in, int64_t out,
                                    at::Generator& generator, at::ScalarType dtype) {
    const auto options = at::TensorOptions().dtype(dtype);
    const double bound = 1 / std::sqrt(static_cast<double>(in));
    return {register_parameter(name + "_weight",
                               at::empty({out, in}, options).uniform_(-bound, bound, generator)),
            register_parameter(name + "_bias",
                               at::empty({out}, options).uniform_(-bound, bound, generator))};
}
at::Tensor Candidate::Linear::operator()(const at::Tensor& value) const {
    return at::linear(value, weight, bias);
}
const Config& Candidate::config() const noexcept { return config_; }
std::string Candidate::representation_id() const { return representation_id_; }
MemorySnapshot Candidate::empty_memory() const {
    return snapshot(at::empty({0, hidden_width}, feature_projection_.options()),
                    at::empty({0, feature_width}, feature_projection_.options()),
                    at::empty({0}, feature_projection_.options()), at::empty({0}, at::kLong),
                    representation_id());
}
MemorySnapshot
Candidate::snapshot(const at::Tensor& keys, const at::Tensor& features,
                    // Los tipos FP e int64 se comprueban antes de copiar. Intercambiarlos falla.
                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                    const at::Tensor& returns, const at::Tensor& ids,
                    const std::string& representation_id) const {
    require(representation_id == this->representation_id(),
            "La representación de memoria no coincide");
    require(keys.defined() && keys.dim() == 2, "La memoria necesita una matriz de claves");
    const auto count = keys.size(0);
    require(count <= config_.max_episodes, "La memoria supera el presupuesto de episodios");
    check_tensor(keys, {count, hidden_width}, feature_projection_, true);
    check_tensor(features, {count, feature_width}, feature_projection_, true);
    check_tensor(returns, {count}, feature_projection_, true);
    require(ids.defined() && ids.device().is_cpu() && ids.scalar_type() == at::kLong &&
                ids.layout() == at::kStrided && ids.sizes() == at::IntArrayRef({count}),
            "Los IDs deben ser un vector int64 en CPU");
    require((ids >= 0).all().item<bool>() &&
                (count < 2 || (ids.slice(0, 1) > ids.slice(0, 0, count - 1)).all().item<bool>()),
            "Los IDs de memoria deben ser no negativos, únicos y crecientes");
    const auto norms = keys.norm(2, -1);
    require(((norms - 1).abs() <= key_tolerance).logical_or(norms == 0).all().item<bool>(),
            "Las claves deben tener norma uno o ser exactamente nulas");
    MemorySnapshot result;
    result.keys_ = keys.detach().clone();
    result.features_ = features.detach().clone();
    result.returns_ = returns.detach().clone();
    result.ids_ = ids.to(feature_projection_.device()).clone();
    result.representation_id_ = representation_id;
    return result;
}
Encoded Candidate::encode(const Inputs& inputs) const {
    require(inputs.prices.defined() && inputs.prices.dim() == 3,
            "Los precios necesitan tres dimensiones");
    const auto batch = inputs.prices.size(0);
    require(batch > 0 && batch <= config_.max_batch, "El lote está vacío o supera el presupuesto");
    check_tensor(inputs.prices, {batch, price_window, config_.dimensions.front()},
                 feature_projection_, true);
    const std::array<at::Tensor, modality_count - 1> blocks{inputs.news, inputs.charts,
                                                            inputs.fundamentals, inputs.macro};
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        check_tensor(blocks.at(i), {batch, config_.dimensions.at(i + 1)}, feature_projection_,
                     true);
    }
    require(inputs.presence.defined() && inputs.presence.layout() == at::kStrided &&
                inputs.presence.scalar_type() == at::kBool &&
                inputs.presence.device() == feature_projection_.device() &&
                inputs.presence.sizes() == at::IntArrayRef({batch, modality_count}) &&
                inputs.presence.all().item<bool>(),
            "Se requieren las cuatro modalidades y macro en cada fila");
    const auto start = at::zeros({1, batch, hidden_width}, feature_projection_.options());
    const auto price_hidden =
        std::get<1>(at::gru(inputs.prices, start, gru_, true, 1, 0., is_training(), false, true))
            .select(0, 0);
    std::vector<at::Tensor> projected{price_hidden};
    std::vector<at::Tensor> flattened{inputs.prices.flatten(1)};
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        projected.push_back(at::silu(modalities_.at(i)(blocks.at(i))));
        flattened.push_back(blocks.at(i));
    }
    const auto fused = at::silu(fusion_(at::cat(projected, -1)));
    at::Tensor features;
    at::Tensor keys;
    {
        const at::NoGradGuard guard;
        features = at::matmul(at::cat(flattened, -1), feature_projection_);
        keys = normalize(at::matmul(features, key_projection_));
    }
    require(at::isfinite(fused).all().item<bool>() && at::isfinite(features).all().item<bool>() &&
                at::isfinite(keys).all().item<bool>(),
            "La codificación excede el rango numérico");
    return {fused, features, keys};
}
at::Tensor Candidate::initial_state(const at::Tensor& fused) const {
    require(fused.defined() && fused.dim() == 2 && fused.size(0) > 0 &&
                fused.size(0) <= config_.max_batch,
            "La fusión debe ser un lote dentro del presupuesto");
    check_tensor(fused, {fused.size(0), feature_width}, feature_projection_);
    return initial_(fused);
}
Read Candidate::read(const at::Tensor& state, const MemorySnapshot& memory) const {
    require(state.defined() && state.dim() == 2 && state.size(0) > 0 &&
                state.size(0) <= config_.max_batch,
            "El estado debe ser un lote dentro del presupuesto");
    const auto batch = state.size(0);
    check_tensor(state, {batch, hidden_width}, feature_projection_);
    require(memory.representation_id_ == representation_id() &&
                memory.keys_.device() == state.device(),
            "La instantánea pertenece a otra representación o dispositivo");
    const auto count = std::min(memory.size(), config_.neighbors);
    if (count == 0) {
        return {at::zeros_like(state), at::empty({batch, 0}, state.options()),
                at::empty({batch, 0}, state.options().dtype(at::kLong)),
                at::zeros({batch, 1}, state.options().dtype(at::kBool))};
    }
    const auto query = normalize(query_(state));
    at::Tensor indices;
    {
        const at::NoGradGuard guard;
        const auto scores = at::matmul(query, memory.keys_.t());
        // Los IDs ya están ordenados. La ordenación estable conserva el desempate.
        indices = at::argsort(scores, true, -1, true).narrow(1, 0, count);
    }
    const auto flat = indices.flatten();
    const auto keys = memory.keys_.index_select(0, flat).reshape({batch, count, hidden_width});
    const auto scores = (keys * query.unsqueeze(1)).sum(-1) / config_.temperature;
    const auto weights = at::softmax(scores, -1);
    const auto features = memory.features_.index_select(0, flat);
    const auto outcomes = memory.returns_.index_select(0, flat).unsqueeze(1);
    const auto values =
        value_(at::cat({features, outcomes}, -1)).reshape({batch, count, hidden_width});
    return {(values * weights.unsqueeze(-1)).sum(1), weights,
            memory.ids_.index_select(0, flat).reshape({batch, count}),
            at::ones({batch, 1}, state.options().dtype(at::kBool))};
}
at::Tensor Candidate::refine(const at::Tensor& state, const at::Tensor& fused,
                             const Read& memory_read) const {
    require(state.defined() && state.dim() == 2 && state.size(0) > 0 &&
                state.size(0) <= config_.max_batch,
            "El estado debe ser un lote dentro del presupuesto");
    const auto batch = state.size(0);
    check_tensor(state, {batch, hidden_width}, feature_projection_);
    check_tensor(fused, {batch, feature_width}, feature_projection_);
    check_tensor(memory_read.values, {batch, hidden_width}, feature_projection_);
    require(memory_read.presence.defined() &&
                memory_read.presence.sizes() == at::IntArrayRef({batch, 1}) &&
                memory_read.presence.scalar_type() == at::kBool &&
                memory_read.presence.device() == state.device(),
            "La máscara de memoria no coincide con el estado");
    const auto joined = at::cat(
        {state, fused, memory_read.values, memory_read.presence.to(state.scalar_type())}, -1);
    return state + step_logit_.sigmoid() * update_(joined).tanh();
}
at::Tensor Candidate::quantiles(const at::Tensor& state) const {
    require(state.defined() && state.dim() == 2 && state.size(0) > 0 &&
                state.size(0) <= config_.max_batch,
            "El estado debe ser un lote dentro del presupuesto");
    check_tensor(state, {state.size(0), hidden_width}, feature_projection_);
    const auto raw = head_(state);
    const auto median = raw.select(1, median_index);
    const auto lower = median - at::softplus(raw.select(1, 1));
    const auto upper = median + at::softplus(raw.select(1, median_index + 1));
    return at::stack({lower - at::softplus(raw.select(1, 0)), lower, median, upper,
                      upper + at::softplus(raw.select(1, quantile_count - 1))},
                     1);
}
Prediction Candidate::forward(const Inputs& inputs, const MemorySnapshot& memory,
                              int64_t refinements) const {
    require(refinements == 1 || refinements == 2 || refinements == 4, "K debe ser 1, 2 o 4");
    auto encoded = encode(inputs);
    auto state = initial_state(encoded.fused);
    Read memory_read;
    for (int64_t step = 0; step < refinements; ++step) {
        memory_read = read(state, memory);
        state = refine(state, encoded.fused, memory_read);
    }
    auto output = quantiles(state);
    require(at::isfinite(output).all().item<bool>(), "La predicción contiene NaN o infinito");
    return {std::move(output), std::move(state), std::move(encoded), std::move(memory_read)};
}
} // namespace mars_titan::candidate
