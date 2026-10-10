#include "mars_titan/ppo_policy.hpp"
#include "mars_titan/ppo_objectives.hpp"
#include "mars_titan/quantile_dqn.hpp"
#include "mars_titan/klpo_terminal.hpp"
#include "mars_titan/simulation_files.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/core/grad_mode.h>
#include <ATen/ops/_cudnn_rnn_flatten_weight.h>
#include <ATen/ops/_use_cudnn_rnn_flatten_weight.h>
#include <ATen/ops/cudnn_is_acceptable.h>
#include <ATen/ops/gru.h>
#include <ATen/ops/smooth_l1_loss.h>
#include <c10/core/DeviceGuard.h>
#include <torch/nn/module.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/optim/adam.h>
#include <torch/serialize/archive.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <istream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mars_titan::learning {
namespace {
constexpr int64_t maximum_samples = 1 << 20;
constexpr int64_t maximum_controlled_rollout = 16384;
constexpr int64_t maximum_epochs = 64;
constexpr int64_t maximum_optimizer_steps = maximum_samples * maximum_epochs;
constexpr int64_t maximum_minibatch = 65536;
constexpr int64_t gru_gates = 3;
constexpr int64_t maximum_hidden_width = 1024;
constexpr std::size_t maximum_observation_width = 32768;
constexpr std::size_t maximum_memory_bytes = std::size_t{2} * 1024 * 1024 * 1024;
constexpr std::streamoff maximum_checkpoint_bytes = std::streamoff{64} * 1024 * 1024;
constexpr double normalization_epsilon = 1e-8;
constexpr double gradient_norm_order = 2;
constexpr double maximum_log_probability = 1e-6;
constexpr double maximum_loss_weight = 100;
constexpr double maximum_gradient_norm = 1e6;
constexpr uint64_t shuffle_seed_offset = 0x9e3779b97f4a7c15ULL;
constexpr uint64_t auxiliary_seed_offset = 0xd1b54a32d192ed03ULL;
constexpr int64_t maximum_auxiliary_samples = 8192;
constexpr int64_t maximum_auxiliary_updates = 64;
constexpr int64_t checkpoint_version = 2;
constexpr int64_t controller_checkpoint_version = 3;
constexpr int64_t terminal_checkpoint_version = 4;
constexpr std::string_view terminal_adam_contract = "klpo_actor_adam_zero_critic_v1";
constexpr double terminal_adam_beta1 = .9;
constexpr double terminal_adam_beta2 = .999;
constexpr double terminal_adam_epsilon = 1e-8;
// Umbral de Huber de QR-DQN. El artículo compara kappa = 0 y kappa = 1 (QR-DQN-0 y QR-DQN-1)
// y aquí se conserva la variante QR-DQN-1.
constexpr double qr_dqn_kappa = 1;
constexpr int64_t metric_count = 6;

struct RecurrentSequence {
    int64_t lane;
    int64_t start;
    int64_t length;
    int64_t history_start;
    int64_t external_length;
};

struct PrefixBatch {
    at::Tensor observations;
    at::Tensor lengths;
};

struct LossBatch {
    at::Tensor actions;
    at::Tensor old_log_probabilities;
    at::Tensor advantages;
    at::Tensor returns;
    at::Tensor old_action_weights = {};
};

at::Tensor long_tensor(std::span<const int64_t> values, const at::Device& device) {
    auto result = at::empty({static_cast<int64_t>(values.size())}, at::kLong);
    std::copy(values.begin(), values.end(), result.data_ptr<int64_t>());
    return result.to(device);
}

// Salidas de la capa final: seis logits y el valor en PPO, seis valores Q en Double DQN y
// N cuantiles por acción en QR-DQN.
int64_t output_count(const PpoArchitecture& architecture) {
    if (!architecture.double_dqn) {
        return ppo_action_count + 1;
    }
    return architecture.quantiles > 0 ? ppo_action_count * architecture.quantiles : ppo_action_count;
}

std::size_t model_parameters(std::size_t width, const PpoArchitecture& architecture) {
    const auto hidden = static_cast<std::size_t>(architecture.hidden_width);
    const auto outputs = static_cast<std::size_t>(output_count(architecture));
    const auto encoder = architecture.kind == PpoNetworkKind::mlp
        ? (width + 1) * hidden + (hidden + 1) * hidden
        : static_cast<std::size_t>(gru_gates) * hidden * (width + hidden + 2);
    return encoder + (hidden + 1) * outputs + (architecture.auxiliary ? hidden + 1 : 0);
}

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::invalid_argument(std::string(message));
    }
}

// Geometría de QR-DQN en la huella. Vacía en el resto para conservar las huellas anteriores.
std::string quantile_geometry(const PpoArchitecture& architecture) {
    if (architecture.quantiles == 0) {
        return {};
    }
    // 32 caracteres bastan para la representación más corta de cualquier double con to_chars.
    constexpr std::size_t alpha_characters = 32;
    std::array<char, alpha_characters> alpha{};
    const auto written = std::to_chars(alpha.data(), std::to_address(alpha.end()), architecture.risk_alpha);
    require(written.ec == std::errc{}, "No se pudo representar alpha en la huella");
    return ":" + std::to_string(architecture.quantiles) + ":" + std::string(alpha.data(), written.ptr);
}

void require_float_shape(const at::Tensor& tensor, at::IntArrayRef shape, const at::Device& device) {
    require(tensor.defined() && tensor.layout() == at::kStrided && tensor.sizes() == shape &&
                tensor.device() == device &&
                (tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kDouble),
            "El tensor PPO no tiene la forma, precisión o dispositivo acordados");
}

void require_finite(const at::Tensor& tensor) {
    require(tensor.numel() == 0 || at::isfinite(tensor).all().item<bool>(),
            "El tensor PPO contiene NaN o infinito");
}

void require_float(const at::Tensor& tensor, at::IntArrayRef shape, const at::Device& device) {
    require_float_shape(tensor, shape, device);
    require_finite(tensor);
}

void require_mask(const at::Tensor& mask, at::IntArrayRef shape, const at::Device& device) {
    require(mask.defined() && mask.layout() == at::kStrided && mask.sizes() == shape &&
                mask.device() == device && mask.scalar_type() == at::kBool,
            "La máscara PPO no es un tensor booleano con la forma acordada");
}

void require_resources(std::size_t width, int64_t samples, std::size_t budget,
                        const PpoArchitecture& architecture) {
    require(width > 0 && width <= maximum_observation_width && samples > 0 &&
                samples <= maximum_samples && budget <= maximum_memory_bytes,
            "El lote PPO supera los límites de dimensiones o memoria");
    constexpr std::size_t checkpoint_metadata_bytes = std::size_t{1024} * 1024;
    constexpr std::size_t auxiliary_parameter_slots = 5;
    const std::size_t parameter_slots = architecture.auxiliary ? auxiliary_parameter_slots :
        architecture.double_dqn ? 4 : 3;
    require(model_parameters(width, architecture) <=
                (static_cast<std::size_t>(maximum_checkpoint_bytes) - checkpoint_metadata_bytes) /
                    (sizeof(float) * parameter_slots),
            "La arquitectura PPO excedería la capacidad de recuperación del checkpoint");
    // Incluye parámetros, Adam, gradientes, tensores del rollout y activaciones temporales.
    const std::size_t activation_bytes = static_cast<std::size_t>(architecture.hidden_width) * 64 + 256;
    const auto bytes_per_sample = width * sizeof(float) * 3 + activation_bytes;
    const std::size_t resident_slots = architecture.double_dqn ? auxiliary_parameter_slots : 4;
    const auto model_bytes = model_parameters(width, architecture) * sizeof(float) * resident_slots;
    require(budget > model_bytes &&
                static_cast<std::size_t>(samples) <= (budget - model_bytes) / bytes_per_sample,
            "El presupuesto PPO no cubre el modelo y los tensores del lote");
}

at::Generator make_sampler(const at::Device& device, uint64_t seed) {
    if (device.is_cpu()) {
        return at::detail::createCPUGenerator(seed);
    }
#if defined(MARS_TITAN_LIBTORCH_CUDA)
    auto global = at::globalContext().defaultGenerator(device);
    const std::lock_guard lock(global.mutex());
    auto generator = global.clone();
    generator.set_current_seed(seed);
    return generator;
#else
    static_cast<void>(seed);
    throw std::invalid_argument("Este ejecutable PPO no enlaza el backend CUDA de LibTorch");
#endif
}

at::Tensor generator_state(at::Generator generator) {
    const std::lock_guard lock(generator.mutex());
    return generator.get_state();
}

void restore_generator(at::Generator& generator, const at::Tensor& state) {
    require(state.device().is_cpu() && state.scalar_type() == at::kByte && state.dim() == 1,
            "El checkpoint PPO contiene un estado aleatorio incompatible");
    const std::lock_guard lock(generator.mutex());
    generator.set_state(state);
}

class PolicyNetwork final : public torch::nn::Module {
public:
    PolicyNetwork(int64_t width, PpoArchitecture architecture, at::Generator generator)
        : architecture_(architecture) {
        const auto initialize = [&](std::string_view name, at::IntArrayRef shape, int64_t inputs) {
            const double bound = 1 / std::sqrt(static_cast<double>(inputs));
            auto tensor = at::empty(shape, at::kFloat);
            tensor.uniform_(-bound, bound, generator);
            return register_parameter(std::string(name), tensor);
        };
        const auto hidden = architecture_.hidden_width;
        if (architecture_.kind == PpoNetworkKind::mlp) {
            first_weight_ = initialize("first_weight", {hidden, width}, width);
            first_bias_ = initialize("first_bias", {hidden}, width);
            second_weight_ = initialize("second_weight", {hidden, hidden}, hidden);
            second_bias_ = initialize("second_bias", {hidden}, hidden);
        } else {
            recurrent_ = {
                initialize("weight_ih_l0", {gru_gates * hidden, width}, hidden),
                initialize("weight_hh_l0", {gru_gates * hidden, hidden}, hidden),
                initialize("bias_ih_l0", {gru_gates * hidden}, hidden),
                initialize("bias_hh_l0", {gru_gates * hidden}, hidden)};
        }
        const auto outputs = output_count(architecture_);
        output_weight_ = initialize("output_weight", {outputs, hidden}, hidden);
        output_bias_ = initialize("output_bias", {outputs}, hidden);
        if (architecture_.auxiliary) {
            auxiliary_weight_ = initialize("auxiliary_weight", {1, hidden}, hidden);
            auxiliary_bias_ = initialize("auxiliary_bias", {1}, hidden);
        }
    }

    [[nodiscard]] PpoForward forward(const at::Tensor& observations) const {
        if (architecture_.kind == PpoNetworkKind::gru) {
            const auto state = at::zeros({observations.size(0), architecture_.hidden_width}, observations.options());
            const auto result = infer(observations, state);
            return {result.logits, result.values};
        }
        if (architecture_.quantiles > 0) {
            // La puntuación de riesgo de cada acción ocupa el lugar de los valores Q.
            const auto scores = quantile_action_scores(quantiles(observations), architecture_.risk_alpha);
            return {scores, std::get<0>(scores.max(1))};
        }
        const auto output = at::linear(shared_features(observations), output_weight_, output_bias_);
        return {output.narrow(1, 0, ppo_action_count), architecture_.double_dqn ?
            std::get<0>(output.max(1)) : output.select(1, ppo_action_count)};
    }

    [[nodiscard]] at::Tensor quantiles(const at::Tensor& observations) const {
        return at::linear(shared_features(observations), output_weight_, output_bias_)
            .view({observations.size(0), ppo_action_count, architecture_.quantiles});
    }

    [[nodiscard]] PpoInference sequence(const at::Tensor& observations, const at::Tensor& state) const {
        const auto [encoded, hidden] = at::gru(observations, state.unsqueeze(0), recurrent_, true,
                                              1, 0., at::GradMode::is_enabled(), false, false);
        const auto output = at::linear(encoded, output_weight_, output_bias_);
        return {output.narrow(2, 0, ppo_action_count), output.select(2, ppo_action_count), hidden.squeeze(0)};
    }

    [[nodiscard]] PpoInference infer(const at::Tensor& observations, const at::Tensor& state) const {
        if (architecture_.kind == PpoNetworkKind::mlp) {
            const auto result = forward(observations);
            return {result.logits, result.values, state};
        }
        auto result = sequence(observations.unsqueeze(0), state);
        result.logits = result.logits.squeeze(0);
        result.values = result.values.squeeze(0);
        return result;
    }

    [[nodiscard]] at::Tensor prefix_state(const PrefixBatch& prefix) const {
        const auto& observations = prefix.observations;
        const auto& lengths = prefix.lengths;
        auto state = at::zeros({observations.size(1), architecture_.hidden_width}, observations.options());
        if (observations.size(0) == 0) {
            return state;
        }
        const auto [encoded, hidden] = at::gru(observations, state.unsqueeze(0), recurrent_, true,
                                              1, 0., at::GradMode::is_enabled(), false, false);
        static_cast<void>(hidden);
        const auto last = (lengths - 1).clamp_min(0).view({-1, 1, 1})
                              .expand({-1, 1, architecture_.hidden_width});
        state = encoded.transpose(0, 1).gather(1, last).squeeze(1);
        return at::where((lengths > 0).unsqueeze(1), state, 0.);
    }

    [[nodiscard]] at::Tensor predict_auxiliary(const at::Tensor& observations) const {
        return at::linear(shared_features(observations), auxiliary_weight_, auxiliary_bias_).squeeze(1);
    }

    [[nodiscard]] std::vector<at::Tensor> policy_parameters() const {
        auto result = architecture_.kind == PpoNetworkKind::gru ? recurrent_ :
            std::vector<at::Tensor>{first_weight_, first_bias_, second_weight_, second_bias_};
        result.push_back(output_weight_);
        result.push_back(output_bias_);
        return result;
    }

    void pack_recurrent_weights() {
#if defined(MARS_TITAN_LIBTORCH_CUDA)
        if (architecture_.kind != PpoNetworkKind::gru || !recurrent_.front().is_cuda() ||
            !at::cudnn_is_acceptable(recurrent_.front()) || !at::_use_cudnn_rnn_flatten_weight()) {
            return;
        }
        const c10::DeviceGuard device_guard(recurrent_.front().device());
        const at::NoGradGuard no_grad;
        constexpr int64_t weights_per_layer = 4;
        constexpr int64_t cudnn_gru_mode = 3;
        // cuDNN conserva vistas del bloque en los mismos parámetros registrados y usados por Adam.
        static_cast<void>(at::_cudnn_rnn_flatten_weight(recurrent_, weights_per_layer,
            recurrent_.front().size(1), cudnn_gru_mode, architecture_.hidden_width, 0, 1, false, false));
#endif
    }

    [[nodiscard]] std::vector<at::Tensor> auxiliary_parameters() const {
        return {first_weight_, first_bias_, second_weight_, second_bias_, auxiliary_weight_, auxiliary_bias_};
    }

    void copy_parameters(const PolicyNetwork& source) {
        const at::NoGradGuard no_grad;
        const auto origin = source.parameters();
        auto destination = parameters();
        require(origin.size() == destination.size(), "La red objetivo no conserva la arquitectura online");
        for (std::size_t index = 0; index < origin.size(); ++index) {
            require(origin[index].sizes() == destination[index].sizes(),
                    "La red objetivo no conserva las formas de los parámetros online");
            destination[index].copy_(origin[index]);
        }
    }

    void validate(int64_t width, const at::Device& device) const {
        const auto hidden = architecture_.hidden_width;
        if (architecture_.kind == PpoNetworkKind::mlp) {
            require_float(first_weight_, {hidden, width}, device);
            require_float(first_bias_, {hidden}, device);
            require_float(second_weight_, {hidden, hidden}, device);
            require_float(second_bias_, {hidden}, device);
        } else {
            require_float(recurrent_.at(0), {gru_gates * hidden, width}, device);
            require_float(recurrent_.at(1), {gru_gates * hidden, hidden}, device);
            require_float(recurrent_.at(2), {gru_gates * hidden}, device);
            require_float(recurrent_.at(3), {gru_gates * hidden}, device);
        }
        const auto outputs = output_count(architecture_);
        require_float(output_weight_, {outputs, hidden}, device);
        require_float(output_bias_, {outputs}, device);
        if (architecture_.auxiliary) {
            require_float(auxiliary_weight_, {1, hidden}, device);
            require_float(auxiliary_bias_, {1}, device);
        }
        for (const auto& parameter : parameters()) {
            require(parameter.scalar_type() == at::kFloat,
                    "El checkpoint PPO debe conservar parámetros FP32");
        }
    }

private:
    [[nodiscard]] at::Tensor shared_features(const at::Tensor& observations) const {
        return at::linear(at::linear(observations, first_weight_, first_bias_).tanh(),
                           second_weight_, second_bias_).tanh();
    }

    PpoArchitecture architecture_;
    std::vector<at::Tensor> recurrent_;
    at::Tensor first_weight_;
    at::Tensor first_bias_;
    at::Tensor second_weight_;
    at::Tensor second_bias_;
    at::Tensor output_weight_;
    at::Tensor output_bias_;
    at::Tensor auxiliary_weight_;
    at::Tensor auxiliary_bias_;
};

constexpr std::array hyperparameter_fields = {
    std::pair{"learning_rate", &PpoHyperparameters::learning_rate},
    std::pair{"gamma", &PpoHyperparameters::gamma},
    std::pair{"gae_lambda", &PpoHyperparameters::gae_lambda},
    std::pair{"clip", &PpoHyperparameters::clip},
    std::pair{"entropy", &PpoHyperparameters::entropy},
    std::pair{"value_weight", &PpoHyperparameters::value_weight},
    std::pair{"gradient_norm", &PpoHyperparameters::gradient_norm}
};

int64_t read_integer(torch::serialize::InputArchive& archive, const std::string& key) {
    c10::IValue value;
    archive.read(key, value);
    require(value.isInt(), "El checkpoint PPO contiene un metadato entero incompatible");
    return value.toInt();
}

PpoHyperparameters read_hyperparameters(torch::serialize::InputArchive& archive) {
    PpoHyperparameters parameters;
    for (const auto& [name, field] : hyperparameter_fields) {
        c10::IValue value;
        archive.read(name, value);
        require(value.isDouble(), "El checkpoint PPO contiene un hiperparámetro incompatible");
        parameters.*field = value.toDouble();
    }
    parameters.epochs = read_integer(archive, "epochs");
    parameters.minibatch_size = read_integer(archive, "minibatch_size");
    parameters.validate();
    return parameters;
}

constexpr std::array objective_fields{
    std::pair{"target_kl", &PpoObjectiveConfig::target_kl},
    std::pair{"beta_initial", &PpoObjectiveConfig::beta_initial},
    std::pair{"beta_min", &PpoObjectiveConfig::beta_min},
    std::pair{"beta_max", &PpoObjectiveConfig::beta_max}};
constexpr std::array controller_integers{
    std::pair{"completed_rollouts", &PpoControllerState::completed_rollouts},
    std::pair{"optimizer_steps", &PpoControllerState::optimizer_steps},
    std::pair{"valid_rows", &PpoControllerState::valid_rows},
    std::pair{"completed_epochs", &PpoControllerState::completed_epochs},
    std::pair{"skipped_epochs", &PpoControllerState::skipped_epochs}};

double read_double(torch::serialize::InputArchive& archive, const char* key) {
    c10::IValue value;
    archive.read(key, value);
    require(value.isDouble(), "El controlador contiene un campo real de otro tipo");
    return value.toDouble();
}

void write_terminal_options(torch::serialize::OutputArchive& archive,
                            const PpoTerminalAdamOptions& options) {
    torch::serialize::OutputArchive state;
    state.write("contract", c10::IValue(std::string(terminal_adam_contract)));
    state.write("learning_rate", c10::IValue(options.learning_rate));
    state.write("gradient_norm", c10::IValue(options.gradient_norm));
    state.write("beta1", c10::IValue(terminal_adam_beta1));
    state.write("beta2", c10::IValue(terminal_adam_beta2));
    state.write("epsilon", c10::IValue(terminal_adam_epsilon));
    state.write("weight_decay", c10::IValue(0.));
    state.write("amsgrad", c10::IValue(false));
    archive.write("terminal_adam", state);
}

PpoTerminalAdamOptions read_terminal_options(torch::serialize::InputArchive& archive) {
    torch::serialize::InputArchive state;
    archive.read("terminal_adam", state);
    constexpr std::size_t expected_fields = 8;
    require(state.keys().size() == expected_fields, "El contrato Adam terminal contiene campos ajenos");
    c10::IValue contract, amsgrad;
    state.read("contract", contract);
    state.read("amsgrad", amsgrad);
    PpoTerminalAdamOptions result{read_double(state, "learning_rate"), read_double(state, "gradient_norm")};
    result.validate();
    require(contract.isString() && contract.toStringRef() == terminal_adam_contract &&
                read_double(state, "beta1") == terminal_adam_beta1 &&
                read_double(state, "beta2") == terminal_adam_beta2 &&
                read_double(state, "epsilon") == terminal_adam_epsilon &&
                read_double(state, "weight_decay") == 0 &&
                amsgrad.isBool() && !amsgrad.toBool(),
            "El archivo no conserva el Adam terminal cerrado");
    return result;
}

PpoObjectiveConfig read_objective(torch::serialize::InputArchive& archive) {
    c10::IValue id, sampler, kl;
    archive.read("id", id);
    archive.read("sampler", sampler);
    archive.read("kl", kl);
    require(id.isString() && sampler.isString() && sampler.toStringRef() == ppo_sampler_contract &&
                kl.isString() && kl.toStringRef() == ppo_kl_contract,
            "El checkpoint no conserva los contratos de objetivo y muestreo");
    PpoObjectiveConfig result;
    result.kind = objective_kind(id.toStringRef());
    for (const auto& [name, field] : objective_fields) { result.*field = read_double(archive, name); }
    result.validate();
    require(result.enabled(), "La versión nueva necesita un objetivo explícito");
    return result;
}

PpoControllerState read_controller(torch::serialize::InputArchive& archive) {
    PpoControllerState result;
    result.beta = read_double(archive, "beta");
    result.last_beta = read_double(archive, "last_beta");
    for (const auto& [name, field] : controller_integers) { result.*field = read_integer(archive, name); }
    c10::IValue kl, exceeded;
    archive.read("full_kl", kl);
    archive.read("threshold_exceeded", exceeded);
    require((kl.isNone() || kl.isDouble()) && exceeded.isBool(),
            "La medición guardada del controlador tiene tipos incompatibles");
    if (kl.isDouble()) { result.full_kl = kl.toDouble(); }
    result.threshold_exceeded = exceeded.toBool();
    return result;
}

void write_controller(torch::serialize::OutputArchive& archive,
                      const PpoObjectiveConfig& config, const PpoControllerState& state) {
    torch::serialize::OutputArchive objective, controller;
    objective.write("id", c10::IValue(std::string(objective_id(config.kind))));
    objective.write("sampler", c10::IValue(std::string(ppo_sampler_contract)));
    objective.write("kl", c10::IValue(std::string(ppo_kl_contract)));
    for (const auto& [name, field] : objective_fields) { objective.write(name, c10::IValue(config.*field)); }
    controller.write("beta", c10::IValue(state.beta));
    controller.write("last_beta", c10::IValue(state.last_beta));
    for (const auto& [name, field] : controller_integers) { controller.write(name, c10::IValue(state.*field)); }
    controller.write("full_kl", state.full_kl ? c10::IValue(*state.full_kl) : c10::IValue{});
    controller.write("threshold_exceeded", c10::IValue(state.threshold_exceeded));
    archive.write("objective", objective);
    archive.write("controller", controller);
}

void validate_stream(std::istream& source) {
    const auto position = source.tellg();
    require(position >= 0, "El checkpoint PPO requiere un flujo binario con posición consultable");
    source.seekg(0, std::ios::end);
    const auto length = source.tellg();
    source.seekg(position);
    require(source.good() && length > position && length - position <= maximum_checkpoint_bytes,
            "El checkpoint PPO está vacío o supera el límite de lectura");
}

void validate_gae(const PpoRollout& rollout) {
    require(rollout.rewards.defined() && rollout.rewards.dim() == 2 &&
                rollout.rewards.numel() > 0 && rollout.rewards.numel() <= maximum_samples,
            "GAE necesita un lote [T,N] no vacío y acotado");
    const auto shape = rollout.rewards.sizes();
    const auto device = rollout.rewards.device();
    require_float(rollout.rewards, shape, device);
    require_float(rollout.old_values, shape, device);
    require_float(rollout.next_values, shape, device);
    require_mask(rollout.reward_valid, shape, device);
    require_mask(rollout.terminated, shape, device);
    require_mask(rollout.truncated, shape, device);
    if (rollout.episode_starts.defined()) {
        require_mask(rollout.episode_starts, shape, device);
    }
}

std::vector<RecurrentSequence> recurrent_sequences(const PpoRollout& rollout) {
    const auto starts = rollout.episode_starts.cpu().contiguous();
    const auto valid = rollout.reward_valid.cpu().contiguous();
    const auto ended = (rollout.terminated | rollout.truncated).cpu().contiguous();
    const auto lengths = rollout.prefix_lengths.cpu().contiguous();
    const auto start_values = starts.accessor<bool, 2>();
    const auto valid_values = valid.accessor<bool, 2>();
    const auto end_values = ended.accessor<bool, 2>();
    const auto prefix_lengths = lengths.accessor<int64_t, 1>();
    std::vector<RecurrentSequence> sequences;
    for (int64_t lane = 0; lane < starts.size(1); ++lane) {
        const auto prefix = prefix_lengths[lane];
        require(prefix >= 0 && prefix <= rollout.prefix_observations.size(0) &&
                    (start_values[0][lane] ? prefix == 0 : prefix > 0),
                "La GRU necesita el prefijo completo o un inicio de episodio explícito");
        int64_t history_start = 0;
        int64_t external_length = prefix;
        for (int64_t time = 0; time < starts.size(0);) {
            if (start_values[time][lane]) {
                history_start = time;
                external_length = 0;
            }
            const auto candidate_end = std::min(time + ppo_sequence_length, starts.size(0));
            int64_t end = time + 1;
            while (end < candidate_end && !start_values[end][lane]) {
                ++end;
            }
            require(external_length + end - history_start <= ppo_maximum_history,
                    "El episodio GRU supera 256 observaciones sin reinicio");
            bool has_valid = false;
            for (int64_t step = time; step < end; ++step) {
                has_valid = has_valid || valid_values[step][lane];
                require(step + 1 == starts.size(0) || !end_values[step][lane] ||
                            start_values[step + 1][lane],
                        "La GRU necesita reiniciar el estado tras terminar o truncar un episodio");
            }
            if (has_valid) {
                sequences.push_back({lane, time, end - time, history_start, external_length});
            }
            time = end;
        }
    }
    return sequences;
}

PpoUpdateStats update_statistics(at::Tensor totals, PpoUpdateStats result, int64_t epochs) {
    totals = (totals / static_cast<double>(result.valid_transitions * epochs)).cpu().contiguous();
    const auto metrics = totals.accessor<double, 1>();
    result.policy_loss = metrics[0];
    result.value_loss = metrics[1];
    result.entropy = metrics[2];
    result.approximate_kl = metrics[3];
    result.clip_fraction = metrics[4];
    constexpr int64_t gradient_index = 5;
    result.gradient_norm = metrics[gradient_index];
    return result;
}

std::size_t adam_steps(const torch::optim::Adam& optimizer) {
    std::optional<int64_t> common;
    for (const auto& [parameter, stored] : optimizer.state()) {
        static_cast<void>(parameter);
        const auto* state = dynamic_cast<const torch::optim::AdamParamState*>(stored.get());
        require(state != nullptr && state->step() > 0 && state->step() <= maximum_optimizer_steps &&
                    (!common || *common == state->step()),
                "El checkpoint PPO no conserva un paso positivo y común de Adam");
        common = state->step();
    }
    return static_cast<std::size_t>(common.value_or(0));
}
}
struct PpoPolicy::Impl {
    std::size_t width;
    PpoHyperparameters parameters;
    std::string device_name;
    at::Device tensor_device;
    std::size_t budget;
    PpoArchitecture architecture;
    PpoObjectiveConfig objective;
    PpoControllerState controller;
    PolicyNetwork network;
    std::unique_ptr<PolicyNetwork> target;
    std::size_t dqn_environment_step = 0;
    std::size_t target_sync_step = 0;
    std::size_t target_interval = 0;
    std::unique_ptr<torch::optim::Adam> optimizer;
    std::unique_ptr<torch::optim::Adam> auxiliary_optimizer;
    std::unique_ptr<torch::optim::Adam> terminal_optimizer;
    PpoTerminalAdamOptions terminal_options;
    bool terminal_failed = false;
    at::Generator sampler;
    at::Generator shuffler;
    at::Generator auxiliary_rng;

    Impl(std::size_t observation_width, PpoHyperparameters hyperparameters, uint64_t seed,
         std::string_view device, std::size_t memory_budget, PpoArchitecture selected_architecture,
         PpoObjectiveConfig selected_objective)
        : width(observation_width), parameters(hyperparameters), device_name(device),
          tensor_device(device_name), budget(memory_budget), architecture(selected_architecture),
          objective(selected_objective),
          network(static_cast<int64_t>(width), architecture, at::detail::createCPUGenerator(seed)),
          sampler(make_sampler(tensor_device, seed)),
          shuffler(at::detail::createCPUGenerator(seed ^ shuffle_seed_offset)) {
        network.to(tensor_device, at::kFloat);
        controller.beta = objective.beta_initial;
        controller.last_beta = objective.beta_initial;
        network.pack_recurrent_weights();
        optimizer = std::make_unique<torch::optim::Adam>(
            network.policy_parameters(), torch::optim::AdamOptions(parameters.learning_rate));
        if (architecture.auxiliary) {
            auxiliary_optimizer = std::make_unique<torch::optim::Adam>(
                network.auxiliary_parameters(), torch::optim::AdamOptions(parameters.learning_rate));
            auxiliary_rng = at::detail::createCPUGenerator(seed ^ auxiliary_seed_offset);
        }
        if (architecture.double_dqn) {
            target = std::make_unique<PolicyNetwork>(static_cast<int64_t>(width), architecture,
                                                     at::detail::createCPUGenerator(seed));
            target->to(tensor_device, at::kFloat);
            for (const auto& parameter : target->parameters()) {
                parameter.set_requires_grad(false);
            }
        }
    }

    void validate_observations(const at::Tensor& observations, bool allow_cpu = false) const {
        require(!terminal_failed, "El actor necesita recuperar un checkpoint confirmado");
        require(observations.defined() && observations.dim() == 2 &&
                    observations.size(1) == static_cast<int64_t>(width) &&
                    observations.scalar_type() == at::kFloat,
                "La política PPO necesita observaciones FP32 [N,D]");
        require_resources(width, observations.size(0), budget, architecture);
        const auto expected_device = allow_cpu && observations.device().is_cpu() ? observations.device() : tensor_device;
        require_float(observations, observations.sizes(), expected_device);
    }

    [[nodiscard]] std::size_t update_budget(const PpoRollout& rollout) const {
        if (!objective.enabled()) { return budget; }
        require(rollout.rewards.defined() && rollout.rewards.numel() > 0 &&
                    rollout.rewards.numel() <= maximum_controlled_rollout,
                "El controlador admite hasta 16384 transiciones por rollout");
        constexpr auto row_bytes = static_cast<std::size_t>(ppo_action_count) *
                                   (2 * sizeof(float) + 16 * sizeof(double));
        const auto rows = static_cast<std::size_t>(rollout.rewards.numel());
        require(rows <= budget / row_bytes && rows * row_bytes < budget,
                "Los pesos y temporales del controlador superan el presupuesto");
        return budget - rows * row_bytes;
    }

    void validate_rollout(const PpoRollout& rollout, const at::Device& expected_device) const {
        require(rollout.rewards.defined() && rollout.rewards.device() == expected_device,
                "El rollout PPO no pertenece al dispositivo de entrada acordado");
        validate_gae(rollout);
        const auto shape = rollout.rewards.sizes();
        require(rollout.observations.defined() && rollout.observations.dim() == 3 &&
                    rollout.observations.size(0) == shape[0] &&
                    rollout.observations.size(1) == shape[1] &&
                    rollout.observations.size(2) == static_cast<int64_t>(width) &&
                    rollout.observations.device() == expected_device,
                "Las observaciones PPO no corresponden a los pasos y carriles del rollout");
        require_resources(width, rollout.rewards.numel(), update_budget(rollout), architecture);
        validate_observations(rollout.observations.flatten(0, 1), expected_device.is_cpu());
        require_float(rollout.old_log_probabilities, shape, expected_device);
        require(rollout.rewards.device() == expected_device && rollout.actions.defined() &&
                    rollout.actions.layout() == at::kStrided && rollout.actions.sizes() == shape &&
                    rollout.actions.device() == expected_device && rollout.actions.scalar_type() == at::kLong,
                "Las acciones PPO no corresponden al dispositivo o a la forma del rollout");
        require(((rollout.actions >= 0) & (rollout.actions < ppo_action_count)).all().item<bool>(),
                "La acción PPO está fuera del intervalo de seis acciones");
        require((rollout.old_log_probabilities <= maximum_log_probability).all().item<bool>(),
                "Las probabilidades anteriores PPO exceden el intervalo numérico admitido");
        require(rollout.old_action_weights.defined() == objective.enabled(),
                "El rollout no conserva la distribución exigida por el objetivo PPO");
        if (objective.enabled()) {
            require(rollout.old_action_weights.sizes() == at::IntArrayRef({shape[0], shape[1], ppo_action_count}),
                    "Los pesos históricos no corresponden al rollout");
            validate_ppo_behavior(rollout.old_action_weights.flatten(0, 1), rollout.actions.flatten(),
                                  rollout.old_log_probabilities.flatten(), rollout.reward_valid.flatten());
        }
        if (architecture.kind == PpoNetworkKind::gru) {
            require_mask(rollout.episode_starts, shape, expected_device);
            require(shape[0] <= ppo_maximum_history && rollout.prefix_observations.defined() &&
                        rollout.prefix_observations.dim() == 3 &&
                        rollout.prefix_observations.size(0) <= ppo_maximum_history &&
                        rollout.prefix_observations.size(1) == shape[1] &&
                        rollout.prefix_observations.size(2) == static_cast<int64_t>(width) &&
                        rollout.prefix_observations.scalar_type() == at::kFloat,
                    "La GRU necesita un rollout y un prefijo FP32 de hasta 256 pasos por carril");
            require_resources(width, rollout.rewards.numel() +
                rollout.prefix_observations.size(0) * shape[1], update_budget(rollout), architecture);
            require_float(rollout.prefix_observations, rollout.prefix_observations.sizes(), expected_device);
            require(rollout.prefix_lengths.defined() && rollout.prefix_lengths.layout() == at::kStrided &&
                        rollout.prefix_lengths.sizes() == at::IntArrayRef({shape[1]}) &&
                        rollout.prefix_lengths.device() == expected_device &&
                        rollout.prefix_lengths.scalar_type() == at::kLong,
                    "La GRU necesita longitudes int64 para los prefijos de cada carril");
        }
    }

    [[nodiscard]] at::Tensor optimize(const PpoForward& output, const LossBatch& batch) {
        const auto& actions = batch.actions;
        const auto& previous = batch.old_log_probabilities;
        const auto& advantages = batch.advantages;
        const auto& returns = batch.returns;
        const auto log_probabilities = output.logits.log_softmax(-1);
        const auto selected = log_probabilities.gather(1, actions.unsqueeze(1)).squeeze(1);
        const auto policy_loss = objective.kind == PpoObjectiveKind::kl_penalty_adaptive
            ? ppo_penalized_objective(log_probabilities, batch.old_action_weights, actions, previous,
                                      advantages, controller.beta).mean()
            : -ppo_clipped_objective(selected, previous, advantages, parameters.clip).mean();
        const auto value_loss = (output.values - returns).square().mean();
        const auto entropy = -(log_probabilities.exp() * log_probabilities).sum(-1).mean();
        const auto loss = policy_loss + parameters.value_weight * value_loss - parameters.entropy * entropy;
        require(at::isfinite(loss).item<bool>(), "La pérdida PPO no es finita");
        optimizer->zero_grad();
        loss.backward();
        const double gradient = torch::nn::utils::clip_grad_norm_(
            network.policy_parameters(), parameters.gradient_norm, gradient_norm_order, true);
        optimizer->step();
        const at::NoGradGuard no_grad;
        const auto log_ratio = selected.to(at::kDouble) - previous;
        const auto ratio = log_ratio.exp();
        const auto approximate_kl = (ratio - 1 - log_ratio).mean();
        const auto clip_fraction = ((ratio - 1).abs() > parameters.clip).to(at::kFloat).mean();
        return at::stack({policy_loss, value_loss, entropy, approximate_kl, clip_fraction,
                          at::full({}, gradient, previous.options())}).detach().to(at::kDouble) *
               static_cast<double>(actions.numel());
    }

    [[nodiscard]] at::Tensor reconstruct_prefix(const PpoRollout& rollout,
                                                 std::span<const RecurrentSequence> sequences) const {
        const at::NoGradGuard no_grad;
        std::vector<int64_t> lengths;
        lengths.reserve(sequences.size());
        for (const auto& sequence : sequences) {
            lengths.push_back(sequence.external_length + sequence.start - sequence.history_start);
        }
        const auto maximum_length = *std::max_element(lengths.begin(), lengths.end());
        auto prefix = at::zeros({maximum_length, static_cast<int64_t>(sequences.size()),
                                 static_cast<int64_t>(width)}, rollout.observations.options());
        for (std::size_t index = 0; index < sequences.size(); ++index) {
            const auto& sequence = sequences[index];
            auto destination = prefix.select(1, static_cast<int64_t>(index));
            if (sequence.external_length > 0) {
                destination.narrow(0, 0, sequence.external_length).copy_(
                    rollout.prefix_observations.select(1, sequence.lane).narrow(0, 0, sequence.external_length));
            }
            const auto internal_length = sequence.start - sequence.history_start;
            if (internal_length > 0) {
                destination.narrow(0, sequence.external_length, internal_length).copy_(
                    rollout.observations.select(1, sequence.lane).narrow(0, sequence.history_start, internal_length));
            }
        }
        return network.prefix_state({prefix, long_tensor(lengths, tensor_device)});
    }

    [[nodiscard]] std::pair<PpoForward, at::Tensor> recurrent_output(const PpoRollout& rollout,
        std::span<const RecurrentSequence> sequences) const {
        std::vector<int64_t> starts;
        std::vector<int64_t> lanes;
        std::vector<int64_t> lengths;
        for (const auto& sequence : sequences) {
            starts.push_back(sequence.start);
            lanes.push_back(sequence.lane);
            lengths.push_back(sequence.length);
        }
        const auto time = at::arange(ppo_sequence_length,
            at::TensorOptions().dtype(at::kLong).device(tensor_device)).unsqueeze(1);
        const auto positions = ((time + long_tensor(starts, tensor_device)).clamp_max(rollout.rewards.size(0) - 1) *
            rollout.rewards.size(1) + long_tensor(lanes, tensor_device)).flatten();
        const auto present = time < long_tensor(lengths, tensor_device);
        const auto valid = present.flatten() & rollout.reward_valid.flatten().index_select(0, positions);
        const auto chosen = valid.nonzero().squeeze(1);
        auto observations = rollout.observations.detach().flatten(0, 1).index_select(0, positions)
                                .reshape({ppo_sequence_length, static_cast<int64_t>(sequences.size()),
                                          static_cast<int64_t>(width)});
        observations = at::where(present.unsqueeze(2), observations, 0.);
        const auto initial = reconstruct_prefix(rollout, sequences);
        const auto sequence_output = network.sequence(observations, initial);
        const PpoForward output{sequence_output.logits.flatten(0, 1).index_select(0, chosen),
                                sequence_output.values.flatten().index_select(0, chosen)};
        const auto indices = positions.index_select(0, chosen);
        return {output, indices};
    }

    [[nodiscard]] at::Tensor recurrent_minibatch(const PpoRollout& rollout,
        std::span<const RecurrentSequence> sequences, const at::Tensor& normalized,
        const at::Tensor& returns) {
        const auto [output, indices] = recurrent_output(rollout, sequences);
        return optimize(output, {rollout.actions.flatten().index_select(0, indices),
            rollout.old_log_probabilities.detach().flatten().index_select(0, indices).to(at::kDouble),
            normalized.index_select(0, indices), returns.index_select(0, indices),
            objective.enabled() ? rollout.old_action_weights.flatten(0, 1).index_select(0, indices) : at::Tensor{}});
    }

    [[nodiscard]] std::optional<double> measure_kl(const PpoRollout& rollout) const {
        const at::NoGradGuard no_grad;
        const auto indices = rollout.reward_valid.flatten().nonzero().squeeze(1);
        if (indices.numel() == 0) { return std::nullopt; }
        auto total = at::zeros({}, at::TensorOptions().dtype(at::kDouble).device(tensor_device));
        int64_t measured = 0;
        const auto accumulate = [&](const at::Tensor& logits, const at::Tensor& rows) {
            const auto historical = rollout.old_action_weights.flatten(0, 1).index_select(0, rows).to(tensor_device);
            const auto current = ppo_behavior_log_probabilities(logits.log_softmax(1).exp());
            const auto previous = ppo_behavior_log_probabilities(historical);
            total.add_(ppo_categorical_kl(current, previous).sum());
            measured += rows.numel();
        };
        if (architecture.kind == PpoNetworkKind::gru) {
            const auto sequences = recurrent_sequences(rollout);
            const auto per_minibatch = static_cast<std::size_t>(parameters.minibatch_size / ppo_sequence_length);
            const auto view = std::span(sequences);
            for (std::size_t offset = 0; offset < sequences.size(); offset += per_minibatch) {
                const auto count = std::min(per_minibatch, sequences.size() - offset);
                int64_t longest = 0;
                for (const auto& sequence : view.subspan(offset, count)) {
                    longest = std::max(longest, sequence.external_length + sequence.start - sequence.history_start);
                }
                require_resources(width, rollout.rewards.numel() +
                    rollout.prefix_observations.size(0) * rollout.rewards.size(1) +
                    (longest + ppo_sequence_length) * static_cast<int64_t>(count), update_budget(rollout), architecture);
                const auto [output, rows] = recurrent_output(rollout, view.subspan(offset, count));
                accumulate(output.logits, rows);
            }
        } else {
            for (int64_t offset = 0; offset < indices.numel(); offset += parameters.minibatch_size) {
                const auto rows = indices.narrow(0, offset, std::min(parameters.minibatch_size, indices.numel() - offset));
                const auto observations = rollout.observations.flatten(0, 1).index_select(0, rows).to(tensor_device);
                accumulate(network.forward(observations).logits, rows);
            }
        }
        require(measured == indices.numel(), "La medición KL perdió filas válidas");
        const auto mean = total.item<double>() / static_cast<double>(measured);
        static_cast<void>(epoch_decision(objective, 1, parameters.epochs, measured, mean));
        return mean;
    }

    [[nodiscard]] bool measured_epoch(const PpoRollout& rollout, int64_t epoch, PpoUpdateStats& result) const {
        result.completed_epochs = epoch + 1;
        result.full_kl = measure_kl(rollout);
        const auto decision = epoch_decision(objective, result.completed_epochs, parameters.epochs,
                                             result.valid_transitions, result.full_kl);
        result.threshold_exceeded = decision.threshold_exceeded;
        result.skipped_epochs = decision.stop_remaining_epochs ? parameters.epochs - result.completed_epochs : 0;
        return decision.stop_remaining_epochs;
    }

    void confirm_controller(const PpoUpdateStats& result) {
        auto next = controller;
        next.last_beta = controller.beta;
        ++next.completed_rollouts;
        next.optimizer_steps = static_cast<int64_t>(optimizer_step_count());
        next.valid_rows = result.valid_transitions;
        next.completed_epochs = result.completed_epochs;
        next.skipped_epochs = result.skipped_epochs;
        next.full_kl = result.full_kl;
        next.threshold_exceeded = result.threshold_exceeded;
        if (objective.kind == PpoObjectiveKind::kl_penalty_adaptive) {
            next.beta = next_beta(objective, controller.beta, result.full_kl, result.valid_transitions);
        }
        next.validate(objective, next.optimizer_steps, parameters.epochs);
        controller = next;
    }

    [[nodiscard]] PpoUpdateStats update_feedforward(const PpoRollout& rollout,
        const PpoAdvantages& gae, const at::Tensor& indices) {
        PpoUpdateStats result;
        result.valid_transitions = indices.numel();
        const auto expected_steps = ((result.valid_transitions + parameters.minibatch_size - 1) /
                                     parameters.minibatch_size) * parameters.epochs;
        require(optimizer_step_count() <= static_cast<std::size_t>(maximum_optimizer_steps - expected_steps),
                "La actualización PPO excedería el límite de pasos de Adam");
        const auto observations = rollout.observations.detach().flatten(0, 1).index_select(0, indices).to(tensor_device);
        const auto actions = rollout.actions.flatten().index_select(0, indices).to(tensor_device);
        const auto old_log_probabilities = rollout.old_log_probabilities.detach().flatten()
                                              .index_select(0, indices).to(tensor_device).to(at::kDouble);
        const auto old_weights = objective.enabled()
            ? rollout.old_action_weights.detach().flatten(0, 1).index_select(0, indices).to(tensor_device) : at::Tensor{};
        const auto advantages = gae.advantages.flatten().index_select(0, indices).to(tensor_device).to(at::kFloat);
        require_float(advantages, actions.sizes(), tensor_device);
        const auto normalized = ((advantages - advantages.mean()) /
                                  advantages.std(false).clamp_min(normalization_epsilon));
        const auto returns = gae.returns.flatten().index_select(0, indices).to(tensor_device).to(at::kFloat);
        require_float(normalized, actions.sizes(), tensor_device);
        require_float(returns, actions.sizes(), tensor_device);
        auto totals = at::zeros({metric_count}, observations.options().dtype(at::kDouble));
        const at::AutoGradMode enable_grad(true);
        for (int64_t epoch = 0; epoch < parameters.epochs; ++epoch) {
            const auto order = at::randperm(result.valid_transitions, shuffler,
                                           at::TensorOptions().dtype(at::kLong)).to(tensor_device);
            for (int64_t offset = 0; offset < result.valid_transitions; offset += parameters.minibatch_size) {
                const auto count = std::min(parameters.minibatch_size, result.valid_transitions - offset);
                const auto minibatch = order.narrow(0, offset, count);
                const auto output = network.forward(observations.index_select(0, minibatch));
                totals.add_(optimize(output, {actions.index_select(0, minibatch),
                    old_log_probabilities.index_select(0, minibatch), normalized.index_select(0, minibatch),
                    returns.index_select(0, minibatch), objective.enabled()
                        ? old_weights.index_select(0, minibatch) : at::Tensor{}}));
                ++result.minibatches;
            }
            if (objective.enabled() && measured_epoch(rollout, epoch, result)) { break; }
        }
        return update_statistics(totals, result, objective.enabled() ? result.completed_epochs : parameters.epochs);
    }

    [[nodiscard]] PpoUpdateStats update_recurrent(const PpoRollout& rollout) {
        const auto sequences = recurrent_sequences(rollout);
        const auto indices = rollout.reward_valid.flatten().nonzero().squeeze(1);
        PpoUpdateStats result;
        result.valid_transitions = indices.numel();
        if (result.valid_transitions == 0) {
            return result;
        }
        const auto per_minibatch = parameters.minibatch_size / ppo_sequence_length;
        const auto sequence_count = static_cast<int64_t>(sequences.size());
        int64_t maximum_prefix = 0;
        for (const auto& sequence : sequences) {
            maximum_prefix = std::max(maximum_prefix,
                sequence.external_length + sequence.start - sequence.history_start);
        }
        const auto minibatch_samples = (maximum_prefix + ppo_sequence_length) *
                                       std::min(per_minibatch, sequence_count);
        require_resources(width, rollout.rewards.numel() +
            rollout.prefix_observations.size(0) * rollout.rewards.size(1) + minibatch_samples,
            update_budget(rollout), architecture);
        const auto expected_steps = ((sequence_count + per_minibatch - 1) / per_minibatch) * parameters.epochs;
        require(optimizer_step_count() <= static_cast<std::size_t>(maximum_optimizer_steps - expected_steps),
                "La actualización GRU excedería el límite de pasos de Adam");
        const auto gae = ppo_gae(rollout, parameters);
        const auto advantages = gae.advantages.flatten().index_select(0, indices).to(at::kFloat);
        const auto selected_returns = gae.returns.flatten().index_select(0, indices).to(at::kFloat);
        require_float(advantages, indices.sizes(), tensor_device);
        require_float(selected_returns, indices.sizes(), tensor_device);
        const auto normalized_values = (advantages - advantages.mean()) /
                                        advantages.std(false).clamp_min(normalization_epsilon);
        require_float(normalized_values, indices.sizes(), tensor_device);
        auto normalized = at::zeros({rollout.rewards.numel()}, rollout.observations.options());
        auto returns = at::zeros_like(normalized);
        normalized.index_copy_(0, indices, normalized_values);
        returns.index_copy_(0, indices, selected_returns);
        auto totals = at::zeros({metric_count}, rollout.rewards.options().dtype(at::kDouble));
        const at::AutoGradMode enable_grad(true);
        for (int64_t epoch = 0; epoch < parameters.epochs; ++epoch) {
            const auto order = at::randperm(sequence_count, shuffler, at::TensorOptions().dtype(at::kLong));
            const auto ordered = order.accessor<int64_t, 1>();
            for (int64_t offset = 0; offset < sequence_count; offset += per_minibatch) {
                const auto count = std::min(per_minibatch, sequence_count - offset);
                std::vector<RecurrentSequence> selected;
                selected.reserve(static_cast<std::size_t>(count));
                for (int64_t index = offset; index < offset + count; ++index) {
                    selected.push_back(sequences.at(static_cast<std::size_t>(ordered[index])));
                }
                totals.add_(recurrent_minibatch(rollout, selected, normalized, returns));
                ++result.minibatches;
            }
            if (objective.enabled() && measured_epoch(rollout, epoch, result)) { break; }
        }
        return update_statistics(totals, result, objective.enabled() ? result.completed_epochs : parameters.epochs);
    }

    void validate_optimizer(torch::optim::Adam& selected_optimizer) {
        require(selected_optimizer.param_groups().size() == 1,
                "El checkpoint PPO altera los grupos del optimizador");
        const auto* options = dynamic_cast<const torch::optim::AdamOptions*>(
            &selected_optimizer.param_groups().front().options());
        require(options != nullptr && *options == torch::optim::AdamOptions(parameters.learning_rate),
                "El checkpoint PPO altera las opciones de Adam");
        const auto& tensors = selected_optimizer.param_groups().front().params();
        require(selected_optimizer.state().empty() || selected_optimizer.state().size() == tensors.size(),
                "El checkpoint PPO contiene un estado parcial de Adam");
        static_cast<void>(adam_steps(selected_optimizer));
        for (const auto& parameter : tensors) {
            if (selected_optimizer.state().empty()) {
                break;
            }
            const auto found = selected_optimizer.state().find(parameter.unsafeGetTensorImpl());
            require(found != selected_optimizer.state().end(), "Falta el estado de Adam de un parámetro PPO");
            auto* state = dynamic_cast<torch::optim::AdamParamState*>(found->second.get());
            require(state != nullptr && state->step() >= 0,
                    "El checkpoint PPO contiene un paso de Adam incompatible");
            state->exp_avg(state->exp_avg().to(tensor_device));
            state->exp_avg_sq(state->exp_avg_sq().to(tensor_device));
            require_float(state->exp_avg(), parameter.sizes(), tensor_device);
            require_float(state->exp_avg_sq(), parameter.sizes(), tensor_device);
            require(state->exp_avg().scalar_type() == at::kFloat &&
                        state->exp_avg_sq().scalar_type() == at::kFloat &&
                        (state->exp_avg_sq() >= 0).all().item<bool>(),
                    "El checkpoint PPO contiene momentos de Adam inválidos");
        }
    }

    void validate_terminal_optimizer() {
        require(terminal_optimizer && !terminal_failed && !objective.enabled() &&
                    !architecture.auxiliary && !architecture.double_dqn && optimizer->state().empty(),
                "El actor terminal no conserva su aislamiento o necesita recuperación");
        terminal_options.validate();
        require(terminal_options.learning_rate == parameters.learning_rate,
                "La tasa del optimizador terminal no corresponde al actor");
        validate_optimizer(*terminal_optimizer);
        const auto expected = network.policy_parameters();
        const auto& actual = terminal_optimizer->param_groups().front().params();
        require(actual.size() == expected.size(), "El Adam terminal no corresponde a sus parámetros");
        for (std::size_t index = 0; index < expected.size(); ++index) {
            require(actual[index].is_same(expected[index]), "El optimizador contiene un parámetro ajeno");
            if (index + 2 >= expected.size() && !terminal_optimizer->state().empty()) {
                const auto& base = terminal_optimizer->state().at(actual[index].unsafeGetTensorImpl());
                const auto* state = dynamic_cast<const torch::optim::AdamParamState*>(base.get());
                require(state != nullptr &&
                            (state->exp_avg().select(0, ppo_action_count) == 0).all().item<bool>() &&
                            (state->exp_avg_sq().select(0, ppo_action_count) == 0).all().item<bool>(),
                        "El crítico contiene momentos de otra actualización");
            }
        }
    }

    [[nodiscard]] at::Tensor critic_bits() const {
        const at::NoGradGuard no_grad;
        const auto weights = network.policy_parameters();
        return at::cat({weights[weights.size() - 2].select(0, ppo_action_count).reshape({-1}),
                        weights.back().select(0, ppo_action_count).reshape({-1})})
            .view(at::kInt).clone();
    }

    [[nodiscard]] std::size_t optimizer_step_count() const {
        return adam_steps(terminal_optimizer ? *terminal_optimizer : *optimizer);
    }
};

void PpoTerminalAdamOptions::validate() const {
    require(std::isfinite(learning_rate) && learning_rate > 0 && learning_rate <= 1 &&
                std::isfinite(gradient_norm) && gradient_norm >= 0 &&
                gradient_norm <= maximum_gradient_norm,
            "Las opciones del Adam terminal no son finitas o exceden sus límites");
}
void PpoHyperparameters::validate() const {
    require(std::isfinite(learning_rate) && learning_rate > 0 && learning_rate <= 1 &&
                std::isfinite(gamma) && gamma >= 0 && gamma <= 1 &&
                std::isfinite(gae_lambda) && gae_lambda >= 0 && gae_lambda <= 1 &&
                std::isfinite(clip) && clip > 0 && clip < 1 &&
                std::isfinite(entropy) && entropy >= 0 && entropy <= maximum_loss_weight &&
                std::isfinite(value_weight) && value_weight >= 0 && value_weight <= maximum_loss_weight &&
                std::isfinite(gradient_norm) && gradient_norm > 0 && gradient_norm <= maximum_gradient_norm &&
                epochs > 0 && epochs <= maximum_epochs &&
                minibatch_size > 0 && minibatch_size <= maximum_minibatch,
            "Los hiperparámetros PPO están fuera de sus límites finitos");
}
at::Tensor ppo_clipped_objective(const at::Tensor& log_probabilities,
                                const at::Tensor& old_log_probabilities,
                                const at::Tensor& advantages, double clip) {
    require(std::isfinite(clip) && clip > 0 && clip < 1 && log_probabilities.defined() &&
                log_probabilities.numel() > 0 && log_probabilities.numel() <= maximum_samples,
            "El objetivo PPO necesita un recorte y un lote válidos");
    require_float(log_probabilities, log_probabilities.sizes(), log_probabilities.device());
    require_float(old_log_probabilities, log_probabilities.sizes(), log_probabilities.device());
    require_float(advantages, log_probabilities.sizes(), log_probabilities.device());
    const auto ratio = (log_probabilities.to(at::kDouble) -
                        old_log_probabilities.detach().to(at::kDouble)).exp();
    require(at::isfinite(ratio).all().item<bool>(), "El cociente PPO desborda su precisión");
    return at::minimum(ratio * advantages.detach(),
                       ratio.clamp(1 - clip, 1 + clip) * advantages.detach());
}
PpoAdvantages ppo_gae(const PpoRollout& rollout, const PpoHyperparameters& parameters) {
    parameters.validate();
    validate_gae(rollout);
    const at::NoGradGuard no_grad;
    const auto rewards = rollout.rewards.to(at::kDouble);
    const auto old_values = rollout.old_values.to(at::kDouble);
    const auto next_values = rollout.next_values.to(at::kDouble);
    const auto advantages = at::zeros_like(rewards);
    auto carry = at::zeros({rewards.size(1)}, rewards.options());
    for (int64_t time = rewards.size(0); time-- > 0;) {
        const auto valid = rollout.reward_valid.select(0, time);
        const auto terminated = rollout.terminated.select(0, time);
        const auto truncated = rollout.truncated.select(0, time);
        const auto bootstrap = at::where(terminated, 0., next_values.select(0, time));
        const auto delta = rewards.select(0, time) + parameters.gamma * bootstrap -
                           old_values.select(0, time);
        auto boundary = terminated | truncated;
        if (rollout.episode_starts.defined() && time + 1 < rewards.size(0)) {
            boundary = boundary | rollout.episode_starts.select(0, time + 1);
        }
        const auto continuation = at::where(boundary, 0., carry);
        carry = at::where(valid, delta + parameters.gamma * parameters.gae_lambda * continuation, 0.);
        advantages.select(0, time).copy_(carry);
    }
    return {advantages, at::where(rollout.reward_valid, advantages + old_values, 0.)};
}
PpoPolicy::PpoPolicy(std::size_t width, const PpoHyperparameters& parameters, uint64_t seed,
                     std::string_view device_name, std::size_t memory_budget, PpoArchitecture architecture,
                     PpoObjectiveConfig objective) {
    parameters.validate();
    architecture.validate();
    objective.validate();
    require(!objective.enabled() || (!architecture.auxiliary && !architecture.double_dqn),
            "El controlador PPO no admite optimizador auxiliar ni Double DQN");
    require(architecture.kind != PpoNetworkKind::gru ||
                parameters.minibatch_size % ppo_sequence_length == 0,
            "La GRU necesita minibatches de un múltiplo de 16 transiciones");
    require(device_name == "cpu" || device_name == "cuda:0", "El dispositivo PPO debe ser cpu o cuda:0");
    require_resources(width, 1, memory_budget, architecture);
    impl_ = std::make_unique<Impl>(width, parameters, seed, device_name, memory_budget, architecture, objective);
}
PpoPolicy::PpoPolicy(PpoPolicy&&) noexcept = default;
PpoPolicy& PpoPolicy::operator=(PpoPolicy&&) noexcept = default;
PpoPolicy::~PpoPolicy() = default;
PpoForward PpoPolicy::forward(const at::Tensor& observations) const {
    impl_->validate_observations(observations);
    const at::NoGradGuard no_grad;
    return impl_->network.forward(observations);
}

at::Tensor PpoPolicy::act(const at::Tensor& observations, bool deterministic) {
    require(!impl_->architecture.double_dqn, "Double DQN debe usar su selector epsilon-greedy");
    const at::NoGradGuard no_grad;
    const auto output = forward(observations);
    const auto log_probabilities = output.logits.log_softmax(-1);
    const auto actions = deterministic ? log_probabilities.argmax(1) :
        at::multinomial(log_probabilities.exp(), 1, true, impl_->sampler).squeeze(1);
    const auto selected = log_probabilities.gather(1, actions.unsqueeze(1)).squeeze(1);
    return at::stack({actions.to(at::kDouble), selected.to(at::kDouble), output.values.to(at::kDouble)}, 1);
}

at::Tensor PpoPolicy::values(const at::Tensor& observations) const { return forward(observations).values; }

PpoUpdateStats PpoPolicy::update(const PpoRollout& rollout) {
    require(!impl_->terminal_optimizer, "El actor terminal no admite actualizaciones PPO");
    require(!impl_->architecture.double_dqn, "Double DQN debe usar su actualización con red objetivo");
    impl_->validate_rollout(rollout, impl_->tensor_device);
    if (impl_->architecture.kind == PpoNetworkKind::gru) {
        const auto result = impl_->update_recurrent(rollout);
        if (impl_->objective.enabled()) { impl_->confirm_controller(result); }
        return result;
    }
    const auto indices = rollout.reward_valid.flatten().nonzero().squeeze(1);
    if (indices.numel() == 0) {
        if (impl_->objective.enabled()) { impl_->confirm_controller({}); }
        return {};
    }
    const auto result = impl_->update_feedforward(rollout, ppo_gae(rollout, impl_->parameters), indices);
    if (impl_->objective.enabled()) { impl_->confirm_controller(result); }
    return result;
}
PpoUpdateStats PpoPolicy::update_from_cpu(const PpoRollout& rollout) {
    require(!impl_->terminal_optimizer, "El actor terminal no admite actualizaciones PPO");
    require(impl_->architecture.kind == PpoNetworkKind::mlp && !impl_->architecture.double_dqn,
            "La actualización desde CPU requiere una política PPO de tipo MLP");
    impl_->validate_rollout(rollout, at::Device(at::kCPU));
    const auto indices = rollout.reward_valid.flatten().nonzero().squeeze(1);
    if (indices.numel() == 0) {
        if (impl_->objective.enabled()) { impl_->confirm_controller({}); }
        return {};
    }
    const auto result = impl_->update_feedforward(rollout, ppo_gae(rollout, impl_->parameters), indices);
    if (impl_->objective.enabled()) { impl_->confirm_controller(result); }
    return result;
}

std::optional<double> PpoPolicy::full_kl(const PpoRollout& rollout) const {
    require(impl_->objective.enabled(), "La KL completa requiere un objetivo explícito");
    require(rollout.observations.defined(), "Faltan observaciones para medir KL");
    const auto source_device = impl_->architecture.kind == PpoNetworkKind::mlp && rollout.observations.device().is_cpu()
        ? at::Device(at::kCPU) : impl_->tensor_device;
    impl_->validate_rollout(rollout, source_device);
    return impl_->measure_kl(rollout);
}

void PpoPolicy::save(std::ostream& destination) const {
    if (impl_->terminal_optimizer) { impl_->validate_terminal_optimizer(); }
    torch::serialize::OutputArchive archive;
    archive.write("schema_version", c10::IValue(impl_->terminal_optimizer ? terminal_checkpoint_version :
        impl_->objective.enabled() ? controller_checkpoint_version : checkpoint_version));
    archive.write("observation_width", c10::IValue(static_cast<int64_t>(impl_->width)));
    archive.write("memory_budget", c10::IValue(static_cast<int64_t>(impl_->budget)));
    archive.write("device", c10::IValue(impl_->device_name));
    archive.write("architecture_kind", c10::IValue(static_cast<int64_t>(impl_->architecture.kind)));
    archive.write("hidden_width", c10::IValue(impl_->architecture.hidden_width));
    archive.write("auxiliary_enabled", c10::IValue(impl_->architecture.auxiliary));
    archive.write("double_dqn", c10::IValue(impl_->architecture.double_dqn));
    if (impl_->architecture.quantiles > 0) {
        // Solo QR-DQN escribe su cabeza. Los archivos de Double DQN conservan sus bytes.
        archive.write("quantiles", c10::IValue(impl_->architecture.quantiles));
        archive.write("risk_alpha", c10::IValue(impl_->architecture.risk_alpha));
    }
    if (impl_->objective.enabled()) {
        impl_->controller.validate(impl_->objective, static_cast<int64_t>(optimizer_steps()), impl_->parameters.epochs);
        write_controller(archive, impl_->objective, impl_->controller);
    }
    torch::serialize::OutputArchive hyperparameters;
    for (const auto& [name, field] : hyperparameter_fields) {
        hyperparameters.write(name, c10::IValue(impl_->parameters.*field));
    }
    hyperparameters.write("epochs", c10::IValue(impl_->parameters.epochs));
    hyperparameters.write("minibatch_size", c10::IValue(impl_->parameters.minibatch_size));
    archive.write("hyperparameters", hyperparameters);
    torch::serialize::OutputArchive network;
    impl_->network.save(network);
    archive.write("network", network);
    torch::serialize::OutputArchive optimizer;
    if (impl_->terminal_optimizer) {
        write_terminal_options(archive, impl_->terminal_options);
        impl_->terminal_optimizer->save(optimizer);
    } else {
        impl_->optimizer->save(optimizer);
    }
    archive.write("optimizer", optimizer);
    archive.write("sampling_rng", generator_state(impl_->sampler), true);
    archive.write("shuffle_rng", generator_state(impl_->shuffler), true);
    if (impl_->architecture.auxiliary) {
        torch::serialize::OutputArchive auxiliary_optimizer;
        impl_->auxiliary_optimizer->save(auxiliary_optimizer);
        archive.write("auxiliary_optimizer", auxiliary_optimizer);
        archive.write("auxiliary_steps", c10::IValue(static_cast<int64_t>(auxiliary_steps())));
        archive.write("auxiliary_rng", generator_state(impl_->auxiliary_rng), true);
    }
    if (impl_->architecture.double_dqn) {
        torch::serialize::OutputArchive target;
        impl_->target->save(target);
        archive.write("target_network", target);
        archive.write("dqn_environment_step", c10::IValue(static_cast<int64_t>(impl_->dqn_environment_step)));
        archive.write("target_sync_step", c10::IValue(static_cast<int64_t>(impl_->target_sync_step)));
        archive.write("target_interval", c10::IValue(static_cast<int64_t>(impl_->target_interval)));
    }
    archive.save_to(destination);
    require(destination.good(), "No se pudo escribir el checkpoint PPO completo");
}

PpoPolicy PpoPolicy::load(std::istream& source, std::string_view device_name) {
    return load_mode(source, device_name, false);
}

PpoPolicy PpoPolicy::load_terminal(std::istream& source, std::string_view device_name) {
    return load_mode(source, device_name, true);
}

PpoPolicy PpoPolicy::load_mode(std::istream& source, std::string_view device_name, bool terminal) {
    validate_stream(source);
    torch::serialize::InputArchive archive;
    archive.load_from(source, at::Device(at::kCPU));
    const auto version = read_integer(archive, "schema_version");
    require(terminal ? version == terminal_checkpoint_version :
                (version == 1 || version == checkpoint_version || version == controller_checkpoint_version),
            "La versión del checkpoint PPO no está admitida");
    PpoArchitecture architecture;
    if (version >= checkpoint_version) {
        const auto kind = read_integer(archive, "architecture_kind");
        require(kind == static_cast<int64_t>(PpoNetworkKind::mlp) ||
                    kind == static_cast<int64_t>(PpoNetworkKind::gru),
                "El checkpoint PPO declara una arquitectura desconocida");
        architecture = {static_cast<PpoNetworkKind>(kind), read_integer(archive, "hidden_width")};
        c10::IValue auxiliary;
        if (archive.try_read("auxiliary_enabled", auxiliary)) {
            require(auxiliary.isBool(), "El checkpoint PPO contiene una bandera auxiliar inválida");
            architecture.auxiliary = auxiliary.toBool();
        }
        c10::IValue double_dqn;
        if (archive.try_read("double_dqn", double_dqn)) {
            require(double_dqn.isBool(), "El checkpoint contiene una bandera Double DQN inválida");
            architecture.double_dqn = double_dqn.toBool();
        }
        c10::IValue quantiles;
        if (archive.try_read("quantiles", quantiles)) {
            c10::IValue risk_alpha;
            archive.read("risk_alpha", risk_alpha);
            require(quantiles.isInt() && risk_alpha.isDouble(), "El checkpoint contiene una cabeza cuantílica inválida");
            architecture.quantiles = quantiles.toInt();
            architecture.risk_alpha = risk_alpha.toDouble();
        }
        architecture.validate();
    }
    const auto width = read_integer(archive, "observation_width");
    const auto budget = read_integer(archive, "memory_budget");
    require(width > 0 && budget > 0, "El checkpoint PPO contiene dimensiones inválidas");
    c10::IValue saved_device;
    archive.read("device", saved_device);
    require(saved_device.isString() && saved_device.toStringRef() == device_name,
            "La recuperación PPO exige el mismo dispositivo del checkpoint");
    torch::serialize::InputArchive hyperparameters;
    archive.read("hyperparameters", hyperparameters);
    const auto parameters = read_hyperparameters(hyperparameters);
    PpoObjectiveConfig objective;
    PpoControllerState controller;
    torch::serialize::InputArchive objective_archive, controller_archive;
    if (version == controller_checkpoint_version) {
        archive.read("objective", objective_archive);
        archive.read("controller", controller_archive);
        objective = read_objective(objective_archive);
        controller = read_controller(controller_archive);
    } else {
        require(!archive.try_read("objective", objective_archive) && !archive.try_read("controller", controller_archive),
                "Un checkpoint anterior contiene un controlador de otra versión");
    }
    PpoPolicy result(static_cast<std::size_t>(width), parameters, 0, device_name,
                     static_cast<std::size_t>(budget), architecture, objective);
    torch::serialize::InputArchive network;
    archive.read("network", network);
    if (terminal) {
        for (const auto& item : result.impl_->network.named_parameters()) {
            at::Tensor saved;
            network.read(item.key(), saved);
            require(saved.defined() && saved.layout() == at::kStrided &&
                        saved.scalar_type() == at::kFloat,
                    "El archivo terminal necesita parámetros FP32 sin conversión");
            require_float(saved, item.value().sizes(), at::Device(at::kCPU));
        }
    }
    result.impl_->network.load(network);
    result.impl_->network.to(result.impl_->tensor_device, at::kFloat);
    result.impl_->network.validate(width, result.impl_->tensor_device);
    result.impl_->network.pack_recurrent_weights();
    torch::serialize::InputArchive optimizer;
    archive.read("optimizer", optimizer);
    if (terminal) {
        result.enable_terminal_adam(read_terminal_options(archive));
        result.impl_->terminal_optimizer->load(optimizer);
        result.impl_->validate_terminal_optimizer();
    } else {
        torch::serialize::InputArchive incompatible;
        require(!archive.try_read("terminal_adam", incompatible),
                "Un checkpoint PPO contiene estado de la ruta terminal");
        result.impl_->optimizer->load(optimizer);
        result.impl_->validate_optimizer(*result.impl_->optimizer);
    }
    if (objective.enabled()) {
        controller.validate(objective, static_cast<int64_t>(result.optimizer_steps()), parameters.epochs);
        result.impl_->controller = controller;
    }
    at::Tensor sampling_state;
    at::Tensor shuffle_state;
    archive.read("sampling_rng", sampling_state, true);
    archive.read("shuffle_rng", shuffle_state, true);
    restore_generator(result.impl_->sampler, sampling_state);
    restore_generator(result.impl_->shuffler, shuffle_state);
    if (architecture.auxiliary) {
        torch::serialize::InputArchive auxiliary_optimizer;
        archive.read("auxiliary_optimizer", auxiliary_optimizer);
        result.impl_->auxiliary_optimizer->load(auxiliary_optimizer);
        result.impl_->validate_optimizer(*result.impl_->auxiliary_optimizer);
        const auto steps = read_integer(archive, "auxiliary_steps");
        require(steps >= 0 && static_cast<std::size_t>(steps) == result.auxiliary_steps(),
                "El checkpoint PPO no conserva el contador del optimizador auxiliar");
        at::Tensor auxiliary_state;
        archive.read("auxiliary_rng", auxiliary_state, true);
        restore_generator(result.impl_->auxiliary_rng, auxiliary_state);
    }
    if (architecture.double_dqn) {
        const auto environment_step = read_integer(archive, "dqn_environment_step");
        const auto target_step = read_integer(archive, "target_sync_step");
        const auto interval = read_integer(archive, "target_interval");
        const auto updates = result.optimizer_steps();
        require(environment_step >= 0 && target_step >= 0 && target_step <= environment_step &&
                    ((updates == 0 && environment_step == 0 && target_step == 0 && interval == 0) ||
                     (updates > 0 && environment_step >= static_cast<int64_t>(dqn_warmup_steps) &&
                      updates <= static_cast<std::size_t>(environment_step) - dqn_warmup_steps + 1 &&
                      interval > 0 && interval <= maximum_samples &&
                      (target_step == 0 || (target_step >= static_cast<int64_t>(dqn_warmup_steps) &&
                                           target_step % interval == 0)))),
                "El checkpoint Double DQN no conserva cursores de actualización y sincronización coherentes");
        torch::serialize::InputArchive target;
        archive.read("target_network", target);
        result.impl_->target->load(target);
        result.impl_->target->to(result.impl_->tensor_device, at::kFloat);
        result.impl_->target->validate(width, result.impl_->tensor_device);
        result.impl_->dqn_environment_step = static_cast<std::size_t>(environment_step);
        result.impl_->target_sync_step = static_cast<std::size_t>(target_step);
        result.impl_->target_interval = static_cast<std::size_t>(interval);
    }
    return result;
}
PpoRandomState PpoPolicy::random_state() const {
    return {generator_state(impl_->sampler), generator_state(impl_->shuffler)};
}

void PpoPolicy::restore_random_state(const PpoRandomState& state) {
    auto sampler = impl_->sampler.clone();
    auto shuffler = impl_->shuffler.clone();
    restore_generator(sampler, state.sampling);
    restore_generator(shuffler, state.shuffle);
    impl_->sampler = std::move(sampler);
    impl_->shuffler = std::move(shuffler);
}

std::size_t PpoPolicy::observation_width() const noexcept { return impl_->width; }
std::size_t PpoPolicy::memory_budget() const noexcept { return impl_->budget; }
std::string PpoPolicy::critic_fingerprint() const {
    require(!impl_->terminal_failed && !impl_->architecture.double_dqn,
            "La política no conserva una fila de valor válida");
    const auto bits = impl_->critic_bits().to(at::kCPU);
    std::string material(static_cast<std::size_t>(bits.numel()) * sizeof(int32_t), '\0');
    std::memcpy(material.data(), bits.const_data_ptr<int32_t>(), material.size());
    return simulation::content_sha256(material);
}
const PpoHyperparameters& PpoPolicy::hyperparameters() const noexcept {
    return impl_->parameters;
}
const std::string& PpoPolicy::device() const noexcept {
    return impl_->device_name;
}
uint64_t PpoPolicy::seed() const { return impl_->sampler.current_seed(); }
std::size_t PpoPolicy::optimizer_steps() const {
    return impl_->optimizer_step_count();
}
void PpoArchitecture::validate() const {
    require((kind == PpoNetworkKind::mlp || kind == PpoNetworkKind::gru) && hidden_width > 0 &&
                hidden_width <= maximum_hidden_width &&
                (kind != PpoNetworkKind::gru || hidden_width == default_ppo_hidden_width) &&
                (!auxiliary || (kind == PpoNetworkKind::mlp && hidden_width == default_ppo_hidden_width)) &&
                (!double_dqn || (kind == PpoNetworkKind::mlp && hidden_width == default_ppo_hidden_width && !auxiliary)),
            "La arquitectura PPO necesita una MLP acotada o una GRU de 64 unidades");
    require((quantiles == 0 && risk_alpha == 1) ||
                (double_dqn && quantiles >= 2 && quantiles <= maximum_quantiles),
            "La cabeza cuantílica solo existe en Double DQN y con un número acotado de cuantiles");
    if (quantiles > 0) {
        // Comprueba que alpha*N sea un número entero de niveles con la misma regla que actúa.
        static_cast<void>(quantile_action_scores(at::zeros({1, ppo_action_count, quantiles}, at::kDouble), risk_alpha));
    }
}

at::Tensor PpoPolicy::initial_state(std::size_t batch_size) const {
    require(batch_size > 0 && batch_size <= static_cast<std::size_t>(maximum_samples),
            "El estado recurrente PPO necesita un número acotado de carriles");
    require_resources(impl_->width, static_cast<int64_t>(batch_size), impl_->budget, impl_->architecture);
    const auto state_width = impl_->architecture.kind == PpoNetworkKind::gru ? impl_->architecture.hidden_width : 0;
    return at::zeros({static_cast<int64_t>(batch_size), state_width},
                      at::TensorOptions().dtype(at::kFloat).device(impl_->tensor_device));
}

PpoInference PpoPolicy::infer(const at::Tensor& observations, const at::Tensor& state,
                              const at::Tensor& episode_starts) const {
    impl_->validate_observations(observations);
    const at::NoGradGuard no_grad;
    auto current = state.defined() ? state : initial_state(static_cast<std::size_t>(observations.size(0)));
    const auto state_width = impl_->architecture.kind == PpoNetworkKind::gru ? impl_->architecture.hidden_width : 0;
    require_float(current, {observations.size(0), state_width}, impl_->tensor_device);
    require(current.scalar_type() == at::kFloat, "El estado recurrente PPO debe conservar precisión FP32");
    if (episode_starts.defined()) {
        require_mask(episode_starts, {observations.size(0)}, impl_->tensor_device);
        current = at::where(episode_starts.unsqueeze(1), 0., current);
    }
    return impl_->network.infer(observations, current);
}

PpoForward PpoPolicy::terminal_forward(const at::Tensor& history,
                                      const at::Tensor& lengths) const {
    require(!impl_->terminal_failed, "El actor necesita recuperar un checkpoint confirmado");
    require(!impl_->architecture.double_dqn && !impl_->architecture.auxiliary &&
                history.defined() && history.layout() == at::kStrided && history.dim() == 3 &&
                history.scalar_type() == at::kFloat && history.device() == impl_->tensor_device &&
                history.size(0) > 0 && history.size(0) <= maximum_klpo_terminal_length &&
                history.size(1) > 0 && history.size(1) <= maximum_klpo_terminal_batch &&
                history.size(2) == static_cast<int64_t>(impl_->width),
            "La historia terminal no conserva arquitectura, dimensiones o precisión");
    const auto time = history.size(0);
    const auto lanes = history.size(1);
    require_resources(impl_->width, time * lanes, impl_->budget, impl_->architecture);
    require(lengths.defined() && lengths.layout() == at::kStrided &&
                lengths.scalar_type() == at::kLong && lengths.device() == impl_->tensor_device &&
                lengths.sizes() == at::IntArrayRef({lanes}) &&
                ((lengths > 0) & (lengths <= time)).all().item<bool>(),
            "Las longitudes no corresponden a historias completas");
    require_finite(history);
    const auto present = at::arange(time, lengths.options()).unsqueeze(1) < lengths;
    const auto observations = at::where(present.unsqueeze(2), history, 0.);
    PpoForward result;
    if (impl_->architecture.kind == PpoNetworkKind::gru) {
        const auto state = initial_state(static_cast<std::size_t>(lanes));
        const auto sequence = impl_->network.sequence(observations, state);
        result = {sequence.logits, sequence.values};
    } else {
        const auto output = impl_->network.forward(observations.flatten(0, 1));
        result = {output.logits.reshape({time, lanes, ppo_action_count}),
                  output.values.reshape({time, lanes})};
    }
    require_finite(result.logits);
    return {at::where(present.unsqueeze(2), result.logits, 0.),
            at::where(present, result.values, 0.)};
}

PpoAction PpoPolicy::act_recurrent(const at::Tensor& observations, const at::Tensor& state,
                                   const at::Tensor& episode_starts, bool deterministic) {
    require(!impl_->architecture.double_dqn, "Double DQN debe usar su selector epsilon-greedy");
    return sample(infer(observations, state, episode_starts), deterministic);
}

PpoAction PpoPolicy::sample(const PpoInference& output, bool deterministic) {
    require(!impl_->architecture.double_dqn, "Double DQN debe usar su selector epsilon-greedy");
    require(output.logits.defined() && output.logits.dim() == 2 && output.logits.size(1) == ppo_action_count &&
                output.logits.scalar_type() == at::kFloat && output.logits.device() == impl_->tensor_device &&
                output.values.defined() && output.values.sizes() == at::IntArrayRef({output.logits.size(0)}) &&
                output.values.device() == impl_->tensor_device && output.next_state.defined() &&
                output.next_state.size(0) == output.logits.size(0),
            "La inferencia no corresponde a la arquitectura o al dispositivo de la política");
    const at::NoGradGuard no_grad;
    const auto log_probabilities = output.logits.log_softmax(-1);
    const auto actions = deterministic ? log_probabilities.argmax(1) :
        at::multinomial(log_probabilities.exp(), 1, true, impl_->sampler).squeeze(1);
    const auto selected = log_probabilities.gather(1, actions.unsqueeze(1)).squeeze(1);
    return {at::stack({actions.to(at::kDouble), selected.to(at::kDouble), output.values.to(at::kDouble)}, 1),
            output.next_state, log_probabilities.exp()};
}
const PpoArchitecture& PpoPolicy::architecture() const noexcept {
    return impl_->architecture;
}
const PpoObjectiveConfig& PpoPolicy::objective() const noexcept { return impl_->objective; }
const PpoControllerState& PpoPolicy::controller_state() const noexcept { return impl_->controller; }
std::size_t PpoPolicy::parameter_count() const noexcept { return model_parameters(impl_->width, impl_->architecture); }
std::string PpoPolicy::parameter_fingerprint() const {
    require(!impl_->terminal_failed, "El actor necesita recuperar un checkpoint confirmado");
    const at::NoGradGuard no_grad;
    impl_->network.validate(static_cast<int64_t>(impl_->width), impl_->tensor_device);
    std::string material = "policy_parameters_fp32_v1\n";
    material += std::to_string(impl_->width) + ":" +
                std::to_string(static_cast<unsigned>(impl_->architecture.kind)) + ":" +
                std::to_string(impl_->architecture.hidden_width) + ":" +
                std::to_string(impl_->architecture.auxiliary) + ":" +
                std::to_string(impl_->architecture.double_dqn) + quantile_geometry(impl_->architecture) + "\n";
    material.reserve(parameter_count() * sizeof(float) + simulation::bytes_per_kibibyte);
    for (const auto& item : impl_->network.named_parameters()) {
        const auto value = item.value().detach().to(at::kCPU).contiguous();
        material.append(item.key()).push_back('\0');
        for (const auto extent : value.sizes()) {
            material.append(std::to_string(extent)).push_back(',');
        }
        material.push_back('\0');
        const auto bytes = static_cast<std::size_t>(value.numel()) * sizeof(float);
        const auto offset = material.size();
        material.resize(offset + bytes);
        const std::span destination(material);
        std::memcpy(destination.subspan(offset, bytes).data(), value.const_data_ptr<float>(), bytes);
    }
    return simulation::content_sha256(material);
}

void PpoPolicy::enable_terminal_adam(PpoTerminalAdamOptions options) {
    options.validate();
    require(!impl_->terminal_optimizer && !impl_->terminal_failed && !impl_->objective.enabled() &&
                !impl_->architecture.auxiliary && !impl_->architecture.double_dqn &&
                impl_->optimizer->state().empty() && options.learning_rate == impl_->parameters.learning_rate,
            "La ruta terminal exige un Adam nuevo y una configuración propia compatible");
    impl_->network.validate(static_cast<int64_t>(impl_->width), impl_->tensor_device);
    auto optimizer = std::make_unique<torch::optim::Adam>(
        impl_->network.policy_parameters(), torch::optim::AdamOptions(options.learning_rate)
            .betas(std::make_tuple(terminal_adam_beta1, terminal_adam_beta2))
            .eps(terminal_adam_epsilon).weight_decay(0).amsgrad(false));
    impl_->terminal_options = options;
    impl_->terminal_optimizer = std::move(optimizer);
    impl_->validate_terminal_optimizer();
}

bool PpoPolicy::terminal_adam_enabled() const noexcept {
    return static_cast<bool>(impl_->terminal_optimizer);
}
PpoTerminalAdamOptions PpoPolicy::terminal_adam_options() const {
    impl_->validate_terminal_optimizer();
    return impl_->terminal_options;
}
void PpoPolicy::terminal_zero_grad() {
    impl_->validate_terminal_optimizer();
    impl_->terminal_optimizer->zero_grad();
}

double PpoPolicy::validate_terminal_gradients() const {
    impl_->validate_terminal_optimizer();
    const at::NoGradGuard no_grad;
    const auto parameters = impl_->network.policy_parameters();
    std::vector<at::Tensor> norms;
    norms.reserve(parameters.size());
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        const auto gradient = parameters[index].grad();
        require(gradient.defined() && gradient.layout() == at::kStrided,
                "Falta un gradiente denso del actor terminal");
        require_float(gradient, parameters[index].sizes(), impl_->tensor_device);
        require(gradient.scalar_type() == at::kFloat, "Los gradientes terminales necesitan FP32");
        if (index + 2 >= parameters.size()) {
            require((gradient.select(0, ppo_action_count) == 0).all().item<bool>(),
                    "La pérdida terminal alcanzó la fila del crítico");
        }
        norms.push_back(at::linalg_vector_norm(gradient, gradient_norm_order,
                                               std::nullopt, false, at::kDouble));
    }
    const auto norm = at::linalg_vector_norm(at::stack(norms), gradient_norm_order).item<double>();
    require(std::isfinite(norm), "La norma global del actor no es finita");
    return norm;
}

std::vector<at::Tensor> PpoPolicy::terminal_gradients() const {
    static_cast<void>(validate_terminal_gradients());
    const at::NoGradGuard no_grad;
    std::vector<at::Tensor> result;
    for (const auto& parameter : impl_->network.policy_parameters()) {
        result.push_back(parameter.grad().detach().to(at::kCPU).clone());
    }
    return result;
}

double PpoPolicy::terminal_step() {
    double norm = validate_terminal_gradients();
    const auto previous = optimizer_steps();
    require(previous < static_cast<std::size_t>(maximum_optimizer_steps),
            "Adam terminal superaría el límite de actualizaciones");
    const auto critic = impl_->critic_bits();
    if (impl_->terminal_options.gradient_norm > 0) {
        norm = torch::nn::utils::clip_grad_norm_(impl_->network.policy_parameters(),
            impl_->terminal_options.gradient_norm, gradient_norm_order, true);
    }
    // A partir de aquí un fallo exige recuperar el checkpoint anterior.
    impl_->terminal_failed = true;
    impl_->terminal_optimizer->step();
    impl_->network.validate(static_cast<int64_t>(impl_->width), impl_->tensor_device);
    require(at::equal(critic, impl_->critic_bits()), "Adam modificó los bits de la fila del crítico");
    impl_->terminal_failed = false;
    try {
        impl_->validate_terminal_optimizer();
        require(optimizer_steps() == previous + 1,
                "Adam terminal no confirmó exactamente una actualización");
    } catch (...) {
        impl_->terminal_failed = true;
        throw;
    }
    return norm;
}

PpoPolicy PpoPolicy::frozen_reference(const PpoRandomState& initial_rng) const {
    require(!impl_->terminal_failed && !impl_->objective.enabled() &&
                !impl_->architecture.auxiliary && !impl_->architecture.double_dqn,
            "La referencia terminal necesita un actor compatible y confirmado");
    if (impl_->terminal_optimizer) { impl_->validate_terminal_optimizer(); }
    PpoPolicy result(impl_->width, impl_->parameters, 0, impl_->device_name,
                     impl_->budget, impl_->architecture);
    result.impl_->network.copy_parameters(impl_->network);
    result.restore_random_state(initial_rng);
    return result;
}

PpoAuxiliaryStats PpoPolicy::consolidate(const at::Tensor& observations,
                                       const at::Tensor& matured_rewards, int64_t steps) {
    require(impl_->architecture.auxiliary && steps > 0 && steps <= maximum_auxiliary_updates,
            "La consolidación requiere una MLP64 auxiliar y entre 1 y 64 pasos explícitos");
    impl_->validate_observations(observations, true);
    require(observations.size(0) <= maximum_auxiliary_samples &&
                auxiliary_steps() <= static_cast<std::size_t>(maximum_optimizer_steps - steps),
            "La consolidación excede 8192 filas o el presupuesto de actualizaciones");
    require_float(matured_rewards, {observations.size(0)}, observations.device());
    const auto rewards = matured_rewards.detach().to(at::kFloat);
    require_float(rewards, matured_rewards.sizes(), observations.device());
    const auto count = std::min(ppo_auxiliary_batch_size, observations.size(0));
    constexpr int64_t auxiliary_metrics = 3;
    auto totals = at::zeros({auxiliary_metrics}, observations.options().device(impl_->tensor_device).dtype(at::kDouble));
    const at::AutoGradMode enable_grad(true);
    for (int64_t step = 0; step < steps; ++step) {
        const auto indices = at::randperm(observations.size(0), impl_->auxiliary_rng,
            at::TensorOptions().dtype(at::kLong)).narrow(0, 0, count).to(observations.device());
        const auto selected = observations.detach().index_select(0, indices).to(impl_->tensor_device);
        const auto targets = rewards.index_select(0, indices).to(impl_->tensor_device);
        const auto loss = (impl_->network.predict_auxiliary(selected) - targets).abs().mean();
        require(at::isfinite(loss).item<bool>(), "El MAE auxiliar no es finito");
        impl_->auxiliary_optimizer->zero_grad();
        loss.backward();
        const auto gradient = torch::nn::utils::clip_grad_norm_(
            impl_->network.auxiliary_parameters(), impl_->parameters.gradient_norm, gradient_norm_order, true);
        impl_->auxiliary_optimizer->step();
        const at::NoGradGuard no_grad;
        const auto after = (impl_->network.predict_auxiliary(selected) - targets).abs().mean();
        require(at::isfinite(after).item<bool>(), "El MAE auxiliar no es finito después de actualizar");
        totals.add_(at::stack({loss.detach().to(at::kDouble), after.to(at::kDouble),
                               at::full({}, gradient, totals.options())}));
    }
    totals = (totals / static_cast<double>(steps)).cpu().contiguous();
    const auto metrics = totals.accessor<double, 1>();
    return {count * steps, steps, metrics[0], metrics[1], metrics[2]};
}
std::size_t PpoPolicy::auxiliary_steps() const {
    return impl_->auxiliary_optimizer ? adam_steps(*impl_->auxiliary_optimizer) : 0;
}
at::Tensor PpoPolicy::state_from_history(const at::Tensor& observations,
                                        const at::Tensor& lengths) const {
    require(!impl_->terminal_failed, "El actor necesita recuperar un checkpoint confirmado");
    require(observations.defined() && observations.dim() == 3 &&
                observations.size(0) <= ppo_maximum_history && observations.size(1) > 0 &&
                observations.size(2) == static_cast<int64_t>(impl_->width) &&
                observations.scalar_type() == at::kFloat,
            "La reconstrucción necesita historia FP32 [P,N,D] de hasta 256 pasos");
    require_resources(impl_->width, std::max(int64_t{1}, observations.size(0)) * observations.size(1),
                       impl_->budget, impl_->architecture);
    require_float(observations, observations.sizes(), impl_->tensor_device);
    require(lengths.defined() && lengths.layout() == at::kStrided &&
                lengths.sizes() == at::IntArrayRef({observations.size(1)}) &&
                lengths.scalar_type() == at::kLong && lengths.device() == impl_->tensor_device,
            "La reconstrucción necesita longitudes int64 por carril en el dispositivo de la política");
    require(((lengths >= 0) & (lengths <= observations.size(0))).all().item<bool>(),
            "Las longitudes de reconstrucción exceden la historia disponible");
    const at::NoGradGuard no_grad;
    return impl_->architecture.kind == PpoNetworkKind::gru ?
        impl_->network.prefix_state({observations, lengths}) :
        initial_state(static_cast<std::size_t>(observations.size(1)));
}
at::Tensor double_dqn_targets(const at::Tensor& rewards, const at::Tensor& terminated,
                             const at::Tensor& online_next, const at::Tensor& target_next, double gamma) {
    require(std::isfinite(gamma) && gamma >= 0 && gamma <= 1 && rewards.defined() && rewards.dim() == 1 &&
                rewards.numel() > 0 && rewards.numel() <= maximum_minibatch,
            "Double DQN necesita un descuento finito y un lote acotado de recompensas");
    const auto device = rewards.device();
    require_float_shape(rewards, rewards.sizes(), device);
    require_float_shape(online_next, {rewards.numel(), ppo_action_count}, device);
    require_mask(terminated, online_next.sizes().slice(0, 1), device);
    require_float_shape(target_next, online_next.sizes(), device);
    const at::NoGradGuard no_grad;
    require_finite(at::cat({rewards.flatten(), online_next.flatten(), target_next.flatten()}));
    const auto actions = online_next.argmax(1, true);
    const auto values = target_next.gather(1, actions).squeeze(1);
    const auto targets = rewards + gamma * (~terminated).to(rewards.scalar_type()) * values;
    require_float(targets, rewards.sizes(), device);
    return targets;
}

PpoAction PpoPolicy::act_double_dqn(const at::Tensor& observations, double epsilon) {
    require(impl_->architecture.double_dqn && std::isfinite(epsilon) && epsilon >= 0 && epsilon <= 1,
            "La acción Double DQN necesita una arquitectura compatible y epsilon entre cero y uno");
    const at::NoGradGuard no_grad;
    const auto output = forward(observations);
    require_float(output.logits, {observations.size(0), ppo_action_count}, impl_->tensor_device);
    const auto greedy = output.logits.argmax(1);
    auto probabilities = at::full(output.logits.sizes(), epsilon / static_cast<double>(ppo_action_count),
                                  output.logits.options().dtype(at::kDouble));
    probabilities.scatter_add_(1, greedy.unsqueeze(1),
                                at::full({observations.size(0), 1}, 1 - epsilon, probabilities.options()));
    const auto actions = epsilon == 0 ? greedy : at::multinomial(probabilities, 1, true, impl_->sampler).squeeze(1);
    const auto selected_logp = probabilities.gather(1, actions.unsqueeze(1)).squeeze(1).log();
    return {at::stack({actions.to(at::kDouble), selected_logp, output.values.to(at::kDouble)}, 1),
            initial_state(static_cast<std::size_t>(observations.size(0))), probabilities};
}

at::Tensor PpoPolicy::action_quantiles(const at::Tensor& observations) const {
    require(impl_->architecture.quantiles > 0, "La política no tiene cabeza cuantílica");
    impl_->validate_observations(observations);
    return impl_->network.quantiles(observations);
}

DqnLoss PpoPolicy::double_dqn_loss(const DqnBatch& batch) const {
    require(impl_->architecture.double_dqn, "La pérdida de valor necesita Double DQN o QR-DQN");
    impl_->validate_observations(batch.observations, true);
    impl_->validate_observations(batch.next_observations, true);
    const auto samples = batch.observations.size(0);
    const auto source_device = batch.observations.device();
    require(samples <= default_ppo_minibatch_size && batch.next_observations.sizes() == batch.observations.sizes() &&
                batch.next_observations.device() == source_device && batch.actions.defined() &&
                batch.actions.layout() == at::kStrided && batch.actions.scalar_type() == at::kLong &&
                batch.actions.sizes() == at::IntArrayRef({samples}) && batch.actions.device() == source_device,
            "Double DQN necesita como máximo 64 transiciones completas en un mismo dispositivo");
    require_float(batch.rewards, {samples}, source_device);
    require_mask(batch.terminated, {samples}, source_device);
    require_mask(batch.reward_valid, {samples}, source_device);
    require(((batch.actions >= 0) & (batch.actions < ppo_action_count)).all().item<bool>(),
            "Las acciones Double DQN exceden las seis opciones admitidas");
    const auto indices = batch.reward_valid.nonzero().squeeze(1);
    DqnLoss result;
    result.valid_transitions = indices.numel();
    if (result.valid_transitions == 0) {
        return result;
    }
    const auto observations = batch.observations.detach().index_select(0, indices).to(impl_->tensor_device);
    const auto following = batch.next_observations.detach().index_select(0, indices).to(impl_->tensor_device);
    const auto actions = batch.actions.index_select(0, indices).to(impl_->tensor_device);
    const auto rewards = batch.rewards.detach().index_select(0, indices).to(impl_->tensor_device, at::kFloat);
    const auto terminated = batch.terminated.index_select(0, indices).to(impl_->tensor_device);
    const auto& architecture = impl_->architecture;
    at::Tensor targets;
    {
        const at::NoGradGuard no_grad;
        targets = architecture.quantiles > 0
            ? quantile_double_targets(rewards, terminated, impl_->network.quantiles(following),
                                      impl_->target->quantiles(following),
                                      {.gamma = impl_->parameters.gamma, .risk_alpha = architecture.risk_alpha})
            : double_dqn_targets(rewards, terminated, impl_->network.forward(following).logits,
                                 impl_->target->forward(following).logits, impl_->parameters.gamma);
    }
    const at::AutoGradMode enable_grad(true);
    if (architecture.quantiles > 0) {
        const auto count = architecture.quantiles;
        const auto index = actions.view({-1, 1, 1}).expand({actions.size(0), 1, count});
        const auto selected = impl_->network.quantiles(observations).gather(1, index).squeeze(1);
        result.loss = quantile_huber_loss(selected, targets, qr_dqn_kappa).mean();
        require(at::isfinite(result.loss).item<bool>(), "La pérdida cuantílica de QR-DQN no es finita");
        return result;
    }
    const auto selected = impl_->network.forward(observations).logits.gather(1, actions.unsqueeze(1)).squeeze(1);
    result.loss = at::smooth_l1_loss(selected, targets);
    require(at::isfinite(result.loss).item<bool>(), "La pérdida SmoothL1 de Double DQN no es finita");
    return result;
}

DqnUpdateStats PpoPolicy::update_double_dqn(const DqnBatch& batch, std::size_t environment_step,
                                            std::size_t target_interval) {
    require(impl_->architecture.double_dqn && environment_step >= dqn_warmup_steps &&
                environment_step <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) &&
                environment_step > impl_->dqn_environment_step && target_interval > 0 &&
                target_interval <= static_cast<std::size_t>(maximum_samples) &&
                (impl_->target_interval == 0 || impl_->target_interval == target_interval) &&
                optimizer_steps() < static_cast<std::size_t>(maximum_optimizer_steps),
            "Double DQN necesita un cursor creciente tras el calentamiento y un intervalo objetivo estable");
    const auto computed = double_dqn_loss(batch);
    DqnUpdateStats result;
    result.valid_transitions = computed.valid_transitions;
    if (result.valid_transitions == 0) {
        return result;
    }
    const at::AutoGradMode enable_grad(true);
    impl_->optimizer->zero_grad();
    computed.loss.backward();
    result.gradient_norm = torch::nn::utils::clip_grad_norm_(
        impl_->network.policy_parameters(), impl_->parameters.gradient_norm, gradient_norm_order, true);
    impl_->optimizer->step();
    result.target_synchronized = environment_step % target_interval == 0;
    if (result.target_synchronized) {
        impl_->target->copy_parameters(impl_->network);
        impl_->target_sync_step = environment_step;
    }
    impl_->dqn_environment_step = environment_step;
    impl_->target_interval = target_interval;
    result.updates = 1;
    result.loss = computed.loss.item<double>();
    return result;
}

std::size_t PpoPolicy::dqn_environment_step() const noexcept { return impl_->dqn_environment_step; }
std::size_t PpoPolicy::target_sync_step() const noexcept { return impl_->target_sync_step; }
}
