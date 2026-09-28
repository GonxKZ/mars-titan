#include "mars_titan/ppo_policy.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/core/grad_mode.h>
#include <torch/nn/module.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/optim/adam.h>
#include <torch/serialize/archive.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <istream>
#include <mutex>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace mars_titan::learning {
namespace {
constexpr int64_t maximum_samples = 1 << 20;
constexpr int64_t maximum_epochs = 64;
constexpr int64_t maximum_optimizer_steps = maximum_samples * maximum_epochs;
constexpr int64_t maximum_minibatch = 65536;
constexpr int64_t hidden_width = 64;
constexpr std::size_t maximum_observation_width = 32768;
constexpr std::size_t maximum_memory_bytes = std::size_t{2} * 1024 * 1024 * 1024;
constexpr std::streamoff maximum_checkpoint_bytes = std::streamoff{64} * 1024 * 1024;
constexpr double normalization_epsilon = 1e-8;
constexpr double maximum_log_probability = 1e-6;
constexpr double maximum_loss_weight = 100;
constexpr double maximum_gradient_norm = 1e6;
constexpr uint64_t shuffle_seed_offset = 0x9e3779b97f4a7c15ULL;
constexpr int64_t checkpoint_version = 1;

std::size_t model_bytes(std::size_t width) {
    const auto hidden = static_cast<std::size_t>(hidden_width);
    const auto outputs = static_cast<std::size_t>(ppo_action_count + 1);
    return ((width + 1) * hidden + (hidden + 1) * hidden + (hidden + 1) * outputs) *
           sizeof(float) * 4;
}

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::invalid_argument(std::string(message));
    }
}

void require_float(const at::Tensor& tensor, at::IntArrayRef shape, const at::Device& device) {
    require(tensor.defined() && tensor.layout() == at::kStrided && tensor.sizes() == shape &&
                tensor.device() == device &&
                (tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kDouble),
            "El tensor PPO no tiene la forma, precisión o dispositivo acordados");
    require(at::isfinite(tensor).all().item<bool>(), "El tensor PPO contiene NaN o infinito");
}

void require_mask(const at::Tensor& mask, at::IntArrayRef shape, const at::Device& device) {
    require(mask.defined() && mask.layout() == at::kStrided && mask.sizes() == shape &&
                mask.device() == device && mask.scalar_type() == at::kBool,
            "La máscara PPO no es un tensor booleano con la forma acordada");
}

void require_resources(std::size_t width, int64_t samples, std::size_t budget) {
    require(width > 0 && width <= maximum_observation_width && samples > 0 &&
                samples <= maximum_samples && budget <= maximum_memory_bytes,
            "El lote PPO supera los límites de dimensiones o memoria");
    // Incluye parámetros, Adam, gradientes, tensores del rollout y activaciones temporales.
    constexpr std::size_t activation_bytes = static_cast<std::size_t>(hidden_width) * 32 + 256;
    const auto bytes_per_sample = width * sizeof(float) * 3 + activation_bytes;
    require(budget > model_bytes(width) &&
                static_cast<std::size_t>(samples) <= (budget - model_bytes(width)) / bytes_per_sample,
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
    PolicyNetwork(int64_t width, at::Generator generator) {
        const auto initialize = [&](std::string_view name, at::IntArrayRef shape, int64_t inputs) {
            const double bound = 1 / std::sqrt(static_cast<double>(inputs));
            auto tensor = at::empty(shape, at::kFloat);
            tensor.uniform_(-bound, bound, generator);
            return register_parameter(std::string(name), tensor);
        };
        first_weight_ = initialize("first_weight", {hidden_width, width}, width);
        first_bias_ = initialize("first_bias", {hidden_width}, width);
        second_weight_ = initialize("second_weight", {hidden_width, hidden_width}, hidden_width);
        second_bias_ = initialize("second_bias", {hidden_width}, hidden_width);
        output_weight_ = initialize("output_weight", {ppo_action_count + 1, hidden_width}, hidden_width);
        output_bias_ = initialize("output_bias", {ppo_action_count + 1}, hidden_width);
    }

    [[nodiscard]] PpoForward forward(const at::Tensor& observations) const {
        const auto first = at::linear(observations, first_weight_, first_bias_).tanh();
        const auto second = at::linear(first, second_weight_, second_bias_).tanh();
        const auto output = at::linear(second, output_weight_, output_bias_);
        return {output.narrow(1, 0, ppo_action_count), output.select(1, ppo_action_count)};
    }

    void validate(int64_t width, const at::Device& device) const {
        require_float(first_weight_, {hidden_width, width}, device);
        require_float(first_bias_, {hidden_width}, device);
        require_float(second_weight_, {hidden_width, hidden_width}, device);
        require_float(second_bias_, {hidden_width}, device);
        require_float(output_weight_, {ppo_action_count + 1, hidden_width}, device);
        require_float(output_bias_, {ppo_action_count + 1}, device);
        for (const auto& parameter : parameters()) {
            require(parameter.scalar_type() == at::kFloat,
                    "El checkpoint PPO debe conservar parámetros FP32");
        }
    }

private:
    at::Tensor first_weight_;
    at::Tensor first_bias_;
    at::Tensor second_weight_;
    at::Tensor second_bias_;
    at::Tensor output_weight_;
    at::Tensor output_bias_;
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
}
}
struct PpoPolicy::Impl {
    std::size_t width;
    PpoHyperparameters parameters;
    std::string device_name;
    at::Device tensor_device;
    std::size_t budget;
    PolicyNetwork network;
    std::unique_ptr<torch::optim::Adam> optimizer;
    at::Generator sampler;
    at::Generator shuffler;

    Impl(std::size_t observation_width, PpoHyperparameters hyperparameters, uint64_t seed,
         std::string_view device, std::size_t memory_budget)
        : width(observation_width), parameters(hyperparameters), device_name(device),
          tensor_device(device_name), budget(memory_budget),
          network(static_cast<int64_t>(width), at::detail::createCPUGenerator(seed)),
          sampler(make_sampler(tensor_device, seed)),
          shuffler(at::detail::createCPUGenerator(seed ^ shuffle_seed_offset)) {
        network.to(tensor_device, at::kFloat);
        optimizer = std::make_unique<torch::optim::Adam>(
            network.parameters(), torch::optim::AdamOptions(parameters.learning_rate));
    }

    void validate_observations(const at::Tensor& observations) const {
        require(observations.defined() && observations.dim() == 2 &&
                    observations.size(1) == static_cast<int64_t>(width) &&
                    observations.scalar_type() == at::kFloat,
                "La política PPO necesita observaciones FP32 [N,D]");
        require_resources(width, observations.size(0), budget);
        require_float(observations, observations.sizes(), tensor_device);
    }

    void validate_rollout(const PpoRollout& rollout) const {
        validate_gae(rollout);
        const auto shape = rollout.rewards.sizes();
        require(rollout.observations.defined() && rollout.observations.dim() == 3 &&
                    rollout.observations.size(0) == shape[0] &&
                    rollout.observations.size(1) == shape[1] &&
                    rollout.observations.size(2) == static_cast<int64_t>(width),
                "Las observaciones PPO no corresponden a los pasos y carriles del rollout");
        require_resources(width, rollout.rewards.numel(), budget);
        validate_observations(rollout.observations.flatten(0, 1));
        require_float(rollout.old_log_probabilities, shape, tensor_device);
        require(rollout.rewards.device() == tensor_device && rollout.actions.defined() &&
                    rollout.actions.layout() == at::kStrided && rollout.actions.sizes() == shape &&
                    rollout.actions.device() == tensor_device && rollout.actions.scalar_type() == at::kLong,
                "Las acciones PPO no corresponden al dispositivo o a la forma del rollout");
        require(((rollout.actions >= 0) & (rollout.actions < ppo_action_count)).all().item<bool>(),
                "La acción PPO está fuera del intervalo de seis acciones");
        require((rollout.old_log_probabilities <= maximum_log_probability).all().item<bool>(),
                "Las probabilidades anteriores PPO exceden el intervalo numérico admitido");
    }

    void validate_optimizer() {
        require(optimizer->param_groups().size() == 1,
                "El checkpoint PPO altera los grupos del optimizador");
        const auto* options = dynamic_cast<const torch::optim::AdamOptions*>(
            &optimizer->param_groups().front().options());
        require(options != nullptr && *options == torch::optim::AdamOptions(parameters.learning_rate),
                "El checkpoint PPO altera las opciones de Adam");
        const auto tensors = network.parameters();
        require(optimizer->state().empty() || optimizer->state().size() == tensors.size(),
                "El checkpoint PPO contiene un estado parcial de Adam");
        static_cast<void>(optimizer_step_count());
        for (const auto& parameter : tensors) {
            if (optimizer->state().empty()) {
                break;
            }
            const auto found = optimizer->state().find(parameter.unsafeGetTensorImpl());
            require(found != optimizer->state().end(), "Falta el estado de Adam de un parámetro PPO");
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

    [[nodiscard]] std::size_t optimizer_step_count() const {
        std::optional<int64_t> common;
        for (const auto& [parameter, stored] : optimizer->state()) {
            static_cast<void>(parameter);
            const auto* state = dynamic_cast<const torch::optim::AdamParamState*>(stored.get());
            require(state != nullptr && state->step() > 0 && state->step() <= maximum_optimizer_steps &&
                        (!common || *common == state->step()),
                    "El checkpoint PPO no conserva un paso positivo y común de Adam");
            common = state->step();
        }
        return static_cast<std::size_t>(common.value_or(0));
    }
};
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
        const auto continuation = at::where(terminated | truncated, 0., carry);
        carry = at::where(valid, delta + parameters.gamma * parameters.gae_lambda * continuation, 0.);
        advantages.select(0, time).copy_(carry);
    }
    return {advantages, at::where(rollout.reward_valid, advantages + old_values, 0.)};
}
PpoPolicy::PpoPolicy(std::size_t width, const PpoHyperparameters& parameters, uint64_t seed,
                     std::string_view device_name, std::size_t memory_budget) {
    parameters.validate();
    require(device_name == "cpu" || device_name == "cuda:0", "El dispositivo PPO debe ser cpu o cuda:0");
    require_resources(width, 1, memory_budget);
    impl_ = std::make_unique<Impl>(width, parameters, seed, device_name, memory_budget);
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
    impl_->validate_rollout(rollout);
    const auto gae = ppo_gae(rollout, impl_->parameters);
    const auto indices = rollout.reward_valid.flatten().nonzero().squeeze(1);
    PpoUpdateStats result;
    result.valid_transitions = indices.numel();
    if (result.valid_transitions == 0) {
        return result;
    }
    const auto expected_steps = ((result.valid_transitions + impl_->parameters.minibatch_size - 1) /
                                 impl_->parameters.minibatch_size) * impl_->parameters.epochs;
    require(optimizer_steps() <= static_cast<std::size_t>(maximum_optimizer_steps - expected_steps),
            "La actualización PPO excedería el límite de pasos de Adam");
    const auto observations = rollout.observations.detach().flatten(0, 1).index_select(0, indices);
    const auto actions = rollout.actions.flatten().index_select(0, indices);
    const auto old_log_probabilities = rollout.old_log_probabilities.detach().flatten()
                                          .index_select(0, indices).to(at::kDouble);
    const auto advantages = gae.advantages.flatten().index_select(0, indices).to(at::kFloat);
    require_float(advantages, actions.sizes(), impl_->tensor_device);
    const auto normalized = ((advantages - advantages.mean()) /
                              advantages.std(false).clamp_min(normalization_epsilon));
    const auto returns = gae.returns.flatten().index_select(0, indices).to(at::kFloat);
    require_float(normalized, actions.sizes(), impl_->tensor_device);
    require_float(returns, actions.sizes(), impl_->tensor_device);
    constexpr int64_t metric_count = 6;
    auto totals = at::zeros({metric_count}, observations.options().dtype(at::kDouble));
    const at::AutoGradMode enable_grad(true);
    const auto& parameters = impl_->parameters;
    for (int64_t epoch = 0; epoch < parameters.epochs; ++epoch) {
        const auto order = at::randperm(result.valid_transitions, impl_->shuffler,
                                       at::TensorOptions().dtype(at::kLong)).to(impl_->tensor_device);
        for (int64_t offset = 0; offset < result.valid_transitions; offset += parameters.minibatch_size) {
            const auto count = std::min(parameters.minibatch_size, result.valid_transitions - offset);
            const auto minibatch = order.narrow(0, offset, count);
            const auto output = impl_->network.forward(observations.index_select(0, minibatch));
            const auto log_probabilities = output.logits.log_softmax(-1);
            const auto selected = log_probabilities.gather(
                1, actions.index_select(0, minibatch).unsqueeze(1)).squeeze(1);
            const auto previous = old_log_probabilities.index_select(0, minibatch);
            const auto policy_loss = -ppo_clipped_objective(
                selected, previous, normalized.index_select(0, minibatch), parameters.clip).mean();
            const auto value_loss = (output.values - returns.index_select(0, minibatch)).square().mean();
            const auto entropy = -(log_probabilities.exp() * log_probabilities).sum(-1).mean();
            const auto loss = policy_loss + parameters.value_weight * value_loss - parameters.entropy * entropy;
            require(at::isfinite(loss).item<bool>(), "La pérdida PPO no es finita");
            impl_->optimizer->zero_grad();
            loss.backward();
            const double gradient = torch::nn::utils::clip_grad_norm_(
                impl_->network.parameters(), parameters.gradient_norm, 2., true);
            impl_->optimizer->step();
            const at::NoGradGuard no_grad;
            const auto log_ratio = selected - previous;
            const auto ratio = log_ratio.exp();
            const auto approximate_kl = (ratio - 1 - log_ratio).mean();
            const auto clip_fraction = ((ratio - 1).abs() > parameters.clip).to(at::kFloat).mean();
            const auto gradient_tensor = at::full({}, gradient, totals.options());
            totals.add_(at::stack({policy_loss, value_loss, entropy, approximate_kl, clip_fraction,
                                  gradient_tensor}).detach().to(at::kDouble) * static_cast<double>(count));
            ++result.minibatches;
        }
    }
    totals = (totals / static_cast<double>(result.valid_transitions * parameters.epochs)).cpu().contiguous();
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
void PpoPolicy::save(std::ostream& destination) const {
    torch::serialize::OutputArchive archive;
    archive.write("schema_version", c10::IValue(checkpoint_version));
    archive.write("observation_width", c10::IValue(static_cast<int64_t>(impl_->width)));
    archive.write("memory_budget", c10::IValue(static_cast<int64_t>(impl_->budget)));
    archive.write("device", c10::IValue(impl_->device_name));
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
    impl_->optimizer->save(optimizer);
    archive.write("optimizer", optimizer);
    archive.write("sampling_rng", generator_state(impl_->sampler), true);
    archive.write("shuffle_rng", generator_state(impl_->shuffler), true);
    archive.save_to(destination);
    require(destination.good(), "No se pudo escribir el checkpoint PPO completo");
}

PpoPolicy PpoPolicy::load(std::istream& source, std::string_view device_name) {
    validate_stream(source);
    torch::serialize::InputArchive archive;
    archive.load_from(source, at::Device(at::kCPU));
    require(read_integer(archive, "schema_version") == checkpoint_version,
            "La versión del checkpoint PPO no está admitida");
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
    PpoPolicy result(static_cast<std::size_t>(width), parameters, 0, device_name,
                     static_cast<std::size_t>(budget));
    torch::serialize::InputArchive network;
    archive.read("network", network);
    result.impl_->network.load(network);
    result.impl_->network.to(result.impl_->tensor_device, at::kFloat);
    result.impl_->network.validate(width, result.impl_->tensor_device);
    torch::serialize::InputArchive optimizer;
    archive.read("optimizer", optimizer);
    result.impl_->optimizer->load(optimizer);
    result.impl_->validate_optimizer();
    at::Tensor sampling_state;
    at::Tensor shuffle_state;
    archive.read("sampling_rng", sampling_state, true);
    archive.read("shuffle_rng", shuffle_state, true);
    restore_generator(result.impl_->sampler, sampling_state);
    restore_generator(result.impl_->shuffler, shuffle_state);
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
}
