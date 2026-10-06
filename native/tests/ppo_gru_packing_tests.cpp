#include "mars_titan/ppo_policy.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/DeviceAccelerator.h>
#include <ATen/Parallel.h>
#include <torch/optim/adam.h>
#include <torch/serialize/archive.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace mars_titan::learning;
using Snapshot = std::map<std::string, at::Tensor>;
constexpr int64_t width = 207;
constexpr int64_t lanes = 2;
constexpr int64_t prefix_ticks = 4;
constexpr int64_t observation_scale = 1000;
constexpr int64_t reset_tick = ppo_sequence_length / 2;
constexpr double prefix_value = 0.25;
constexpr uint64_t seed = 719;
// El reinicio produce tres secuencias, dos minibatches por época y cuatro pasos Adam.
constexpr std::size_t updates_per_rollout = 4;
constexpr std::size_t lifecycle_repetitions = 5;
constexpr std::uintmax_t maximum_snapshot_bytes = std::uintmax_t{8} * 1024 * 1024;
constexpr std::uintmax_t maximum_checkpoint_bytes = std::uintmax_t{64} * 1024 * 1024;
constexpr std::array parameter_names = {
    "weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0", "output_weight", "output_bias"};

void require(bool condition, std::string_view reason) {
    if (!condition) {
        throw std::runtime_error(std::string(reason));
    }
}

void capture(Snapshot& result, const std::string& name, const at::Tensor& value) {
    require(!result.contains(name), "El diagnóstico contiene una clave duplicada");
    result.emplace(name, value.detach().cpu().clone());
}

void equal(const Snapshot& expected, const Snapshot& actual) {
    require(expected.size() == actual.size(), "Cambió el número de tensores del diagnóstico");
    for (const auto& [name, value] : expected) {
        const auto found = actual.find(name);
        require(found != actual.end(), "Falta un tensor del diagnóstico");
        require(value.scalar_type() == found->second.scalar_type() &&
                    at::equal(value, found->second), "El diagnóstico altera valores o precisión");
    }
}

void require_packed(std::span<const at::Tensor, 4> parameters) {
    std::array<std::pair<std::size_t, std::size_t>, 4> intervals;
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        require(parameters[0].is_alias_of(parameters[index]),
                "Los pesos GRU no comparten el almacenamiento compactado");
        const auto& parameter = parameters[index];
        require(parameter.scalar_type() == at::kFloat && parameter.is_contiguous() &&
                    parameter.storage_offset() >= 0 && parameter.numel() > 0,
                "Un peso compactado no conserva una vista FP32 contigua");
        const auto begin = static_cast<std::size_t>(parameter.storage_offset());
        const auto count = static_cast<std::size_t>(parameter.numel());
        const auto capacity = parameter.storage().nbytes() / sizeof(float);
        require(begin <= capacity && count <= capacity - begin,
                "Un peso excede los límites de su almacenamiento");
        intervals.at(index) = {begin, begin + count};
        for (std::size_t previous = 0; previous < index; ++previous) {
            require(intervals.at(previous).second <= begin ||
                        intervals.at(index).second <= intervals.at(previous).first,
                    "La compactación solapa dos parámetros GRU");
        }
    }
}

void packing_intervals_reject_every_overlapping_pair() {
    constexpr int64_t elements_per_weight = 2;
    constexpr std::size_t weight_count = 4;
    const auto storage = at::arange(static_cast<int64_t>(weight_count) * elements_per_weight,
                                    at::kFloat);
    std::array<at::Tensor, weight_count> weights;
    for (std::size_t index = 0; index < weights.size(); ++index) {
        weights.at(index) = storage.narrow(0, static_cast<int64_t>(index) * elements_per_weight,
                                            elements_per_weight);
    }
    require_packed(weights);
    for (std::size_t first = 0; first < weights.size(); ++first) {
        for (std::size_t second = first + 1; second < weights.size(); ++second) {
            for (int64_t shift = 0; shift < elements_per_weight; ++shift) {
                auto overlapping = weights;
                overlapping.at(second) = storage.narrow(0,
                    static_cast<int64_t>(first) * elements_per_weight + shift, elements_per_weight);
                bool rejected = false;
                try {
                    require_packed(overlapping);
                } catch (const std::runtime_error&) {
                    rejected = true;
                }
                require(rejected, "Se aceptó un solapamiento completo o parcial entre pesos GRU");
            }
        }
    }
}

Snapshot inspect(const PpoPolicy& policy, bool expect_packed) {
    std::stringstream saved;
    policy.save(saved);
    torch::serialize::InputArchive archive;
    archive.load_from(saved, at::Device(at::kCPU));
    c10::IValue schema;
    archive.read("schema_version", schema);
    require(schema.isInt() && schema.toInt() == 2, "Cambió el esquema del checkpoint GRU");
    torch::serialize::InputArchive network;
    archive.read("network", network);
    require(network.keys().size() == parameter_names.size(),
            "Cambió el registro de parámetros GRU");
    Snapshot result;
    std::vector<at::Tensor> parameters;
    for (const auto* name : parameter_names) {
        at::Tensor parameter;
        network.read(name, parameter);
        require(parameter.scalar_type() == at::kFloat, "Un parámetro GRU dejó de ser FP32");
        parameters.push_back(parameter);
        capture(result, std::string("parameter_") + name, parameter);
    }
    // La carga en CPU conserva los alias del archivo sin exponer la red privada.
    if (expect_packed) {
        require_packed(std::span<const at::Tensor>(parameters).first<4>());
    } else if (policy.device() == "cpu") {
        for (std::size_t index = 1; index < 4; ++index) {
            require(!parameters[0].is_alias_of(parameters[index]),
                    "La ruta CPU cambió el almacenamiento de sus parámetros");
        }
    }
    const auto random = policy.random_state();
    capture(result, "sampling_rng", random.sampling);
    capture(result, "shuffle_rng", random.shuffle);
    capture(result, "optimizer_steps", at::scalar_tensor(
        static_cast<int64_t>(policy.optimizer_steps()), at::kLong));
    if (policy.optimizer_steps() != 0) {
        torch::serialize::InputArchive saved_optimizer;
        archive.read("optimizer", saved_optimizer);
        torch::optim::Adam optimizer(parameters,
            torch::optim::AdamOptions(policy.hyperparameters().learning_rate));
        optimizer.load(saved_optimizer);
        require(optimizer.state().size() == parameters.size(), "Falta un estado de Adam");
        for (std::size_t index = 0; index < parameters.size(); ++index) {
            const auto found = optimizer.state().find(parameters[index].unsafeGetTensorImpl());
            require(found != optimizer.state().end(), "Adam perdió el vínculo con un parámetro");
            const auto* state =
                dynamic_cast<const torch::optim::AdamParamState*>(found->second.get());
            require(state != nullptr, "El estado recuperado no pertenece a Adam");
            const std::string name = parameter_names.at(index);
            capture(result, "adam_mean_" + name, state->exp_avg());
            capture(result, "adam_square_" + name, state->exp_avg_sq());
            capture(result, "adam_step_" + name, at::scalar_tensor(state->step(), at::kLong));
        }
    }
    return result;
}

PpoRollout rollout(PpoPolicy& policy) {
    const at::Device device(policy.device());
    const auto real = at::TensorOptions().dtype(at::kFloat).device(device);
    const auto precise = real.dtype(at::kDouble);
    const auto integral = real.dtype(at::kLong);
    const auto boolean = real.dtype(at::kBool);
    constexpr std::array shape{ppo_sequence_length, lanes};
    PpoRollout result;
    result.observations = (at::arange(ppo_sequence_length * lanes * width, real) / observation_scale)
                              .reshape({ppo_sequence_length, lanes, width});
    result.actions = (at::arange(ppo_sequence_length * lanes, integral) % ppo_action_count)
                         .reshape(shape);
    result.old_log_probabilities = at::zeros(shape, precise);
    result.old_values = at::zeros(shape, precise);
    result.rewards = at::where(result.actions % 2 == 0, 1., -1.);
    result.next_values = at::zeros(shape, precise);
    result.reward_valid = at::ones(shape, boolean);
    result.reward_valid[-1][1].fill_(false);
    result.terminated = at::zeros(shape, boolean);
    result.terminated[-1].fill_(true);
    result.truncated = at::zeros(shape, boolean);
    result.episode_starts = at::zeros(shape, boolean);
    result.episode_starts[reset_tick][0].fill_(true);
    result.prefix_observations = at::full({prefix_ticks, lanes, width}, prefix_value, real);
    result.prefix_lengths = at::tensor({prefix_ticks, prefix_ticks - 1}, integral);
    auto state = policy.state_from_history(result.prefix_observations, result.prefix_lengths);
    for (int64_t time = 0; time < ppo_sequence_length; ++time) {
        const auto output =
            policy.infer(result.observations[time], state, result.episode_starts[time]);
        state = output.next_state;
        result.old_values[time].copy_(output.values);
        result.old_log_probabilities[time].copy_(output.logits.log_softmax(-1)
            .gather(1, result.actions[time].unsqueeze(1)).squeeze(1));
    }
    return result;
}

void append(Snapshot& destination, const std::string& stage, const Snapshot& source) {
    for (const auto& [name, value] : source) {
        std::string key = stage;
        key.append("_").append(name);
        capture(destination, key, value);
    }
}

struct LifecycleSample {
    std::string_view operation;
    std::string_view phase;
    std::size_t repetition;
    double seconds = 0;
};

template<class Operation>
PpoPolicy measure_policy(std::vector<LifecycleSample>* samples, std::string_view device,
                          LifecycleSample sample, Operation create) {
    if (samples == nullptr) {
        return create();
    }
    if (device == "cuda:0") {
        at::accelerator::synchronizeDevice(0);
    }
    const auto started = std::chrono::steady_clock::now();
    auto policy = create();
    if (device == "cuda:0") {
        at::accelerator::synchronizeDevice(0);
    }
    sample.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    samples->push_back(sample);
    return policy;
}

std::string read_checkpoint(const std::filesystem::path& source) {
    require(std::filesystem::file_size(source) <= maximum_checkpoint_bytes,
            "El checkpoint de referencia supera el límite de lectura");
    std::ifstream stream(source, std::ios::binary);
    require(stream.good(), "No se pudo abrir el checkpoint de referencia");
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void write_checkpoint(const std::filesystem::path& destination, const std::string& bytes) {
    require(!std::filesystem::exists(destination), "El checkpoint de salida ya existe");
    std::ofstream stream(destination, std::ios::binary);
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    stream.close();
    require(stream.good(), "No se pudo escribir el checkpoint completo");
}

Snapshot check(std::string_view device, bool expect_packed,
    const std::filesystem::path& checkpoint_input, const std::filesystem::path& checkpoint_output,
    std::vector<LifecycleSample>* lifecycle) {
    PpoHyperparameters parameters;
    parameters.gamma = 0;
    parameters.epochs = 2;
    parameters.minibatch_size = ppo_sequence_length * lanes;
    const auto cpu_random = at::detail::getDefaultCPUGenerator().get_state();
    const auto create = [&] {
        return PpoPolicy(width, parameters, seed, device, default_ppo_memory_bytes,
                         {PpoNetworkKind::gru, default_ppo_hidden_width});
    };
    auto policy = measure_policy(lifecycle, device, {"construction", "first", 0}, create);
    require(at::equal(cpu_random, at::detail::getDefaultCPUGenerator().get_state()),
            "La inicialización consume el generador global de CPU");
    Snapshot result;
    require(policy.optimizer_steps() == 0, "Una política nueva contiene pasos Adam");
    const auto initial = inspect(policy, expect_packed);
    append(result, "initial", initial);
    const auto samples = rollout(policy);
    const auto observations = samples.observations[0];
    const auto output = policy.infer(observations);
    capture(result, "initial_logits", output.logits);
    capture(result, "initial_values", output.values);
    capture(result, "initial_hidden", output.next_state);
    capture(result, "initial_action", policy.act_recurrent(observations, output.next_state).packed);
    const auto statistics = policy.update(samples);
    require(statistics.valid_transitions == ppo_sequence_length * lanes - 1,
            "El relleno contribuye a las transiciones válidas");
    require(statistics.minibatches == static_cast<int64_t>(updates_per_rollout) &&
                policy.optimizer_steps() == updates_per_rollout,
            "El primer update no aplica exactamente cuatro pasos Adam");
    const auto updated = inspect(policy, expect_packed);
    bool changed_parameter = false;
    bool changed_moment = false;
    for (const auto* name : parameter_names) {
        const std::string key = std::string("parameter_") + name;
        changed_parameter |= !at::equal(initial.at(key), updated.at(key));
        changed_moment |= updated.at(std::string("adam_square_") + name).ne(0).any().item<bool>();
    }
    require(changed_parameter && changed_moment, "Adam no modifica parámetros y momentos");
    append(result, "updated", updated);
    std::stringstream saved;
    policy.save(saved);
    if (!checkpoint_output.empty()) {
        write_checkpoint(checkpoint_output, saved.str());
    }
    const auto checkpoint = checkpoint_input.empty() ? saved.str() : read_checkpoint(checkpoint_input);
    const auto load = [&] {
        std::istringstream stream(checkpoint);
        return PpoPolicy::load(stream, device);
    };
    auto restored = measure_policy(lifecycle, device, {"load_from_memory", "first", 0}, load);
    equal(inspect(policy, expect_packed), inspect(restored, expect_packed));
    const auto original_action = policy.act_recurrent(observations, output.next_state);
    const auto restored_action = restored.act_recurrent(observations, output.next_state);
    require(at::equal(original_action.packed, restored_action.packed) &&
                at::equal(original_action.next_state, restored_action.next_state),
            "La recuperación altera la siguiente acción o estado recurrente");
    capture(result, "resumed_action", restored_action.packed);
    const auto next_statistics = policy.update(samples);
    const auto restored_statistics = restored.update(samples);
    require(next_statistics.minibatches == static_cast<int64_t>(updates_per_rollout) &&
                restored_statistics.minibatches == static_cast<int64_t>(updates_per_rollout) &&
                policy.optimizer_steps() == 2 * updates_per_rollout &&
                restored.optimizer_steps() == 2 * updates_per_rollout,
            "La siguiente actualización no alcanza exactamente ocho pasos Adam");
    const auto next = inspect(policy, expect_packed);
    equal(next, inspect(restored, expect_packed));
    append(result, "next_update", next);
    const auto final_output = restored.infer(observations, output.next_state);
    capture(result, "final_logits", final_output.logits);
    capture(result, "final_values", final_output.values);
    capture(result, "final_hidden", final_output.next_state);
    if (lifecycle != nullptr) {
        for (std::size_t repetition = 0; repetition <= lifecycle_repetitions; ++repetition) {
            const auto phase = repetition == 0 ? "warmup" : "measured";
            auto fresh = measure_policy(lifecycle, device, {"construction", phase, repetition}, create);
            equal(initial, inspect(fresh, expect_packed));
            auto loaded = measure_policy(lifecycle, device, {"load_from_memory", phase, repetition}, load);
            equal(updated, inspect(loaded, expect_packed));
        }
    }
    return result;
}

void write_lifecycle(const std::filesystem::path& destination,
                     const std::vector<LifecycleSample>& samples) {
    require(!std::filesystem::exists(destination), "La salida de tiempos ya existe");
    std::ofstream stream(destination);
    stream << "operation,phase,repetition,seconds\n"
           << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto& sample : samples) {
        stream << sample.operation << ',' << sample.phase << ',' << sample.repetition
               << ',' << sample.seconds << '\n';
    }
    stream.close();
    require(stream.good(), "No se pudieron guardar los tiempos completos");
}

void write_snapshot(const Snapshot& snapshot, const std::filesystem::path& destination) {
    require(!std::filesystem::exists(destination), "El diagnóstico de salida ya existe");
    torch::serialize::OutputArchive archive;
    for (const auto& [name, value] : snapshot) {
        archive.write(name, value, true);
    }
    archive.save_to(destination.string());
}

Snapshot read_snapshot(const std::filesystem::path& source) {
    require(std::filesystem::file_size(source) <= maximum_snapshot_bytes,
            "El diagnóstico de referencia supera el límite de lectura");
    torch::serialize::InputArchive archive;
    archive.load_from(source.string(), at::Device(at::kCPU));
    Snapshot result;
    for (const auto& name : archive.keys()) {
        at::Tensor value;
        archive.read(name, value, true);
        result.emplace(name, value);
    }
    return result;
}
}

int main(int argc, char** argv) {
    try {
        std::string device = "cpu";
        std::filesystem::path snapshot;
        std::filesystem::path reference;
        std::filesystem::path checkpoint_input;
        std::filesystem::path checkpoint_output;
        std::filesystem::path lifecycle_output;
        bool expect_packed = false;
        const std::span<char*> arguments(argv, static_cast<std::size_t>(argc));
        for (std::size_t index = 1; index < arguments.size(); ++index) {
            const std::string_view argument(arguments[index]);
            if (argument == "--expect-packed") {
                expect_packed = true;
            } else {
                require(index + 1 < arguments.size(), "Falta el valor del argumento de prueba");
                const std::string value(arguments[++index]);
                if (argument == "--device") {
                    device = value;
                } else if (argument == "--snapshot") {
                    snapshot = value;
                } else if (argument == "--reference") {
                    reference = value;
                } else if (argument == "--checkpoint-input") {
                    checkpoint_input = value;
                } else if (argument == "--checkpoint-output") {
                    checkpoint_output = value;
                } else if (argument == "--benchmark-lifecycle") {
                    lifecycle_output = value;
                } else {
                    throw std::invalid_argument("Argumento de prueba desconocido");
                }
            }
        }
        require(device == "cpu" || device == "cuda:0", "La prueba requiere cpu o cuda:0");
        require(!expect_packed || device == "cuda:0", "La compactación se comprueba en CUDA");
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        at::globalContext().setDeterministicAlgorithms(true, false);
        at::globalContext().setBenchmarkCuDNN(false);
        at::globalContext().setDeterministicCuDNN(true);
        at::globalContext().setFloat32Precision(at::Float32Backend::GENERIC, at::Float32Op::ALL,
                                               at::Float32Precision::IEEE);
        packing_intervals_reject_every_overlapping_pair();
        std::vector<LifecycleSample> lifecycle;
        lifecycle.reserve(2 * (lifecycle_repetitions + 2));
        const auto result = check(device, expect_packed, checkpoint_input, checkpoint_output,
                                  lifecycle_output.empty() ? nullptr : &lifecycle);
        if (!reference.empty()) {
            equal(read_snapshot(reference), result);
        }
        if (!snapshot.empty()) {
            write_snapshot(result, snapshot);
        }
        if (!lifecycle_output.empty()) {
            write_lifecycle(lifecycle_output, lifecycle);
        }
        std::cout << "GRU comprobada en " << device << ", " << result.size()
                  << " tensores de parámetros, Adam, RNG y salidas\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
