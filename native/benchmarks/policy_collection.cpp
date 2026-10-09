// Coste sin aprendizaje del recorrido de las políticas de la etapa RL sobre cintas reconstruidas.
//
// Mide por separado el paso del entorno, la copia de observaciones, la inferencia del actor
// y del crítico con sus copias de vuelta, las ventajas GAE, la recogida completa de PPO con
// `PpoTrainer::advance` antes de su primera actualización, una oleada de KLPO y el forward
// y backward de un minilote PPO. Ninguna medida crea un paso de Adam: al terminar se exige
// que los parámetros conserven su huella y que no haya pasos de optimizador.

#include "mars_titan/klpo_collection.hpp"
#include "mars_titan/policy_tapes.hpp"
#include "mars_titan/ppo_training.hpp"
#include "mars_titan/simulation_files.hpp"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <ATen/record_function.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#if defined(MARS_TITAN_LIBTORCH_CUDA)
#include <torch/cuda.h>
#endif

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using mars_titan::learning::KlpoCollectionOptions;
using mars_titan::learning::KlpoCollectionPhase;
using mars_titan::learning::KlpoTerminalCollector;
using mars_titan::learning::PpoArchitecture;
using mars_titan::learning::PpoHyperparameters;
using mars_titan::learning::PpoPolicy;
using mars_titan::learning::PpoTrainer;
using mars_titan::learning::PpoTrainingConfig;
using mars_titan::learning::PolicyTapeRole;
using mars_titan::simulation::BatchInput;
using mars_titan::simulation::FinancialBatch;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr std::size_t default_environments = 16;
constexpr std::size_t default_ticks = 512;
constexpr std::size_t default_warmup = 32;
constexpr std::size_t captured_observations = 256;
constexpr std::size_t rollout_ticks = 64;
constexpr std::size_t gradient_rows = 64;
constexpr std::size_t evaluation_repeats = 3;
constexpr std::size_t maximum_trainer_rollout = 16384;
constexpr std::size_t diagnostic_transitions = 32;
constexpr std::size_t trainer_rollout_bytes = std::size_t{256} * 1024 * 1024;
constexpr double microseconds = 1e6;
constexpr double stage_capital = 1'000'000;
constexpr double stage_cost_bps = 10;
constexpr double stage_participation = 0.01;
constexpr double stage_score_scale = 0.01;
constexpr double stage_ruin_penalty = -20;
constexpr double greedy_epsilon = 0.05;
constexpr double invalid_fraction = 0.05;
constexpr double terminal_fraction = 0.01;
constexpr uint64_t seed = 42;
constexpr int64_t action_count = mars_titan::learning::ppo_action_count;
constexpr double percentile_50 = 0.5;
constexpr double percentile_95 = 0.95;
constexpr double percentile_99 = 0.99;

struct Options {
    std::vector<std::string> train;
    std::string validation;
    std::string device = "cpu";
    std::string output;
    // Traza de Kineto para atribuir el tiempo a operadores. Sus tiempos no se informan.
    std::string profile;
    std::size_t environments = default_environments;
    std::size_t ticks = default_ticks;
    std::size_t warmup = default_warmup;
    std::size_t workers = 1;
    int threads = 1;
    // Fases omitidas: inference, gae, gradient, trainer, klpo o evaluation.
    std::set<std::string, std::less<>> skip;
};

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

std::size_t count(std::string_view text) {
    std::size_t value = 0;
    const auto parsed = std::from_chars(text.begin(), text.end(), value);
    require(parsed.ec == std::errc{} && parsed.ptr == text.end() && value > 0, "Se necesita un entero positivo");
    return value;
}

Options parse(std::span<char*> arguments) {
    Options result;
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const std::string_view name(arguments[index]);
        require(++index < arguments.size(), "Falta el valor de un argumento");
        const std::string_view value(arguments[index]);
        if (name == "--train-tape") {
            result.train.emplace_back(value);
        } else if (name == "--validation-tape") {
            result.validation = value;
        } else if (name == "--device") {
            result.device = value;
        } else if (name == "--output") {
            result.output = value;
        } else if (name == "--profile") {
            result.profile = value;
        } else if (name == "--environments") {
            result.environments = count(value);
        } else if (name == "--ticks") {
            result.ticks = count(value);
        } else if (name == "--warmup") {
            result.warmup = count(value);
        } else if (name == "--workers") {
            result.workers = count(value);
        } else if (name == "--skip") {
            require(value == "inference" || value == "gae" || value == "gradient" || value == "trainer" ||
                        value == "klpo" || value == "evaluation",
                    "Fase desconocida");
            result.skip.emplace(value);
        } else if (name == "--threads") {
            result.threads = static_cast<int>(count(value));
        } else {
            throw std::invalid_argument("Argumento desconocido");
        }
    }
    require(!result.train.empty() && !result.validation.empty() && !result.output.empty() &&
                (result.device == "cpu" || result.device == "cuda:0"),
            "Se necesitan cintas de ajuste y validación, una salida y el dispositivo cpu o cuda:0");
    return result;
}

void synchronize(const std::string& device) {
#if defined(MARS_TITAN_LIBTORCH_CUDA)
    if (device != "cpu") {
        torch::cuda::synchronize(0);
    }
#else
    require(device == "cpu", "Este ejecutable no enlaza el backend CUDA de LibTorch");
#endif
}

Json summary(std::vector<double> seconds, double items_per_call) {
    require(!seconds.empty(), "La medida necesita repeticiones");
    std::sort(seconds.begin(), seconds.end());
    const auto at = [&](double quantile) {
        const auto position = static_cast<std::size_t>(quantile * static_cast<double>(seconds.size() - 1));
        return seconds.at(position) * microseconds;
    };
    const auto total = std::accumulate(seconds.begin(), seconds.end(), 0.);
    return Json{{"calls", seconds.size()},
                {"p50_us", at(percentile_50)},
                {"p95_us", at(percentile_95)},
                {"p99_us", at(percentile_99)},
                {"mean_us", total / static_cast<double>(seconds.size()) * microseconds},
                {"items_per_second", items_per_call * static_cast<double>(seconds.size()) / total}};
}

template<class Function> double timed(Function&& function) {
    const auto start = Clock::now();
    std::forward<Function>(function)();
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Huella de bytes, tipo y forma de tensores CPU, para comparar recorridos antes y después.
std::string tensor_digest(const std::vector<at::Tensor>& tensors) {
    std::string bytes;
    for (const auto& tensor : tensors) {
        if (!tensor.defined()) {
            bytes += "undefined;";
            continue;
        }
        const auto value = tensor.detach().to(at::kCPU).contiguous();
        bytes += std::string(c10::toString(value.scalar_type())) + ":";
        for (const auto size : value.sizes()) {
            bytes += std::to_string(size) + ",";
        }
        bytes.append(static_cast<const char*>(value.const_data_ptr()), value.nbytes());
    }
    return mars_titan::simulation::content_sha256(bytes);
}

mars_titan::simulation::Parameters stage_parameters() {
    mars_titan::simulation::Parameters result;
    result.capital = stage_capital;
    result.cost_bps = stage_cost_bps;
    result.participation = stage_participation;
    result.score_scale = stage_score_scale;
    result.ruin_penalty = stage_ruin_penalty;
    return result;
}

// Los entornos recorren en ciclo las cintas de ajuste, como en el esquema 4 de PPO y en KLPO.
std::vector<BatchInput> lanes(const std::vector<BatchInput>& tapes, std::size_t environments) {
    std::vector<BatchInput> result;
    result.reserve(environments);
    for (std::size_t lane = 0; lane < environments; ++lane) {
        result.push_back(tapes[lane % tapes.size()]);
    }
    return result;
}

at::Tensor host_rows(std::span<const float> values, std::size_t rows, std::size_t width, bool pinned) {
    auto result = at::empty({static_cast<int64_t>(rows), static_cast<int64_t>(width)},
                            at::TensorOptions().dtype(at::kFloat).pinned_memory(pinned));
    std::memcpy(result.data_ptr<float>(), values.data(), values.size_bytes());
    return result;
}

struct Environment {
    Json report;
    std::vector<std::vector<float>> observations;
    std::size_t width = 0;
};

// Paso contable con un ciclo fijo de las seis acciones y reinicio de los entornos terminados.
Environment measure_environment(const std::vector<BatchInput>& inputs, const Options& options) {
    FinancialBatch batch(inputs, options.workers);
    std::vector<uint8_t> actions(batch.size());
    std::vector<std::size_t> finished;
    std::vector<double> seconds;
    Environment result;
    result.width = batch.observation_width();
    for (std::size_t step = 0; step < options.warmup + options.ticks; ++step) {
        for (std::size_t lane = 0; lane < actions.size(); ++lane) {
            actions[lane] = static_cast<uint8_t>((lane + step) % action_count);
        }
        if (result.observations.size() < captured_observations) {
            const auto view = batch.observations();
            result.observations.emplace_back(view.begin(), view.end());
        }
        const auto elapsed = timed([&] {
            const auto& transition = batch.step(actions);
            finished.clear();
            for (std::size_t lane = 0; lane < actions.size(); ++lane) {
                if (transition.terminated[lane] != 0 || transition.truncated[lane] != 0) {
                    finished.push_back(lane);
                }
            }
            if (!finished.empty()) {
                batch.reset(finished);
            }
        });
        if (step >= options.warmup) {
            seconds.push_back(elapsed);
        }
    }
    result.report = summary(seconds, static_cast<double>(batch.size()));
    result.report["observation_width"] = result.width;
    return result;
}

PpoArchitecture architecture(bool double_dqn) {
    PpoArchitecture result;
    result.double_dqn = double_dqn;
    return result;
}

// Copia, acción con muestreo y bootstrap del crítico, como en cada paso de la recogida PPO.
Json measure_inference(const Environment& environment, const Options& options, bool double_dqn,
                       bool pinned) {
    const auto rows = options.environments;
    PpoPolicy policy(environment.width, PpoHyperparameters{}, seed, options.device,
                     mars_titan::learning::default_ppo_memory_bytes, architecture(double_dqn));
    const auto fingerprint = policy.parameter_fingerprint();
    const at::Device device(options.device);
    auto state = policy.initial_state(rows);
    std::vector<double> copy;
    std::vector<double> act;
    std::vector<double> bootstrap;
    const auto calls = options.warmup + options.ticks;
    for (std::size_t call = 0; call < calls; ++call) {
        const auto& current = environment.observations[call % environment.observations.size()];
        const auto& following = environment.observations[(call + 1) % environment.observations.size()];
        at::Tensor observation;
        at::Tensor next;
        const auto copy_seconds = timed([&] {
            observation = host_rows(current, rows, environment.width, pinned).to(device, pinned);
            next = host_rows(following, rows, environment.width, pinned).to(device, pinned);
            synchronize(options.device);
        });
        mars_titan::learning::PpoAction chosen;
        const auto act_seconds = timed([&] {
            chosen = double_dqn ? policy.act_double_dqn(observation, greedy_epsilon)
                                : policy.act_recurrent(observation, state);
            const auto packed = chosen.packed.to(at::kCPU).contiguous();
            require(std::isfinite(*packed.const_data_ptr<double>()), "Acción no finita");
        });
        const auto bootstrap_seconds = timed([&] {
            const auto values = policy.infer(next, chosen.next_state).values.to(at::kCPU, at::kDouble).contiguous();
            require(std::isfinite(*values.const_data_ptr<double>()), "Valor no finito");
        });
        if (call >= options.warmup) {
            copy.push_back(copy_seconds);
            act.push_back(act_seconds);
            bootstrap.push_back(bootstrap_seconds);
        }
    }
    require(policy.parameter_fingerprint() == fingerprint && policy.optimizer_steps() == 0,
            "La inferencia cambió los parámetros");
    const auto lanes_per_call = static_cast<double>(rows);
    return Json{{"architecture", double_dqn ? "double_dqn_mlp64" : "ppo_mlp64"},
                {"pinned_host_memory", pinned},
                {"copy_two_observation_batches", summary(copy, lanes_per_call)},
                {"act_and_copy_back", summary(act, lanes_per_call)},
                {"bootstrap_and_copy_back", summary(bootstrap, lanes_per_call)}};
}

// GAE FP64 en CPU de un recorrido de 1.024 transiciones con 16 entornos.
Json measure_gae(const Options& options) {
    const auto time = static_cast<int64_t>(rollout_ticks);
    const auto lanes_count = static_cast<int64_t>(options.environments);
    auto generator = at::detail::createCPUGenerator(seed);
    mars_titan::learning::PpoRollout rollout;
    rollout.rewards = at::randn({time, lanes_count}, generator, at::kDouble);
    rollout.old_values = at::randn({time, lanes_count}, generator, at::kDouble);
    rollout.next_values = at::randn({time, lanes_count}, generator, at::kDouble);
    rollout.reward_valid = at::rand({time, lanes_count}, generator) > invalid_fraction;
    rollout.terminated = at::rand({time, lanes_count}, generator) > 1 - terminal_fraction;
    rollout.truncated = at::zeros({time, lanes_count}, at::kBool);
    rollout.observations = at::zeros({time, lanes_count, 1}, at::kFloat);
    rollout.actions = at::zeros({time, lanes_count}, at::kLong);
    rollout.old_log_probabilities = at::zeros({time, lanes_count}, at::kDouble);
    const PpoHyperparameters parameters;
    std::vector<double> seconds;
    for (std::size_t call = 0; call < options.warmup + options.ticks; ++call) {
        const auto elapsed = timed([&] { static_cast<void>(mars_titan::learning::ppo_gae(rollout, parameters)); });
        if (call >= options.warmup) {
            seconds.push_back(elapsed);
        }
    }
    return summary(seconds, static_cast<double>(time * lanes_count));
}

// Recogida PPO de producción: PpoTrainer::advance antes de completar su primer recorrido.
// En CPU solo se admite el diagnóstico de 32 transiciones: un carril y 31 pasos, sin tiempos.
Json measure_trainer(const std::vector<BatchInput>& all_inputs, const Options& options) {
    const bool diagnostic = options.device == "cpu";
    const auto inputs = diagnostic ? std::vector<BatchInput>{all_inputs.front()} : all_inputs;
    PpoTrainingConfig config;
    config.total_transitions = diagnostic ? diagnostic_transitions : maximum_trainer_rollout;
    config.rollout_transitions = config.total_transitions;
    config.rollout_bytes = trainer_rollout_bytes;
    config.workers = options.workers;
    config.seed = seed;
    const auto last_tick = config.rollout_transitions / inputs.size();
    const auto warmup = diagnostic ? 0 : options.warmup;
    const auto ticks = diagnostic ? last_tick - 1 : options.ticks;
    require(warmup + ticks < last_tick, "La medida alcanzaría la primera actualización PPO");
    // Los brazos PPO de la etapa declaran objetivos explícitos y guardan los seis pesos.
    mars_titan::learning::PpoObjectiveConfig objective;
    objective.kind = mars_titan::learning::PpoObjectiveKind::clip_full_kl;
    PpoTrainer trainer(inputs, config, PpoHyperparameters{}, options.device, diagnostic, {}, objective);
    const auto fingerprint = trainer.policy().parameter_fingerprint();
    std::vector<double> seconds;
    for (std::size_t tick = 0; tick < warmup + ticks; ++tick) {
        const auto elapsed = timed([&] { require(trainer.advance(), "La recogida terminó antes de lo previsto"); });
        if (tick >= warmup) {
            seconds.push_back(elapsed);
        }
    }
    require(trainer.optimizer_steps() == 0 && trainer.policy().optimizer_steps() == 0 &&
                trainer.policy().parameter_fingerprint() == fingerprint &&
                trainer.partial_ticks() == warmup + ticks,
            "La recogida PPO llegó a una actualización");
    const auto state = trainer.snapshot();
    const auto& rollout = state.rollout;
    Json sessions = Json::array();
    for (const auto& session : state.environment.sessions) {
        sessions.push_back(mars_titan::simulation::snapshot_json(session));
    }
    auto result = summary(seconds, static_cast<double>(inputs.size()));
    result["diagnostic"] = diagnostic;
    result["lanes"] = inputs.size();
    result["rollout_sha256"] = tensor_digest({rollout.observations, rollout.actions, rollout.old_log_probabilities,
        rollout.old_values, rollout.rewards, rollout.next_values, rollout.reward_valid, rollout.terminated,
        rollout.truncated, rollout.old_action_weights});
    result["environment_sha256"] = mars_titan::simulation::content_sha256(sessions.dump());
    result["sampler_sha256"] = tensor_digest({trainer.policy().random_state().sampling});
    result["episodes"] = state.episodes;
    return result;
}

// Una oleada KLPO hasta la fase ready y el forward y backward de su objetivo, sin Adam.
Json measure_klpo(const std::vector<BatchInput>& inputs, const Options& options) {
    KlpoCollectionOptions collection;
    collection.device = options.device;
    collection.workers = options.workers;
    collection.seed = seed;
    KlpoTerminalCollector collector(inputs, collection);
    const auto fingerprint = collector.policy().parameter_fingerprint();
    std::vector<double> seconds;
    while (collector.phase() == KlpoCollectionPhase::collecting) {
        seconds.push_back(timed([&] { static_cast<void>(collector.collect_tick()); }));
    }
    require(collector.phase() == KlpoCollectionPhase::ready, "La oleada KLPO no quedó lista");
    std::size_t transitions = 0;
    for (const auto& input : inputs) {
        transitions += input.tape->close_times.size() - 1;
    }
    const auto ticks = static_cast<double>(seconds.size());
    const auto total = std::accumulate(seconds.begin(), seconds.end(), 0.);
    Json result{{"wave_ticks", seconds.size()},
                {"wave_transitions", transitions},
                {"collection_tick", summary(seconds, static_cast<double>(transitions) / ticks)},
                {"collection_wave_seconds", total}};
    const auto objective_seconds = timed([&] {
        const auto objective = collector.objective();
        objective.mean.backward();
        synchronize(options.device);
    });
    require(collector.policy().parameter_fingerprint() == fingerprint && collector.policy().optimizer_steps() == 0,
            "La oleada KLPO cambió el actor");
    result["records_sha256"] =
        mars_titan::simulation::content_sha256(mars_titan::learning::serialize_klpo_batch(collector.records()));
    result["sampler_sha256"] = tensor_digest({collector.policy().random_state().sampling});
    result["objective_forward_backward_seconds"] = objective_seconds;
    return result;
}

// Forward y backward de un minilote del objetivo PPO recortado, sin optimizador ni recorte.
// Las comprobaciones de terminal_forward añaden dos sincronizaciones, como las de la actualización.
Json measure_gradient(const Environment& environment, const Options& options) {
    PpoPolicy policy(environment.width, PpoHyperparameters{}, seed, options.device);
    const auto fingerprint = policy.parameter_fingerprint();
    const at::Device device(options.device);
    std::vector<float> rows;
    for (std::size_t index = 0; rows.size() < gradient_rows * environment.width; ++index) {
        const auto& source = environment.observations[index % environment.observations.size()];
        rows.insert(rows.end(), source.begin(), source.begin() + static_cast<std::ptrdiff_t>(environment.width));
    }
    // terminal_forward conserva el autograd del llamante. Con un paso de historia es la MLP del minilote.
    const auto observations =
        host_rows(rows, gradient_rows, environment.width, false).to(device).unsqueeze(0);
    const auto lengths = at::ones({static_cast<int64_t>(gradient_rows)}, at::kLong).to(device);
    auto generator = at::detail::createCPUGenerator(seed);
    const auto size = static_cast<int64_t>(gradient_rows);
    const auto actions = at::randint(action_count, {size}, generator, at::kLong).to(device);
    const auto previous = at::full({size}, -std::log(static_cast<double>(action_count)), at::kDouble).to(device);
    const auto advantages = at::randn({size}, generator, at::kFloat).to(device);
    const auto returns = at::randn({size}, generator, at::kFloat).to(device);
    const PpoHyperparameters parameters;
    std::vector<double> seconds;
    for (std::size_t call = 0; call < options.warmup + options.ticks; ++call) {
        const auto elapsed = timed([&] {
            const auto output = policy.terminal_forward(observations, lengths);
            const auto logp = output.logits.squeeze(0).log_softmax(-1);
            const auto selected = logp.gather(1, actions.unsqueeze(1)).squeeze(1);
            const auto policy_loss = -mars_titan::learning::ppo_clipped_objective(
                selected, previous, advantages, parameters.clip).mean();
            const auto value_loss = (output.values.squeeze(0) - returns).square().mean();
            const auto entropy = -(logp.exp() * logp).sum(-1).mean();
            const auto loss = policy_loss + parameters.value_weight * value_loss - parameters.entropy * entropy;
            loss.backward();
            synchronize(options.device);
        });
        if (call >= options.warmup) {
            seconds.push_back(elapsed);
        }
    }
    // Forward sin gradiente del mismo minilote: Double DQN añade dos por actualización.
    const auto rows_2d = observations.squeeze(0);
    std::vector<double> forward;
    for (std::size_t call = 0; call < options.warmup + options.ticks; ++call) {
        const auto elapsed = timed([&] {
            static_cast<void>(policy.infer(rows_2d));
            synchronize(options.device);
        });
        if (call >= options.warmup) {
            forward.push_back(elapsed);
        }
    }
    require(policy.parameter_fingerprint() == fingerprint && policy.optimizer_steps() == 0,
            "El backward cambió los parámetros");
    return Json{{"forward_backward", summary(seconds, 1)}, {"forward_no_grad", summary(forward, 1)},
                {"rows", gradient_rows}};
}

// Evaluación de un carril con argmax sobre la cinta de validación, como la selección.
Json measure_evaluation(const BatchInput& validation, std::size_t width, const Options& options) {
    PpoPolicy policy(width, PpoHyperparameters{}, seed, options.device);
    const auto transitions = static_cast<double>(validation.tape->close_times.size() - 1);
    std::vector<double> seconds;
    seconds.reserve(evaluation_repeats);
    for (std::size_t repeat = 0; repeat < evaluation_repeats; ++repeat) {
        seconds.push_back(timed([&] {
            static_cast<void>(mars_titan::learning::evaluate_policy(policy, {validation}, options.workers));
        }));
    }
    require(policy.optimizer_steps() == 0, "La evaluación cambió la política");
    return summary(seconds, transitions);
}

void configure(const Options& options) {
    // Igual que los binarios de política: hilos acotados, algoritmos deterministas y FP32 IEEE.
    require(setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 0) == 0, "No se pudo fijar el espacio de cuBLAS");
    at::set_num_threads(options.threads);
    if (at::get_num_interop_threads() != 1) {
        at::set_num_interop_threads(1);
    }
    at::globalContext().setDeterministicAlgorithms(true, false);
    at::globalContext().setBenchmarkCuDNN(false);
    at::globalContext().setDeterministicCuDNN(true);
    at::globalContext().setFloat32Precision(at::Float32Backend::GENERIC, at::Float32Op::ALL,
                                            at::Float32Precision::IEEE);
#if defined(MARS_TITAN_LIBTORCH_CUDA)
    require(options.device == "cpu" || torch::cuda::is_available(), "CUDA no está disponible");
#endif
}

// Activa Kineto durante su vida si se pide una traza y la guarda al terminar.
class Profile {
  public:
    explicit Profile(const Options& options) : path_(options.profile) {
        if (path_.empty()) {
            return;
        }
        namespace profiler = torch::autograd::profiler;
        std::set<torch::profiler::impl::ActivityType> activities{torch::profiler::impl::ActivityType::CPU};
        if (options.device != "cpu") {
            activities.insert(torch::profiler::impl::ActivityType::CUDA);
        }
        const torch::profiler::impl::ProfilerConfig config(torch::profiler::impl::ProfilerState::KINETO);
        profiler::prepareProfiler(config, activities);
        profiler::enableProfiler(config, activities);
    }
    Profile(const Profile&) = delete;
    Profile& operator=(const Profile&) = delete;
    Profile(Profile&&) = delete;
    Profile& operator=(Profile&&) = delete;
    ~Profile() {
        if (!path_.empty()) {
            torch::autograd::profiler::disableProfiler()->save(path_);
        }
    }

  private:
    std::string path_;
};

int run(const Options& options) {
    configure(options);
    const auto parameters = stage_parameters();
    std::vector<BatchInput> train;
    train.reserve(options.train.size());
    for (const auto& folder : options.train) {
        train.push_back(mars_titan::learning::load_policy_tape(folder, PolicyTapeRole::train, parameters).input);
    }
    const auto validation =
        mars_titan::learning::load_policy_tape(options.validation, PolicyTapeRole::validation, parameters).input;
    const auto inputs = lanes(train, options.environments);
    const auto started = Clock::now();
    Json report;
    {
    const Profile profile(options);
    auto environment = [&] {
        RECORD_USER_SCOPE("phase_environment");
        return measure_environment(inputs, options);
    }();
    report = Json{{"kind", "rl_stage_policy_collection_benchmark"},
                {"learning", "none_no_optimizer_steps"},
                {"device", options.device},
                {"environments", options.environments},
                {"assets", train.front().tape->assets.size()},
                {"train_tapes", options.train.size()},
                {"ticks", options.ticks},
                {"warmup", options.warmup},
                {"workers", options.workers},
                {"torch_threads", at::get_num_threads()},
                {"torch_interop_threads", at::get_num_interop_threads()},
                {"profiled", !options.profile.empty()},
                {"environment_step", environment.report}};
    const auto wanted = [&](std::string_view phase) { return !options.skip.contains(phase); };
    Json inference = Json::array();
    for (const bool double_dqn : {false, true}) {
        if (!wanted("inference")) {
            break;
        }
        for (const bool pinned : {false, true}) {
            if (pinned && options.device == "cpu") {
                continue;
            }
            RECORD_USER_SCOPE(double_dqn ? "phase_inference_double_dqn" : "phase_inference_ppo");
            inference.push_back(measure_inference(environment, options, double_dqn, pinned));
        }
    }
    report["inference"] = inference;
    if (wanted("gae")) {
        RECORD_USER_SCOPE("phase_gae");
        report["gae_cpu_fp64"] = measure_gae(options);
    }
    if (wanted("gradient")) {
        RECORD_USER_SCOPE("phase_minibatch_gradient");
        report["ppo_minibatch"] = measure_gradient(environment, options);
    }
    if (wanted("trainer")) {
        RECORD_USER_SCOPE("phase_ppo_trainer");
        report["ppo_trainer_collection"] = measure_trainer(inputs, options);
    }
    if (wanted("klpo")) {
        RECORD_USER_SCOPE("phase_klpo_wave");
        report["klpo_wave"] = measure_klpo(inputs, options);
    }
    if (wanted("evaluation")) {
        RECORD_USER_SCOPE("phase_evaluation");
        report["evaluation_single_lane"] = measure_evaluation(validation, environment.width, options);
    }
    }
    report["total_seconds"] = std::chrono::duration<double>(Clock::now() - started).count();
    report["executable_peak_rss_bytes"] = mars_titan::simulation::process_memory_high_water();
    std::ofstream(options.output) << report.dump(2) << '\n';
    std::cout << report.dump() << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse(std::span(argv, static_cast<std::size_t>(argc))));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
