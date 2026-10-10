// Coste sin aprendizaje del recorrido de las políticas de la etapa RL sobre cintas reconstruidas.
//
// Mide por separado el paso del entorno, la copia de observaciones, la inferencia del actor
// y del crítico con sus copias de vuelta, las ventajas GAE, la recogida completa de PPO con
// `PpoTrainer::advance` antes de su primera actualización, el primer paso de un entrenador
// recién creado, que también admite la CPU, una oleada de KLPO y el forward
// y backward de un minilote PPO. También mide la acción y la pérdida de valor de Double DQN y
// de sus cabezas cuantílicas sobre transiciones recogidas de la cinta, y el forward y backward
// de los cuatro objetivos de grupo sobre la misma oleada que KLPO. Ninguna medida crea un paso
// de Adam: al terminar se exige que los parámetros conserven su huella y que no haya pasos de
// optimizador. Con `--repeat` las recogidas PPO y KLPO se repiten para medir varios procesos
// simultáneos.

#include "mars_titan/group_waves.hpp"
#include "mars_titan/klpo_collection.hpp"
#include "mars_titan/policy_tapes.hpp"
#include "mars_titan/ppo_training.hpp"
#include "mars_titan/quantile_dqn.hpp"
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
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
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
constexpr std::size_t first_tick_repeats = 64;
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
    // Repeticiones de las recogidas PPO y KLPO para medir procesos simultáneos en régimen estable.
    std::size_t repeat = 1;
    // Fases omitidas: inference, gae, gradient, trainer, first_tick, klpo, evaluation,
    // value_minibatch o group.
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
                        value == "first_tick" || value == "klpo" || value == "evaluation" ||
                        value == "value_minibatch" || value == "group",
                    "Fase desconocida");
            result.skip.emplace(value);
        } else if (name == "--threads") {
            result.threads = static_cast<int>(count(value));
        } else if (name == "--repeat") {
            result.repeat = count(value);
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
    // Acción, recompensa y cierre de cada paso capturado, para formar transiciones de la cinta.
    std::vector<std::vector<uint8_t>> actions;
    std::vector<std::vector<double>> rewards;
    std::vector<std::vector<uint8_t>> closed;
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
        const bool captured = result.observations.size() < captured_observations;
        if (captured) {
            const auto view = batch.observations();
            result.observations.emplace_back(view.begin(), view.end());
            result.actions.push_back(actions);
        }
        const auto elapsed = timed([&] {
            const auto& transition = batch.step(actions);
            if (captured) {
                result.rewards.push_back(transition.rewards);
                std::vector<uint8_t> closed(actions.size());
                for (std::size_t lane = 0; lane < actions.size(); ++lane) {
                    closed[lane] = static_cast<uint8_t>(transition.terminated[lane] != 0 ||
                                                        transition.truncated[lane] != 0 ||
                                                        transition.reward_valid[lane] == 0);
                }
                result.closed.push_back(std::move(closed));
            }
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

// Arquitecturas de la acción medidas, con el nombre que publica el informe.
struct Head {
    std::string_view name;
    bool double_dqn = false;
    int64_t quantiles = 0;
    double risk_alpha = 1;
};
constexpr std::array<Head, 4> heads{
    Head{"ppo_mlp64"}, Head{"double_dqn_mlp64", true},
    Head{"qr_dqn_mlp64", true, mars_titan::learning::qr_dqn_quantiles},
    Head{"qr_dqn_cvar_mlp64", true, mars_titan::learning::qr_dqn_quantiles,
         mars_titan::learning::qr_dqn_cvar_alpha}};

PpoArchitecture architecture(const Head& head) {
    PpoArchitecture result;
    result.double_dqn = head.double_dqn;
    result.quantiles = head.quantiles;
    result.risk_alpha = head.risk_alpha;
    return result;
}

// Copia, acción con muestreo y bootstrap del crítico, como en cada paso de la recogida PPO.
Json measure_inference(const Environment& environment, const Options& options, const Head& head,
                       bool pinned) {
    const auto rows = options.environments;
    const bool double_dqn = head.double_dqn;
    PpoPolicy policy(environment.width, PpoHyperparameters{}, seed, options.device,
                     mars_titan::learning::default_ppo_memory_bytes, architecture(head));
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
    return Json{{"architecture", head.name},
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

// Primer paso de un entrenador recién creado con todos los entornos. Es la única recogida a escala
// completa que admite la CPU: su diagnóstico de 32 transiciones cubre dos pasos de 16 entornos y el
// segundo ya cerraría el recorrido. El primer paso no tiene un bootstrap anterior que reutilizar, así
// que muestrea con su propio forward, como la recogida anterior a esa reutilización.
Json measure_first_tick(const std::vector<BatchInput>& inputs, const Options& options) {
    const bool diagnostic = options.device == "cpu";
    PpoTrainingConfig config;
    config.total_transitions = 2 * inputs.size();
    config.rollout_transitions = config.total_transitions;
    config.rollout_bytes = trainer_rollout_bytes;
    config.workers = options.workers;
    config.seed = seed;
    require(!diagnostic || config.total_transitions <= diagnostic_transitions,
            "El primer paso en CPU necesita como mucho 16 entornos");
    mars_titan::learning::PpoObjectiveConfig objective;
    objective.kind = mars_titan::learning::PpoObjectiveKind::clip_full_kl;
    std::vector<double> seconds;
    for (std::size_t call = 0; call < options.warmup + first_tick_repeats; ++call) {
        PpoTrainer trainer(inputs, config, PpoHyperparameters{}, options.device, diagnostic, {}, objective);
        const auto fingerprint = trainer.policy().parameter_fingerprint();
        const auto elapsed = timed([&] { require(trainer.advance(), "La recogida terminó antes de lo previsto"); });
        require(trainer.optimizer_steps() == 0 && trainer.policy().optimizer_steps() == 0 &&
                    trainer.policy().parameter_fingerprint() == fingerprint && trainer.partial_ticks() == 1,
                "El primer paso PPO llegó a una actualización");
        if (call >= options.warmup) {
            seconds.push_back(elapsed);
        }
    }
    auto result = summary(seconds, static_cast<double>(inputs.size()));
    result["diagnostic"] = diagnostic;
    result["lanes"] = inputs.size();
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

// Pérdida de valor de un minilote de 64 transiciones de la cinta, con los dos forwards sin
// gradiente de sus objetivos y el backward, como cada actualización de Double DQN o QR-DQN sin
// el paso de Adam. Las transiciones son pares consecutivos capturados al recorrer la cinta con el
// ciclo fijo de acciones, con su recompensa contable real.
Json measure_value_minibatch(const Environment& environment, const Options& options, const Head& head) {
    PpoPolicy policy(environment.width, PpoHyperparameters{}, seed, options.device,
                     mars_titan::learning::default_ppo_memory_bytes, architecture(head));
    const auto fingerprint = policy.parameter_fingerprint();
    std::vector<float> current;
    std::vector<float> following;
    std::vector<int64_t> actions;
    std::vector<float> rewards;
    const auto lanes_count = environment.actions.front().size();
    for (std::size_t tick = 0; tick + 1 < environment.observations.size() && actions.size() < gradient_rows;
         ++tick) {
        for (std::size_t lane = 0; lane < lanes_count && actions.size() < gradient_rows; ++lane) {
            if (environment.closed[tick][lane] != 0) {
                continue;
            }
            const auto row = [&](const std::vector<float>& values) {
                return std::span(values).subspan(lane * environment.width, environment.width);
            };
            const auto now = row(environment.observations[tick]);
            const auto next = row(environment.observations[tick + 1]);
            current.insert(current.end(), now.begin(), now.end());
            following.insert(following.end(), next.begin(), next.end());
            actions.push_back(environment.actions[tick][lane]);
            rewards.push_back(static_cast<float>(environment.rewards[tick][lane]));
        }
    }
    require(actions.size() == gradient_rows, "La cinta no ofrece 64 transiciones válidas");
    const at::Device device(options.device);
    const auto size = static_cast<int64_t>(gradient_rows);
    const mars_titan::learning::DqnBatch batch{
        host_rows(current, gradient_rows, environment.width, false).to(device),
        host_rows(following, gradient_rows, environment.width, false).to(device),
        at::tensor(actions, at::kLong).to(device),
        at::tensor(rewards, at::kFloat).to(device),
        at::zeros({size}, at::kBool).to(device),
        at::ones({size}, at::kBool).to(device)};
    std::vector<double> seconds;
    double loss = 0;
    for (std::size_t call = 0; call < options.warmup + options.ticks; ++call) {
        const auto elapsed = timed([&] {
            const auto computed = policy.double_dqn_loss(batch);
            computed.loss.backward();
            synchronize(options.device);
            loss = computed.loss.item<double>();
        });
        if (call >= options.warmup) {
            seconds.push_back(elapsed);
        }
    }
    require(std::isfinite(loss) && policy.parameter_fingerprint() == fingerprint && policy.optimizer_steps() == 0,
            "La pérdida de valor cambió los parámetros");
    return Json{{"architecture", head.name},
                {"rows", gradient_rows},
                {"loss_forward_backward", summary(seconds, 1)},
                {"parameters", policy.parameter_count()}};
}

// Una oleada recogida como KLPO y el forward y backward de los cuatro objetivos de grupo y del
// objetivo KLPO por bloques de 8 episodios, como el controlador, sin Adam. Con la traza se mide
// además lo que añaden los diagnósticos de la oleada.
Json measure_group(const std::vector<BatchInput>& inputs, const Options& options) {
    KlpoCollectionOptions collection;
    collection.device = options.device;
    collection.workers = options.workers;
    collection.seed = seed;
    KlpoTerminalCollector collector(inputs, collection);
    while (collector.phase() == KlpoCollectionPhase::collecting) {
        static_cast<void>(collector.collect_tick());
    }
    require(collector.phase() == KlpoCollectionPhase::ready, "La oleada de grupo no quedó lista");
    const auto& records = collector.records();
    PpoPolicy actor(collector.policy().observation_width(), PpoHyperparameters{}, seed, options.device);
    const auto fingerprint = actor.parameter_fingerprint();
    require(fingerprint == records.reference_sha256, "El actor no parte de la referencia de la oleada");
    std::size_t transitions = 0;
    for (const auto& episode : records.episodes) {
        transitions += episode.steps.size();
    }
    constexpr std::size_t block = 8;
    constexpr std::size_t repeats = 5;
    const auto blocks = [&](const std::function<void(const mars_titan::learning::KlpoEpisodeBatch&, std::size_t)>&
                                function) {
        for (std::size_t begin = 0; begin < records.episodes.size(); begin += block) {
            const auto end = std::min(records.episodes.size(), begin + block);
            mars_titan::learning::KlpoEpisodeBatch part{records.fold, records.reference_sha256, records.beta,
                                                        records.gamma, records.observation_width,
                                                        records.max_bytes, {}};
            part.episodes.assign(records.episodes.begin() + static_cast<std::ptrdiff_t>(begin),
                                 records.episodes.begin() + static_cast<std::ptrdiff_t>(end));
            function(part, begin);
        }
        synchronize(options.device);
    };
    const auto measured = [&](const std::function<void()>& function) {
        function();
        std::vector<double> seconds;
        seconds.reserve(repeats);
        for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
            seconds.push_back(timed(function));
        }
        return summary(seconds, static_cast<double>(transitions));
    };
    Json result{{"wave_transitions", transitions},
                {"episodes", records.episodes.size()},
                {"block_episodes", block},
                {"records_sha256", mars_titan::simulation::content_sha256(
                                       mars_titan::learning::serialize_klpo_batch(records))}};
    result["klpo_terminal_token_full_v1"] = measured([&] {
        blocks([&](const auto& part, std::size_t) {
            const auto objective = mars_titan::learning::klpo_episode_objective(actor, part);
            if (!objective.no_policy_decisions) {
                (objective.per_episode.sum() / static_cast<double>(records.episodes.size())).backward();
            }
        });
    });
    for (const auto* id : {"grpo_outcome_v1", "dr_grpo_outcome_v1", "dapo_outcome_static_v1", "gspo_outcome_v1"}) {
        const auto config = mars_titan::learning::published_group_objective(id);
        for (const bool traced : {false, true}) {
            auto row = measured([&] {
                const auto wave = mars_titan::learning::group_wave(records, config);
                mars_titan::learning::GroupObjectiveTrace merged;
                blocks([&](const auto& part, std::size_t begin) {
                    mars_titan::learning::GroupObjectiveTrace trace;
                    const auto loss = mars_titan::learning::group_block_loss(actor, part, begin, wave, config,
                                                                             traced ? &trace : nullptr);
                    if (loss.defined()) {
                        loss.backward();
                        if (traced) {
                            mars_titan::learning::accumulate_group_trace(merged, trace, config.ratio());
                        }
                    }
                });
                if (traced) {
                    static_cast<void>(mars_titan::learning::group_wave_trace(wave));
                }
            });
            result[std::string(id) + (traced ? "_with_trace" : "")] = std::move(row);
        }
    }
    require(actor.parameter_fingerprint() == fingerprint && actor.optimizer_steps() == 0,
            "Los objetivos de grupo cambiaron al actor");
    return result;
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

// Las repeticiones parten de la misma semilla y deben reproducir todas las huellas de la primera.
bool same_collection(const Json& first, const Json& other) {
    for (const auto& [phase, values] : first.items()) {
        for (const auto& [key, value] : values.items()) {
            if (key.ends_with("_sha256") && other.at(phase).at(key) != value) {
                return false;
            }
        }
    }
    return true;
}

// Intervalo de reloj de pared de las repeticiones y tiempos de cada una, para sumar el caudal
// de varios procesos simultáneos sobre su intervalo común.
Json repeated(const std::vector<Json>& passes, std::chrono::system_clock::time_point first,
              std::chrono::system_clock::time_point last) {
    const auto unix_seconds = [](std::chrono::system_clock::time_point point) {
        return std::chrono::duration<double>(point.time_since_epoch()).count();
    };
    Json trainer = Json::array();
    Json waves = Json::array();
    for (const auto& pass : passes) {
        if (pass.contains("ppo_trainer_collection")) {
            trainer.push_back(pass.at("ppo_trainer_collection").at("p50_us"));
        }
        if (pass.contains("klpo_wave")) {
            waves.push_back(pass.at("klpo_wave").at("collection_wave_seconds"));
        }
    }
    return Json{{"count", passes.size()},
                {"start_unix_seconds", unix_seconds(first)},
                {"end_unix_seconds", unix_seconds(last)},
                {"trainer_p50_us", trainer},
                {"klpo_wave_seconds", waves}};
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
    for (const auto& head : heads) {
        if (!wanted("inference")) {
            break;
        }
        for (const bool pinned : {false, true}) {
            if (pinned && options.device == "cpu") {
                continue;
            }
            RECORD_USER_SCOPE(head.double_dqn ? "phase_inference_value" : "phase_inference_ppo");
            inference.push_back(measure_inference(environment, options, head, pinned));
        }
    }
    report["inference"] = inference;
    if (wanted("value_minibatch")) {
        RECORD_USER_SCOPE("phase_value_minibatch");
        Json values = Json::array();
        for (const auto& head : heads) {
            if (head.double_dqn) {
                values.push_back(measure_value_minibatch(environment, options, head));
            }
        }
        report["value_minibatch"] = values;
    }
    if (wanted("gae")) {
        RECORD_USER_SCOPE("phase_gae");
        report["gae_cpu_fp64"] = measure_gae(options);
    }
    if (wanted("gradient")) {
        RECORD_USER_SCOPE("phase_minibatch_gradient");
        report["ppo_minibatch"] = measure_gradient(environment, options);
    }
    if (wanted("first_tick")) {
        RECORD_USER_SCOPE("phase_ppo_first_tick");
        report["ppo_trainer_first_tick"] = measure_first_tick(inputs, options);
    }
    const auto first = std::chrono::system_clock::now();
    std::vector<Json> passes;
    for (std::size_t repeat = 0; repeat < options.repeat; ++repeat) {
        Json pass = Json::object();
        if (wanted("trainer")) {
            RECORD_USER_SCOPE("phase_ppo_trainer");
            pass["ppo_trainer_collection"] = measure_trainer(inputs, options);
        }
        if (wanted("klpo")) {
            RECORD_USER_SCOPE("phase_klpo_wave");
            pass["klpo_wave"] = measure_klpo(inputs, options);
        }
        require(passes.empty() || same_collection(passes.front(), pass),
                "Una repetición cambió las huellas de la recogida");
        passes.push_back(std::move(pass));
    }
    report.update(passes.front());
    if (options.repeat > 1) {
        report["repeats"] = repeated(passes, first, std::chrono::system_clock::now());
    }
    if (wanted("group")) {
        RECORD_USER_SCOPE("phase_group_objectives");
        report["group_objectives"] = measure_group(inputs, options);
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
