#include "mars_titan/ppo_experiment.hpp"
#include "mars_titan/ppo_checkpoints.hpp"
#include "mars_titan/ppo_inputs.hpp"
#include "mars_titan/ppo_training.hpp"
#include "mars_titan/simulation_files.hpp"

#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/AllocatorConfig.h>
#include <torch/cuda.h>
#include <torch/version.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fcntl.h>
#include <initializer_list>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace mars_titan::learning {
namespace {
using Json = nlohmann::json;
using simulation::atomic_json_file;
using simulation::content_sha256;
using simulation::parse_bounded_json;
using simulation::read_bounded_file;
using simulation::read_json_int64;
constexpr std::size_t maximum_sources = 12;
constexpr std::size_t maximum_transitions = std::size_t{1} << 20;
constexpr std::size_t maximum_rollout = 16384;
constexpr std::size_t diagnostic_transitions = 32;
constexpr std::size_t minimum_vram_bytes = std::size_t{256} * 1024 * 1024;
constexpr std::size_t maximum_vram_bytes = std::size_t{6} * 1024 * 1024 * 1024;
constexpr std::size_t maximum_process_memory_bytes = std::size_t{12} * 1024 * 1024 * 1024;
constexpr double maximum_vram_fraction = 0.75;
constexpr std::array allowed_seeds{uint64_t{42}, uint64_t{43}, uint64_t{44}};
constexpr std::string_view selection_metric = "ruin_count_then_mean_log_growth";
constexpr std::string_view lock_name = "mars-titan-scientific-gpu.lock";

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::invalid_argument(std::string(message));
    }
}

void require_fields(const Json& object, std::initializer_list<std::string_view> names) {
    require(object.is_object() && object.size() == names.size() &&
                std::ranges::all_of(names, [&](auto name) { return object.contains(std::string(name)); }),
            "El documento PPO no conserva los campos de su esquema");
}

std::size_t count(const Json& value) {
    const auto parsed = read_json_int64(value);
    require(parsed >= 0, "Un contador PPO no puede ser negativo");
    return static_cast<std::size_t>(parsed);
}

double finite_number(const Json& value) {
    require(value.is_number(), "El parámetro PPO necesita un número");
    const auto result = value.get<double>();
    require(std::isfinite(result), "El parámetro PPO necesita un número finito");
    return result;
}

bool boolean(const Json& value) {
    require(value.is_boolean(), "El parámetro PPO necesita un booleano");
    return value.get<bool>();
}

struct ExperimentConfig {
    Json document;
    PpoTrainingConfig training;
    PpoHyperparameters hyperparameters;
    simulation::Parameters environment;
    std::size_t environments = 0;
    std::size_t checkpoint_transitions = 0;
    double min_delta = 0;
    std::size_t patience = 0;
    bool early_stopping = false;
};

ExperimentConfig configuration(const std::filesystem::path& path) {
    ExperimentConfig result;
    result.document = parse_bounded_json(read_bounded_file(path, simulation::maximum_manifest_bytes));
    const auto& document = result.document;
    require_fields(document, {"schema_version", "training", "environments", "hyperparameters",
                              "environment", "checkpoint_transitions", "selection", "final_test_opened"});
    require(read_json_int64(document.at("schema_version")) == 1 &&
                !boolean(document.at("final_test_opened")), "La versión PPO o el cierre del test no es válido");
    const auto& training = document.at("training");
    require_fields(training, {"total_transitions", "rollout_transitions", "workers", "seed", "rollout_bytes"});
    result.training = {count(training.at("total_transitions")), count(training.at("rollout_transitions")),
                       count(training.at("workers")), static_cast<uint64_t>(count(training.at("seed"))),
                       count(training.at("rollout_bytes"))};
    const auto& hyper = document.at("hyperparameters");
    require_fields(hyper, {"learning_rate", "gamma", "gae_lambda", "clip", "entropy", "value_weight",
                          "gradient_norm", "epochs", "minibatch_size"});
    result.hyperparameters = {finite_number(hyper.at("learning_rate")), finite_number(hyper.at("gamma")),
        finite_number(hyper.at("gae_lambda")), finite_number(hyper.at("clip")), finite_number(hyper.at("entropy")),
        finite_number(hyper.at("value_weight")), finite_number(hyper.at("gradient_norm")),
        read_json_int64(hyper.at("epochs")), read_json_int64(hyper.at("minibatch_size"))};
    result.hyperparameters.validate();
    const auto& environment = document.at("environment");
    require_fields(environment, {"capital", "cost_bps", "participation", "score_scale", "ruin_penalty"});
    result.environment = {finite_number(environment.at("capital")), finite_number(environment.at("cost_bps")),
        finite_number(environment.at("participation")), finite_number(environment.at("score_scale")),
        finite_number(environment.at("ruin_penalty"))};
    const auto& selection = document.at("selection");
    require_fields(selection, {"min_delta", "patience", "early_stopping", "metric"});
    require(selection.at("metric") == selection_metric, "La métrica de selección PPO no está admitida");
    result.min_delta = finite_number(selection.at("min_delta"));
    result.patience = count(selection.at("patience"));
    result.early_stopping = boolean(selection.at("early_stopping"));
    result.environments = count(document.at("environments"));
    result.checkpoint_transitions = count(document.at("checkpoint_transitions"));
    require(result.environments > 0 && result.environments <= simulation::maximum_environments &&
                result.training.total_transitions >= result.environments &&
                result.training.total_transitions <= maximum_transitions &&
                result.training.total_transitions % result.environments == 0 &&
                result.training.rollout_transitions >= result.environments &&
                result.training.rollout_transitions <= maximum_rollout &&
                result.training.rollout_transitions <= result.training.total_transitions &&
                result.training.rollout_transitions % result.environments == 0 &&
                result.checkpoint_transitions > 0 && result.checkpoint_transitions <= maximum_transitions &&
                result.min_delta >= 0 && result.patience > 0 && result.patience <= maximum_transitions &&
                std::ranges::find(allowed_seeds, result.training.seed) != allowed_seeds.end(),
            "El presupuesto, las réplicas o la semilla PPO no están admitidos");
    return result;
}

bool contains(const std::filesystem::path& parent, const std::filesystem::path& child) {
    const auto first = std::filesystem::weakly_canonical(parent);
    const auto second = std::filesystem::weakly_canonical(child);
    auto a = first.begin();
    auto b = second.begin();
    while (a != first.end() && b != second.end() && *a == *b) {
        ++a;
        ++b;
    }
    return a == first.end();
}

void validate_options(const PpoExperimentOptions& options, const ExperimentConfig& config) {
    require(!options.output.empty() && !options.train_tapes.empty() && !options.validation_tapes.empty() &&
                options.train_tapes.size() <= maximum_sources && options.validation_tapes.size() <= maximum_sources &&
                config.environments % options.train_tapes.size() == 0,
            "Se necesitan fuentes acotadas y réplicas equilibradas entre todos los escenarios");
    require(options.device == "cpu" || options.device == "cuda:0", "El dispositivo PPO debe ser cpu o cuda:0");
    if (options.device == "cpu") {
        require(options.diagnostic && config.training.total_transitions <= diagnostic_transitions &&
                    !options.gpu_lease_fd && !options.vram_budget_bytes && !options.vram_total_bytes,
                "La CPU solo admite el diagnóstico explícito de hasta 32 transiciones");
    } else {
        require(!options.diagnostic, "El diagnóstico PPO se ejecuta únicamente en CPU");
    }
    require(!contains(options.output, options.config), "La configuración debe quedar fuera de la salida PPO");
    for (const auto* sources : {&options.train_tapes, &options.validation_tapes}) {
        for (const auto& source : *sources) {
            require(!contains(source, options.output) && !contains(options.output, source),
                    "La salida PPO debe permanecer separada de sus fuentes");
        }
    }
}

double validate_gpu_lease(const PpoExperimentOptions& options) {
    if (!options.gpu_lease_fd || !options.vram_budget_bytes || !options.vram_total_bytes) {
        throw std::invalid_argument("CUDA necesita el bloqueo heredado y su presupuesto de VRAM");
    }
    require(*options.gpu_lease_fd >= 0 && *options.vram_budget_bytes >= minimum_vram_bytes &&
                *options.vram_budget_bytes <= maximum_vram_bytes && *options.vram_total_bytes > 0,
            "CUDA necesita el bloqueo heredado y un presupuesto entre 256 MiB y 6 GiB");
    const auto descriptor = options.gpu_lease_fd.value();
    const auto fraction = static_cast<double>(options.vram_budget_bytes.value()) /
                          static_cast<double>(options.vram_total_bytes.value());
    require(fraction <= maximum_vram_fraction, "El presupuesto PPO no puede superar el 75 % de la VRAM");
    const auto* configured = std::getenv("XDG_RUNTIME_DIR");
    const std::filesystem::path runtime = configured != nullptr ? configured :
        "/tmp/mars-titan-" + std::to_string(getuid());
    require(runtime.is_absolute(), "El directorio del bloqueo GPU debe ser absoluto");
    struct stat directory {};
    struct stat inherited {};
    struct stat current {};
    const auto lock = runtime / lock_name;
    require(lstat(runtime.c_str(), &directory) == 0 && S_ISDIR(directory.st_mode) &&
                directory.st_uid == getuid() && (directory.st_mode & (S_IWGRP | S_IWOTH)) == 0 &&
                fstat(descriptor, &inherited) == 0 && S_ISREG(inherited.st_mode) &&
                inherited.st_uid == getuid() && inherited.st_nlink == 1 &&
                (inherited.st_mode & (S_IWGRP | S_IWOTH)) == 0 &&
                lstat(lock.c_str(), &current) == 0 && S_ISREG(current.st_mode) &&
                current.st_dev == inherited.st_dev && current.st_ino == inherited.st_ino &&
                current.st_uid == inherited.st_uid,
            "El descriptor heredado no corresponde al bloqueo GPU privado de esta ejecución");
    require(flock(descriptor, LOCK_EX | LOCK_NB) == 0,
            "No se puede confirmar el bloqueo exclusivo de la GPU");
    require(lstat(lock.c_str(), &current) == 0 && current.st_dev == inherited.st_dev &&
                current.st_ino == inherited.st_ino && S_ISREG(current.st_mode),
            "El archivo del bloqueo GPU cambió durante la admisión");
    return fraction;
}

void configure_runtime(const PpoExperimentOptions& options) {
    if (options.device == "cuda:0") {
#if defined(MARS_TITAN_LIBTORCH_CUDA)
        const auto fraction = validate_gpu_lease(options);
        const std::string cublas_workspace = ":4096:8";
        const auto* workspace = std::getenv("CUBLAS_WORKSPACE_CONFIG");
        require(workspace == nullptr || std::string_view(workspace) == cublas_workspace,
                "CUBLAS_WORKSPACE_CONFIG debe ser :4096:8 para esta configuración PPO");
        require(setenv("CUBLAS_WORKSPACE_CONFIG", cublas_workspace.c_str(), 0) == 0,
                "No se pudo configurar el espacio de trabajo determinista de cuBLAS");
        constexpr std::size_t fraction_buffer_size = 64;
        std::array<char, fraction_buffer_size> fraction_text{};
        const auto formatted = std::to_chars(fraction_text.begin(), fraction_text.end(),
            std::nextafter(fraction, 0.), std::chars_format::general, std::numeric_limits<double>::max_digits10);
        require(formatted.ec == std::errc{}, "No se pudo representar el presupuesto de VRAM");
        c10::CachingAllocator::setAllocatorSettings("per_process_memory_fraction:" +
            std::string(fraction_text.data(), formatted.ptr));
        require(torch::cuda::is_available(), "CUDA no está disponible en el ejecutable PPO");
#else
        static_cast<void>(validate_gpu_lease(options));
        throw std::invalid_argument("El ejecutable PPO no enlaza el backend CUDA de LibTorch");
#endif
    }
    at::set_num_threads(1);
    if (at::get_num_interop_threads() != 1) {
        at::set_num_interop_threads(1);
    }
    at::globalContext().setDeterministicAlgorithms(true, false);
    at::globalContext().setBenchmarkCuDNN(false);
    at::globalContext().setDeterministicCuDNN(true);
    at::globalContext().setFloat32Precision(at::Float32Backend::GENERIC, at::Float32Op::ALL,
                                           at::Float32Precision::IEEE);
}

struct InputFootprint {
    std::size_t assets;
    std::size_t sessions;
    std::size_t actions;
    std::size_t context_fields;
};

void account_memory(std::size_t items, std::size_t width, std::size_t& bytes) {
    if (width == 0) {
        return;
    }
    require(bytes <= simulation::default_batch_bytes &&
                items <= (simulation::default_batch_bytes - bytes) / width,
            "Las réplicas PPO superan el presupuesto de memoria de contextos, estados y buffers");
    bytes += items * width;
}

void account_input(const InputFootprint& input, std::size_t replicas, std::size_t& bytes) {
    constexpr std::size_t financial_fields = 6;
    constexpr std::size_t maximum_field_string = 256;
    constexpr std::size_t live_context_copies = 4;
    constexpr std::size_t live_state_copies = 8;
    require(input.assets > 0 && input.assets <= simulation::maximum_assets && input.sessions >= 2 &&
                input.sessions <= simulation::maximum_sessions && input.actions <= simulation::maximum_actions &&
                input.context_fields <= simulation::maximum_context_fields,
            "Las dimensiones declaradas de las fuentes PPO exceden sus límites");
    const auto context_values = input.sessions * input.context_fields;
    account_memory(replicas * live_context_copies, context_values * sizeof(simulation::ContextValue), bytes);
    account_memory(replicas * live_context_copies,
                    input.context_fields * (sizeof(simulation::ContextField) + 2 * maximum_field_string), bytes);
    const auto width = input.assets * financial_fields + 2 + 3 * input.context_fields;
    account_memory(replicas, 4 * width * sizeof(float) + 4 * (sizeof(double) + 3 * sizeof(uint8_t)), bytes);
    account_memory(replicas, input.assets * live_state_copies *
        (sizeof(mt_position_v1) + sizeof(mt_trade_v1) + sizeof(std::size_t) + sizeof(double) + sizeof(uint32_t) + 1), bytes);
    account_memory(replicas, input.sessions * 4 * sizeof(std::vector<std::size_t>), bytes);
    account_memory(replicas, input.actions * live_state_copies *
        (sizeof(simulation::Receivable) + sizeof(std::size_t) + 1), bytes);
}

void preflight_memory(const PpoExperimentOptions& options, const ExperimentConfig& config) {
    std::size_t total = 0;
    std::size_t observation_width = 0;
    for (const auto* paths : {&options.train_tapes, &options.validation_tapes}) {
        const auto replicas = paths == &options.train_tapes ? config.environments / paths->size() : 1;
        for (const auto& path : *paths) {
            const auto manifest = parse_bounded_json(read_bounded_file(path / "manifest.json", simulation::maximum_manifest_bytes));
            const auto partition = paths == &options.train_tapes ? "train" : "validation";
            require(read_json_int64(manifest.at("schema_version")) == 1 &&
                        !boolean(manifest.at("final_test_opened")) &&
                        manifest.at("identity").at("domain") == "synthetic" &&
                        manifest.at("identity").at("partition") == partition,
                    "La fuente PPO no corresponde a su partición o pretende abrir el test");
            std::size_t fields = 0;
            const auto context_path = path / "context.json";
            simulation::require_safe_path(context_path);
            if (std::filesystem::exists(context_path)) {
                const auto context = parse_bounded_json(read_bounded_file(context_path, simulation::maximum_manifest_bytes));
                require(context.at("fields").is_array(), "Falta el esquema del contexto PPO");
                fields = context.at("fields").size();
            }
            const InputFootprint footprint{count(manifest.at("assets")), count(manifest.at("sessions")),
                manifest.at("actions").size(), fields};
            account_input(footprint, replicas, total);
            constexpr std::size_t financial_fields = 6;
            observation_width = std::max(observation_width, footprint.assets * financial_fields + 2 + 3 * fields);
        }
    }
    constexpr std::size_t scalar_bytes = sizeof(int64_t) + 4 * sizeof(double) + 3 * sizeof(bool);
    account_memory(config.training.rollout_transitions, 3 * (observation_width * sizeof(float) + scalar_bytes), total);
}

struct LoadedSource {
    simulation::BatchInput input;
    Json identity;
    std::optional<int64_t> generator_seed;
};

LoadedSource load_source(const std::filesystem::path& path, std::string_view partition,
                         const simulation::Parameters& parameters) {
    const auto manifest_bytes = read_bounded_file(path / "manifest.json", simulation::maximum_manifest_bytes);
    const auto manifest = parse_bounded_json(manifest_bytes);
    LoadedSource result{load_ppo_input(path), Json::object(), std::nullopt};
    result.input.parameters = parameters;
    require(result.input.tape->partition == partition && result.input.tape->domain == "synthetic" &&
                result.input.tape->source_sha256 == content_sha256(manifest_bytes),
            "La fuente PPO no corresponde a su partición o cambió durante la lectura");
    const auto& source = manifest.at("identity").at("source");
    if (source.is_object() && source.contains("generator")) {
        require(source.at("generator").is_object() && source.at("generator").contains("seed"),
                "La procedencia del generador necesita una semilla explícita");
        const auto seed = read_json_int64(source.at("generator").at("seed"));
        require(seed >= 0, "La semilla del escenario no puede ser negativa");
        result.generator_seed = seed;
    }
    result.identity = Json{{"manifest_sha256", result.input.tape->source_sha256},
        {"market_sha256", manifest.at("file_sha256")},
        {"context_sha256", result.input.context ? Json(result.input.context->source_sha256) : Json(nullptr)},
        {"generator_seed", result.generator_seed ? Json(*result.generator_seed) : Json(nullptr)}};
    return result;
}

void require_compatible(const simulation::BatchInput& reference, const simulation::BatchInput& input) {
    require(reference.tape->assets == input.tape->assets && reference.tape->currency == input.tape->currency &&
                reference.tape->domain == input.tape->domain && reference.tape->parent_id == input.tape->parent_id &&
                reference.context.has_value() == input.context.has_value() &&
                (!reference.context || reference.context->fields == input.context->fields),
            "Entrenamiento y validación no conservan activos, moneda, padre o campos de contexto");
}

struct ExperimentInputs {
    std::vector<simulation::BatchInput> training;
    std::vector<simulation::BatchInput> validation;
    Json identity;
};

ExperimentInputs load_inputs(const PpoExperimentOptions& options, const ExperimentConfig& config) {
    ExperimentInputs result;
    std::vector<simulation::BatchInput> sources;
    std::set<std::string> hashes;
    std::set<std::string> training_market_hashes;
    std::set<int64_t> training_seeds;
    Json training = Json::array();
    Json validation = Json::array();
    for (const auto& path : options.train_tapes) {
        auto loaded = load_source(path, "train", config.environment);
        require(hashes.insert(loaded.input.tape->source_sha256).second,
                "El entrenamiento PPO duplica una fuente");
        if (!sources.empty()) {
            require_compatible(sources.front(), loaded.input);
        }
        if (loaded.generator_seed) {
            training_seeds.insert(*loaded.generator_seed);
        }
        training_market_hashes.insert(loaded.identity.at("market_sha256").get<std::string>());
        training.push_back(std::move(loaded.identity));
        sources.push_back(std::move(loaded.input));
    }
    for (const auto& path : options.validation_tapes) {
        auto loaded = load_source(path, "validation", config.environment);
        require_compatible(sources.front(), loaded.input);
        require(hashes.insert(loaded.input.tape->source_sha256).second &&
                    !training_market_hashes.contains(loaded.identity.at("market_sha256").get<std::string>()) &&
                    (!loaded.generator_seed || !training_seeds.contains(*loaded.generator_seed)),
                "La validación PPO reutiliza datos o semillas de entrenamiento");
        validation.push_back(std::move(loaded.identity));
        result.validation.push_back(std::move(loaded.input));
    }
    std::size_t replicated_bytes = 0;
    const auto replicas = config.environments / sources.size();
    for (const auto& input : sources) {
        account_input({input.tape->assets.size(), input.tape->close_times.size(), input.tape->actions.size(),
                       input.context ? input.context->fields.size() : 0}, replicas, replicated_bytes);
    }
    for (const auto& input : result.validation) {
        account_input({input.tape->assets.size(), input.tape->close_times.size(), input.tape->actions.size(),
                       input.context ? input.context->fields.size() : 0}, 1, replicated_bytes);
    }
    result.training.reserve(config.environments);
    for (std::size_t index = 0; index < config.environments; ++index) {
        result.training.push_back(sources[index % sources.size()]);
    }
    result.identity = Json{{"train", training}, {"validation", validation}};
    return result;
}

Json experiment_identity(const PpoExperimentOptions& options, const ExperimentConfig& config,
                         const ExperimentInputs& inputs) {
    return Json{{"schema_version", 1}, {"kind", "native_ppo"}, {"configuration", config.document},
        {"sources", inputs.identity}, {"device", options.device}, {"diagnostic", options.diagnostic},
        {"architecture", "mlp_64_64_tanh_6_1_v1"}, {"torch_version", TORCH_VERSION},
        {"native_source_sha256", MARS_TITAN_NATIVE_SOURCE_SHA256},
        {"native_build_sha256", MARS_TITAN_NATIVE_BUILD_SHA256},
        {"precision", "weights_fp32_gae_ratio_fp64"}, {"deterministic", true},
        {"evaluation_policy", "greedy_argmax"}, {"rollout_policy", "categorical_sampling"},
        {"parent_frozen", true}, {"final_test_opened", false}};
}

struct Progress {
    std::string status = "running";
    std::size_t evaluations = 0;
    std::size_t stale_evaluations = 0;
    std::optional<std::size_t> evaluated_optimizer_steps;
    Json best = nullptr;
};

Json progress_json(const Progress& progress) {
    return Json{{"status", progress.status}, {"evaluations", progress.evaluations},
        {"stale_evaluations", progress.stale_evaluations}, {"best", progress.best},
        {"evaluated_optimizer_steps", progress.evaluated_optimizer_steps ?
            Json(*progress.evaluated_optimizer_steps) : Json(nullptr)}};
}

Json state_json(const PpoTrainingState& state, const ExperimentConfig& config, const Progress& progress) {
    Json sessions = Json::array();
    for (const auto& session : state.environment.sessions) {
        sessions.push_back(simulation::snapshot_json(session));
    }
    return Json{{"schema_version", 1}, {"configuration", config.document},
        {"device", state.device}, {"diagnostic", state.diagnostic},
        {"transitions", state.transitions}, {"optimizer_steps", state.optimizer_steps},
        {"invalid_transitions", state.invalid_transitions}, {"episodes", state.episodes},
        {"reset_lanes", state.reset_lanes}, {"sessions", sessions},
        {"context_sources", state.environment.context_sources}, {"progress", progress_json(progress)}};
}

PpoTrainingState restore_state(const PpoCheckpointBundle& bundle, const ExperimentConfig& config,
                                const PpoExperimentOptions& options) {
    const auto& metadata = bundle.metadata;
    require_fields(metadata, {"schema_version", "configuration", "device", "diagnostic", "transitions",
        "optimizer_steps", "invalid_transitions", "episodes", "reset_lanes", "sessions", "context_sources", "progress"});
    require(read_json_int64(metadata.at("schema_version")) == 1 &&
                metadata.at("configuration") == config.document && metadata.at("device") == options.device &&
                boolean(metadata.at("diagnostic")) == options.diagnostic,
            "El checkpoint PPO no conserva la configuración del experimento");
    require(metadata.at("sessions").is_array() && metadata.at("sessions").size() == config.environments &&
                metadata.at("context_sources").is_array() &&
                metadata.at("context_sources").size() == config.environments &&
                metadata.at("reset_lanes").is_array() && metadata.at("reset_lanes").size() <= config.environments,
            "El checkpoint PPO no conserva los carriles y sus fuentes");
    PpoTrainingState state;
    state.config = config.training;
    state.hyperparameters = config.hyperparameters;
    state.device = options.device;
    state.diagnostic = options.diagnostic;
    state.transitions = count(metadata.at("transitions"));
    state.optimizer_steps = count(metadata.at("optimizer_steps"));
    state.invalid_transitions = count(metadata.at("invalid_transitions"));
    state.episodes = count(metadata.at("episodes"));
    for (const auto& lane : metadata.at("reset_lanes")) {
        state.reset_lanes.push_back(count(lane));
    }
    for (const auto& session : metadata.at("sessions")) {
        state.environment.sessions.push_back(simulation::read_snapshot(session));
    }
    state.environment.context_sources = metadata.at("context_sources").get<std::vector<std::string>>();
    state.policy_archive = bundle.policy_archive;
    state.rollout = deserialize_rollout(bundle.rollout_archive);
    return state;
}

Progress restore_progress(const Json& value, const PpoTrainingState& state,
                           const ExperimentConfig& config, std::size_t validation_episodes) {
    require_fields(value, {"status", "evaluations", "stale_evaluations", "best", "evaluated_optimizer_steps"});
    Progress progress;
    progress.status = value.at("status").get<std::string>();
    require(progress.status == "running" || progress.status == "paused" || progress.status == "completed" ||
                progress.status == "early_stopped", "El estado del experimento PPO no está admitido");
    progress.evaluations = count(value.at("evaluations"));
    progress.stale_evaluations = count(value.at("stale_evaluations"));
    require(progress.evaluations <= state.optimizer_steps + 1 &&
                progress.stale_evaluations <= progress.evaluations,
            "La selección PPO contiene contadores de evaluación incoherentes");
    if (!value.at("evaluated_optimizer_steps").is_null()) {
        progress.evaluated_optimizer_steps = count(value.at("evaluated_optimizer_steps"));
        require(*progress.evaluated_optimizer_steps <= state.optimizer_steps,
                "La evaluación PPO contiene actualizaciones futuras");
    }
    progress.best = value.at("best");
    require(progress.evaluated_optimizer_steps.has_value() == (progress.evaluations != 0) &&
                progress.best.is_null() == (progress.evaluations == 0),
            "La selección PPO no conserva una evaluación inicial completa");
    if (!progress.best.is_null()) {
        require_fields(progress.best, {"ruin_count", "mean_log_growth", "transitions", "optimizer_steps", "episodes"});
        require(count(progress.best.at("transitions")) <= state.transitions &&
                    count(progress.best.at("optimizer_steps")) <= state.optimizer_steps &&
                    count(progress.best.at("ruin_count")) <= count(progress.best.at("episodes")) &&
                    count(progress.best.at("episodes")) == validation_episodes &&
                    progress.stale_evaluations < progress.evaluations &&
                    progress.evaluations > 0 && progress.evaluated_optimizer_steps.has_value(),
                "La selección PPO no corresponde al cursor confirmado");
        static_cast<void>(finite_number(progress.best.at("mean_log_growth")));
    } else {
        require(progress.evaluations == 0, "Falta el candidato de una evaluación PPO completa");
    }
    require(progress.status != "completed" || state.transitions == state.config.total_transitions,
            "El checkpoint PPO declara completado un presupuesto pendiente");
    if (progress.status == "completed" || progress.status == "early_stopped") {
        require(progress.evaluated_optimizer_steps && *progress.evaluated_optimizer_steps == state.optimizer_steps,
                "El checkpoint PPO declara finalizada una validación pendiente");
    }
    require(progress.status != "early_stopped" ||
                (config.early_stopping && progress.stale_evaluations >= config.patience),
            "La parada temprana del checkpoint PPO no corresponde al criterio declarado");
    return progress;
}

void require_ram_budget() {
    require(simulation::process_memory_high_water() <= maximum_process_memory_bytes,
            "El proceso PPO supera el límite de 12 GiB de RAM observado al confirmar su estado");
}

std::string current_utc() {
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    require(gmtime_r(&now, &utc) != nullptr, "No se pudo obtener la fecha UTC del proceso");
    constexpr std::size_t timestamp_bytes = 32;
    std::array<char, timestamp_bytes> text{};
    const auto length = std::strftime(text.data(), text.size(), "%Y-%m-%dT%H:%M:%SZ", &utc);
    require(length != 0, "No se pudo representar la fecha UTC del proceso");
    return {text.data(), length};
}

class ExperimentRun {
public:
    ExperimentRun(const PpoExperimentOptions& options, ExperimentConfig config, ExperimentInputs inputs,
                   const std::function<bool()>& stop, std::chrono::steady_clock::time_point started)
        : options_(options), config_(std::move(config)), inputs_(std::move(inputs)),
          identity_(experiment_identity(options_, config_, inputs_)),
          trainer_(inputs_.training, config_.training, config_.hyperparameters, options_.device, options_.diagnostic),
          store_(options_.output, identity_, options_.resume), stop_(stop), started_(started) {}

    Json run() {
        if (options_.resume && has_recent_checkpoint()) {
            const auto bundle = store_.load_latest();
            auto state = restore_state(bundle, config_, options_);
            progress_ = restore_progress(bundle.metadata.at("progress"), state, config_, inputs_.validation.size());
            if (!progress_.best.is_null()) {
                const auto selected = store_.load_best();
                require(selected.metadata.at("progress").at("best") == progress_.best &&
                            count(selected.metadata.at("transitions")) == count(progress_.best.at("transitions")) &&
                            count(selected.metadata.at("optimizer_steps")) == count(progress_.best.at("optimizer_steps")),
                        "El mejor checkpoint PPO no corresponde a la selección confirmada");
            }
            trainer_.restore(state);
            last_checkpoint_ = trainer_.transitions();
            receipt_ = bundle.receipt;
            receipt_.erase("discarded");
            if (progress_.status == "completed" || progress_.status == "early_stopped") {
                return publish_report();
            }
            progress_.status = "running";
        } else {
            save(false);
        }
        try {
            if (!evaluate_pending()) {
                return pause();
            }
            if (config_.early_stopping && progress_.stale_evaluations >= config_.patience) {
                progress_.status = "early_stopped";
                return save(false);
            }
            while (trainer_.transitions() < config_.training.total_transitions) {
                if (stop_requested()) {
                    return pause();
                }
                static_cast<void>(trainer_.advance());
                if (!evaluate_pending()) {
                    return pause();
                }
                if (config_.early_stopping && progress_.stale_evaluations >= config_.patience) {
                    progress_.status = "early_stopped";
                    return save(false);
                }
                if (trainer_.transitions() - last_checkpoint_ >= config_.checkpoint_transitions) {
                    save(false);
                }
            }
            progress_.status = "completed";
            return save(false);
        } catch (const std::exception& error) {
            progress_.status = "failed";
            try {
                static_cast<void>(publish_report());
            } catch (const std::exception& report_error) {
                throw std::runtime_error(std::string(error.what()) +
                    ". Tampoco se pudo registrar el fallo: " + report_error.what());
            }
            throw;
        }
    }

private:
    bool has_recent_checkpoint() const {
        const auto envelope = parse_bounded_json(read_bounded_file(options_.output / "ppo-index.json",
                                                                   simulation::maximum_manifest_bytes));
        const auto& index = envelope.at("payload");
        if (!index.at("recent").empty()) {
            return true;
        }
        require(index.at("best").is_null() && index.at("retired").empty(),
                "Un índice PPO sin estados recientes no puede conservar una selección anterior");
        return false;
    }

    bool stop_requested() const {
        return (stop_ && stop_()) || (options_.stop_after && trainer_.transitions() >= *options_.stop_after);
    }

    Json pause() {
        progress_.status = "paused";
        return save(false);
    }

    bool evaluate_pending() {
        if (progress_.evaluated_optimizer_steps && *progress_.evaluated_optimizer_steps == trainer_.optimizer_steps()) {
            return true;
        }
        const auto evaluation = evaluate_policy(trainer_.policy(), inputs_.validation, config_.training.workers, stop_);
        if (evaluation.paused) {
            return false;
        }
        require(evaluation.incomplete == 0 && evaluation.episodes == inputs_.validation.size() &&
                    std::isfinite(evaluation.mean_log_growth),
                "La validación PPO está incompleta y no permite seleccionar un checkpoint");
        const bool improved = progress_.best.is_null() ||
            evaluation.ruined < count(progress_.best.at("ruin_count")) ||
            (evaluation.ruined == count(progress_.best.at("ruin_count")) &&
             evaluation.mean_log_growth > finite_number(progress_.best.at("mean_log_growth")) + config_.min_delta);
        ++progress_.evaluations;
        progress_.evaluated_optimizer_steps = trainer_.optimizer_steps();
        if (improved) {
            progress_.best = Json{{"ruin_count", evaluation.ruined}, {"mean_log_growth", evaluation.mean_log_growth},
                {"transitions", trainer_.transitions()}, {"optimizer_steps", trainer_.optimizer_steps()},
                {"episodes", evaluation.episodes}};
            progress_.stale_evaluations = 0;
        } else {
            ++progress_.stale_evaluations;
        }
        save(improved);
        return true;
    }

    Json save(bool select_best) {
        require_ram_budget();
        const auto state = trainer_.snapshot();
        require_ram_budget();
        receipt_ = store_.save(state_json(state, config_, progress_), state.policy_archive,
                               serialize_rollout(state.rollout), select_best);
        last_checkpoint_ = trainer_.transitions();
        return publish_report();
    }

    Json publish_report() const {
        const bool stopped_early = progress_.status == "early_stopped";
        auto selection = config_.document.at("selection");
        selection["policy"] = "greedy_argmax";
        Json report{{"schema_version", 1}, {"kind", "native_ppo"},
            {"activity", "rl"}, {"model", "ppo"}, {"backend", "native_libtorch"},
            {"seed", config_.training.seed}, {"partition", "train"}, {"updated_at", current_utc()},
            {"global_step", trainer_.transitions()}, {"total_steps", config_.training.total_transitions},
            {"status", stopped_early ? "completed" : progress_.status},
            {"stopping_reason", stopped_early ? Json("early_stop") :
                progress_.status == "completed" ? Json("budget_exhausted") :
                progress_.status == "paused" ? Json("requested_pause") : Json(nullptr)},
            {"identity_sha256", content_sha256(identity_.dump())}, {"domain", options_.diagnostic ? "technical" : "synthetic"},
            {"device", options_.device}, {"diagnostic", options_.diagnostic},
            {"transitions", trainer_.transitions()}, {"optimizer_steps", trainer_.optimizer_steps()},
            {"partial_ticks", trainer_.partial_ticks()}, {"invalid_transitions", trainer_.invalid_transitions()},
            {"evaluations", progress_.evaluations}, {"stale_evaluations", progress_.stale_evaluations},
            {"best", progress_.best}, {"selection", selection},
            {"checkpoint", receipt_}, {"parent_frozen", true}, {"final_test_opened", false},
            {"resources", Json{{"ram_peak_bytes", simulation::process_memory_high_water()},
#if defined(__linux__)
                {"ram_peak_method", "procfs_VmHWM"},
#else
                {"ram_peak_method", "getrusage_RUSAGE_SELF"},
#endif
                {"ram_guard_bytes", maximum_process_memory_bytes},
                {"ram_guard_scope", "before_runtime_and_checkpoints"},
                {"rollout_budget_bytes", config_.training.rollout_bytes},
                {"vram_budget_bytes", options_.vram_budget_bytes ? Json(*options_.vram_budget_bytes) : Json(nullptr)},
                {"vram_total_bytes", options_.vram_total_bytes ? Json(*options_.vram_total_bytes) : Json(nullptr)},
                {"vram_peak_bytes", nullptr}}},
            {"invocation_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count()}};
        atomic_json_file(options_.output / "run.json", report);
        return report;
    }

    const PpoExperimentOptions& options_;
    ExperimentConfig config_;
    ExperimentInputs inputs_;
    Json identity_;
    PpoTrainer trainer_;
    PpoCheckpointStore store_;
    const std::function<bool()>& stop_;
    std::chrono::steady_clock::time_point started_;
    Progress progress_;
    std::size_t last_checkpoint_ = 0;
    Json receipt_ = nullptr;
};
}

Json run_ppo_experiment(const PpoExperimentOptions& options, const std::function<bool()>& stop_requested) {
    const auto started = std::chrono::steady_clock::now();
    const auto config = configuration(options.config);
    validate_options(options, config);
    // La admisión se comprueba antes de cualquier tensor CUDA o creación de la salida.
    if (options.device == "cuda:0") {
        static_cast<void>(validate_gpu_lease(options));
    }
    preflight_memory(options, config);
    auto inputs = load_inputs(options, config);
    require_ram_budget();
    configure_runtime(options);
    ExperimentRun run(options, config, std::move(inputs), stop_requested, started);
    return run.run();
}
}
