#include "mars_titan/ppo_experiment.hpp"
#include "mars_titan/decision_trace.hpp"
#include "mars_titan/learning_hold.hpp"
#include "mars_titan/policy_evaluation.hpp"
#include "mars_titan/policy_tapes.hpp"
#include "mars_titan/ppo_checkpoints.hpp"
#include "mars_titan/ppo_inputs.hpp"
#include "mars_titan/ppo_training.hpp"
#include "mars_titan/simulation_files.hpp"
#include "accurate_sum.hpp"

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
#include <sstream>
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
constexpr std::size_t maximum_adaptive_sources = 512;
constexpr std::size_t adaptive_environments = 16;
constexpr std::size_t adaptive_markov_dimensions = 3;
constexpr std::size_t adaptive_macro_concepts = 3;
constexpr std::size_t macro_catalog_concepts = 140;
constexpr std::size_t market_numeric_columns = 6;
constexpr std::size_t maximum_catalog_bytes = std::size_t{512} * 1024 * 1024;
constexpr std::size_t maximum_transitions = std::size_t{1} << 20;
constexpr std::size_t maximum_selection_evaluations = 4096;
constexpr std::size_t maximum_rollout = 16384;
constexpr std::size_t diagnostic_transitions = 32;
constexpr std::size_t minimum_vram_bytes = std::size_t{256} * 1024 * 1024;
constexpr std::size_t maximum_vram_bytes = std::size_t{6} * 1024 * 1024 * 1024;
constexpr std::size_t maximum_process_memory_bytes = std::size_t{12} * 1024 * 1024 * 1024;
constexpr double maximum_vram_fraction = 0.75;
constexpr std::array allowed_seeds{uint64_t{42}, uint64_t{43}, uint64_t{44}};
constexpr std::string_view selection_metric = "ruin_count_then_mean_log_growth";
// Variante declarada que descuenta la venta final al último cierre. No sustituye a la anterior.
constexpr std::string_view liquidated_selection_metric =
    "ruin_count_then_mean_liquidated_log_growth";
constexpr std::string_view lock_name = "mars-titan-scientific-gpu.lock";
constexpr int64_t reconstructed_schema = 4;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::invalid_argument(std::string(message));
    }
}

void require_fields(const Json& object, std::initializer_list<std::string_view> names) {
    require(object.is_object() && object.size() == names.size() &&
                std::ranges::all_of(names,
                                    [&](auto name) { return object.contains(std::string(name)); }),
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
    PpoObjectiveConfig objective;
    simulation::Parameters environment;
    std::size_t environments = 0;
    std::size_t checkpoint_transitions = 0;
    double min_delta = 0;
    bool liquidated_selection = false;
    std::size_t patience = 0;
    bool early_stopping = false;
    std::size_t min_transitions = 0;
    int64_t schema_version = 1;
    PpoLearningOptions learning;
    std::size_t evaluation_transitions = 0;
    Json markov_identity = nullptr;
    std::optional<std::filesystem::path> markov_path;
};

// El esquema 3 introduce cursores de validación y el 4 conserva su selección sobre cintas
// reconstruidas por ventana walk-forward, sin contexto observable ni mundos técnicos.
bool cursor_selection(const ExperimentConfig& config) { return config.schema_version >= 3; }
bool reconstructed(const ExperimentConfig& config) {
    return config.schema_version == reconstructed_schema;
}

bool markov_variant(std::string_view variant) {
    return variant == "ppo_hmm" || variant == "ppo_episodic_hmm" || variant == "ppo_recent_aux" ||
           variant == "ppo_replay_aux";
}

void configure_agent(ExperimentConfig& config, const std::filesystem::path& path) {
    const auto& agent = config.document.at("agent");
    require_fields(agent, {"variant", "trading_field", "markov_fields", "hmm_file"});
    auto& options = config.learning;
    options.enabled = true;
    options.environments = adaptive_environments;
    options.variant = agent.at("variant").get<std::string>();
    if (reconstructed(config)) {
        // Las cintas reconstruidas no tienen calentamiento ni contexto: todas las sesiones
        // son decisiones y la etapa solo compara PPO y Double DQN con la misma representación.
        require((options.variant == "ppo" || options.variant == "double_dqn") &&
                    agent.at("trading_field").is_null() && agent.at("markov_fields") == Json::array() &&
                    agent.at("hmm_file").is_null(),
                "El esquema 4 admite PPO o Double DQN sin contexto, máscara ni HMM");
        return;
    }
    constexpr std::array<std::string_view, 9> variants{
        "ppo",       "ppo_window",       "ppo_gru",        "ppo_episodic",
        "ppo_hmm",   "ppo_episodic_hmm", "ppo_recent_aux", "ppo_replay_aux",
        "double_dqn"};
    require(std::ranges::find(variants, options.variant) != variants.end(),
            "La variante adaptativa PPO no está admitida");
    options.trading_field = count(agent.at("trading_field"));
    require(agent.at("markov_fields").is_array() &&
                agent.at("markov_fields").size() <= adaptive_markov_dimensions,
            "El contexto HMM necesita como máximo tres campos");
    for (const auto& field : agent.at("markov_fields")) {
        options.markov_fields.push_back(count(field));
    }
    if (!markov_variant(options.variant)) {
        require(agent.at("hmm_file").is_null() && options.markov_fields.empty(),
                "Esta variante no consume parámetros HMM");
        return;
    }
    require_fields(agent.at("hmm_file"), {"path", "sha256"});
    const auto relative = agent.at("hmm_file").at("path").get<std::string>();
    require(!relative.empty(), "Falta la ruta del HMM ajustado con entrenamiento");
    config.markov_path = path.parent_path() / relative;
    const auto bytes = read_bounded_file(*config.markov_path, simulation::maximum_manifest_bytes);
    const auto digest = content_sha256(bytes);
    require(agent.at("hmm_file").at("sha256") == digest, "El HMM no conserva su SHA256 declarado");
    const auto model = parse_bounded_json(bytes);
    require(read_json_int64(model.at("schema_version")) == 1 && model.at("fit_split") == "train" &&
                model.at("inference") == "forward_filter_only" &&
                model.at("feature_indices") == options.markov_fields &&
                options.markov_fields.size() == adaptive_markov_dimensions &&
                std::set<std::size_t>(options.markov_fields.begin(), options.markov_fields.end())
                        .size() == adaptive_markov_dimensions,
            "El HMM no conserva ajuste de entrenamiento y filtrado temporal");
    const auto& parameters = model.at("parameters");
    require_fields(parameters,
                   {"states", "dimensions", "prior", "transitions", "means", "variances"});
    simulation::MarkovParameters parsed;
    parsed.states = count(parameters.at("states"));
    parsed.dimensions = count(parameters.at("dimensions"));
    require(parsed.states == 2 && parsed.dimensions == options.markov_fields.size(),
            "El experimento necesita un HMM de dos estados y tres variables");
    for (const auto& [name, size] :
         std::array{std::pair{"prior", parsed.states},
                    std::pair{"transitions", parsed.states * parsed.states},
                    std::pair{"means", parsed.states * parsed.dimensions},
                    std::pair{"variances", parsed.states * parsed.dimensions}}) {
        require(parameters.at(name).is_array() && parameters.at(name).size() == size,
                "Los parámetros HMM no conservan sus dimensiones");
        for (const auto& value : parameters.at(name)) {
            static_cast<void>(finite_number(value));
        }
    }
    parsed.prior = parameters.at("prior").get<std::vector<double>>();
    parsed.transitions = parameters.at("transitions").get<std::vector<double>>();
    parsed.means = parameters.at("means").get<std::vector<double>>();
    parsed.variances = parameters.at("variances").get<std::vector<double>>();
    const simulation::MarkovFilter validated(parsed);
    static_cast<void>(validated);
    options.markov = std::move(parsed);
    config.markov_identity = Json{{"sha256", digest}, {"model", model}};
}

PpoObjectiveConfig objective_configuration(const Json& value) {
    require(value.is_object() && value.contains("id") && value.at("id").is_string(),
            "El objetivo PPO necesita una identidad explícita");
    PpoObjectiveConfig result;
    result.kind = objective_kind(value.at("id").get<std::string>());
    require(result.enabled(), "El bloque de objetivo no puede migrar la política legacy");
    if (result.kind == PpoObjectiveKind::kl_penalty_adaptive) {
        require_fields(value, {"schema_version", "id", "target_kl", "beta_initial", "beta_min", "beta_max"});
        result.beta_initial = finite_number(value.at("beta_initial"));
        result.beta_min = finite_number(value.at("beta_min"));
        result.beta_max = finite_number(value.at("beta_max"));
    } else if (result.kind == PpoObjectiveKind::clip_kl_epoch_stop) {
        require_fields(value, {"schema_version", "id", "target_kl"});
    } else {
        require_fields(value, {"schema_version", "id"});
    }
    require(read_json_int64(value.at("schema_version")) == 1, "El objetivo PPO usa otro esquema");
    if (value.contains("target_kl")) { result.target_kl = finite_number(value.at("target_kl")); }
    result.validate();
    return result;
}

ExperimentConfig configuration(const std::filesystem::path& path) {
    ExperimentConfig result;
    result.document =
        parse_bounded_json(read_bounded_file(path, simulation::maximum_manifest_bytes));
    const auto& document = result.document;
    auto base_document = document;
    if (document.contains("policy_objective")) {
        require(read_json_int64(document.at("schema_version")) >= 2,
                "El objetivo explícito necesita una configuración adaptativa nueva");
        result.objective = objective_configuration(document.at("policy_objective"));
        base_document.erase("policy_objective");
    }
    result.schema_version = read_json_int64(document.at("schema_version"));
    if (result.schema_version >= 2 && result.schema_version <= reconstructed_schema) {
        require_fields(base_document, {"schema_version", "training", "environments", "hyperparameters",
                                  "environment", "checkpoint_transitions", "selection",
                                  "final_test_opened", "agent", "evaluation_transitions"});
        configure_agent(result, path);
        require(!result.objective.enabled() ||
                    (result.learning.variant != "double_dqn" && result.learning.variant != "ppo_recent_aux" &&
                     result.learning.variant != "ppo_replay_aux"),
                "El controlador PPO no admite Double DQN ni actualizaciones auxiliares");
    } else {
        require_fields(base_document,
                       {"schema_version", "training", "environments", "hyperparameters",
                        "environment", "checkpoint_transitions", "selection", "final_test_opened"});
    }
    require((result.schema_version >= 1 && result.schema_version <= reconstructed_schema) &&
                !boolean(document.at("final_test_opened")),
            "La versión PPO o el cierre del test no es válido");
    const auto& training = document.at("training");
    require_fields(
        training, {"total_transitions", "rollout_transitions", "workers", "seed", "rollout_bytes"});
    result.training = {count(training.at("total_transitions")),
                       count(training.at("rollout_transitions")), count(training.at("workers")),
                       static_cast<uint64_t>(count(training.at("seed"))),
                       count(training.at("rollout_bytes"))};
    const auto& hyper = document.at("hyperparameters");
    require_fields(hyper, {"learning_rate", "gamma", "gae_lambda", "clip", "entropy",
                           "value_weight", "gradient_norm", "epochs", "minibatch_size"});
    result.hyperparameters = {
        finite_number(hyper.at("learning_rate")),   finite_number(hyper.at("gamma")),
        finite_number(hyper.at("gae_lambda")),      finite_number(hyper.at("clip")),
        finite_number(hyper.at("entropy")),         finite_number(hyper.at("value_weight")),
        finite_number(hyper.at("gradient_norm")),   read_json_int64(hyper.at("epochs")),
        read_json_int64(hyper.at("minibatch_size"))};
    result.hyperparameters.validate();
    const auto& environment = document.at("environment");
    require_fields(environment,
                   {"capital", "cost_bps", "participation", "score_scale", "ruin_penalty"});
    result.environment = {finite_number(environment.at("capital")),
                          finite_number(environment.at("cost_bps")),
                          finite_number(environment.at("participation")),
                          finite_number(environment.at("score_scale")),
                          finite_number(environment.at("ruin_penalty"))};
    const auto& selection = document.at("selection");
    if (cursor_selection(result)) {
        require_fields(selection,
                       {"min_delta", "patience", "early_stopping", "metric", "min_transitions"});
        result.min_transitions = count(selection.at("min_transitions"));
    } else {
        require_fields(selection, {"min_delta", "patience", "early_stopping", "metric"});
    }
    require(selection.at("metric") == selection_metric ||
                selection.at("metric") == liquidated_selection_metric,
            "La métrica de selección PPO no está admitida");
    result.liquidated_selection = selection.at("metric") == liquidated_selection_metric;
    result.min_delta = finite_number(selection.at("min_delta"));
    result.patience = count(selection.at("patience"));
    result.early_stopping = boolean(selection.at("early_stopping"));
    result.environments = count(document.at("environments"));
    result.checkpoint_transitions = count(document.at("checkpoint_transitions"));
    if (result.learning.enabled) {
        result.evaluation_transitions = count(document.at("evaluation_transitions"));
        require(
            result.environments == adaptive_environments &&
                result.training.rollout_transitions != 0 &&
                result.evaluation_transitions >= result.training.rollout_transitions &&
                result.evaluation_transitions <= maximum_transitions &&
                result.evaluation_transitions % result.training.rollout_transitions == 0,
            "La evaluación adaptativa necesita 16 entornos y un intervalo múltiplo del recorrido");
    }
    require(!cursor_selection(result) ||
                (result.min_transitions <= result.training.total_transitions &&
                 result.min_transitions % result.evaluation_transitions == 0 &&
                 1 + (result.training.total_transitions + result.evaluation_transitions - 1) /
                         result.evaluation_transitions <= maximum_selection_evaluations &&
                 (result.early_stopping || result.min_transitions == 0) &&
                 (!result.early_stopping || result.learning.variant != "double_dqn" ||
                  result.min_transitions >= dqn_learning_warmup)),
            "El selector debe respetar mínimo, intervalo, presupuesto, calentamiento y hasta 4096 evaluaciones");
    require(result.environments > 0 && result.environments <= simulation::maximum_environments &&
                result.training.total_transitions >= result.environments &&
                result.training.total_transitions <= maximum_transitions &&
                result.training.total_transitions % result.environments == 0 &&
                result.training.rollout_transitions >= result.environments &&
                result.training.rollout_transitions <= maximum_rollout &&
                result.training.rollout_transitions <= result.training.total_transitions &&
                result.training.rollout_transitions % result.environments == 0 &&
                result.checkpoint_transitions > 0 &&
                result.checkpoint_transitions <= maximum_transitions && result.min_delta >= 0 &&
                result.patience > 0 && result.patience <= maximum_transitions &&
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

void require_device(const PpoExperimentOptions& options, std::size_t total_transitions) {
    require(options.device == "cpu" || options.device == "cuda:0",
            "El dispositivo PPO debe ser cpu o cuda:0");
    if (options.device == "cpu") {
        require(options.diagnostic && total_transitions <= diagnostic_transitions &&
                    !options.gpu_lease_fd && !options.vram_budget_bytes &&
                    !options.vram_total_bytes,
                "La CPU solo admite el diagnóstico explícito de hasta 32 transiciones");
    } else {
        require(!options.diagnostic, "El diagnóstico PPO se ejecuta únicamente en CPU");
    }
}

void validate_options(const PpoExperimentOptions& options, const ExperimentConfig& config) {
    const auto source_limit = config.learning.enabled ? maximum_adaptive_sources : maximum_sources;
    if (options.audit_run) {
        require(
            config.learning.enabled && !options.audit_run->empty() && !options.output.empty() &&
                options.train_tapes.empty() && options.validation_tapes.empty() &&
                !options.audit_tapes.empty() && options.audit_tapes.size() <= source_limit &&
                !options.stop_after,
            "La auditoría requiere esquema 2, fuentes reservadas y ninguna fuente de selección");
        require(!contains(*options.audit_run, options.output) &&
                    !contains(options.output, *options.audit_run),
                "La auditoría debe guardar sus resultados fuera de la ejecución seleccionada");
    } else {
        require(options.audit_tapes.empty() && !options.output.empty() &&
                    !options.train_tapes.empty() && !options.validation_tapes.empty() &&
                    options.train_tapes.size() <= source_limit &&
                    options.validation_tapes.size() <= source_limit &&
                    (reconstructed(config) ? options.validation_tapes.size() == 1
                     : config.learning.enabled
                         ? options.train_tapes.size() >= config.environments
                         : config.environments % options.train_tapes.size() == 0),
                "Se necesitan fuentes acotadas y réplicas equilibradas entre todos los escenarios");
    }
    require_device(options, config.training.total_transitions);
    require(!contains(options.output, options.config),
            "La configuración debe quedar fuera de la salida PPO");
    if (config.markov_path) {
        require(!contains(options.output, *config.markov_path),
                "Los parámetros HMM deben quedar fuera de la salida PPO");
    }
    for (const auto* sources :
         {&options.train_tapes, &options.validation_tapes, &options.audit_tapes}) {
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
    require(fraction <= maximum_vram_fraction,
            "El presupuesto PPO no puede superar el 75 % de la VRAM");
    const auto* configured = std::getenv("XDG_RUNTIME_DIR");
    const std::filesystem::path runtime =
        configured != nullptr ? configured : "/tmp/mars-titan-" + std::to_string(getuid());
    require(runtime.is_absolute(), "El directorio del bloqueo GPU debe ser absoluto");
    struct stat directory{};
    struct stat inherited{};
    struct stat current{};
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
        const auto formatted =
            std::to_chars(fraction_text.begin(), fraction_text.end(), std::nextafter(fraction, 0.),
                          std::chars_format::general, std::numeric_limits<double>::max_digits10);
        require(formatted.ec == std::errc{}, "No se pudo representar el presupuesto de VRAM");
        c10::CachingAllocator::setAllocatorSettings(
            "per_process_memory_fraction:" + std::string(fraction_text.data(), formatted.ptr));
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
                input.sessions <= simulation::maximum_sessions &&
                input.actions <= simulation::maximum_actions &&
                input.context_fields <= simulation::maximum_context_fields,
            "Las dimensiones declaradas de las fuentes PPO exceden sus límites");
    const auto context_values = input.sessions * input.context_fields;
    account_memory(replicas * live_context_copies,
                   context_values * sizeof(simulation::ContextValue), bytes);
    account_memory(replicas * live_context_copies,
                   input.context_fields *
                       (sizeof(simulation::ContextField) + 2 * maximum_field_string),
                   bytes);
    const auto width = input.assets * financial_fields + 2 + 3 * input.context_fields;
    account_memory(replicas, 4 * width * sizeof(float) + 4 * (sizeof(double) + 3 * sizeof(uint8_t)),
                   bytes);
    account_memory(replicas,
                   input.assets * live_state_copies *
                       (sizeof(mt_position_v1) + sizeof(mt_trade_v1) + sizeof(std::size_t) +
                        sizeof(double) + sizeof(uint32_t) + 1),
                   bytes);
    account_memory(replicas, input.sessions * 4 * sizeof(std::vector<std::size_t>), bytes);
    account_memory(replicas,
                   input.actions * live_state_copies *
                       (sizeof(simulation::Receivable) + sizeof(std::size_t) + 1),
                   bytes);
}

void validate_source_role(const Json& manifest, std::string_view partition, bool adaptive,
                          bool audit = false) {
    const auto& identity = manifest.at("identity");
    require(read_json_int64(manifest.at("schema_version")) == 1 &&
                !boolean(manifest.at("final_test_opened")) &&
                identity.at("domain") == "synthetic" && identity.at("partition") == partition,
            "La fuente PPO no corresponde a su partición o pretende abrir el test");
    const auto& source = identity.at("source");
    if (source.is_object()) {
        if (source.contains("evaluator_only")) {
            require(boolean(source.at("evaluator_only")) == audit,
                    "La fuente de auditoría no puede participar en entrenamiento o selección");
        }
        if (source.contains("generator") && source.at("generator").is_object() &&
            source.at("generator").contains("split")) {
            require(source.at("generator").at("split") == (audit ? "audit" : partition),
                    "La partición de auditoría no puede participar en entrenamiento o selección");
        }
    }
    if (adaptive) {
        require(source.is_object() && source.at("generator").is_object() &&
                    source.at("generator").at("split") == (audit ? "audit" : partition) &&
                    source.at("analysis_domain") == "technical" &&
                    !boolean(source.at("real_corpus_compatible")),
                "La fuente adaptativa necesita procedencia técnica y una partición explícita");
    }
}

void preflight_memory(const PpoExperimentOptions& options, const ExperimentConfig& config) {
    std::size_t total = 0;
    std::size_t observation_width = 0;
    for (const auto* paths :
         {&options.train_tapes, &options.validation_tapes, &options.audit_tapes}) {
        const auto replicas =
            paths == &options.train_tapes && !config.learning.enabled && !paths->empty()
                ? config.environments / paths->size()
                : 1;
        for (const auto& path : *paths) {
            const auto manifest = parse_bounded_json(
                read_bounded_file(path / "manifest.json", simulation::maximum_manifest_bytes));
            const auto partition = paths == &options.train_tapes ? "train" : "validation";
            if (reconstructed(config)) {
                require_policy_tape_manifest(manifest, paths == &options.train_tapes
                                                           ? PolicyTapeRole::train
                                                       : paths == &options.audit_tapes
                                                           ? PolicyTapeRole::evaluation
                                                           : PolicyTapeRole::validation);
            } else {
                validate_source_role(manifest, partition, config.learning.enabled,
                                     paths == &options.audit_tapes);
            }
            std::size_t fields = 0;
            const auto context_path = path / "context.json";
            simulation::require_safe_path(context_path);
            if (std::filesystem::exists(context_path)) {
                const auto context = parse_bounded_json(
                    read_bounded_file(context_path, simulation::maximum_manifest_bytes));
                require(context.at("fields").is_array(), "Falta el esquema del contexto PPO");
                fields = context.at("fields").size();
            }
            const InputFootprint footprint{count(manifest.at("assets")),
                                           count(manifest.at("sessions")),
                                           manifest.at("actions").size(), fields};
            account_input(footprint, replicas, total);
            if (config.learning.enabled) {
                require(footprint.assets <= simulation::maximum_rows / footprint.sessions,
                        "La cinta adaptativa supera el número máximo de filas");
                account_memory(footprint.assets * footprint.sessions,
                               market_numeric_columns * sizeof(double), total);
                account_memory(footprint.sessions, 3 * sizeof(int64_t), total);
            }
            constexpr std::size_t financial_fields = 6;
            observation_width =
                std::max(observation_width, footprint.assets * financial_fields + 2 + 3 * fields);
        }
    }
    constexpr std::size_t scalar_bytes = sizeof(int64_t) + 4 * sizeof(double) + 3 * sizeof(bool);
    if (!options.audit_run) {
        account_memory(config.training.rollout_transitions,
                       3 * (observation_width * sizeof(float) + scalar_bytes), total);
    }
    require(total <= maximum_catalog_bytes,
            "El catálogo adaptativo supera 512 MiB de memoria prevista");
}

struct LoadedSource {
    simulation::BatchInput input;
    Json identity;
    std::optional<int64_t> generator_seed;
};

LoadedSource load_source(const std::filesystem::path& path, std::string_view partition,
                         const simulation::Parameters& parameters, bool adaptive,
                         bool audit = false) {
    const auto manifest_bytes =
        read_bounded_file(path / "manifest.json", simulation::maximum_manifest_bytes);
    const auto manifest = parse_bounded_json(manifest_bytes);
    validate_source_role(manifest, partition, adaptive, audit);
    LoadedSource result{load_ppo_input(path), Json::object(), std::nullopt};
    result.input.parameters = parameters;
    require(result.input.tape->partition == partition && result.input.tape->domain == "synthetic" &&
                result.input.tape->source_sha256 == content_sha256(manifest_bytes),
            "La fuente PPO no corresponde a su partición o cambió durante la lectura");
    const auto& source = manifest.at("identity").at("source");
    if (adaptive) {
        if (!result.input.context) {
            throw std::invalid_argument("La fuente adaptativa no contiene contexto");
        }
        const auto warmup = count(source.at("decision_start"));
        require(warmup > 0 && warmup < result.input.tape->close_times.size() - 1 &&
                    warmup == count(source.at("generator").at("warmup_sessions")) &&
                    count(source.at("warmup_action")) == 1 &&
                    !boolean(source.at("warmup_policy_loss")) &&
                    count(source.at("generator").at("sessions")) ==
                        result.input.tape->close_times.size() &&
                    count(source.at("generator").at("assets")) == result.input.tape->assets.size(),
                "El calentamiento no conserva las dimensiones de su procedencia");
        const auto& context = *result.input.context;
        constexpr std::array<std::string_view, adaptive_macro_concepts> macro_fields{
            "us_treasury_2y", "us_treasury_10y", "us_curve_10y_2y"};
        require(source.at("modality_contract") == "technical_context_only" &&
                    source.at("simulated_macro").is_array() &&
                    source.at("simulated_macro").size() == macro_fields.size(),
                "La procedencia no conserva la cobertura macro técnica declarada");
        for (std::size_t index = 0; index < macro_fields.size(); ++index) {
            require(source.at("simulated_macro").at(index) == macro_fields.at(index) &&
                        std::ranges::any_of(context.fields,
                                            [&](const auto& field) {
                                                return field.name == macro_fields.at(index);
                                            }),
                    "Falta un concepto macro declarado entre los campos observables");
        }
        const auto width = context.fields.size();
        for (std::size_t session = 0; session < result.input.tape->close_times.size(); ++session) {
            require(context.values[session * width + width - 1].value ==
                        (session >= warmup ? 1.0F : 0.0F),
                    "El contexto modifica el límite de calentamiento declarado");
        }
    }
    if (source.is_object() && source.contains("generator")) {
        require(source.at("generator").is_object() && source.at("generator").contains("seed"),
                "La procedencia del generador necesita una semilla explícita");
        const auto seed = read_json_int64(source.at("generator").at("seed"));
        require(seed >= 0 && (!adaptive ||
                              static_cast<uint64_t>(seed) <= std::numeric_limits<uint32_t>::max()),
                "La semilla del escenario no pertenece al rango admitido");
        result.generator_seed = seed;
    }
    result.identity = Json{
        {"manifest_sha256", result.input.tape->source_sha256},
        {"market_sha256", manifest.at("file_sha256")},
        {"context_sha256",
         result.input.context ? Json(result.input.context->source_sha256) : Json(nullptr)},
        {"generator_seed", result.generator_seed ? Json(*result.generator_seed) : Json(nullptr)}};
    return result;
}

void require_learning_fields(const simulation::BatchInput& input,
                             const PpoLearningOptions& options) {
    if (!options.enabled) {
        return;
    }
    if (!input.context || !options.trading_field) {
        throw std::invalid_argument(
            "El escenario adaptativo necesita contexto y campo de admisión de operaciones");
    }
    const auto& context = *input.context;
    const auto field = *options.trading_field;
    require(field + 1 == context.fields.size() && context.fields[field].name == "trading_enabled" &&
                context.fields[field].unit == "boolean",
            "trading_enabled debe ser el último campo booleano del contexto");
    bool enabled = false;
    for (std::size_t session = 0; session < input.tape->close_times.size(); ++session) {
        const auto& flag = context.values[session * context.fields.size() + field];
        require(flag.present && (flag.value == 0 || flag.value == 1) &&
                    (!enabled || flag.value == 1),
                "La admisión de operaciones contiene valores ausentes o reinicia el calentamiento");
        enabled = flag.value == 1;
    }
    require(enabled, "El mundo no contiene decisiones posteriores al calentamiento");
    for (const auto index : options.markov_fields) {
        require(index < context.fields.size() && index != field,
                "El HMM contiene un índice de contexto no admitido");
    }
}

void require_compatible(const simulation::BatchInput& reference,
                        const simulation::BatchInput& input) {
    require(reference.tape->assets == input.tape->assets &&
                reference.tape->currency == input.tape->currency &&
                reference.tape->domain == input.tape->domain &&
                reference.tape->parent_id == input.tape->parent_id &&
                reference.tape->instruments == input.tape->instruments &&
                reference.context.has_value() == input.context.has_value() &&
                (!reference.context || reference.context->fields == input.context->fields),
            "Entrenamiento y validación no conservan activos, moneda, padre, reglas o contexto");
}

struct ExperimentInputs {
    std::vector<simulation::BatchInput> training;
    std::vector<simulation::BatchInput> validation;
    Json identity;
};

// Ajuste y validación de una política por ventana: cintas reconstruidas en orden temporal,
// con los mismos activos, reglas y base histórica. Los entornos recorren el ajuste en ciclo.
ExperimentInputs load_reconstructed_inputs(const PpoExperimentOptions& options,
                                           const ExperimentConfig& config) {
    std::vector<PolicyTape> tapes;
    for (const auto& path : options.train_tapes) {
        tapes.push_back(load_policy_tape(path, PolicyTapeRole::train, config.environment));
    }
    for (const auto& path : options.validation_tapes) {
        tapes.push_back(load_policy_tape(path, PolicyTapeRole::validation, config.environment));
    }
    require_policy_sequence(tapes);
    ExperimentInputs result;
    Json training = Json::array();
    Json validation = Json::array();
    std::size_t bytes = 0;
    for (auto& tape : tapes) {
        const auto& market = *tape.input.tape;
        require(market.close_times.size() <= static_cast<std::size_t>(ppo_maximum_history),
                "Una cinta reconstruida supera la historia recurrente acotada");
        account_input({market.assets.size(), market.close_times.size(), market.actions.size(), 0},
                      1, bytes);
        const bool train = tape.identity.at("role") == "train";
        (train ? training : validation).push_back(std::move(tape.identity));
        (train ? result.training : result.validation).push_back(std::move(tape.input));
    }
    result.identity = Json{{"train", training}, {"validation", validation}};
    return result;
}

ExperimentInputs load_inputs(const PpoExperimentOptions& options, const ExperimentConfig& config) {
    if (reconstructed(config)) {
        return load_reconstructed_inputs(options, config);
    }
    ExperimentInputs result;
    std::vector<simulation::BatchInput> sources;
    std::set<std::string> hashes;
    std::set<std::string> training_market_hashes;
    std::set<int64_t> training_seeds;
    std::set<int64_t> validation_seeds;
    Json training = Json::array();
    Json validation = Json::array();
    for (const auto& path : options.train_tapes) {
        auto loaded = load_source(path, "train", config.environment, config.learning.enabled);
        require_learning_fields(loaded.input, config.learning);
        require(hashes.insert(loaded.input.tape->source_sha256).second,
                "El entrenamiento PPO duplica una fuente");
        if (!sources.empty()) {
            require_compatible(sources.front(), loaded.input);
        }
        if (loaded.generator_seed) {
            const auto inserted = training_seeds.insert(*loaded.generator_seed).second;
            require(!config.learning.enabled || inserted,
                    "El entrenamiento adaptativo repite una semilla de mundo");
        }
        training_market_hashes.insert(loaded.identity.at("market_sha256").get<std::string>());
        training.push_back(std::move(loaded.identity));
        sources.push_back(std::move(loaded.input));
    }
    for (const auto& path : options.validation_tapes) {
        auto loaded = load_source(path, "validation", config.environment, config.learning.enabled);
        require_learning_fields(loaded.input, config.learning);
        require_compatible(sources.front(), loaded.input);
        if (config.learning.enabled) {
            require(loaded.generator_seed && validation_seeds.insert(*loaded.generator_seed).second,
                    "La validación adaptativa necesita semillas únicas por mundo");
        }
        require(hashes.insert(loaded.input.tape->source_sha256).second &&
                    !training_market_hashes.contains(
                        loaded.identity.at("market_sha256").get<std::string>()) &&
                    (!loaded.generator_seed || !training_seeds.contains(*loaded.generator_seed)),
                "La validación PPO reutiliza datos o semillas de entrenamiento");
        validation.push_back(std::move(loaded.identity));
        result.validation.push_back(std::move(loaded.input));
    }
    std::size_t replicated_bytes = 0;
    const auto replicas =
        config.learning.enabled ? std::size_t{1} : config.environments / sources.size();
    for (const auto& input : sources) {
        account_input({input.tape->assets.size(), input.tape->close_times.size(),
                       input.tape->actions.size(),
                       input.context ? input.context->fields.size() : 0},
                      replicas, replicated_bytes);
    }
    for (const auto& input : result.validation) {
        account_input({input.tape->assets.size(), input.tape->close_times.size(),
                       input.tape->actions.size(),
                       input.context ? input.context->fields.size() : 0},
                      1, replicated_bytes);
    }
    if (config.learning.enabled) {
        result.training = std::move(sources);
    } else {
        result.training.reserve(config.environments);
        for (std::size_t index = 0; index < config.environments; ++index) {
            result.training.push_back(sources[index % sources.size()]);
        }
    }
    if (config.learning.markov) {
        const auto& fitted = config.markov_identity.at("model").at("train_manifest_sha256");
        require(fitted.is_array() && fitted.size() == training.size(),
                "El ajuste HMM no corresponde al catálogo de entrenamiento");
        std::set<std::string> expected;
        for (const auto& record : training) {
            expected.insert(record.at("manifest_sha256").get<std::string>());
        }
        for (const auto& digest : fitted) {
            require(expected.erase(digest.get<std::string>()) == 1,
                    "El HMM se ajustó con fuentes ajenas, repetidas o reservadas");
        }
        require(expected.empty(), "El HMM no cubre las fuentes de entrenamiento declaradas");
    }
    result.identity = Json{{"train", training}, {"validation", validation}};
    return result;
}

Json experiment_identity(const PpoExperimentOptions& options, const ExperimentConfig& config,
                         const ExperimentInputs& inputs) {
    Json result{{"schema_version", config.schema_version},
                {"kind", "native_ppo"},
                {"configuration", config.document},
                {"sources", inputs.identity},
                {"device", options.device},
                {"diagnostic", options.diagnostic},
                {"architecture", "mlp_64_64_tanh_6_1_v1"},
                {"torch_version", TORCH_VERSION},
                {"native_source_sha256", MARS_TITAN_NATIVE_SOURCE_SHA256},
                {"native_build_sha256", MARS_TITAN_NATIVE_BUILD_SHA256},
                {"precision", "weights_fp32_gae_ratio_fp64"},
                {"deterministic", true},
                {"evaluation_policy", "greedy_argmax"},
                {"rollout_policy", "categorical_sampling"},
                {"parent_frozen", true},
                {"final_test_opened", false}};
    if (config.learning.enabled) {
        result["architecture"] = "adaptive_policy_v1";
        result["agent_variant"] = config.learning.variant;
        result["markov"] = config.markov_identity;
        result["fold"] = "experimental";
        result["representation"] = "fixed_projection_1729_v1";
        result["projection_seed"] = policy_projection_seed;
        result["reservoir_seed"] = config.training.seed;
        const auto& reference = inputs.training.front();
        Json fields = Json::array();
        if (reference.context) {
            for (const auto& field : reference.context->fields) {
                fields.push_back(Json{{"name", field.name}, {"unit", field.unit}});
            }
        }
        result["observation_schema"] = Json{{"assets", reference.tape->assets},
                                            {"currency", reference.tape->currency},
                                            {"domain", reference.tape->domain},
                                            {"parent_id", reference.tape->parent_id},
                                            {"context_fields", fields}};
        if (reconstructed(config)) {
            // Cada ventana reajusta el predictor: la política se liga a la base histórica.
            result["observation_schema"]["parent_id"] = nullptr;
            result["observation_schema"]["historical_basis"] = reference.tape->historical_basis;
            result["sources_contract"] = simulation::reconstructed_tape_contract;
        }
    }
    if (config.objective.enabled()) {
        result["policy_objective"] = config.document.at("policy_objective");
        result["sampler_contract"] = ppo_sampler_contract;
        result["kl_contract"] = "categorical_behavior_to_current_kl_v1";
        result["kl_measurement"] = "complete_valid_rows_after_epoch_v1";
    }
    return result;
}

struct Progress {
    std::string status = "running";
    std::size_t evaluations = 0;
    std::size_t stale_evaluations = 0;
    std::vector<std::size_t> evaluation_cursors;
    std::optional<std::size_t> evaluated_optimizer_steps;
    std::optional<std::size_t> evaluated_transitions;
    std::string pause_reason = "requested_pause";
    Json best = nullptr;
};

Json progress_json(const Progress& progress, const ExperimentConfig& config) {
    Json result{{"status", progress.status},
                {"evaluations", progress.evaluations},
                {"stale_evaluations", progress.stale_evaluations},
                {"best", progress.best},
                {"evaluated_optimizer_steps", progress.evaluated_optimizer_steps
                                                  ? Json(*progress.evaluated_optimizer_steps)
                                                  : Json(nullptr)}};
    if (config.learning.enabled) {
        result["evaluated_transitions"] =
            progress.evaluated_transitions ? Json(*progress.evaluated_transitions) : Json(nullptr);
        result["pause_reason"] = progress.pause_reason;
    }
    if (cursor_selection(config)) {
        result["evaluation_cursors"] = progress.evaluation_cursors;
    }
    return result;
}

Json learning_identity(const ExperimentConfig& config) {
    return Json{{"agent", config.document.at("agent")},
                {"environments", config.environments},
                {"fold", "experimental"},
                {"representation", "fixed_projection_1729_v1"},
                {"projection_seed", policy_projection_seed},
                {"reservoir_seed", config.training.seed},
                {"hmm_sha256", config.markov_identity.is_null()
                                   ? Json(nullptr)
                                   : config.markov_identity.at("sha256")}};
}

Json controller_json(const PpoControllerState& state) {
    return Json{{"beta", state.beta}, {"last_beta", state.last_beta},
                {"completed_rollouts", state.completed_rollouts}, {"optimizer_steps", state.optimizer_steps},
                {"valid_rows", state.valid_rows}, {"completed_epochs", state.completed_epochs},
                {"skipped_epochs", state.skipped_epochs}, {"threshold_exceeded", state.threshold_exceeded},
                {"full_kl", state.full_kl ? Json(*state.full_kl) : Json(nullptr)}};
}

PpoControllerState controller_state(const Json& value, const ExperimentConfig& config, int64_t adam_steps) {
    require_fields(value, {"beta", "last_beta", "completed_rollouts", "optimizer_steps", "valid_rows",
                          "completed_epochs", "skipped_epochs", "threshold_exceeded", "full_kl"});
    require(value.at("beta").is_number_float() && value.at("last_beta").is_number_float() &&
                (value.at("full_kl").is_null() || value.at("full_kl").is_number_float()),
            "El controlador guardado necesita campos reales con su tipo original");
    PpoControllerState result;
    result.beta = finite_number(value.at("beta"));
    result.last_beta = finite_number(value.at("last_beta"));
    result.completed_rollouts = read_json_int64(value.at("completed_rollouts"));
    result.optimizer_steps = read_json_int64(value.at("optimizer_steps"));
    result.valid_rows = read_json_int64(value.at("valid_rows"));
    result.completed_epochs = read_json_int64(value.at("completed_epochs"));
    result.skipped_epochs = read_json_int64(value.at("skipped_epochs"));
    result.threshold_exceeded = boolean(value.at("threshold_exceeded"));
    if (!value.at("full_kl").is_null()) { result.full_kl = finite_number(value.at("full_kl")); }
    result.validate(config.objective, adam_steps, config.hyperparameters.epochs);
    return result;
}

Json state_json(const PpoTrainingState& state, const ExperimentConfig& config,
                const Progress& progress) {
    Json sessions = Json::array();
    for (const auto& session : state.environment.sessions) {
        sessions.push_back(simulation::snapshot_json(session));
    }
    Json result{{"schema_version", config.schema_version},
                {"configuration", config.document},
                {"device", state.device},
                {"diagnostic", state.diagnostic},
                {"transitions", state.transitions},
                {"optimizer_steps", state.optimizer_steps},
                {"invalid_transitions", state.invalid_transitions},
                {"episodes", state.episodes},
                {"reset_lanes", state.reset_lanes},
                {"sessions", sessions},
                {"context_sources", state.environment.context_sources},
                {"progress", progress_json(progress, config)}};
    if (config.learning.enabled) {
        result["source_indices"] = state.source_indices;
        result["next_source"] = state.next_source;
        result["observed_transitions"] = state.observed_transitions;
        result["learning"] = learning_identity(config);
    }
    if (config.objective.enabled()) { result["policy_controller"] = controller_json(state.controller); }
    return result;
}

PpoTrainingState restore_state(const PpoCheckpointBundle& bundle, const ExperimentConfig& config,
                               const PpoExperimentOptions& options) {
    const auto& metadata = bundle.metadata;
    auto legacy_metadata = metadata;
    if (config.objective.enabled()) {
        require(legacy_metadata.erase("policy_controller") == 1,
                "Falta el controlador PPO en los metadatos del checkpoint");
    }
    if (config.learning.enabled) {
        for (const auto* name :
             {"source_indices", "next_source", "observed_transitions", "learning"}) {
            require(legacy_metadata.erase(name) == 1,
                    "Falta estado adaptativo en el checkpoint PPO");
        }
    }
    require_fields(legacy_metadata,
                   {"schema_version", "configuration", "device", "diagnostic", "transitions",
                    "optimizer_steps", "invalid_transitions", "episodes", "reset_lanes", "sessions",
                    "context_sources", "progress"});
    require(read_json_int64(metadata.at("schema_version")) == config.schema_version &&
                (config.objective.enabled() ? metadata.at("configuration").dump() == config.document.dump()
                                            : metadata.at("configuration") == config.document) &&
                metadata.at("device") == options.device &&
                boolean(metadata.at("diagnostic")) == options.diagnostic,
            "El checkpoint PPO no conserva la configuración del experimento");
    require(metadata.at("sessions").is_array() &&
                metadata.at("sessions").size() == config.environments &&
                metadata.at("context_sources").is_array() &&
                metadata.at("context_sources").size() == config.environments &&
                metadata.at("reset_lanes").is_array() &&
                metadata.at("reset_lanes").size() <= config.environments,
            "El checkpoint PPO no conserva los carriles y sus fuentes");
    PpoTrainingState state;
    state.config = config.training;
    state.hyperparameters = config.hyperparameters;
    state.device = options.device;
    state.diagnostic = options.diagnostic;
    state.transitions = count(metadata.at("transitions"));
    state.optimizer_steps = count(metadata.at("optimizer_steps"));
    state.objective = config.objective;
    if (config.objective.enabled()) {
        state.controller = controller_state(metadata.at("policy_controller"), config,
                                             read_json_int64(metadata.at("optimizer_steps")));
    }
    state.invalid_transitions = count(metadata.at("invalid_transitions"));
    state.episodes = count(metadata.at("episodes"));
    for (const auto& lane : metadata.at("reset_lanes")) {
        state.reset_lanes.push_back(count(lane));
    }
    for (const auto& session : metadata.at("sessions")) {
        state.environment.sessions.push_back(simulation::read_snapshot(session));
    }
    state.environment.context_sources =
        metadata.at("context_sources").get<std::vector<std::string>>();
    state.policy_archive = bundle.policy_archive;
    if (config.learning.enabled) {
        require(metadata.at("source_indices").is_array() &&
                    metadata.at("source_indices").size() == config.environments,
                "Faltan los índices del catálogo asignados a cada entorno");
        for (const auto& value : metadata.at("source_indices")) {
            state.source_indices.push_back(count(value));
        }
        state.next_source = count(metadata.at("next_source"));
        state.observed_transitions = count(metadata.at("observed_transitions"));
        state.learning = config.learning;
        require(metadata.at("learning") == learning_identity(config),
                "El checkpoint adaptativo cambió la representación o los parámetros del agente");
        restore_training_buffer(bundle.rollout_archive, state);
    } else {
        state.rollout = deserialize_rollout(bundle.rollout_archive);
    }
    return state;
}

void validate_convergence_progress(const Json& progress, const ExperimentConfig& config,
                                  std::size_t transitions, std::size_t optimizer_steps) {
    if (!cursor_selection(config)) {
        return;
    }
    const auto evaluations = count(progress.at("evaluations"));
    const auto stale = count(progress.at("stale_evaluations"));
    const auto& cursor = progress.at("evaluated_transitions");
    const auto& history = progress.at("evaluation_cursors");
    require(transitions <= config.training.total_transitions && history.is_array() &&
                history.size() <= maximum_selection_evaluations && history.size() == evaluations &&
                cursor.is_null() == history.empty(),
            "El selector no conserva sus cursores de validación completos y acotados");
    if (cursor.is_null()) {
        require(transitions == 0 && stale == 0 && progress.at("best").is_null() &&
                    progress.at("status") != "completed" && progress.at("status") != "early_stopped",
                "El selector vacío contiene resultados o un estado terminal");
        return;
    }
    const auto evaluated = count(cursor);
    const auto interval = config.evaluation_transitions;
    require(count(history.front()) == 0 && count(history.back()) == evaluated &&
                evaluated <= transitions && transitions - evaluated < interval + config.environments,
            "El selector no conserva los cursores inicial y confirmado de validación");
    const auto best_transition = count(progress.at("best").at("transitions"));
    std::size_t eligible = 0;
    std::size_t previous = 0;
    bool selected = false;
    for (std::size_t index = 0; index < history.size(); ++index) {
        const auto current = count(history.at(index));
        require(current <= evaluated &&
                    (index == 0 ||
                     (current > previous && current - previous < interval + config.environments &&
                      (current - previous >= interval || current == config.training.total_transitions))),
                "Los cursores no respetan el orden y la programación de validaciones completas");
        selected = selected || current == best_transition;
        eligible += current > std::max(config.min_transitions, best_transition) ? 1 : 0;
        previous = current;
    }
    require(selected && stale == eligible,
            "La paciencia incluye validaciones anteriores al mínimo o a la mejor selección");
    const auto status = progress.at("status").get<std::string>();
    if (status == "completed" || status == "early_stopped") {
        require(evaluated == transitions &&
                    count(progress.at("evaluated_optimizer_steps")) == optimizer_steps,
                "La selección terminal tiene una validación pendiente");
    }
    require(status != "early_stopped" ||
                (config.early_stopping && optimizer_steps > 0 && evaluated > config.min_transitions &&
                 transitions < config.training.total_transitions && stale >= config.patience),
            "La parada no cumple el mínimo y la paciencia declarados");
}

Progress restore_progress(const Json& value, const PpoTrainingState& state,
                          const ExperimentConfig& config,
                          std::span<const simulation::BatchInput> validation) {
    if (cursor_selection(config)) {
        require_fields(value,
                       {"status", "evaluations", "stale_evaluations", "best", "evaluation_cursors",
                        "evaluated_optimizer_steps", "evaluated_transitions", "pause_reason"});
    } else if (config.learning.enabled) {
        require_fields(value,
                       {"status", "evaluations", "stale_evaluations", "best",
                        "evaluated_optimizer_steps", "evaluated_transitions", "pause_reason"});
    } else {
        require_fields(value, {"status", "evaluations", "stale_evaluations", "best",
                               "evaluated_optimizer_steps"});
    }
    validate_convergence_progress(value, config, state.transitions, state.optimizer_steps);
    Progress progress;
    progress.status = value.at("status").get<std::string>();
    require(progress.status == "running" || progress.status == "paused" ||
                progress.status == "completed" || progress.status == "early_stopped",
            "El estado del experimento PPO no está admitido");
    progress.evaluations = count(value.at("evaluations"));
    progress.stale_evaluations = count(value.at("stale_evaluations"));
    if (cursor_selection(config)) {
        for (const auto& cursor : value.at("evaluation_cursors")) {
            progress.evaluation_cursors.push_back(count(cursor));
        }
    }
    if (config.learning.enabled) {
        progress.pause_reason = value.at("pause_reason").get<std::string>();
        require(progress.pause_reason == "requested_pause" ||
                    progress.pause_reason == "trace_capacity",
                "El motivo de pausa de la traza no está admitido");
    }
    if (config.learning.enabled && !value.at("evaluated_transitions").is_null()) {
        progress.evaluated_transitions = count(value.at("evaluated_transitions"));
        require(*progress.evaluated_transitions <= state.transitions,
                "La evaluación adaptativa contiene transiciones futuras");
    }
    require(!config.learning.enabled ||
                progress.evaluated_transitions.has_value() == (progress.evaluations != 0),
            "Falta el cursor de la última evaluación adaptativa completa");
    require((cursor_selection(config) || progress.evaluations <= state.optimizer_steps + 1) &&
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
        // La variante liquidada añade su puntuación al candidato. El resto del esquema no cambia.
        auto fields = progress.best;
        if (config.liquidated_selection) {
            static_cast<void>(finite_number(fields.at("mean_liquidated_log_growth")));
            fields.erase("mean_liquidated_log_growth");
        }
        if (config.learning.enabled) {
            require_fields(fields,
                           {"ruin_count", "mean_log_growth", "mean_max_drawdown",
                            "validation_metrics", "transitions", "optimizer_steps", "episodes"});
            const auto drawdown = finite_number(progress.best.at("mean_max_drawdown"));
            require(drawdown >= 0 && drawdown <= 1,
                    "La selección adaptativa contiene una caída máxima inválida");
            const auto& metrics = progress.best.at("validation_metrics");
            require(metrics.is_array() && metrics.size() == validation.size(),
                    "La selección no conserva las métricas de cada mundo de validación");
            simulation::AccurateSum drawdown_sum;
            for (std::size_t index = 0; index < validation.size(); ++index) {
                const auto& row = metrics.at(index);
                require_fields(row, {"manifest_sha256", "net_return", "max_drawdown", "costs",
                                     "turnover", "steps", "completed"});
                const auto decline = finite_number(row.at("max_drawdown"));
                require(row.at("manifest_sha256") == validation[index].tape->source_sha256 &&
                            finite_number(row.at("net_return")) >= -1 && decline >= 0 &&
                            decline <= 1 && finite_number(row.at("costs")) >= 0 &&
                            finite_number(row.at("turnover")) >= 0 && row.at("completed") == true &&
                            count(row.at("steps")) > 0 &&
                            count(row.at("steps")) < validation[index].tape->close_times.size(),
                        "Una métrica de validación no corresponde al mundo confirmado");
                require(drawdown_sum.add(decline), "La suma de caídas máximas no es finita");
            }
            const auto mean_drawdown = drawdown_sum.value() / static_cast<double>(validation.size());
            constexpr double metric_tolerance = 1e-12;
            require(std::abs(mean_drawdown - drawdown) <= metric_tolerance,
                    "La caída máxima media no coincide con las métricas guardadas");
        } else {
            require_fields(fields, {"ruin_count", "mean_log_growth", "transitions",
                                    "optimizer_steps", "episodes"});
        }
        require(count(progress.best.at("transitions")) <= state.transitions &&
                    count(progress.best.at("optimizer_steps")) <= state.optimizer_steps &&
                    count(progress.best.at("ruin_count")) <= count(progress.best.at("episodes")) &&
                    count(progress.best.at("episodes")) == validation.size() &&
                    progress.stale_evaluations < progress.evaluations && progress.evaluations > 0 &&
                    progress.evaluated_optimizer_steps.has_value(),
                "La selección PPO no corresponde al cursor confirmado");
        static_cast<void>(finite_number(progress.best.at("mean_log_growth")));
    } else {
        require(progress.evaluations == 0, "Falta el candidato de una evaluación PPO completa");
    }
    require(progress.status != "completed" || state.transitions == state.config.total_transitions,
            "El checkpoint PPO declara completado un presupuesto pendiente");
    if (progress.status == "completed" || progress.status == "early_stopped") {
        require(progress.evaluated_optimizer_steps &&
                    *progress.evaluated_optimizer_steps == state.optimizer_steps,
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

double elapsed_seconds(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

class InvocationTimer {
  public:
    explicit InvocationTimer(double& seconds) : seconds_(&seconds) {}
    ~InvocationTimer() { *seconds_ += elapsed_seconds(started_); }
    InvocationTimer(const InvocationTimer&) = delete;
    InvocationTimer& operator=(const InvocationTimer&) = delete;
    InvocationTimer(InvocationTimer&&) = delete;
    InvocationTimer& operator=(InvocationTimer&&) = delete;

  private:
    double* seconds_;
    std::chrono::steady_clock::time_point started_ = std::chrono::steady_clock::now();
};

class ExperimentRun {
  public:
    ExperimentRun(const PpoExperimentOptions& options, ExperimentConfig config,
                  ExperimentInputs inputs, const std::function<bool()>& stop,
                  std::chrono::steady_clock::time_point started)
        : options_(options), config_(std::move(config)), inputs_(std::move(inputs)),
          identity_(experiment_identity(options_, config_, inputs_)),
          trainer_(inputs_.training, config_.training, config_.hyperparameters, options_.device,
                   options_.diagnostic, config_.learning, config_.objective),
          store_(options_.output, identity_, options_.resume), stop_(stop), started_(started) {}

    Json run() {
        if (options_.resume && has_recent_checkpoint()) {
            const auto bundle = store_.load_latest();
            auto state = restore_state(bundle, config_, options_);
            progress_ = restore_progress(bundle.metadata.at("progress"), state, config_,
                                         inputs_.validation);
            if (!progress_.best.is_null()) {
                const auto selected = store_.load_best();
                require(selected.metadata.at("progress").at("best") == progress_.best &&
                            count(selected.metadata.at("transitions")) ==
                                count(progress_.best.at("transitions")) &&
                            count(selected.metadata.at("optimizer_steps")) ==
                                count(progress_.best.at("optimizer_steps")),
                        "El mejor checkpoint PPO no corresponde a la selección confirmada");
            }
            trainer_.restore(state);
            initialize_trace(state.observed_transitions);
            setup_seconds_ = elapsed_seconds(started_);
            last_checkpoint_ = trainer_.transitions();
            receipt_ = bundle.receipt;
            receipt_.erase("discarded");
            if (progress_.status == "completed" || progress_.status == "early_stopped") {
                record_selected_policy();
                return publish_report();
            }
            progress_.status = "running";
            progress_.pause_reason = "requested_pause";
        } else {
            initialize_trace(0);
            setup_seconds_ = elapsed_seconds(started_);
            save(false);
        }
        try {
            if (!evaluate_pending()) {
                return pause();
            }
            if (config_.early_stopping && progress_.stale_evaluations >= config_.patience &&
                (!cursor_selection(config_) ||
                 (trainer_.optimizer_steps() > 0 &&
                  trainer_.transitions() < config_.training.total_transitions))) {
                progress_.status = "early_stopped";
                record_selected_policy();
                return save(false);
            }
            while (trainer_.transitions() < config_.training.total_transitions) {
                if (stop_requested()) {
                    return pause();
                }
                {
                    const InvocationTimer measured(training_seconds_);
                    static_cast<void>(
                        trainer_.advance([&](std::span<const DecisionRecord> records) {
                            if (trace_) {
                                trace_->append(records);
                            }
                        }));
                }
                if (!evaluate_pending()) {
                    return pause();
                }
                if (config_.early_stopping && progress_.stale_evaluations >= config_.patience &&
                    (!cursor_selection(config_) ||
                     (trainer_.optimizer_steps() > 0 &&
                      trainer_.transitions() < config_.training.total_transitions))) {
                    progress_.status = "early_stopped";
                    record_selected_policy();
                    return save(false);
                }
                if (trainer_.transitions() - last_checkpoint_ >= config_.checkpoint_transitions) {
                    save(false);
                }
            }
            progress_.status = "completed";
            record_selected_policy();
            return save(false);
        } catch (const TraceCapacityError&) {
            progress_.pause_reason = "trace_capacity";
            return pause();
        } catch (const std::exception& error) {
            progress_.status = "failed";
            try {
                static_cast<void>(publish_report());
            } catch (const std::exception& report_error) {
                throw std::runtime_error(
                    std::string(error.what()) +
                    ". Tampoco se pudo registrar el fallo: " + report_error.what());
            }
            throw;
        }
    }

  private:
    // Huella de la política elegida, la misma que identifica su evaluación posterior.
    void record_selected_policy() {
        if (reconstructed(config_) && !progress_.best.is_null()) {
            selected_policy_sha256_ = content_sha256(store_.load_best().policy_archive);
        }
    }

    void initialize_trace(std::size_t confirmed) {
        if (!config_.learning.enabled) {
            return;
        }
        const auto directory = options_.output / "trace";
        const bool resume = std::filesystem::exists(directory);
        require(resume || confirmed == 0, "Falta la traza de decisiones del checkpoint confirmado");
        trace_ = std::make_unique<TraceWriter>(directory, content_sha256(identity_.dump()), resume,
                                               static_cast<uint64_t>(confirmed));
        confirmed_trace_cursor_ = static_cast<uint64_t>(confirmed);
    }

    bool has_recent_checkpoint() const {
        const auto envelope = parse_bounded_json(read_bounded_file(
            options_.output / "ppo-index.json", simulation::maximum_manifest_bytes));
        const auto& index = envelope.at("payload");
        if (!index.at("recent").empty()) {
            return true;
        }
        require(index.at("best").is_null() && index.at("retired").empty(),
                "Un índice PPO sin estados recientes no puede conservar una selección anterior");
        return false;
    }

    bool stop_requested() const {
        return (stop_ && stop_()) ||
               (options_.stop_after && trainer_.transitions() >= *options_.stop_after);
    }

    Json pause() {
        progress_.status = "paused";
        return save(false);
    }

    bool evaluate_pending() {
        if (cursor_selection(config_)
                ? progress_.evaluated_transitions &&
                      *progress_.evaluated_transitions == trainer_.transitions()
                : progress_.evaluated_optimizer_steps &&
                      *progress_.evaluated_optimizer_steps == trainer_.optimizer_steps()) {
            return true;
        }
        if (config_.learning.enabled && progress_.evaluations != 0) {
            if (!progress_.evaluated_transitions) {
                throw std::invalid_argument("Falta el cursor de la última evaluación completa");
            }
            if (trainer_.transitions() != config_.training.total_transitions &&
                trainer_.transitions() - *progress_.evaluated_transitions <
                    config_.evaluation_transitions) {
                return true;
            }
        }
        PpoEvaluation evaluation;
        {
            const InvocationTimer measured(evaluation_seconds_);
            evaluation = evaluate_policy(trainer_.policy(), inputs_.validation,
                                         config_.training.workers, stop_, config_.learning);
        }
        if (evaluation.paused) {
            return false;
        }
        const auto score = config_.liquidated_selection ? evaluation.mean_liquidated_log_growth
                                                        : evaluation.mean_log_growth;
        const auto score_field =
            config_.liquidated_selection ? "mean_liquidated_log_growth" : "mean_log_growth";
        require(evaluation.incomplete == 0 && evaluation.episodes == inputs_.validation.size() &&
                    std::isfinite(evaluation.mean_log_growth) && std::isfinite(score),
                "La validación PPO está incompleta y no permite seleccionar un checkpoint");
        double mean_drawdown = 0;
        Json validation_metrics = Json::array();
        if (config_.learning.enabled) {
            require(evaluation.metrics.size() == evaluation.episodes,
                    "La evaluación adaptativa no conserva una métrica por mundo");
            simulation::AccurateSum drawdown_sum;
            for (std::size_t index = 0; index < evaluation.metrics.size(); ++index) {
                const auto& metrics = evaluation.metrics[index];
                if (!metrics.max_drawdown || !metrics.net_return) {
                    throw std::invalid_argument(
                        "La evaluación no tiene retorno o caída máxima valorables");
                }
                const auto drawdown = *metrics.max_drawdown;
                require(metrics.completed && std::isfinite(drawdown) && drawdown >= 0 &&
                            drawdown <= 1 && std::isfinite(*metrics.net_return) &&
                            std::isfinite(metrics.costs) && std::isfinite(metrics.turnover),
                        "La evaluación adaptativa no permite calcular la caída máxima media");
                require(drawdown_sum.add(drawdown), "La suma de caídas máximas no es finita");
                validation_metrics.push_back(
                    Json{{"manifest_sha256", inputs_.validation[index].tape->source_sha256},
                         {"net_return", *metrics.net_return},
                         {"max_drawdown", drawdown},
                         {"costs", metrics.costs},
                         {"turnover", metrics.turnover},
                         {"steps", metrics.steps},
                         {"completed", metrics.completed}});
            }
            mean_drawdown = drawdown_sum.value() / static_cast<double>(evaluation.episodes);
        }
        const bool improved =
            progress_.best.is_null() ||
            evaluation.ruined < count(progress_.best.at("ruin_count")) ||
            (evaluation.ruined == count(progress_.best.at("ruin_count")) &&
             score > finite_number(progress_.best.at(score_field)) + config_.min_delta);
        if (cursor_selection(config_)) {
            require(progress_.evaluation_cursors.size() < maximum_selection_evaluations,
                    "El selector supera el límite de cursores de validación");
            progress_.evaluation_cursors.push_back(trainer_.transitions());
        }
        ++progress_.evaluations;
        progress_.evaluated_optimizer_steps = trainer_.optimizer_steps();
        progress_.evaluated_transitions = trainer_.transitions();
        if (improved) {
            progress_.best = Json{{"ruin_count", evaluation.ruined},
                                  {"mean_log_growth", evaluation.mean_log_growth},
                                  {"transitions", trainer_.transitions()},
                                  {"optimizer_steps", trainer_.optimizer_steps()},
                                  {"episodes", evaluation.episodes}};
            if (config_.liquidated_selection) {
                progress_.best["mean_liquidated_log_growth"] = evaluation.mean_liquidated_log_growth;
            }
            if (config_.learning.enabled) {
                progress_.best["mean_max_drawdown"] = mean_drawdown;
                progress_.best["validation_metrics"] = std::move(validation_metrics);
            }
            progress_.stale_evaluations = 0;
        } else if (!cursor_selection(config_) || trainer_.transitions() > config_.min_transitions) {
            ++progress_.stale_evaluations;
        } else {
            progress_.stale_evaluations = 0;
        }
        save(improved);
        return true;
    }

    Json save(bool select_best) {
        {
            const InvocationTimer measured(checkpoint_seconds_);
            require_ram_budget();
            const auto state = trainer_.snapshot();
            require_ram_budget();
            if (trace_) {
                require(trace_->cursor() == state.observed_transitions,
                        "La traza no coincide con las decisiones observadas del checkpoint");
                trace_->flush();
            }
            receipt_ = store_.save(state_json(state, config_, progress_), state.policy_archive,
                                   config_.learning.enabled ? serialize_training_buffer(state)
                                                            : serialize_rollout(state.rollout),
                                   select_best);
            if (trace_) {
                confirmed_trace_cursor_ = static_cast<uint64_t>(state.observed_transitions);
            }
            last_checkpoint_ = trainer_.transitions();
        }
        return publish_report();
    }

    Json publish_report() const {
        const bool stopped_early = progress_.status == "early_stopped";
        auto selection = config_.document.at("selection");
        selection["policy"] = "greedy_argmax";
        Json report{
            {"schema_version", config_.schema_version},
            {"kind", "native_ppo"},
            {"activity", "rl"},
            {"model", "ppo"},
            {"backend", "native_libtorch"},
            {"seed", config_.training.seed},
            {"partition", "train"},
            {"updated_at", current_utc()},
            {"global_step", trainer_.transitions()},
            {"total_steps", config_.training.total_transitions},
            {"status", stopped_early ? "completed" : progress_.status},
            {"stopping_reason", stopped_early                     ? Json("early_stop")
                                : progress_.status == "completed" ? Json("budget_exhausted")
                                : progress_.status == "paused"    ? Json("requested_pause")
                                                                  : Json(nullptr)},
            {"identity_sha256", content_sha256(identity_.dump())},
            {"domain", options_.diagnostic ? "technical"
                       : reconstructed(config_) ? "real"
                                                : "synthetic"},
            {"device", options_.device},
            {"diagnostic", options_.diagnostic},
            {"transitions", trainer_.transitions()},
            {"optimizer_steps", trainer_.optimizer_steps()},
            {"partial_ticks", trainer_.partial_ticks()},
            {"invalid_transitions", trainer_.invalid_transitions()},
            {"evaluations", progress_.evaluations},
            {"stale_evaluations", progress_.stale_evaluations},
            {"best", progress_.best},
            {"selection", selection},
            {"checkpoint", receipt_},
            {"parent_frozen", true},
            {"final_test_opened", false},
            {"resources",
             Json{{"ram_peak_bytes", simulation::process_memory_high_water()},
#if defined(__linux__)
                  {"ram_peak_method", "procfs_VmHWM"},
#else
                  {"ram_peak_method", "getrusage_RUSAGE_SELF"},
#endif
                  {"ram_guard_bytes", maximum_process_memory_bytes},
                  {"ram_guard_scope", "before_runtime_and_checkpoints"},
                  {"rollout_budget_bytes", config_.training.rollout_bytes},
                  {"vram_budget_bytes",
                   options_.vram_budget_bytes ? Json(*options_.vram_budget_bytes) : Json(nullptr)},
                  {"vram_total_bytes",
                   options_.vram_total_bytes ? Json(*options_.vram_total_bytes) : Json(nullptr)},
                  {"vram_peak_bytes", nullptr}}},
            {"invocation_seconds",
             std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count()}};
        if (config_.learning.enabled) {
            report["model"] = config_.learning.variant;
            report["agent_variant"] = config_.learning.variant;
            if (!reconstructed(config_)) {
                report["analysis_domain"] = "technical";
                report["macro_coverage"] = Json{{"simulated_concepts", adaptive_macro_concepts},
                                                {"catalog_concepts", macro_catalog_concepts}};
            }
            report["observed_transitions"] = trainer_.observed_transitions();
            report["evaluation_transitions"] = config_.evaluation_transitions;
            report["timings"] = Json{{"training_seconds", training_seconds_},
                                     {"evaluation_seconds", evaluation_seconds_},
                                     {"checkpoint_seconds", checkpoint_seconds_},
                                     {"setup_seconds", setup_seconds_}};
            report["parameters"] = trainer_.policy().parameter_count();
            report["auxiliary_steps"] = trainer_.policy().auxiliary_steps();
            report["auxiliary_samples"] = trainer_.auxiliary_samples();
            report["catalog_train_sources"] = inputs_.training.size();
            report["catalog_validation_sources"] = inputs_.validation.size();
            if (trace_) {
                report["trace"] = Json{{"path", "trace"},
                                       {"identity_sha256", content_sha256(identity_.dump())},
                                       {"confirmed_cursor", confirmed_trace_cursor_},
                                       {"reserved_bytes", trace_->bytes()},
                                       {"budget_bytes", default_trace_bytes}};
            }
            if (progress_.status == "paused") {
                report["stopping_reason"] = progress_.pause_reason;
            }
        }
        if (cursor_selection(config_)) {
            report["evaluation_cursors"] = progress_.evaluation_cursors;
        }
        if (config_.objective.enabled()) {
            report["policy_objective"] = config_.document.at("policy_objective");
            report["policy_controller"] = controller_json(trainer_.policy().controller_state());
        }
        if (reconstructed(config_)) {
            report["selected_policy_sha256"] = selected_policy_sha256_;
            report["sources"] = inputs_.identity;
        }
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
    std::unique_ptr<TraceWriter> trace_;
    uint64_t confirmed_trace_cursor_ = 0;
    double setup_seconds_ = 0;
    double training_seconds_ = 0;
    double evaluation_seconds_ = 0;
    double checkpoint_seconds_ = 0;
    Json selected_policy_sha256_ = nullptr;
};

Json audit_seal(const Json& value) {
    return Json{{"payload", value}, {"sha256", content_sha256(value.dump())}};
}

Json read_audit_state(const std::filesystem::path& path) {
    const auto envelope =
        parse_bounded_json(read_bounded_file(path, simulation::maximum_manifest_bytes));
    require_fields(envelope, {"payload", "sha256"});
    require(content_sha256(envelope.at("payload").dump()) ==
                envelope.at("sha256").get<std::string>(),
            "El estado confirmado de auditoría perdió su integridad");
    return envelope.at("payload");
}

std::vector<simulation::BatchInput> audit_sources(const PpoExperimentOptions& options,
                                                  const ExperimentConfig& config,
                                                  const Json& selected_identity, Json& receipts) {
    std::set<std::string> used_hashes;
    std::set<int64_t> used_seeds;
    for (const auto* partition : {"train", "validation"}) {
        for (const auto& source : selected_identity.at("sources").at(partition)) {
            used_hashes.insert(source.at("manifest_sha256").get<std::string>());
            used_seeds.insert(read_json_int64(source.at("generator_seed")));
        }
    }
    const auto& schema = selected_identity.at("observation_schema");
    std::vector<simulation::BatchInput> inputs;
    std::size_t diagnostic_steps = 0;
    constexpr std::size_t cost_count = 3;
    for (const auto& path : options.audit_tapes) {
        auto loaded = load_source(path, "validation", config.environment, true, true);
        require(loaded.generator_seed && used_seeds.insert(*loaded.generator_seed).second &&
                    used_hashes.insert(loaded.input.tape->source_sha256).second,
                "La auditoría reutiliza una fuente o semilla de selección");
        require_learning_fields(loaded.input, config.learning);
        const auto& tape = *loaded.input.tape;
        if (!loaded.input.context) {
            throw std::invalid_argument("La auditoría requiere contexto observable");
        }
        const auto& context = *loaded.input.context;
        require(schema.at("assets") == tape.assets && schema.at("currency") == tape.currency &&
                    schema.at("domain") == tape.domain &&
                    schema.at("parent_id") == tape.parent_id &&
                    schema.at("context_fields").size() == context.fields.size(),
                "La auditoría cambió los activos, el padre o el esquema de observaciones");
        for (std::size_t index = 0; index < context.fields.size(); ++index) {
            require(schema.at("context_fields").at(index) ==
                        Json{{"name", context.fields[index].name},
                             {"unit", context.fields[index].unit}},
                    "La auditoría cambió un campo de contexto o su unidad");
        }
        if (!inputs.empty()) {
            require_compatible(inputs.front(), loaded.input);
        }
        for (std::size_t at = 0; at + 1 < tape.close_times.size(); ++at) {
            if (context.values[at * context.fields.size() + context.fields.size() - 1].value == 1) {
                diagnostic_steps += cost_count;
            }
        }
        receipts.push_back(std::move(loaded.identity));
        inputs.push_back(std::move(loaded.input));
    }
    require(!options.diagnostic || diagnostic_steps <= diagnostic_transitions,
            "La auditoría diagnóstica CPU admite hasta 32 decisiones posteriores al calentamiento");
    return inputs;
}

// Selección cerrada de una ejecución: identidad, almacén bloqueado y checkpoint elegido.
struct ClosedSelection {
    Json identity;
    std::unique_ptr<PpoCheckpointStore> store;
    PpoCheckpointBundle selected;
};

ClosedSelection closed_selection(const PpoExperimentOptions& options,
                                 const ExperimentConfig& config) {
    if (!options.audit_run) {
        throw std::invalid_argument("Falta la ejecución cuya selección se va a auditar");
    }
    const auto record = parse_bounded_json(read_bounded_file(*options.audit_run / "identity.json",
                                                             simulation::maximum_manifest_bytes));
    ClosedSelection result{record.at("identity"), nullptr, {}};
    const auto& selected_identity = result.identity;
    require(selected_identity.at("schema_version") == config.schema_version &&
                selected_identity.at("configuration") == config.document &&
                selected_identity.at("markov") == config.markov_identity &&
                selected_identity.at("device") == options.device &&
                selected_identity.at("diagnostic") == options.diagnostic &&
                selected_identity.at("native_source_sha256") == MARS_TITAN_NATIVE_SOURCE_SHA256 &&
                selected_identity.at("native_build_sha256") == MARS_TITAN_NATIVE_BUILD_SHA256 &&
                selected_identity.at("parent_frozen") == true &&
                selected_identity.at("final_test_opened") == false,
            "La auditoría no conserva la identidad y configuración de la selección");
    result.store = std::make_unique<PpoCheckpointStore>(*options.audit_run, selected_identity, true);
    auto& frozen = *result.store;
    const auto latest = frozen.load_latest();
    const auto& progress = latest.metadata.at("progress");
    validate_convergence_progress(progress, config, count(latest.metadata.at("transitions")),
                                  count(latest.metadata.at("optimizer_steps")));
    const auto status = progress.at("status").get<std::string>();
    require((status == "completed" &&
             count(latest.metadata.at("transitions")) == config.training.total_transitions) ||
                (status == "early_stopped" && config.early_stopping &&
                 count(progress.at("stale_evaluations")) >= config.patience),
            "La auditoría necesita una selección finalizada y cerrada");
    require(progress.at("evaluated_optimizer_steps") == latest.metadata.at("optimizer_steps") &&
                !progress.at("best").is_null(),
            "La selección cerrada conserva una evaluación pendiente");
    result.selected = frozen.load_best();
    const auto& selected = result.selected;
    require(selected.metadata.at("configuration") == config.document &&
                selected.metadata.at("schema_version") == config.schema_version &&
                selected.metadata.at("device") == options.device &&
                selected.metadata.at("diagnostic") == options.diagnostic &&
                selected.metadata.at("progress").at("best") == progress.at("best") &&
                selected.metadata.at("transitions") == progress.at("best").at("transitions") &&
                selected.metadata.at("optimizer_steps") ==
                    progress.at("best").at("optimizer_steps"),
            "El checkpoint elegido no corresponde a la selección cerrada");
    return result;
}

PpoPolicy selected_policy(const ClosedSelection& selection, const ExperimentConfig& config,
                          const PpoExperimentOptions& options) {
    std::istringstream archive(selection.selected.policy_archive);
    auto policy = PpoPolicy::load(archive, options.device);
    require(policy.seed() == config.training.seed &&
                policy.hyperparameters() == config.hyperparameters &&
                policy.objective() == config.objective &&
                policy.optimizer_steps() == count(selection.selected.metadata.at("optimizer_steps")),
            "La política congelada no corresponde a los parámetros de la selección");
    return policy;
}

// Evaluación de la política elegida en cintas reconstruidas posteriores a su selección.
Json run_reconstructed_evaluation(const PpoExperimentOptions& options,
                                  const ExperimentConfig& config,
                                  const std::function<bool()>& stop) {
    const auto selection = closed_selection(options, config);
    preflight_memory(options, config);
    FrozenEvaluationRequest request{options.output, options.resume, nullptr, {},
                                    config.learning, config.training.workers};
    Json tapes = Json::array();
    for (const auto& path : options.audit_tapes) {
        auto tape = load_policy_tape(path, PolicyTapeRole::evaluation, config.environment);
        require_after_selection(selection.identity.at("sources"), tape);
        require(selection.identity.at("observation_schema").at("assets") ==
                    tape.input.tape->assets,
                "La evaluación cambia los activos que observa la política");
        tapes.push_back(tape.identity);
        request.tapes.push_back(std::move(tape));
    }
    require_policy_sequence(request.tapes);
    require_ram_budget();
    configure_runtime(options);
    const auto policy = selected_policy(selection, config, options);
    request.identity = Json{{"schema_version", 1},
                            {"kind", "native_ppo_reconstructed_evaluation"},
                            {"selected_identity_sha256", content_sha256(selection.identity.dump())},
                            {"selected_checkpoint", selection.selected.receipt},
                            {"policy_sha256", content_sha256(selection.selected.policy_archive)},
                            {"optimizer_steps", policy.optimizer_steps()},
                            {"transitions", selection.selected.metadata.at("transitions")},
                            {"tapes", tapes},
                            {"cost_bps", frozen_evaluation_costs},
                            {"seed", config.training.seed},
                            {"device", options.device},
                            {"diagnostic", options.diagnostic},
                            {"native_source_sha256", MARS_TITAN_NATIVE_SOURCE_SHA256},
                            {"native_build_sha256", MARS_TITAN_NATIVE_BUILD_SHA256},
                            {"final_test_opened", false}};
    return run_frozen_evaluation(policy, request, stop);
}

Json run_audit(const PpoExperimentOptions& options, const ExperimentConfig& config,
               const std::function<bool()>& stop, std::chrono::steady_clock::time_point started) {
    const auto selection = closed_selection(options, config);
    const auto& selected_identity = selection.identity;
    const auto& selected = selection.selected;
    // Hasta este punto no se abre ningún archivo de los mundos reservados.
    preflight_memory(options, config);
    Json receipts = Json::array();
    const auto inputs = audit_sources(options, config, selected_identity, receipts);
    require_ram_budget();
    configure_runtime(options);
    const auto policy = selected_policy(selection, config, options);
    constexpr std::array costs{0., 10., 25.};
    const Json identity{{"schema_version", 1},
                        {"kind", "native_ppo_audit"},
                        {"selected_identity_sha256", content_sha256(selected_identity.dump())},
                        {"selected_checkpoint", selected.receipt},
                        {"policy_sha256", content_sha256(selected.policy_archive)},
                        {"configuration", config.document},
                        {"sources", receipts},
                        {"cost_bps", costs},
                        {"seed", config.training.seed},
                        {"device", options.device},
                        {"diagnostic", options.diagnostic},
                        {"native_source_sha256", MARS_TITAN_NATIVE_SOURCE_SHA256},
                        {"native_build_sha256", MARS_TITAN_NATIVE_BUILD_SHA256},
                        {"memory_sensitivity", "frozen_observation_and_hidden"},
                        {"final_test_opened", false}};
    const auto identity_hash = content_sha256(identity.dump());
    simulation::OutputLock lock(options.output, options.resume);
    Json state{{"schema_version", 1},
               {"identity_sha256", identity_hash},
               {"status", "paused"},
               {"confirmed_cursor", 0},
               {"metrics", Json::array()}};
    if (options.resume) {
        require(read_audit_state(options.output / "identity.json") == identity,
                "La auditoría pertenece a otra selección, costes o mundos reservados");
        const auto state_path = options.output / "audit-state.json";
        if (std::filesystem::exists(state_path)) {
            state = read_audit_state(state_path);
        }
    } else {
        atomic_json_file(options.output / "identity.json", audit_seal(identity));
    }
    require_fields(state,
                   {"schema_version", "identity_sha256", "status", "confirmed_cursor", "metrics"});
    const auto total = inputs.size() * costs.size();
    require(state.at("schema_version") == 1 && state.at("identity_sha256") == identity_hash &&
                (state.at("status") == "paused" || state.at("status") == "completed") &&
                state.at("metrics").is_array() && state.at("metrics").size() <= total &&
                (state.at("status") != "completed" || state.at("metrics").size() == total),
            "El estado de auditoría no conserva su identidad o presupuesto");
    std::size_t completed = state.at("metrics").size();
    auto confirmed = static_cast<uint64_t>(count(state.at("confirmed_cursor")));
    require((completed % inputs.size()) % config.environments == 0,
            "La auditoría conserva un grupo de mundos sin confirmar entero");
    std::size_t counted_decisions = 0;
    for (std::size_t at = 0; at < completed; ++at) {
        const auto& row = state.at("metrics").at(at);
        require_fields(row, {"manifest_sha256", "cost_bps", "net_return", "max_drawdown", "costs",
                             "turnover", "steps", "completed", "invalid_reason", "trace_first",
                             "trace_last"});
        require(row.at("manifest_sha256") == inputs[at % inputs.size()].tape->source_sha256 &&
                    finite_number(row.at("cost_bps")) == costs.at(at / inputs.size()) &&
                    row.at("completed").is_boolean() && row.at("invalid_reason").is_string() &&
                    finite_number(row.at("costs")) >= 0 && finite_number(row.at("turnover")) >= 0 &&
                    count(row.at("steps")) > 0 &&
                    count(row.at("steps")) < inputs[at % inputs.size()].tape->close_times.size() &&
                    count(row.at("trace_first")) > 0 &&
                    count(row.at("trace_first")) <= count(row.at("trace_last")) &&
                    count(row.at("trace_last")) <= confirmed,
                "Las métricas confirmadas de auditoría corresponden a otro mundo o coste");
        if (boolean(row.at("completed"))) {
            const auto drawdown = finite_number(row.at("max_drawdown"));
            require(finite_number(row.at("net_return")) >= -1 && drawdown >= 0 && drawdown <= 1,
                    "Las métricas confirmadas de auditoría contienen magnitudes inválidas");
        } else {
            require(row.at("net_return").is_null() && row.at("max_drawdown").is_null(),
                    "Una valoración incompleta no puede conservar retorno o caída máxima");
        }
        counted_decisions += count(row.at("steps"));
    }
    require(counted_decisions == confirmed,
            "La traza de auditoría no concuerda con los pasos de los mundos confirmados");
    const auto trace_path = options.output / "trace";
    const bool trace_resume = std::filesystem::exists(trace_path);
    require(trace_resume || confirmed == 0, "Falta la traza confirmada de la auditoría");
    TraceWriter trace(trace_path, identity_hash, trace_resume, confirmed);
    const auto setup_seconds = elapsed_seconds(started);
    double evaluation_seconds = 0;
    const auto publish = [&](std::string_view reason, std::string_view status_override = "") {
        Json report{
            {"schema_version", 2},
            {"kind", "native_ppo_audit"},
            {"activity", "evaluation"},
            {"phase", "evaluation"},
            {"partition", "audit"},
            {"domain", "synthetic"},
            {"analysis_domain", "technical"},
            {"status", status_override.empty() ? state.at("status") : Json(status_override)},
            {"model", config.learning.variant},
            {"backend", "native_libtorch"},
            {"identity_sha256", identity_hash},
            {"seed", config.training.seed},
            {"device", options.device},
            {"diagnostic", options.diagnostic},
            {"confirmed_episodes", completed},
            {"total_episodes", total},
            {"confirmed_decisions", confirmed},
            {"stopping_reason", reason},
            {"updated_at", current_utc()},
            {"final_test_opened", false},
            {"macro_coverage", Json{{"simulated_concepts", adaptive_macro_concepts},
                                    {"catalog_concepts", macro_catalog_concepts}}},
            {"invocation_seconds", elapsed_seconds(started)},
            {"timings",
             Json{{"setup_seconds", setup_seconds}, {"evaluation_seconds", evaluation_seconds}}}};
        atomic_json_file(options.output / "run.json", report);
        report["identity"] = identity;
        report["metrics"] = state.at("metrics");
        report["trace"] = Json{{"path", "trace"},
                               {"identity_sha256", identity_hash},
                               {"confirmed_cursor", confirmed},
                               {"reserved_bytes", trace.bytes()}};
        atomic_json_file(options.output / "audit.json", report);
        return report;
    };
    const auto confirm = [&](Json next) {
        const auto next_completed = next.at("metrics").size();
        const auto next_confirmed = static_cast<uint64_t>(count(next.at("confirmed_cursor")));
        trace.flush();
        atomic_json_file(options.output / "audit-state.json", audit_seal(next));
        state.swap(next);
        completed = next_completed;
        confirmed = next_confirmed;
    };
    if (!std::filesystem::exists(options.output / "audit-state.json")) {
        confirm(state);
    }
    auto learning = config.learning;
    learning.fold = "audit";
    try {
        while (completed < total) {
            if (stop && stop()) {
                return publish("requested_pause");
            }
            const auto cost = costs.at(completed / inputs.size());
            const auto begin = completed % inputs.size();
            const auto end = std::min(begin + config.environments, inputs.size());
            std::vector<simulation::BatchInput> chunk(
                inputs.begin() + static_cast<std::ptrdiff_t>(begin),
                inputs.begin() + static_cast<std::ptrdiff_t>(end));
            for (auto& input : chunk) {
                input.parameters.cost_bps = cost;
            }
            PpoEvaluation evaluation;
            {
                const InvocationTimer measured(evaluation_seconds);
                evaluation = evaluate_policy(
                    policy, chunk, config.training.workers, stop, learning,
                    [&](std::span<const DecisionRecord> records) {
                        std::vector<DecisionRecord> numbered(records.begin(), records.end());
                        for (std::size_t offset = 0; offset < numbered.size(); ++offset) {
                            numbered[offset].decision_id = trace.cursor() + offset + 1;
                        }
                        trace.append(numbered);
                    },
                    true);
            }
            if (evaluation.paused) {
                trace.rewind_to(confirmed);
                return publish("requested_pause");
            }
            require(evaluation.metrics.size() == chunk.size(),
                    "La auditoría no conserva una métrica por mundo completado");
            auto next = state;
            for (std::size_t index = 0; index < chunk.size(); ++index) {
                const auto& metrics = evaluation.metrics[index];
                next["metrics"].push_back(Json{
                    {"manifest_sha256", chunk[index].tape->source_sha256},
                    {"cost_bps", cost},
                    {"net_return", metrics.net_return ? Json(*metrics.net_return) : Json(nullptr)},
                    {"max_drawdown",
                     metrics.max_drawdown ? Json(*metrics.max_drawdown) : Json(nullptr)},
                    {"costs", metrics.costs},
                    {"turnover", metrics.turnover},
                    {"steps", metrics.steps},
                    {"completed", metrics.completed},
                    {"invalid_reason", metrics.invalid_reason},
                    {"trace_first", confirmed + 1},
                    {"trace_last", trace.cursor()}});
            }
            next["confirmed_cursor"] = trace.cursor();
            next["status"] = next.at("metrics").size() == total ? "completed" : "paused";
            confirm(next);
            static_cast<void>(publish("group_confirmed"));
        }
        return publish("audit_completed");
    } catch (const TraceCapacityError&) {
        trace.rewind_to(confirmed);
        return publish("trace_capacity");
    } catch (const std::exception&) {
        static_cast<void>(publish("evaluation_failed", "failed"));
        throw;
    }
}
} // namespace

void require_policy_device(const PpoExperimentOptions& options, std::size_t total_transitions) {
    require_device(options, total_transitions);
}

void admit_policy_gpu(const PpoExperimentOptions& options) {
    if (options.device == "cuda:0") {
        static_cast<void>(validate_gpu_lease(options));
    }
}

void configure_policy_runtime(const PpoExperimentOptions& options) { configure_runtime(options); }

Json run_ppo_experiment(const PpoExperimentOptions& options,
                        const std::function<bool()>& stop_requested) {
    const auto started = std::chrono::steady_clock::now();
    const auto config = configuration(options.config);
    validate_options(options, config);
    // La admisión se comprueba antes de cualquier tensor CUDA o creación de la salida.
    if (options.device == "cuda:0") {
        static_cast<void>(validate_gpu_lease(options));
    }
    if (reconstructed(config)) {
        // Ajustar o evaluar sobre el histórico reconstruido son usos científicos de datos
        // reales. Ambos se detienen antes de leer cintas o crear salidas si la protección lo pide.
        require_learning_allowed(options.audit_run
                                     ? "la evaluación nativa sobre cintas reconstruidas"
                                     : "el entrenamiento nativo sobre cintas reconstruidas");
        if (options.audit_run) {
            return run_reconstructed_evaluation(options, config, stop_requested);
        }
    } else if (options.audit_run) {
        return run_audit(options, config, stop_requested, started);
    }
    // La auditoría congelada no ajusta parámetros. El entrenamiento se detiene antes de leer
    // fuentes o crear la salida si la protección local no lo permite.
    require_learning_allowed("el entrenamiento PPO nativo");
    preflight_memory(options, config);
    auto inputs = load_inputs(options, config);
    require_ram_budget();
    configure_runtime(options);
    ExperimentRun run(options, config, std::move(inputs), stop_requested, started);
    return run.run();
}
} // namespace mars_titan::learning
