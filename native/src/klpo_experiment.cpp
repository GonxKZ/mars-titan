#include "mars_titan/klpo_experiment.hpp"
#include "mars_titan/build_identity.hpp"
#include "mars_titan/learning_hold.hpp"
#include "mars_titan/learning_traces.hpp"
#include "mars_titan/policy_evaluation.hpp"
#include "mars_titan/policy_tapes.hpp"
#include "mars_titan/ppo_training.hpp"
#include "mars_titan/simulation_files.hpp"

#include <torch/version.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mars_titan::learning {
namespace {
using Json = nlohmann::json;
using simulation::atomic_json_file;
using simulation::content_sha256;
using simulation::parse_bounded_json;
using simulation::read_bounded_file;
using simulation::read_json_int64;

constexpr std::string_view experiment_kind = "native_klpo_terminal";
constexpr std::string_view controller_contract = "klpo_full_fresh_waves_v1";
// Los objetivos de grupo usan el mismo esquema y las mismas oleadas. Solo cambian el tipo, el
// objetivo y el controlador, de modo que el resto de la comparación queda fijado.
constexpr std::string_view group_experiment_kind = "native_group_relative";
constexpr std::string_view selection_metric = "ruin_count_then_mean_log_growth";
constexpr std::string_view liquidated_selection_metric =
    "ruin_count_then_mean_liquidated_log_growth";
constexpr std::array allowed_seeds{uint64_t{42}, uint64_t{43}, uint64_t{44}};
constexpr std::size_t maximum_transitions = std::size_t{1} << 20;
constexpr std::size_t maximum_patience = std::size_t{1} << 20;
constexpr std::string_view best_prefix = "best-actor-";

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::invalid_argument(std::string(message));
    }
}

void require_fields(const Json& object, std::initializer_list<std::string_view> names) {
    require(object.is_object() && object.size() == names.size() &&
                std::ranges::all_of(names,
                                    [&](auto name) { return object.contains(std::string(name)); }),
            "La configuración KLPO no conserva los campos de su esquema");
}

std::size_t count(const Json& value) {
    const auto parsed = read_json_int64(value);
    require(parsed >= 0, "Un contador KLPO no puede ser negativo");
    return static_cast<std::size_t>(parsed);
}

double finite_number(const Json& value) {
    require(value.is_number() && std::isfinite(value.get<double>()),
            "El parámetro KLPO necesita un número finito");
    return value.get<double>();
}

struct KlpoExperimentConfig {
    Json document;
    std::size_t total_transitions = 0;
    std::size_t environments = 0;
    std::size_t evaluation_transitions = 0;
    std::size_t patience = 0;
    double min_delta = 0;
    bool liquidated_selection = false;
    simulation::Parameters environment;
    KlpoLearningConfig learning;
};

KlpoExperimentConfig configuration(const std::filesystem::path& path) {
    KlpoExperimentConfig result;
    result.document =
        parse_bounded_json(read_bounded_file(path, simulation::maximum_manifest_bytes));
    const auto& document = result.document;
    require_fields(document,
                   {"schema_version", "kind", "objective", "controller", "training",
                    "environments", "adam", "terminal", "confirmed_updates_per_reference",
                    "gradient_block_episodes", "environment", "evaluation_transitions",
                    "selection", "final_test_opened"});
    const bool group = document.at("kind") == group_experiment_kind;
    require(read_json_int64(document.at("schema_version")) == 1 &&
                (group ? document.at("controller") == group_relative_controller &&
                             document.at("objective").is_string()
                       : document.at("kind") == experiment_kind &&
                             document.at("objective") == klpo_terminal_contract &&
                             document.at("controller") == controller_contract) &&
                document.at("final_test_opened") == false,
            "La configuración no declara un objetivo y un controlador de oleadas admitidos");
    const auto& training = document.at("training");
    require_fields(training, {"total_transitions", "seed", "workers"});
    result.total_transitions = count(training.at("total_transitions"));
    const auto seed = static_cast<uint64_t>(count(training.at("seed")));
    const auto workers = count(training.at("workers"));
    result.environments = count(document.at("environments"));
    result.evaluation_transitions = count(document.at("evaluation_transitions"));
    const auto& adam = document.at("adam");
    require_fields(adam, {"learning_rate", "gradient_norm"});
    const auto& terminal = document.at("terminal");
    require_fields(terminal, {"beta", "gamma"});
    const auto& environment = document.at("environment");
    require_fields(environment,
                   {"capital", "cost_bps", "participation", "score_scale", "ruin_penalty"});
    result.environment = {finite_number(environment.at("capital")),
                          finite_number(environment.at("cost_bps")),
                          finite_number(environment.at("participation")),
                          finite_number(environment.at("score_scale")),
                          finite_number(environment.at("ruin_penalty"))};
    const auto& selection = document.at("selection");
    require_fields(selection, {"metric", "min_delta", "patience", "early_stopping"});
    require(selection.at("metric") == selection_metric ||
                selection.at("metric") == liquidated_selection_metric,
            "La métrica de selección KLPO no está admitida");
    // La etapa compara con presupuesto fijo. La parada temprana no está implementada.
    require(selection.at("early_stopping") == false,
            "KLPO conserva el presupuesto fijo y no admite parada temprana");
    result.liquidated_selection = selection.at("metric") == liquidated_selection_metric;
    result.min_delta = finite_number(selection.at("min_delta"));
    result.patience = count(selection.at("patience"));
    require(result.total_transitions > 0 && result.total_transitions <= maximum_transitions &&
                std::ranges::find(allowed_seeds, seed) != allowed_seeds.end() &&
                result.environments > 0 && result.environments <= maximum_klpo_episodes &&
                result.evaluation_transitions > 0 &&
                result.evaluation_transitions <= result.total_transitions &&
                result.min_delta >= 0 && result.patience > 0 &&
                result.patience <= maximum_patience,
            "El presupuesto, los entornos, la semilla o la selección KLPO no están admitidos");
    auto& learning = result.learning;
    learning.collection.seed = seed;
    learning.collection.workers = workers;
    learning.collection.beta = finite_number(terminal.at("beta"));
    learning.collection.gamma = finite_number(terminal.at("gamma"));
    // Misma representación adaptativa que PPO con la variante ppo, sin máscara ni HMM.
    learning.collection.context.enabled = true;
    learning.collection.context.variant = "ppo";
    learning.collection.context.fold = learning.collection.fold;
    learning.collection.context.environments = result.environments;
    learning.adam = {.learning_rate = finite_number(adam.at("learning_rate")),
                     .gradient_norm = finite_number(adam.at("gradient_norm"))};
    learning.confirmed_updates_per_reference = count(document.at("confirmed_updates_per_reference"));
    learning.gradient_block_episodes = count(document.at("gradient_block_episodes"));
    if (group) {
        // Las constantes salen de la identidad publicada y no de la configuración. beta solo
        // pertenece a KLPO y se exige igual para que la recogida sea idéntica.
        learning.group = published_group_objective(document.at("objective").get<std::string>());
        require(learning.collection.beta == 1 && learning.collection.gamma == 1,
                "Los objetivos de grupo recogen como KLPO, con beta y gamma iguales a uno");
    }
    learning.validate();
    return result;
}

bool contains(const std::filesystem::path& parent, const std::filesystem::path& child) {
    const auto first = std::filesystem::weakly_canonical(parent);
    const auto second = std::filesystem::weakly_canonical(child);
    const auto [left, right] = std::ranges::mismatch(first, second);
    return left == first.end();
}

void validate_options(const PpoExperimentOptions& options, const KlpoExperimentConfig& config) {
    require(!options.source_check || options.device == "cpu",
            "La comprobación de fuentes no usa la GPU y se ejecuta en CPU con --diagnostic");
    if (options.initial_evaluation) {
        require(!options.output.empty() && options.train_tapes.empty() &&
                    options.validation_tapes.empty() && !options.audit_tapes.empty() &&
                    options.audit_tapes.size() <= maximum_klpo_episodes && !options.stop_after,
                "La evaluación inicial KLPO solo admite cintas de evaluación acotadas y una salida");
    } else if (options.audit_run) {
        require(!options.output.empty() && options.train_tapes.empty() &&
                    options.validation_tapes.empty() && !options.audit_tapes.empty() &&
                    options.audit_tapes.size() <= maximum_klpo_episodes && !options.stop_after &&
                    !contains(*options.audit_run, options.output) &&
                    !contains(options.output, *options.audit_run),
                "La evaluación KLPO necesita la ejecución elegida, sus cintas y otra salida");
    } else {
        require(!options.output.empty() && options.audit_tapes.empty() &&
                    !options.train_tapes.empty() &&
                    options.train_tapes.size() <= config.environments &&
                    options.validation_tapes.size() == 1,
                "KLPO necesita entre una y tantas cintas de ajuste como entornos y una validación");
    }
    // Los modos sin aprendizaje no consumen transiciones: la CPU solo exige el diagnóstico.
    const bool learning = !options.initial_evaluation && !options.source_check;
    require_policy_device(options, learning ? config.total_transitions : 0);
    require(!contains(options.output, options.config),
            "La configuración debe quedar fuera de la salida KLPO");
    for (const auto* sources :
         {&options.train_tapes, &options.validation_tapes, &options.audit_tapes}) {
        for (const auto& source : *sources) {
            require(!contains(source, options.output) && !contains(options.output, source),
                    "La salida KLPO debe permanecer separada de sus cintas");
        }
    }
}

Json seal(const Json& value) { return Json{{"payload", value}, {"sha256", content_sha256(value.dump())}}; }

Json unseal(const std::filesystem::path& path) {
    const auto envelope =
        parse_bounded_json(read_bounded_file(path, simulation::maximum_manifest_bytes));
    require(envelope.is_object() && envelope.size() == 2 &&
                content_sha256(envelope.at("payload").dump()) ==
                    envelope.at("sha256").get<std::string>(),
            "Un registro de la ejecución KLPO perdió su integridad");
    return envelope.at("payload");
}

std::string actor_archive(const PpoCheckpointBundle& bundle) {
    const auto split = count(bundle.metadata.at("actor_archive_bytes"));
    require(split > 0 && split < bundle.policy_archive.size(),
            "El estado KLPO no separa actor y referencia");
    return bundle.policy_archive.substr(0, split);
}

PpoPolicy load_actor(const std::string& bytes, const PpoExperimentOptions& options) {
    std::istringstream stream(bytes);
    return PpoPolicy::load_terminal(stream, options.device);
}

PpoLearningOptions evaluation_context(const KlpoExperimentConfig& config) {
    return config.learning.collection.context;
}

double elapsed_seconds(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

// Cintas de ajuste y validación de un experimento KLPO, con los carriles y el presupuesto en
// oleadas completas. El ajuste y la comprobación de fuentes las leen con esta misma función.
struct KlpoFitTapes {
    std::vector<simulation::BatchInput> lanes;
    std::vector<simulation::BatchInput> validation;
    Json sources;
    std::size_t wave_transitions = 0;
    std::size_t planned_waves = 0;
    std::size_t evaluation_waves = 1;
};

KlpoFitTapes load_fit_tapes(const PpoExperimentOptions& options,
                            const KlpoExperimentConfig& config) {
    std::vector<PolicyTape> tapes;
    tapes.reserve(options.train_tapes.size() + 1);
    for (const auto& path : options.train_tapes) {
        tapes.push_back(load_policy_tape(path, PolicyTapeRole::train, config.environment));
    }
    tapes.push_back(load_policy_tape(options.validation_tapes.front(), PolicyTapeRole::validation,
                                     config.environment));
    require_policy_sequence(tapes);
    for (const auto& tape : tapes) {
        require(tape.input.tape->close_times.size() <= static_cast<std::size_t>(ppo_maximum_history),
                "Una cinta reconstruida supera la historia recurrente acotada");
    }
    KlpoFitTapes result;
    Json train = Json::array();
    for (std::size_t index = 0; index + 1 < tapes.size(); ++index) {
        train.push_back(tapes[index].identity);
    }
    result.sources = Json{{"train", train}, {"validation", Json::array({tapes.back().identity})}};
    result.validation = {tapes.back().input};
    // Cada entorno recorre una cinta de ajuste en ciclo y forma un episodio por oleada.
    for (std::size_t lane = 0; lane < config.environments; ++lane) {
        const auto& input = tapes[lane % (tapes.size() - 1)].input;
        result.wave_transitions += input.tape->close_times.size() - 1;
        result.lanes.push_back(input);
    }
    // Cada cinta de ajuste necesita al menos dos carriles para formar un grupo con línea base.
    // Con 16 entornos esto admite hasta ocho cintas.
    require(!config.learning.group || config.environments >= 2 * (tapes.size() - 1),
            "Los objetivos de grupo necesitan al menos dos carriles por cinta de ajuste");
    result.planned_waves = config.total_transitions / result.wave_transitions;
    require(result.planned_waves > 0, "El presupuesto KLPO no admite ni una oleada completa");
    result.evaluation_waves =
        std::max<std::size_t>(1, config.evaluation_transitions / result.wave_transitions);
    return result;
}

// Identidad de un experimento KLPO sin la del controlador, que solo existe al ajustar. El
// ajuste la completa con `controller_identity_sha256` y la comprobación de fuentes la publica
// tal cual. Las claves de nlohmann se ordenan, así que añadir una no cambia el resto del texto.
Json klpo_sources_identity(const PpoExperimentOptions& options, const KlpoExperimentConfig& config,
                           const Json& sources, std::size_t wave_transitions,
                           std::size_t planned_waves, std::size_t evaluation_waves) {
    auto identity = Json{{"schema_version", 1},
                         {"kind", "native_klpo"},
                         {"configuration", config.document},
                         {"sources", sources},
                         {"sources_contract", simulation::reconstructed_tape_contract},
                         {"lane_rule", "train_tape_index_is_lane_modulo_train_tapes"},
                         {"budget_rule", "complete_waves_within_declared_transitions"},
                         {"wave_transitions", wave_transitions},
                         {"planned_waves", planned_waves},
                         {"evaluation_waves", evaluation_waves},
                         {"device", options.device},
                         {"diagnostic", options.diagnostic},
                         {"representation", "adaptive_context_ppo_fixed_projection_1729_v1"},
                         {"evaluation_policy", "greedy_argmax"},
                         {"rollout_policy", "categorical_sampling"},
                         {"torch_version", TORCH_VERSION},
                         {"native_source_sha256", build::native_source_sha256()},
                         {"native_build_sha256", build::native_build_sha256()},
                         {"parent_frozen", true},
                         {"final_test_opened", false}};
    if (config.learning.group) {
        identity["group_rule"] = std::string(group_wave_rule);
    }
    return identity;
}

class KlpoRun {
  public:
    KlpoRun(const PpoExperimentOptions& options, KlpoExperimentConfig config,
            const std::function<bool()>& stop)
        : options_(options), config_(std::move(config)), stop_(stop),
          started_(std::chrono::steady_clock::now()) {}
    // Guarda referencias a las opciones y a la parada de la orden: no se copia ni se mueve.
    KlpoRun(const KlpoRun&) = delete;
    KlpoRun& operator=(const KlpoRun&) = delete;
    KlpoRun(KlpoRun&&) = delete;
    KlpoRun& operator=(KlpoRun&&) = delete;
    ~KlpoRun() = default;

    Json run() {
        load_tapes();
        configure_policy_runtime(options_);
        auto learning = config_.learning;
        learning.collection.device = options_.device;
        controller_ = std::make_unique<KlpoLearningController>(lanes_, learning);
        store_ = std::make_unique<PpoCheckpointStore>(options_.output, controller_->identity(),
                                                      options_.resume);
        identity_ = experiment_identity();
        identity_sha256_ = content_sha256(identity_.dump());
        const auto experiment_path = options_.output / "experiment.json";
        if (options_.resume && std::filesystem::exists(experiment_path)) {
            require(unseal(experiment_path) == identity_,
                    "La salida pertenece a otro experimento KLPO, cinta o compilación");
        } else {
            atomic_json_file(experiment_path, seal(identity_));
        }
        if (options_.resume && has_recent_checkpoint()) {
            controller_->restore(store_->load_latest());
            state_ = std::filesystem::exists(options_.output / "selection.json")
                         ? unseal(options_.output / "selection.json")
                         : fresh_state();
            validate_state();
        } else {
            static_cast<void>(controller_->save(*store_));
            state_ = fresh_state();
            write_state(state_);
        }
        remove_unreferenced_best();
        if (state_.at("status") == "completed") {
            return publish("completed");
        }
        open_learning_traces();
        return loop();
    }

  private:
    void load_tapes() {
        auto loaded = load_fit_tapes(options_, config_);
        lanes_ = std::move(loaded.lanes);
        validation_ = std::move(loaded.validation);
        sources_ = std::move(loaded.sources);
        wave_transitions_ = loaded.wave_transitions;
        planned_waves_ = loaded.planned_waves;
        evaluation_waves_ = loaded.evaluation_waves;
    }

    Json experiment_identity() const {
        auto identity = klpo_sources_identity(options_, config_, sources_, wave_transitions_,
                                              planned_waves_, evaluation_waves_);
        identity["controller_identity_sha256"] = store_->identity_sha256();
        return identity;
    }

    bool has_recent_checkpoint() const {
        const auto envelope = parse_bounded_json(read_bounded_file(
            options_.output / "ppo-index.json", simulation::maximum_manifest_bytes));
        return !envelope.at("payload").at("recent").empty();
    }

    // Trazas de aprendizaje de #448. El paso es la oleada. La actualización de la oleada w
    // lleva la marca (w, 0) y se escribe antes del paso de Adam, que la confirma al publicar
    // los contadores. Su validación lleva (w, 1) y la confirma `evaluated_waves`. Al reanudar
    // se borran las partes posteriores al estado recuperado, también si el corte llegó entre
    // la parte y su confirmación.
    TraceCommit confirmed_commit() const {
        const auto consumed = static_cast<uint64_t>(controller_->counters().consumed_waves);
        const auto& evaluated = state_.at("evaluated_waves");
        return {consumed, !evaluated.is_null() && count(evaluated) == consumed ? 1U : 0U};
    }

    void open_learning_traces() {
        if (!options_.learning_traces.enabled()) {
            return;
        }
        const Json identity{{"family", config_.learning.group
                                           ? config_.learning.group->id()
                                           : std::string("klpo_terminal")},
                            {"run_identity_sha256", identity_sha256_}};
        learning_traces_ = std::make_unique<LearningTraceWriter>(
            options_.output / "traces", options_.learning_traces, identity, "wave",
            confirmed_commit());
    }

    void record_update(const KlpoGradientSummary& summary) {
        const auto wave = static_cast<uint64_t>(controller_->counters().consumed_waves) + 1;
        if (!learning_traces_ || !learning_traces_->due(wave - 1, wave)) {
            return;
        }
        learning_traces_->scalar(wave, "train", "policy.surrogate", summary.surrogate);
        learning_traces_->scalar(wave, "train", "rl.grad_l2_total", summary.gradient_norm);
        learning_traces_->scalar(wave, "train", "rl.episodes",
                                 static_cast<double>(summary.episodes));
        learning_traces_->scalar(wave, "train", "rl.sampled_decisions",
                                 static_cast<double>(summary.sampled_decisions));
        learning_traces_->scalar(wave, "train", "rl.gradient_blocks",
                                 static_cast<double>(summary.blocks));
        learning_traces_->flush({wave, 0});
    }

    Json fresh_state() const {
        return Json{{"schema_version", 1},          {"identity_sha256", identity_sha256_},
                    {"status", "running"},          {"wave_steps", Json::array()},
                    {"evaluations", 0},             {"stale_evaluations", 0},
                    {"evaluated_waves", nullptr},   {"best", nullptr}};
    }

    void write_state(const Json& next) {
        atomic_json_file(options_.output / "selection.json", seal(next));
        state_ = next;
    }

    // La selección se confirma después de cada oleada consumida y antes de recoger otra.
    void validate_state() const {
        const auto counters = controller_->counters();
        const auto phase = controller_->phase();
        const auto& steps = state_.at("wave_steps");
        const auto consumed = static_cast<std::size_t>(counters.consumed_waves);
        require(state_.at("schema_version") == 1 && state_.at("identity_sha256") == identity_sha256_ &&
                    steps.is_array() && consumed <= planned_waves_ &&
                    (steps.size() == consumed ||
                     (phase == KlpoLearningPhase::consumed && steps.size() + 1 == consumed)),
                "La selección KLPO no corresponde a las oleadas confirmadas");
        for (const auto& value : steps) {
            require(count(value) > 0 && count(value) <= wave_transitions_,
                    "Una oleada confirmada declara pasos imposibles");
        }
        const auto evaluations = count(state_.at("evaluations"));
        const auto& evaluated = state_.at("evaluated_waves");
        require(evaluated.is_null() == (evaluations == 0) &&
                    state_.at("best").is_null() == (evaluations == 0) &&
                    (evaluated.is_null() || count(evaluated) <= consumed) &&
                    count(state_.at("stale_evaluations")) < std::max<std::size_t>(evaluations, 1),
                "La selección KLPO contiene evaluaciones incoherentes");
        if (!state_.at("best").is_null()) {
            const auto& best = state_.at("best");
            const auto bytes = read_bounded_file(options_.output / best.at("actor_file").get<std::string>(),
                                                 maximum_ppo_archive_bytes);
            require(content_sha256(bytes) == best.at("actor_sha256").get<std::string>() &&
                        count(best.at("wave")) <= count(evaluated),
                    "La mejor política KLPO no conserva su huella");
        }
    }

    // Una mejor política publicada sin su selección confirmada se descarta al reanudar.
    void remove_unreferenced_best() const {
        const auto kept = state_.at("best").is_null()
                              ? std::string{}
                              : state_.at("best").at("actor_file").get<std::string>();
        for (const auto& entry : std::filesystem::directory_iterator(options_.output)) {
            const auto name = entry.path().filename().string();
            if (name.starts_with(best_prefix) && name != kept) {
                std::filesystem::remove(entry.path());
            }
        }
    }

    std::size_t confirmed_transitions(std::size_t waves) const {
        std::size_t result = 0;
        for (std::size_t index = 0; index < waves; ++index) {
            result += count(state_.at("wave_steps").at(index));
        }
        return result;
    }

    std::size_t collected() const {
        const auto& steps = state_.at("wave_steps");
        return confirmed_transitions(steps.size()) +
               (controller_->phase() == KlpoLearningPhase::consumed ? 0
                                                                     : controller_->collected_steps());
    }

    bool stop_requested() const {
        return (stop_ && stop_()) || (options_.stop_after && collected() >= *options_.stop_after);
    }

    bool evaluation_due(std::size_t wave) const {
        return wave == 0 || wave % evaluation_waves_ == 0 || wave == planned_waves_;
    }

    // Evalúa el actor confirmado con argmax en validación. Devuelve falso si se pausa.
    bool evaluate(std::size_t wave) {
        const auto bytes = actor_archive(controller_->snapshot());
        const auto actor = load_actor(bytes, options_);
        const auto evaluation = evaluate_policy(actor, validation_, config_.learning.collection.workers,
                                                stop_, evaluation_context(config_));
        if (evaluation.paused) {
            return false;
        }
        const auto score = config_.liquidated_selection ? evaluation.mean_liquidated_log_growth
                                                        : evaluation.mean_log_growth;
        require(evaluation.incomplete == 0 && evaluation.episodes == validation_.size() &&
                    std::isfinite(evaluation.mean_log_growth) && std::isfinite(score),
                "La validación KLPO está incompleta y no permite seleccionar");
        auto next = state_;
        const auto& best = state_.at("best");
        const bool improved =
            best.is_null() || evaluation.ruined < count(best.at("ruin_count")) ||
            (evaluation.ruined == count(best.at("ruin_count")) &&
             score > finite_number(best.at("score")) + config_.min_delta);
        next["evaluations"] = count(state_.at("evaluations")) + 1;
        next["evaluated_waves"] = wave;
        if (learning_traces_) {
            // Cada validación completa se registra antes de confirmar la selección.
            learning_traces_->scalar(wave, "validation", "rl.selection_score", score);
            learning_traces_->scalar(wave, "validation", "rl.mean_log_growth",
                                     evaluation.mean_log_growth);
            if (config_.liquidated_selection) {
                learning_traces_->scalar(wave, "validation", "rl.mean_liquidated_log_growth",
                                         evaluation.mean_liquidated_log_growth);
            }
            learning_traces_->scalar(wave, "validation", "rl.ruin_count",
                                     static_cast<double>(evaluation.ruined));
            learning_traces_->scalar(wave, "validation", "rl.episodes",
                                     static_cast<double>(evaluation.episodes));
            learning_traces_->scalar(wave, "validation", "rl.improved", improved ? 1. : 0.);
            learning_traces_->flush({wave, 1});
        }
        if (improved) {
            const auto digest = content_sha256(bytes);
            const auto name = std::string(best_prefix) + digest + ".pt";
            simulation::atomic_binary_file(options_.output / name, bytes, maximum_ppo_archive_bytes);
            next["best"] = Json{{"wave", wave},
                                {"transitions", confirmed_transitions(wave)},
                                {"optimizer_steps", controller_->counters().confirmed_updates},
                                {"ruin_count", evaluation.ruined},
                                {"episodes", evaluation.episodes},
                                {"mean_log_growth", evaluation.mean_log_growth},
                                {"mean_liquidated_log_growth", evaluation.mean_liquidated_log_growth},
                                {"score", score},
                                {"actor_sha256", digest},
                                {"actor_fingerprint", actor.parameter_fingerprint()},
                                {"actor_file", name}};
            next["stale_evaluations"] = 0;
        } else {
            next["stale_evaluations"] = count(state_.at("stale_evaluations")) + 1;
        }
        write_state(next);
        remove_unreferenced_best();
        return true;
    }

    Json pause() {
        auto next = state_;
        next["status"] = "paused";
        write_state(next);
        return publish("paused");
    }

    Json loop() {
        if (state_.at("evaluated_waves").is_null() && !evaluate(0)) {
            return pause();
        }
        while (true) {
            const auto consumed = static_cast<std::size_t>(controller_->counters().consumed_waves);
            switch (controller_->phase()) {
            case KlpoLearningPhase::consumed: {
                if (state_.at("wave_steps").size() < consumed) {
                    auto next = state_;
                    next["wave_steps"].push_back(controller_->collected_steps());
                    write_state(next);
                }
                if (evaluation_due(consumed) && count(state_.at("evaluated_waves")) < consumed &&
                    !evaluate(consumed)) {
                    return pause();
                }
                if (consumed == planned_waves_) {
                    auto next = state_;
                    next["status"] = "completed";
                    write_state(next);
                    return publish("completed");
                }
                if (stop_requested()) {
                    return pause();
                }
                controller_->start_next_wave(*store_);
                break;
            }
            case KlpoLearningPhase::collecting:
                if (stop_requested()) {
                    static_cast<void>(controller_->save(*store_));
                    return pause();
                }
                static_cast<void>(controller_->collect_tick());
                break;
            case KlpoLearningPhase::ready:
                // La pausa se confirma antes de la actualización, nunca dentro de ella.
                if (stop_requested()) {
                    static_cast<void>(controller_->save(*store_));
                    return pause();
                }
                // Ruta real de Adam, pendiente de ejecución bajo el bloqueo de aprendizaje.
                static_cast<void>(controller_->update_ready(
                    *store_, {}, [&](const KlpoGradientSummary& summary) { record_update(summary); }));
                require(controller_->phase() == KlpoLearningPhase::consumed,
                        "La actualización KLPO no consumió la oleada completa");
                break;
            case KlpoLearningPhase::blocked:
                throw std::invalid_argument("Una oleada KLPO perdió la valoración de un episodio");
            }
        }
    }

    Json publish(std::string_view status) const {
        const auto counters = controller_->counters();
        auto selection = config_.document.at("selection");
        selection["policy"] = "greedy_argmax";
        selection["partition"] = "validation";
        const auto waves = state_.at("wave_steps").size();
        Json report{{"schema_version", 1},
                    {"kind", "native_klpo"},
                    {"activity", "rl"},
                    {"model", config_.learning.group ? Json(config_.learning.group->id())
                                                     : Json("klpo_terminal")},
                    {"backend", "native_libtorch"},
                    {"seed", config_.learning.collection.seed},
                    {"partition", "train"},
                    {"status", status},
                    {"stopping_reason", status == "completed" ? Json("budget_exhausted")
                                                              : Json("requested_pause")},
                    {"identity_sha256", identity_sha256_},
                    {"domain", options_.diagnostic ? "technical" : "real"},
                    {"device", options_.device},
                    {"diagnostic", options_.diagnostic},
                    {"transitions", confirmed_transitions(waves)},
                    {"collected_transitions", collected()},
                    {"budget_transitions", config_.total_transitions},
                    {"wave_transitions", wave_transitions_},
                    {"planned_waves", planned_waves_},
                    {"consumed_waves", counters.consumed_waves},
                    {"optimizer_steps", counters.confirmed_updates},
                    {"reference_version", counters.reference_version},
                    {"evaluations", state_.at("evaluations")},
                    {"stale_evaluations", state_.at("stale_evaluations")},
                    {"best", state_.at("best")},
                    {"selection", selection},
                    {"parent_frozen", true},
                    {"final_test_opened", false},
                    {"invocation_seconds", elapsed_seconds(started_)}};
        atomic_json_file(options_.output / "run.json", report);
        return report;
    }

    const PpoExperimentOptions& options_;
    KlpoExperimentConfig config_;
    const std::function<bool()>& stop_;
    std::chrono::steady_clock::time_point started_;
    std::vector<simulation::BatchInput> lanes_;
    std::vector<simulation::BatchInput> validation_;
    Json sources_;
    std::size_t wave_transitions_ = 0;
    std::size_t planned_waves_ = 0;
    std::size_t evaluation_waves_ = 1;
    std::unique_ptr<KlpoLearningController> controller_;
    std::unique_ptr<PpoCheckpointStore> store_;
    Json identity_;
    std::string identity_sha256_;
    Json state_;
    std::unique_ptr<LearningTraceWriter> learning_traces_;
};

// Evaluación de la política KLPO elegida en cintas posteriores a su selección.
Json run_evaluation(const std::filesystem::path& run, const PpoExperimentOptions& options,
                    const KlpoExperimentConfig& config, const std::function<bool()>& stop) {
    const auto selected = unseal(run / "experiment.json");
    require(selected.at("kind") == "native_klpo" &&
                selected.at("configuration") == config.document &&
                selected.at("device") == options.device &&
                selected.at("diagnostic") == options.diagnostic &&
                selected.at("native_source_sha256") == build::native_source_sha256() &&
                selected.at("native_build_sha256") == build::native_build_sha256() &&
                selected.at("final_test_opened") == false,
            "La evaluación no conserva la configuración y la compilación de la selección");
    const simulation::OutputLock frozen(run, true);
    const auto selection = unseal(run / "selection.json");
    require(selection.at("identity_sha256") == content_sha256(selected.dump()) &&
                selection.at("status") == "completed" &&
                selection.at("evaluated_waves") == selected.at("planned_waves") &&
                !selection.at("best").is_null(),
            "La evaluación KLPO necesita una selección cerrada");
    const auto& best = selection.at("best");
    const auto bytes = read_bounded_file(run / best.at("actor_file").get<std::string>(),
                                         maximum_ppo_archive_bytes);
    require(content_sha256(bytes) == best.at("actor_sha256").get<std::string>(),
            "La política KLPO elegida no conserva su huella");
    FrozenEvaluationRequest request{options.output, options.resume, nullptr, {},
                                    evaluation_context(config),
                                    config.learning.collection.workers,
                                    frozen_costs(options.evaluation_costs), options.decisions};
    Json tapes = Json::array();
    for (const auto& path : options.audit_tapes) {
        auto tape = load_policy_tape(path, PolicyTapeRole::evaluation, config.environment);
        require_after_selection(selected.at("sources"), tape);
        tapes.push_back(tape.identity);
        request.tapes.push_back(std::move(tape));
    }
    require_policy_sequence(request.tapes);
    configure_policy_runtime(options);
    const auto actor = load_actor(bytes, options);
    require(actor.parameter_fingerprint() == best.at("actor_fingerprint").get<std::string>() &&
                actor.optimizer_steps() == count(best.at("optimizer_steps")),
            "El actor KLPO cargado no corresponde a la selección");
    request.identity = Json{{"schema_version", 1},
                            {"kind", "native_klpo_reconstructed_evaluation"},
                            {"selected_identity_sha256", content_sha256(selected.dump())},
                            {"policy_sha256", best.at("actor_sha256")},
                            {"optimizer_steps", best.at("optimizer_steps")},
                            {"transitions", best.at("transitions")},
                            {"tapes", tapes},
                            {"cost_bps", request.costs},
                            {"seed", config.learning.collection.seed},
                            {"device", options.device},
                            {"diagnostic", options.diagnostic},
                            {"decisions", options.decisions},
                            {"native_source_sha256", build::native_source_sha256()},
                            {"native_build_sha256", build::native_build_sha256()},
                            {"final_test_opened", false}};
    return run_frozen_evaluation(actor, request, stop);
}
// Evaluación congelada del actor inicial de la configuración, sin controlador ni Adam terminal.
// Ese actor no ha aprendido nada y su evaluación solo sirve para comprobaciones técnicas, como
// la paridad entre dispositivos. Se construye como el actor de la recogida de un ajuste, con los
// pesos del generador de CPU y la semilla de la configuración.
Json run_initial_evaluation(const PpoExperimentOptions& options, const KlpoExperimentConfig& config,
                            const std::function<bool()>& stop) {
    FrozenEvaluationRequest request{options.output, options.resume, nullptr, {},
                                    evaluation_context(config),
                                    config.learning.collection.workers,
                                    frozen_costs(options.evaluation_costs), options.decisions};
    Json tapes = Json::array();
    std::vector<simulation::BatchInput> inputs;
    for (const auto& path : options.audit_tapes) {
        auto tape = load_policy_tape(path, PolicyTapeRole::evaluation, config.environment);
        tapes.push_back(tape.identity);
        inputs.push_back(tape.input);
        request.tapes.push_back(std::move(tape));
    }
    require_policy_sequence(request.tapes);
    configure_policy_runtime(options);
    auto collection = config.learning.collection;
    collection.device = options.device;
    const auto width = policy_observation_width(inputs, collection.context, collection.seed,
                                                collection.workers);
    const auto actor = initial_klpo_actor(width, collection);
    request.identity = Json{{"schema_version", 1},
                            {"kind", "native_klpo_initial_evaluation"},
                            {"weights", "initial_from_configuration_seed"},
                            {"evaluation_only", evaluation_only()},
                            {"configuration_sha256", content_sha256(config.document.dump())},
                            {"parameter_fingerprint", actor.parameter_fingerprint()},
                            {"optimizer_steps", actor.optimizer_steps()},
                            {"transitions", 0},
                            {"tapes", tapes},
                            {"cost_bps", request.costs},
                            {"seed", collection.seed},
                            {"device", options.device},
                            {"diagnostic", options.diagnostic},
                            {"decisions", options.decisions},
                            {"native_source_sha256", build::native_source_sha256()},
                            {"native_build_sha256", build::native_build_sha256()},
                            {"final_test_opened", false}};
    return run_frozen_evaluation(actor, request, stop);
}

// Lee y valida las cintas de ajuste y validación como el ajuste, monta la recogida sin
// controlador y publica en sources.json las fuentes, las oleadas y el actor inicial.
Json run_source_check(const PpoExperimentOptions& options, const KlpoExperimentConfig& config) {
    const auto loaded = load_fit_tapes(options, config);
    configure_policy_runtime(options);
    auto collection = config.learning.collection;
    collection.device = options.device;
    // La recogida monta los entornos y el actor inicial sin dar ningún paso.
    const KlpoTerminalCollector collector(loaded.lanes, collection);
    require(collector.policy().observation_width() ==
                policy_observation_width(loaded.validation, collection.context, collection.seed,
                                         collection.workers),
            "La validación utiliza otro esquema de observación que el ajuste");
    const auto identity =
        klpo_sources_identity(options, config, loaded.sources, loaded.wave_transitions,
                              loaded.planned_waves, loaded.evaluation_waves);
    const Json report{{"schema_version", 1},
                      {"kind", "native_policy_source_check"},
                      {"status", "completed"},
                      {"evaluation_only", evaluation_only()},
                      {"identity", identity},
                      {"identity_sha256", content_sha256(identity.dump())},
                      {"observation_width", collector.policy().observation_width()},
                      {"initial_parameter_fingerprint", collector.records().reference_sha256},
                      {"transitions", 0},
                      {"optimizer_steps", 0},
                      {"final_test_opened", false}};
    const simulation::OutputLock lock(options.output, false);
    atomic_json_file(options.output / "sources.json", seal(report));
    return report;
}
} // namespace

Json run_klpo_experiment(const PpoExperimentOptions& options, const std::function<bool()>& stop) {
    const bool without_learning = options.initial_evaluation || options.source_check;
    if (without_learning) {
        // Nada aprende: el cerrojo se activa antes de leer la configuración y la protección no
        // se consulta, ni para rebajarla ni para pedirla.
        enter_evaluation_only();
    }
    auto config = configuration(options.config);
    validate_options(options, config);
    admit_policy_gpu(options);
    if (options.initial_evaluation) {
        return run_initial_evaluation(options, config, stop);
    }
    if (options.source_check) {
        return run_source_check(options, config);
    }
    // Ajustar o evaluar sobre el histórico reconstruido se detiene antes de leer cintas o
    // crear salidas si la protección local no lo permite.
    const bool group = config.learning.group.has_value();
    require_learning_allowed(options.audit_run ? "la evaluación KLPO nativa sobre cintas reales"
                             : group           ? "el entrenamiento nativo con objetivo de grupo"
                                               : "el entrenamiento KLPO nativo");
    if (const auto& run = options.audit_run) {
        return run_evaluation(*run, options, config, stop);
    }
    KlpoRun run(options, std::move(config), stop);
    return run.run();
}

} // namespace mars_titan::learning
