#include "mars_titan/cohort_execution.hpp"
#include "mars_titan/simulation_files.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

namespace {
using namespace mars_titan::cohorts;
constexpr std::size_t digest_width = 64;
constexpr std::int64_t tick = 10;
constexpr double target_a = 5;
constexpr double target_b = 6;
constexpr double second_bias = 8;
constexpr double third_bias = 24;
constexpr std::size_t steady_pending = 6;
constexpr std::size_t logical_asset_limit = 8192;
constexpr std::int64_t phase_end = 3 * tick;
constexpr std::int64_t phase_close = 4 * tick;
constexpr std::size_t insufficient_pairs = 125;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}
template <typename Function> void rejected(Function&& function, std::string_view message) {
    try {
        std::forward<Function>(function)();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}
class TemporaryDirectory {
  public:
    TemporaryDirectory() {
        auto pattern = (std::filesystem::temp_directory_path() / "mars-cohort-XXXXXX").string();
        const auto* created = ::mkdtemp(pattern.data());
        require(created != nullptr, "No se puede crear el directorio de prueba");
        path = created;
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    TemporaryDirectory(TemporaryDirectory&&) = delete;
    TemporaryDirectory& operator=(TemporaryDirectory&&) = delete;
    std::filesystem::path path;
};

Definition definition() {
    return Definition{{std::string(digest_width, 'a'), std::string(digest_width, 'b'),
                       std::string(digest_width, 'c'), std::string(digest_width, 'd')},
                      {{"return", 1}, {"event", 2}},
                      Json{{"bias", 0.0}, {"updates", 0}},
                      {}};
}
Callbacks callbacks() {
    return {[](std::span<const Observation> observations, const Task&, const Json& state) {
                std::vector<double> values;
                for (const auto& row : observations) {
                    values.push_back(row.features.front() + state.at("bias").get<double>());
                }
                return values;
            },
            [](const Json& state, std::span<const ResolvedFeedback> outcomes) {
                auto next = state;
                auto bias = state.at("bias").get<double>();
                for (const auto& item : outcomes) {
                    bias += item.label.value - item.prediction.value;
                }
                next["bias"] = bias;
                next["updates"] = state.at("updates").get<std::size_t>() + outcomes.size();
                return next;
            }};
}
Cohort cohort(std::size_t cursor, bool reverse = false) {
    const auto at = static_cast<std::int64_t>(cursor + 1) * tick;
    Cohort result{cursor, at, {{"US/A", at, {1}}, {"US/B", at, {2}}}};
    if (reverse) {
        std::ranges::reverse(result.observations);
    }
    return result;
}
std::vector<Feedback> matured(const Executor& executor, std::int64_t cutoff, bool reverse = false) {
    std::vector<Feedback> result;
    for (const auto& prediction : executor.pending()) {
        const auto available =
            prediction.decision_at + static_cast<std::int64_t>(prediction.task.horizon) * tick;
        if (available <= cutoff) {
            result.push_back(
                {prediction.id, 0, available, prediction.asset == "US/A" ? target_a : target_b});
        }
    }
    if (reverse) {
        std::ranges::reverse(result);
    }
    return result;
}
Commit advance(Executor& executor, std::size_t batch = 1, bool reverse = false,
               const std::function<void(Boundary)>& hook = {}) {
    const auto current = cohort(executor.cursor(), reverse);
    return executor.step(current, matured(executor, current.cutoff, reverse), batch, hook);
}

void whole_cohort_precedes_feedback_and_preserves_the_original_prediction() {
    TemporaryDirectory directory;
    Executor run(directory.path / "run", definition(), callbacks());
    const auto first = advance(run);
    require(first.predictions.size() == 4 && first.applied.empty(), "Faltan decisiones iniciales");
    const auto second = advance(run);
    for (const auto& prediction : second.predictions) {
        require(prediction.value == (prediction.asset == "US/A" ? 1 : 2),
                "La cohorte vio feedback antes de terminar todas sus predicciones");
    }
    require(run.snapshot().at("state").at("bias") == second_bias,
            "No se aplicó el feedback maduro");
    const auto third = advance(run);
    for (const auto& prediction : third.predictions) {
        require(prediction.value == second_bias + (prediction.asset == "US/A" ? 1 : 2),
                "El estado siguiente no se utilizó en la decisión posterior");
    }
    require(run.snapshot().at("state").at("bias") == third_bias,
            "El error no usa la salida original conservada");
    require(third.applied.size() == 4, "Los horizontes no maduran por separado");
    const auto original = run.record(1);
    require(original.at("predictions").at(0).at("value") == 1,
            "Se ha reescrito la primera predicción después del aprendizaje");
    require(replay_id(third.applied.front(), 0) != third.applied.front().id &&
                replay_id(third.applied.front(), 0) != replay_id(third.applied.front(), 1),
            "El replay comparte identidad con el feedback o con otra exposición");
}

void asset_order_batch_size_and_future_suffix_preserve_the_prefix() {
    TemporaryDirectory directory;
    Executor first(directory.path / "first", definition(), callbacks());
    Executor second(directory.path / "second", definition(), callbacks());
    for (std::size_t i = 0; i < 3; ++i) {
        advance(first, 1);
        advance(second, 2, true);
        require(first.snapshot() == second.snapshot(),
                "El orden físico altera el estado confirmado");
        require(first.record(i + 1) == second.record(i + 1),
                "El orden físico altera las decisiones");
    }
    const auto prefix = second.record(2);
    advance(second, 2);
    require(second.record(2) == prefix && first.record(2) == prefix,
            "El sufijo futuro modifica una decisión anterior");
}

void recovery_at_every_boundary_matches_an_uninterrupted_run() {
    TemporaryDirectory directory;
    Executor reference(directory.path / "reference", definition(), callbacks());
    for (std::size_t i = 0; i < 3; ++i) {
        advance(reference);
    }
    const std::array boundaries{Boundary::before_predictions, Boundary::predictions_ready,
                                Boundary::record_written,     Boundary::feedback_applied,
                                Boundary::checkpoint_written, Boundary::before_commit,
                                Boundary::committed};
    for (const auto boundary : boundaries) {
        const auto output = directory.path / std::to_string(static_cast<unsigned int>(boundary));
        {
            Executor interrupted(output, definition(), callbacks());
            advance(interrupted);
            advance(interrupted);
            rejected(
                [&] {
                    advance(interrupted, 1, false, [boundary](Boundary point) {
                        if (point == boundary) {
                            throw std::runtime_error("Interrupción de prueba");
                        }
                    });
                },
                "La frontera no interrumpe la transacción");
            rejected([&] { static_cast<void>(interrupted.snapshot()); },
                     "Una instancia con fallo ambiguo debe exigir reapertura");
        }
        Executor resumed(output, definition(), callbacks(), true);
        require(resumed.cursor() == (boundary == Boundary::committed ? 3 : 2),
                "Se mezclan generaciones alrededor de latest.json");
        if (resumed.cursor() == 2) {
            advance(resumed, 2, true);
        }
        require(resumed.snapshot() == reference.snapshot(), "La recuperación cambia el estado");
        require(resumed.record(3) == reference.record(3),
                "La recuperación cambia la decisión emitida");
    }
}

void incompatible_identities_corrupt_files_and_concurrent_writers_fail_closed() {
    TemporaryDirectory directory;
    const auto output = directory.path / "run";
    {
        Executor writer(output, definition(), callbacks());
        advance(writer);
        rejected([&] { Executor other(output, definition(), callbacks(), true); },
                 "Falta exclusión entre escritores");
    }
    for (std::size_t field = 0; field < 4; ++field) {
        auto wrong = definition();
        std::array fields{&wrong.identity.source_sha256, &wrong.identity.view_sha256,
                          &wrong.identity.representation_sha256, &wrong.identity.model_sha256};
        fields.at(field)->assign(digest_width, 'f');
        rejected([&] { Executor other(output, wrong, callbacks(), true); },
                 "Se admite estado de otra identidad");
    }
    mars_titan::simulation::atomic_json_file(output / "checkpoint-1.json", Json{{"corrupt", true}});
    rejected([&] { Executor other(output, definition(), callbacks(), true); },
             "La corrupción no puede provocar un reinicio silencioso");
}

void admission_rejects_future_duplicate_nonfinite_and_revised_inputs() {
    TemporaryDirectory directory;
    Executor run(directory.path / "run", definition(), callbacks());
    auto invalid = cohort(0);
    invalid.observations.front().available_at += 1;
    rejected([&] { run.step(invalid, {}); }, "Se admite una observación futura");
    invalid = cohort(0);
    invalid.observations.back().asset = invalid.observations.front().asset;
    rejected([&] { run.step(invalid, {}); }, "Se admite un activo duplicado");
    invalid = cohort(0);
    invalid.observations.front().features.front() = std::numeric_limits<double>::quiet_NaN();
    rejected([&] { run.step(invalid, {}); }, "Se admite una entrada no finita");
    rejected([&] { run.step(cohort(0), {}, 0); }, "Se admite un lote físico vacío");
    advance(run);
    auto feedback = matured(run, cohort(1).cutoff);
    feedback.front().revision = 1;
    rejected([&] { run.step(cohort(1), feedback); }, "Se admite una revisión no soportada");
    feedback = matured(run, cohort(1).cutoff);
    feedback.push_back(feedback.front());
    rejected([&] { run.step(cohort(1), feedback); }, "Se aplica un feedback dos veces");
    feedback = matured(run, cohort(1).cutoff);
    const auto previous = feedback;
    run.step(cohort(1), feedback);
    const auto state = run.snapshot();
    rejected([&] { run.step(cohort(2), previous); }, "Se vuelve a aplicar feedback confirmado");
    require(run.snapshot() == state, "Una entrada rechazada cambia el estado");
}

void capacity_and_retention_remain_bounded() {
    TemporaryDirectory directory;
    auto tiny = definition();
    tiny.limits.max_pending = 3;
    Executor limited(directory.path / "limited", tiny, callbacks());
    rejected([&] { advance(limited); }, "La cola excede su capacidad");
    auto disk = definition();
    disk.limits.max_log_bytes = 1;
    Executor exhausted(directory.path / "exhausted", disk, callbacks());
    rejected([&] { advance(exhausted); }, "El registro excede el presupuesto de disco");
    Executor run(directory.path / "run", definition(), callbacks());
    constexpr std::size_t steps = 12;
    for (std::size_t i = 0; i < steps; ++i) {
        advance(run);
    }
    std::size_t checkpoints = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory.path / "run")) {
        if (entry.path().filename().string().starts_with("checkpoint-")) {
            ++checkpoints;
        }
    }
    require(checkpoints == 2, "Los checkpoints crecen con la historia");
    require(run.pending().size() == steady_pending, "Los pendientes no se liberan al madurar");
    rejected([&] { static_cast<void>(run.record(1, 1)); }, "La lectura histórica supera su límite");
    require(run.record(steps).at("generation") == steps, "Falta el último registro confirmado");
}

void spliced_counters_are_rejected_even_if_checkpoint_checksums_are_recomputed() {
    TemporaryDirectory directory;
    const auto output = directory.path / "run";
    {
        Executor run(output, definition(), callbacks());
        advance(run);
        advance(run);
    }
    using namespace mars_titan::simulation;
    auto head =
        parse_bounded_json(read_bounded_file(output / "latest.json", maximum_manifest_bytes));
    for (std::size_t generation = 1; generation <= 2; ++generation) {
        const auto path = output / ("checkpoint-" + std::to_string(generation) + ".json");
        auto state = parse_bounded_json(read_bounded_file(path, maximum_checkpoint_bytes));
        state["issued"] = state.at("issued").get<std::size_t>() + 2;
        state["applied"] = state.at("applied").get<std::size_t>() + 2;
        atomic_json_file(path, state);
        head[generation == 2 ? "checkpoint_sha256" : "previous_sha256"] =
            content_sha256(state.dump());
    }
    atomic_json_file(output / "latest.json", head);
    rejected([&] { Executor run(output, definition(), callbacks(), true); },
             "Los contadores desplazados no concuerdan con el registro inmutable");
}

void process_death_recovers_the_correct_generation() {
    TemporaryDirectory directory;
    Executor reference(directory.path / "reference", definition(), callbacks());
    for (std::size_t i = 0; i < 3; ++i) {
        advance(reference);
    }
    for (const auto boundary : {Boundary::record_written, Boundary::committed}) {
        const auto output = directory.path / std::to_string(static_cast<unsigned int>(boundary));
        {
            Executor first(output, definition(), callbacks());
            advance(first);
            advance(first);
        }
        const auto process = ::fork();
        require(process >= 0, "No se puede crear el proceso de recuperación");
        if (process == 0) {
            Executor interrupted(output, definition(), callbacks(), true);
            advance(interrupted, 1, false, [boundary](Boundary at) {
                if (boundary == at) {
                    ::raise(SIGKILL);
                }
            });
            ::_exit(1);
        }
        int status = 0;
        require(::waitpid(process, &status, 0) == process && WIFSIGNALED(status) &&
                    WTERMSIG(status) == SIGKILL,
                "El proceso no terminó en la frontera pedida");
        Executor resumed(output, definition(), callbacks(), true);
        require(resumed.cursor() == (boundary == Boundary::committed ? 3 : 2),
                "SIGKILL recupera una generación parcial");
        if (resumed.cursor() == 2) {
            advance(resumed);
        }
        require(resumed.snapshot() == reference.snapshot(),
                "SIGKILL cambia los efectos confirmados");
    }
}

void invalid_callback_outputs_do_not_become_confirmed_state() {
    TemporaryDirectory directory;
    auto invalid_prediction = callbacks();
    invalid_prediction.predict = [](std::span<const Observation> rows, const Task&, const Json&) {
        return std::vector<double>(rows.size(), std::numeric_limits<double>::infinity());
    };
    {
        Executor run(directory.path / "prediction", definition(), invalid_prediction);
        rejected([&] { advance(run); }, "El predictor publica valores no finitos");
    }
    Executor prediction(directory.path / "prediction", definition(), callbacks(), true);
    require(prediction.cursor() == 0, "La salida inválida avanzó el cursor");
    auto invalid_update = callbacks();
    invalid_update.update = [](const Json&, std::span<const ResolvedFeedback>) {
        return Json{{"bias", std::numeric_limits<double>::quiet_NaN()}};
    };
    {
        Executor run(directory.path / "update", definition(), invalid_update);
        rejected([&] { advance(run); }, "El estado no finito se serializa como null");
    }
    Executor update(directory.path / "update", definition(), callbacks(), true);
    require(update.cursor() == 0, "El estado inválido avanzó el cursor");
    advance(update);
    require(update.cursor() == 1, "No se puede retomar el registro preparado antes del fallo");
}

Definition prepared_definition() {
    auto result = definition();
    result.prediction_mode = PredictionMode::prepared;
    result.initial_state["fast_steps"] = 0;
    return result;
}
Callbacks prepared_callbacks() {
    auto result = callbacks();
    result.predict = {};
    result.prepare = [](std::span<const Observation> rows, std::span<const Task> tasks,
                        std::int64_t cutoff, const Json& before, std::size_t batch_rows) {
        require(batch_rows > 0 && rows.front().available_at <= cutoff,
                "Falta el corte o el tamaño físico en prepare");
        PreparedCohort proposal{{}, before};
        const auto steps = before.at("fast_steps").get<std::size_t>();
        for (const auto& row : rows) {
            for (const auto& task : tasks) {
                require(task.horizon > 0, "Falta una tarea de la cohorte");
                proposal.values.push_back(row.features.front() + before.at("bias").get<double>() +
                                          static_cast<double>(steps));
            }
        }
        proposal.proposed_state["fast_steps"] = steps + 1;
        return proposal;
    };
    return result;
}

void prepared_state_is_advanced_once_and_feedback_follows_all_predictions() {
    TemporaryDirectory directory;
    std::size_t preparations = 0;
    auto operators = prepared_callbacks();
    const auto prepare = operators.prepare;
    operators.prepare = [&](auto rows, auto tasks, auto cutoff, const Json& before, auto batch) {
        ++preparations;
        return prepare(rows, tasks, cutoff, before, batch);
    };
    Executor run(directory.path / "prepared", prepared_definition(), operators);
    advance(run);
    const auto second = advance(run);
    require(preparations == 2 && run.snapshot().at("state").at("fast_steps") == 2,
            "El estado rápido se escribió por tarea o por lote");
    for (const auto& prediction : second.predictions) {
        require(prediction.value == (prediction.asset == "US/A" ? 2 : 3),
                "Una predicción ve feedback o la propuesta de otra fila");
    }
    require(run.snapshot().at("state").at("bias") == second_bias,
            "El feedback ha perdido la predicción emitida original");
    Executor permuted(directory.path / "permuted", prepared_definition(), prepared_callbacks());
    advance(permuted, 2, true);
    advance(permuted, 2, true);
    require(run.snapshot() == permuted.snapshot(), "La propuesta depende del lote físico");
}

void prepared_state_recovers_at_every_boundary() {
    TemporaryDirectory directory;
    Executor reference(directory.path / "reference", prepared_definition(), prepared_callbacks());
    advance(reference);
    advance(reference);
    const std::array boundaries{Boundary::before_predictions, Boundary::predictions_ready,
                                Boundary::record_written,     Boundary::feedback_applied,
                                Boundary::checkpoint_written, Boundary::before_commit,
                                Boundary::committed};
    for (const auto boundary : boundaries) {
        const auto output = directory.path / std::to_string(static_cast<unsigned int>(boundary));
        {
            Executor interrupted(output, prepared_definition(), prepared_callbacks());
            advance(interrupted);
            rejected(
                [&] {
                    advance(interrupted, 1, false, [boundary](Boundary point) {
                        if (point == boundary) {
                            throw std::runtime_error("Interrupción con propuesta de estado");
                        }
                    });
                },
                "La frontera no interrumpió la sesión");
        }
        Executor resumed(output, prepared_definition(), prepared_callbacks(), true);
        require(resumed.cursor() == (boundary == Boundary::committed ? 2 : 1),
                "Se recuperó una propuesta no confirmada");
        if (resumed.cursor() == 1) {
            advance(resumed, 2, true);
        }
        require(resumed.snapshot() == reference.snapshot(),
                "La propuesta no se recupera exactamente");
    }
}

void invalid_prepared_outputs_and_mixed_contracts_are_rejected() {
    TemporaryDirectory directory;
    for (std::size_t variant = 0; variant < 4; ++variant) {
        auto operators = prepared_callbacks();
        const auto prepare = operators.prepare;
        operators.prepare = [prepare, variant](auto rows, auto tasks, auto cutoff,
                                               const Json& before, auto batch) {
            auto result = prepare(rows, tasks, cutoff, before, batch);
            if (variant == 0) {
                result.values.pop_back();
            } else if (variant == 1) {
                result.values.front() = std::numeric_limits<double>::quiet_NaN();
            } else if (variant == 2) {
                result.proposed_state["fast_steps"] = std::numeric_limits<double>::infinity();
            } else {
                result.proposed_state["oversized"] = std::string(2 * mebibyte, 'x');
            }
            return result;
        };
        const auto output = directory.path / std::to_string(variant);
        {
            Executor run(output, prepared_definition(), operators);
            rejected([&] { advance(run); }, "Se aceptó una propuesta inválida");
            require(!std::filesystem::exists(output / "record-1.json"),
                    "La propuesta inválida dejó predicciones selladas");
        }
        Executor recovered(output, prepared_definition(), prepared_callbacks(), true);
        require(recovered.cursor() == 0, "Una propuesta inválida llegó a latest");
    }
    rejected([&] { Executor run(directory.path / "missing", prepared_definition(), callbacks()); },
             "Prepare es obligatorio en su modo");
    auto both = prepared_callbacks();
    both.predict = callbacks().predict;
    rejected([&] { Executor run(directory.path / "both", prepared_definition(), both); },
             "Dos predictores comparten una misma identidad");
    {
        auto classic = prepared_definition();
        classic.prediction_mode = PredictionMode::stateless;
        Executor run(directory.path / "identity", classic, callbacks());
        advance(run);
    }
    rejected(
        [&] {
            Executor run(directory.path / "identity", prepared_definition(), prepared_callbacks(),
                         true);
        },
        "Prepare recupera una identidad clásica");
}
void prepared_mode_accepts_the_full_candidate_group_without_relaxing_classic_limits() {
    TemporaryDirectory directory;
    auto classic = definition();
    classic.limits.max_assets = logical_asset_limit;
    rejected([&] { Executor invalid(directory.path / "classic", classic, callbacks()); },
             "El modo clásico amplió su límite sin otra identidad");
    auto prepared = prepared_definition();
    prepared.limits.max_assets = logical_asset_limit;
    Executor run(directory.path / "prepared", prepared, prepared_callbacks());
    Cohort group{0, tick, {}};
    constexpr std::size_t candidates = 5676;
    for (std::size_t index = 0; index < candidates; ++index) {
        group.observations.push_back({"US/A" + std::to_string(index), 1, {1.0}});
    }
    const auto committed = run.step(group, {}, 256);
    require(committed.predictions.size() == 2 * candidates &&
                run.snapshot().at("state").at("fast_steps") == 1,
            "La cohorte completa perdió flujos o repitió la preparación");
}

Definition financial_definition() {
    auto result = definition();
    result.prediction_mode = PredictionMode::financial;
    result.tasks = {{"residual", 1}};
    result.phase = {"validation", 1, tick, phase_end, phase_close, std::string(digest_width, 'e')};
    result.initial_state = Json{{"observed", 0}, {"labels", 0}, {"excluded", 0}, {"closed", 0}};
    return result;
}
Callbacks financial_callbacks() {
    Callbacks result;
    result.prepare_event = [](EventKind kind, std::span<const Observation> rows,
                              std::span<const Task>, std::int64_t, const Json& state, std::size_t) {
        PreparedCohort prepared{{}, state};
        if (kind == EventKind::decision) {
            for (const auto& row : rows) {
                prepared.values.push_back(row.features.front() +
                                          state.at("observed").get<double>());
            }
        }
        prepared.proposed_state["observed"] = state.at("observed").get<std::size_t>() + rows.size();
        return prepared;
    };
    result.resolve = [](const Json& state, std::span<const ResolvedFeedback> labels,
                        std::span<const ResolvedPrefixExclusion> excluded,
                        std::span<const AdministrativeFinalization> closed) {
        auto result_state = state;
        result_state["labels"] = state.at("labels").get<std::size_t>() + labels.size();
        result_state["excluded"] = state.at("excluded").get<std::size_t>() + excluded.size();
        result_state["closed"] = state.at("closed").get<std::size_t>() + closed.size();
        return result_state;
    };
    return result;
}
Cohort financial_decision() {
    auto result = cohort(0);
    result.prefix_exclusions = {{"US/A",
                                 {"residual", 1},
                                 tick,
                                 PrefixReason::insufficient_pairs,
                                 insufficient_pairs,
                                 std::nullopt,
                                 std::string(digest_width, 'f')}};
    return result;
}
void financial_exclusion_preserves_emission_and_recovery() {
    TemporaryDirectory directory;
    const auto output = directory.path / "run";
    {
        auto contract = financial_definition();
        contract.limits.max_pending = 1;
        Executor run(output, contract, financial_callbacks());
        const auto decision = run.step(financial_decision(), {});
        require(decision.predictions.size() == 2 && decision.excluded.size() == 1 &&
                    decision.excluded.front().prediction.value == 1 && run.pending().size() == 1,
                "La exclusión del prefijo eliminó una emisión o no liberó su pendiente");
        const auto before = run.snapshot();
        auto duplicate = financial_decision();
        duplicate.cursor = 1;
        duplicate.cutoff = 2 * tick;
        duplicate.kind = EventKind::settlement;
        duplicate.observations.clear();
        rejected([&] { run.step(duplicate, {}); }, "La exclusión del prefijo se aplicó dos veces");
        require(run.snapshot() == before, "El rechazo cambió la generación confirmada");
    }
    auto contract = financial_definition();
    contract.limits.max_pending = 1;
    Executor restored(output, contract, financial_callbacks(), true);
    Cohort close{1, phase_close, {}};
    close.kind = EventKind::settlement;
    close.close_phase = true;
    const auto committed = restored.step(close, {});
    require(committed.predictions.empty() && committed.applied.empty() &&
                committed.finalized.size() == 1 && restored.pending().empty() &&
                restored.snapshot().at("state").at("observed") == 2,
            "El cierre administrativo volvió a predecir o fabricó feedback");
}
void financial_warmup_and_phase_boundaries_are_explicit() {
    TemporaryDirectory directory;
    Executor run(directory.path / "run", financial_definition(), financial_callbacks());
    auto warmup = cohort(0);
    warmup.cutoff = 2;
    warmup.kind = EventKind::warmup;
    for (auto& row : warmup.observations) {
        row.available_at = 2;
    }
    require(run.step(warmup, {}).predictions.empty() && run.pending().empty(),
            "El calentamiento creó predicciones pendientes");
    auto decision = cohort(0);
    decision.cursor = 1;
    const auto issued = run.step(decision, {});
    require(issued.predictions.front().value == 3, "El calentamiento no avanzó el estado rápido");
    Cohort settlement{2, phase_end, {}};
    settlement.kind = EventKind::settlement;
    const std::array crossing{Feedback{issued.predictions.front().id, 0, phase_end, 1}};
    rejected([&] { run.step(settlement, crossing); }, "El label cruza el corte de la fase");
    settlement.close_phase = true;
    rejected([&] { run.step(settlement, {}); }, "Se adelantó el cierre administrativo");
    settlement.cutoff = phase_close;
    settlement.close_phase = false;
    rejected([&] { run.step(settlement, {}); },
             "El settlement terminal dejó la fase sin cierre posible");
    settlement.close_phase = true;
    require(run.step(settlement, {}).finalized.size() == 2, "El rechazo impidió cerrar la fase");
}
void financial_recovery_preserves_exclusions_and_finalizations() {
    TemporaryDirectory directory;
    const std::array boundaries{Boundary::before_predictions, Boundary::predictions_ready,
                                Boundary::record_written,     Boundary::feedback_applied,
                                Boundary::checkpoint_written, Boundary::before_commit,
                                Boundary::committed};
    for (const bool close : {false, true}) {
        for (const auto boundary : boundaries) {
            const auto output =
                directory.path /
                (std::to_string(close) + "-" + std::to_string(static_cast<unsigned int>(boundary)));
            auto event = financial_decision();
            if (close) {
                event = {1, phase_close, {}};
                event.kind = EventKind::settlement;
                event.close_phase = true;
            }
            {
                Executor run(output, financial_definition(), financial_callbacks());
                if (close) {
                    run.step(financial_decision(), {});
                }
                rejected(
                    [&] {
                        run.step(event, {}, 1, [boundary](Boundary current) {
                            if (current == boundary) {
                                throw std::runtime_error("Corte financiero de prueba");
                            }
                        });
                    },
                    "La frontera financiera no interrumpió la transacción");
            }
            Executor resumed(output, financial_definition(), financial_callbacks(), true);
            if (boundary != Boundary::committed) {
                resumed.step(event, {});
            }
            require(resumed.snapshot().at("excluded") == 1 &&
                        resumed.snapshot().at("finalized") == (close ? 1 : 0),
                    "La recuperación perdió o duplicó una resolución");
        }
    }
}
} // namespace

int main() {
    try {
        whole_cohort_precedes_feedback_and_preserves_the_original_prediction();
        asset_order_batch_size_and_future_suffix_preserve_the_prefix();
        recovery_at_every_boundary_matches_an_uninterrupted_run();
        incompatible_identities_corrupt_files_and_concurrent_writers_fail_closed();
        admission_rejects_future_duplicate_nonfinite_and_revised_inputs();
        capacity_and_retention_remain_bounded();
        spliced_counters_are_rejected_even_if_checkpoint_checksums_are_recomputed();
        process_death_recovers_the_correct_generation();
        invalid_callback_outputs_do_not_become_confirmed_state();
        prepared_state_is_advanced_once_and_feedback_follows_all_predictions();
        prepared_state_recovers_at_every_boundary();
        invalid_prepared_outputs_and_mixed_contracts_are_rejected();
        prepared_mode_accepts_the_full_candidate_group_without_relaxing_classic_limits();
        financial_exclusion_preserves_emission_and_recovery();
        financial_warmup_and_phase_boundaries_are_explicit();
        financial_recovery_preserves_exclusions_and_finalizations();
        std::cout << "Comprobaciones de cohortes completadas\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
