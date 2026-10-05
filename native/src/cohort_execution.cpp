#include "mars_titan/cohort_execution.hpp"
#include "mars_titan/simulation_files.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

namespace mars_titan::cohorts {
namespace {
using simulation::atomic_binary_file;
using simulation::atomic_json_file;
using simulation::content_sha256;
using simulation::parse_bounded_json;
using simulation::read_bounded_file;
using simulation::read_json_int64;
constexpr std::size_t digest_width = 64;
constexpr std::size_t maximum_tasks = 16;
constexpr std::size_t maximum_assets = 4096;
constexpr std::size_t maximum_pending = 100000;
constexpr std::size_t maximum_cohorts = 1000000;
constexpr std::size_t maximum_json_depth = 32;
constexpr std::size_t control_bytes = std::size_t{64} * 1024;
constexpr std::size_t maximum_buffer = 64 * mebibyte;
constexpr std::size_t maximum_state = 16 * mebibyte;
constexpr std::size_t temporary_allowance = 64;
constexpr std::size_t prediction_fields = 7;
constexpr std::size_t checkpoint_fields = 10;
constexpr std::size_t record_fields = 11;
constexpr std::size_t head_fields = 5;

bool digest(std::string_view text) {
    return text.size() == digest_width && std::ranges::all_of(text, [](char value) {
               return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
           });
}
bool token(std::string_view text) {
    constexpr std::size_t maximum_token = 128;
    return !text.empty() && text.size() <= maximum_token &&
           std::ranges::all_of(text, [](char value) {
               return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
                      (value >= '0' && value <= '9') || value == '/' || value == '_' ||
                      value == '-' || value == '.' || value == ':';
           });
}
std::size_t count(const Json& value, std::size_t maximum) {
    const auto integer = read_json_int64(value);
    if (integer < 0 || static_cast<std::uint64_t>(integer) > maximum) {
        throw std::invalid_argument("El contador supera el presupuesto de la cohorte");
    }
    return static_cast<std::size_t>(integer);
}
double number(const Json& value) {
    if (!value.is_number() || !std::isfinite(value.get<double>())) {
        throw std::invalid_argument("La cohorte necesita valores numéricos finitos");
    }
    return value.get<double>();
}
void finite_json(const Json& value, std::size_t depth, std::size_t& remaining) {
    if (depth > maximum_json_depth || remaining == 0 || value.is_binary() || value.is_discarded()) {
        throw std::invalid_argument("El estado JSON excede su profundidad o su número de nodos");
    }
    --remaining;
    if (value.is_number_float() && !std::isfinite(value.get<double>())) {
        throw std::invalid_argument("El estado contiene un valor no finito");
    }
    if (value.is_structured()) {
        for (const auto& child : value) {
            finite_json(child, depth + 1, remaining);
        }
    }
}
std::string bounded_json(const Json& value, std::size_t maximum) {
    auto nodes = maximum;
    finite_json(value, 0, nodes);
    auto bytes = value.dump();
    if (bytes.size() > maximum) {
        throw std::invalid_argument("El estado o registro supera su presupuesto de bytes");
    }
    return bytes;
}
void check_state(const Json& state, const Limits& limits) {
    if (!state.is_object()) {
        throw std::invalid_argument("El estado explícito debe ser un objeto JSON");
    }
    static_cast<void>(bounded_json(state, limits.max_state_bytes));
}
void check_limits(const Limits& limits) {
    const auto positive = [](std::size_t value, std::uint64_t maximum) {
        return value > 0 && value <= maximum;
    };
    constexpr std::uint64_t maximum_log = std::uint64_t{16} * 1024 * mebibyte;
    if (!positive(limits.feature_width, maximum_assets) ||
        !positive(limits.max_assets, maximum_assets) ||
        !positive(limits.max_pending, maximum_pending) ||
        !positive(limits.max_state_bytes, maximum_state) ||
        !positive(limits.max_checkpoint_bytes, maximum_buffer) ||
        !positive(limits.max_record_bytes, maximum_buffer) ||
        !positive(limits.max_log_bytes, maximum_log) ||
        !positive(limits.max_cohorts, maximum_cohorts) ||
        limits.feature_width > maximum_buffer / sizeof(double) / limits.max_assets) {
        throw std::invalid_argument("Los límites del ejecutor no son válidos");
    }
}
Json configuration(Definition& definition) {
    check_limits(definition.limits);
    check_state(definition.initial_state, definition.limits);
    const auto& id = definition.identity;
    if (!digest(id.source_sha256) || !digest(id.view_sha256) || !digest(id.representation_sha256) ||
        !digest(id.model_sha256) || definition.tasks.empty() ||
        definition.tasks.size() > maximum_tasks) {
        throw std::invalid_argument(
            "Falta la identidad del origen, vista, representación o modelo");
    }
    std::ranges::sort(definition.tasks, {},
                      [](const Task& task) { return std::tie(task.name, task.horizon); });
    Json tasks = Json::array();
    std::set<std::pair<std::string, std::uint32_t>> keys;
    for (const auto& task : definition.tasks) {
        if (!token(task.name) || task.horizon == 0 || task.horizon > maximum_cohorts ||
            !keys.emplace(task.name, task.horizon).second) {
            throw std::invalid_argument("La tarea o su horizonte no tiene una identidad única");
        }
        tasks.push_back(Json{{"name", task.name}, {"horizon", task.horizon}});
    }
    const auto& limits = definition.limits;
    return Json{{"schema_version", 1},
                {"kind", "cohort_execution"},
                {"source_sha256", id.source_sha256},
                {"view_sha256", id.view_sha256},
                {"representation_sha256", id.representation_sha256},
                {"model_sha256", id.model_sha256},
                {"initial_state_sha256", content_sha256(definition.initial_state.dump())},
                {"label_revision", 0},
                {"tasks", std::move(tasks)},
                {"limits", Json{{"feature_width", limits.feature_width},
                                {"max_assets", limits.max_assets},
                                {"max_pending", limits.max_pending},
                                {"max_state_bytes", limits.max_state_bytes},
                                {"max_checkpoint_bytes", limits.max_checkpoint_bytes},
                                {"max_record_bytes", limits.max_record_bytes},
                                {"max_log_bytes", limits.max_log_bytes},
                                {"max_cohorts", limits.max_cohorts}}}};
}
std::vector<Observation> canonical_observations(const Cohort& cohort, const Limits& limits) {
    if (cohort.cutoff <= 0 || cohort.observations.empty() ||
        cohort.observations.size() > limits.max_assets) {
        throw std::invalid_argument("El corte o el número de activos no es válido");
    }
    for (const auto& row : cohort.observations) {
        if (!token(row.asset) || row.available_at < 0 || row.available_at > cohort.cutoff ||
            row.features.size() != limits.feature_width ||
            !std::ranges::all_of(row.features, [](double value) { return std::isfinite(value); })) {
            throw std::invalid_argument(
                "La observación es futura, no finita o tiene otra dimensión");
        }
    }
    auto result = cohort.observations;
    std::ranges::sort(result, {}, &Observation::asset);
    if (std::ranges::adjacent_find(result, {}, &Observation::asset) != result.end()) {
        throw std::invalid_argument("La cohorte contiene un activo duplicado");
    }
    return result;
}
Json observation_json(const Cohort& cohort, std::span<const Observation> rows) {
    Json values = Json::array();
    for (const auto& row : rows) {
        values.push_back(Json{
            {"asset", row.asset}, {"available_at", row.available_at}, {"features", row.features}});
    }
    return Json{
        {"cursor", cohort.cursor}, {"cutoff", cohort.cutoff}, {"observations", std::move(values)}};
}
std::string prediction_id(std::string_view identity, const Prediction& prediction) {
    return content_sha256(Json{{"kind", "decision"},
                               {"identity_sha256", identity},
                               {"decision_at", prediction.decision_at},
                               {"asset", prediction.asset},
                               {"task", prediction.task.name},
                               {"horizon", prediction.task.horizon}}
                              .dump());
}
std::string feedback_id(std::string_view prediction) {
    return content_sha256(
        Json{{"kind", "feedback"}, {"prediction_id", prediction}, {"revision", 0}}.dump());
}
Json prediction_json(const Prediction& prediction) {
    return Json{{"id", prediction.id},
                {"asset", prediction.asset},
                {"task", prediction.task.name},
                {"horizon", prediction.task.horizon},
                {"generation", prediction.generation},
                {"decision_at", prediction.decision_at},
                {"value", prediction.value}};
}
Prediction read_prediction(const Json& value, const Definition& definition,
                           std::string_view identity) {
    if (!value.is_object() || value.size() != prediction_fields) {
        throw std::invalid_argument("La predicción conservada no cumple su esquema");
    }
    Prediction result{value.at("id").get<std::string>(),
                      value.at("asset").get<std::string>(),
                      {value.at("task").get<std::string>(),
                       static_cast<std::uint32_t>(count(value.at("horizon"), maximum_cohorts))},
                      count(value.at("generation"), definition.limits.max_cohorts),
                      read_json_int64(value.at("decision_at")),
                      number(value.at("value"))};
    if (!token(result.asset) || result.generation == 0 || result.decision_at <= 0 ||
        std::ranges::find(definition.tasks, result.task) == definition.tasks.end() ||
        result.id != prediction_id(identity, result)) {
        throw std::invalid_argument("La predicción perdió su identidad o su tarea");
    }
    return result;
}
Json outcome_json(const ResolvedFeedback& outcome) {
    return Json{{"id", outcome.id},
                {"prediction", prediction_json(outcome.prediction)},
                {"revision", outcome.label.revision},
                {"available_at", outcome.label.available_at},
                {"target", outcome.label.value},
                {"error", outcome.label.value - outcome.prediction.value}};
}
struct Stored {
    std::size_t cursor = 0;
    std::int64_t last_at = 0;
    Json state;
    std::map<std::string, Prediction> pending;
    std::size_t issued = 0;
    std::size_t applied = 0;
    std::size_t log_bytes = 0;
    std::string record_sha256;
};
Json stored_json(const Stored& stored, std::string_view identity) {
    Json pending = Json::array();
    for (const auto& [key, prediction] : stored.pending) {
        static_cast<void>(key);
        pending.push_back(prediction_json(prediction));
    }
    return Json{{"schema_version", 1},           {"identity_sha256", identity},
                {"cursor", stored.cursor},       {"last_at", stored.last_at},
                {"state", stored.state},         {"pending", std::move(pending)},
                {"issued", stored.issued},       {"applied", stored.applied},
                {"log_bytes", stored.log_bytes}, {"record_sha256", stored.record_sha256}};
}
Stored read_stored(const Json& value, const Definition& definition, std::string_view identity) {
    const auto& limits = definition.limits;
    if (!value.is_object() || value.size() != checkpoint_fields ||
        value.at("schema_version") != 1 || value.at("identity_sha256") != identity ||
        !value.at("pending").is_array() || value.at("pending").size() > limits.max_pending) {
        throw std::invalid_argument("El checkpoint no conserva el esquema, identidad o capacidad");
    }
    check_state(value.at("state"), limits);
    Stored result;
    result.cursor = count(value.at("cursor"), limits.max_cohorts);
    result.last_at = read_json_int64(value.at("last_at"));
    result.state = value.at("state");
    result.issued = count(value.at("issued"), std::numeric_limits<std::size_t>::max());
    result.applied = count(value.at("applied"), result.issued);
    result.log_bytes = count(value.at("log_bytes"), limits.max_log_bytes);
    result.record_sha256 = value.at("record_sha256").get<std::string>();
    for (const auto& row : value.at("pending")) {
        auto prediction = read_prediction(row, definition, identity);
        if (prediction.generation > result.cursor || prediction.decision_at > result.last_at ||
            !result.pending.emplace(prediction.id, std::move(prediction)).second) {
            throw std::invalid_argument("Los pendientes mezclan generaciones o repiten decisiones");
        }
    }
    const auto maximum_issued =
        static_cast<std::uint64_t>(result.cursor) * limits.max_assets * definition.tasks.size();
    if (result.issued > maximum_issued || result.issued < result.cursor * definition.tasks.size() ||
        result.issued % definition.tasks.size() != 0 ||
        result.issued - result.applied != result.pending.size() ||
        (result.cursor == 0 && (result.last_at != 0 || result.issued != 0 ||
                                result.log_bytes != 0 || !result.record_sha256.empty())) ||
        (result.cursor > 0 && (result.last_at <= 0 || !digest(result.record_sha256)))) {
        throw std::invalid_argument("El checkpoint no concilia decisiones, pendientes y feedback");
    }
    return result;
}
std::filesystem::path generation_path(const std::filesystem::path& output, std::string_view kind,
                                      std::size_t generation) {
    return output / (std::string(kind) + "-" + std::to_string(generation) + ".json");
}
std::string seal(const std::filesystem::path& path, const Json& value, std::size_t maximum) {
    const auto bytes = bounded_json(value, maximum);
    simulation::require_safe_path(path);
    if (std::filesystem::exists(path)) {
        if (read_bounded_file(path, maximum) != bytes) {
            throw std::invalid_argument("El archivo inmutable pertenece a otra decisión o estado");
        }
    } else {
        atomic_binary_file(path, bytes, maximum, false);
    }
    return content_sha256(bytes);
}
void notify(const std::function<void(Boundary)>& fault, Boundary boundary) {
    if (fault) {
        fault(boundary);
    }
}
} // namespace

struct Executor::Impl {
    Impl(const std::filesystem::path& directory, Definition requested, Callbacks operators,
         bool resume)
        : output(directory), definition(std::move(requested)), callbacks(std::move(operators)) {
        const auto config = configuration(definition);
        identity = content_sha256(config.dump());
        if (!callbacks.predict || !callbacks.update) {
            throw std::invalid_argument("Falta una función de predicción o actualización");
        }
        lock = std::make_unique<simulation::OutputLock>(output, resume);
        const auto identity_file = output / "identity.json";
        if (resume) {
            if (parse_bounded_json(read_bounded_file(identity_file, control_bytes)) != config) {
                throw std::invalid_argument("La ejecución pertenece a otro origen, vista o modelo");
            }
        } else {
            atomic_json_file(identity_file, config, control_bytes);
        }
        if (resume && std::filesystem::exists(output / "latest.json")) {
            restore();
        } else {
            if (std::filesystem::exists(generation_path(output, "record", 1))) {
                throw std::invalid_argument("Hay decisiones sin una generación inicial confirmada");
            }
            stored.state = definition.initial_state;
            checkpoint_sha256 =
                seal(generation_path(output, "checkpoint", 0), stored_json(stored, identity),
                     definition.limits.max_checkpoint_bytes);
            publish_head(stored, checkpoint_sha256, "");
        }
        inventory();
        prune();
    }
    void healthy() const {
        if (failed || busy) {
            throw std::logic_error("La instancia necesita reapertura o está dentro de una cohorte");
        }
    }
    void publish_head(const Stored& next, std::string_view current,
                      std::string_view previous) const {
        atomic_json_file(output / "latest.json",
                         Json{{"schema_version", 1},
                              {"identity_sha256", identity},
                              {"generation", next.cursor},
                              {"checkpoint_sha256", current},
                              {"previous_sha256", previous}},
                         control_bytes);
    }
    Stored load_checkpoint(std::size_t generation, std::string_view checksum) const {
        if (!digest(checksum)) {
            throw std::invalid_argument("Falta la huella del checkpoint confirmado");
        }
        const auto bytes = read_bounded_file(generation_path(output, "checkpoint", generation),
                                             definition.limits.max_checkpoint_bytes);
        if (content_sha256(bytes) != checksum) {
            throw std::invalid_argument("La huella del checkpoint no coincide con latest.json");
        }
        auto result = read_stored(parse_bounded_json(bytes), definition, identity);
        if (result.cursor != generation) {
            throw std::invalid_argument("El checkpoint pertenece a otra generación");
        }
        return result;
    }
    Json load_record(std::size_t generation, std::string_view checksum) const {
        const auto bytes = read_bounded_file(generation_path(output, "record", generation),
                                             definition.limits.max_record_bytes);
        if (content_sha256(bytes) != checksum) {
            throw std::invalid_argument("La huella del registro de decisiones ha cambiado");
        }
        auto record = parse_bounded_json(bytes);
        if (!record.is_object() || record.size() != record_fields ||
            record.at("schema_version") != 1 || record.at("identity_sha256") != identity ||
            count(record.at("generation"), definition.limits.max_cohorts) != generation ||
            !record.at("predictions").is_array() || !record.at("feedback").is_array() ||
            !digest(record.at("observations_sha256").get<std::string>())) {
            throw std::invalid_argument("El registro no identifica la generación confirmada");
        }
        const auto previous = record.at("previous_sha256").get<std::string>();
        if ((generation == 1 && !previous.empty()) || (generation > 1 && !digest(previous))) {
            throw std::invalid_argument("La cadena de decisiones está rota");
        }
        return record;
    }
    void check_transition(const Stored& before, const Json& record, const Stored& after) const {
        if (record.at("previous_sha256") != before.record_sha256 ||
            record.at("state_before_sha256") != content_sha256(before.state.dump()) ||
            read_json_int64(record.at("cutoff")) != after.last_at ||
            count(record.at("issued_total"), after.issued) != after.issued ||
            count(record.at("applied_total"), after.applied) != after.applied ||
            after.last_at <= before.last_at || after.issued < before.issued ||
            after.issued - before.issued != record.at("predictions").size() ||
            after.applied < before.applied ||
            after.applied - before.applied != record.at("feedback").size()) {
            throw std::invalid_argument("El checkpoint y el registro mezclan transiciones");
        }
        auto expected = before.pending;
        for (const auto& row : record.at("feedback")) {
            const auto& value = row.at("prediction");
            const auto id = value.at("id").get<std::string>();
            const auto previous = expected.find(id);
            if (previous == expected.end() || prediction_json(previous->second) != value ||
                row.at("id") != feedback_id(id) || row.at("revision") != 0 ||
                read_json_int64(row.at("available_at")) > after.last_at ||
                read_json_int64(row.at("available_at")) <= previous->second.decision_at ||
                number(row.at("error")) != number(row.at("target")) - previous->second.value) {
                throw std::invalid_argument(
                    "El feedback confirmado no conserva su predicción original");
            }
            expected.erase(previous);
        }
        for (const auto& row : record.at("predictions")) {
            auto prediction = read_prediction(row, definition, identity);
            if (prediction.generation != after.cursor || prediction.decision_at != after.last_at ||
                !expected.emplace(prediction.id, std::move(prediction)).second) {
                throw std::invalid_argument(
                    "La cohorte confirmada duplica o desplaza una decisión");
            }
        }
        Stored expected_queue;
        expected_queue.pending = std::move(expected);
        if (stored_json(expected_queue, identity).at("pending") !=
            stored_json(after, identity).at("pending")) {
            throw std::invalid_argument(
                "La cola no corresponde a las decisiones y feedback confirmados");
        }
    }
    void restore() {
        const auto head =
            parse_bounded_json(read_bounded_file(output / "latest.json", control_bytes));
        if (!head.is_object() || head.size() != head_fields || head.at("schema_version") != 1 ||
            head.at("identity_sha256") != identity) {
            throw std::invalid_argument("La cabecera de recuperación no identifica la ejecución");
        }
        const auto generation = count(head.at("generation"), definition.limits.max_cohorts);
        checkpoint_sha256 = head.at("checkpoint_sha256").get<std::string>();
        stored = load_checkpoint(generation, checkpoint_sha256);
        const auto previous = head.at("previous_sha256").get<std::string>();
        if (generation == 0) {
            if (!previous.empty() || stored.state != definition.initial_state) {
                throw std::invalid_argument("La generación inicial no conserva su estado");
            }
        } else {
            const auto before = load_checkpoint(generation - 1, previous);
            const auto record = load_record(generation, stored.record_sha256);
            check_transition(before, record, stored);
            const auto size = read_bounded_file(generation_path(output, "record", generation),
                                                definition.limits.max_record_bytes)
                                  .size();
            if (stored.log_bytes < before.log_bytes ||
                stored.log_bytes - before.log_bytes != size) {
                throw std::invalid_argument("Los bytes del registro no concilian");
            }
        }
    }
    void inventory() const {
        std::size_t entries = 0;
        for (const auto& entry : std::filesystem::directory_iterator(output)) {
            if (++entries > definition.limits.max_cohorts + temporary_allowance) {
                throw std::invalid_argument("El directorio excede el presupuesto de archivos");
            }
            const auto name = entry.path().filename().string();
            const bool temporary =
                name.find(".pending-") != std::string::npos &&
                (name.starts_with(".record-") || name.starts_with(".checkpoint-") ||
                 name.starts_with(".latest.json.") || name.starts_with(".identity.json.") ||
                 name.starts_with(".report.json."));
            if (temporary) {
                simulation::require_safe_path(entry.path());
                if (!entry.is_regular_file()) {
                    throw std::invalid_argument("Un temporal no es un archivo regular");
                }
                std::filesystem::remove(entry.path());
            }
        }
    }
    void prune() const {
        if (stored.cursor < 2) {
            return;
        }
        const auto obsolete = generation_path(output, "checkpoint", stored.cursor - 2);
        if (std::filesystem::exists(obsolete)) {
            const auto value = parse_bounded_json(
                read_bounded_file(obsolete, definition.limits.max_checkpoint_bytes));
            if (value.at("identity_sha256") != identity ||
                count(value.at("cursor"), definition.limits.max_cohorts) != stored.cursor - 2) {
                throw std::invalid_argument("El checkpoint obsoleto pertenece a otra ejecución");
            }
            std::filesystem::remove(obsolete);
        }
    }
    std::vector<ResolvedFeedback> resolve(std::span<const Feedback> labels,
                                          std::int64_t cutoff) const {
        if (labels.size() > stored.pending.size()) {
            throw std::invalid_argument("El feedback supera la cola pendiente");
        }
        std::vector<ResolvedFeedback> result;
        std::set<std::string> seen;
        for (const auto& label : labels) {
            const auto found = stored.pending.find(label.prediction_id);
            if (found == stored.pending.end() || !seen.insert(label.prediction_id).second ||
                label.revision != 0 || !std::isfinite(label.value) || label.available_at > cutoff ||
                label.available_at <= found->second.decision_at ||
                found->second.decision_at >= cutoff ||
                !std::isfinite(label.value - found->second.value)) {
                throw std::invalid_argument(
                    "El feedback es futuro, repetido, desconocido o tiene otra revisión");
            }
            result.push_back({feedback_id(label.prediction_id), found->second, label});
        }
        std::ranges::sort(result, {}, [](const ResolvedFeedback& item) {
            return std::tuple{item.label.available_at, item.prediction.decision_at,
                              item.prediction.asset, item.prediction.task.name,
                              item.prediction.task.horizon};
        });
        return result;
    }
    std::vector<Prediction> predict(const Cohort& cohort, std::span<const Observation> observations,
                                    std::size_t batch_rows) const {
        std::vector<Prediction> result;
        result.reserve(observations.size() * definition.tasks.size());
        for (const auto& task : definition.tasks) {
            for (std::size_t start = 0; start < observations.size(); start += batch_rows) {
                const auto batch =
                    observations.subspan(start, std::min(batch_rows, observations.size() - start));
                const auto values = callbacks.predict(batch, task, stored.state);
                if (values.size() != batch.size() || !std::ranges::all_of(values, [](double value) {
                        return std::isfinite(value);
                    })) {
                    throw std::invalid_argument(
                        "El predictor no devuelve una salida finita por activo");
                }
                for (std::size_t i = 0; i < values.size(); ++i) {
                    Prediction prediction{
                        "", batch[i].asset, task, stored.cursor + 1, cohort.cutoff, values[i]};
                    prediction.id = prediction_id(identity, prediction);
                    result.push_back(std::move(prediction));
                }
            }
        }
        std::ranges::sort(result, {}, [](const Prediction& prediction) {
            return std::tie(prediction.asset, prediction.task.name, prediction.task.horizon);
        });
        return result;
    }
    Commit step(const Cohort& cohort, std::span<const Feedback> labels, std::size_t batch_rows,
                const std::function<void(Boundary)>& fault) {
        healthy();
        const auto& limits = definition.limits;
        if (cohort.cursor != stored.cursor || cohort.cutoff <= stored.last_at ||
            stored.cursor >= limits.max_cohorts || batch_rows == 0 ||
            batch_rows > limits.max_assets) {
            throw std::invalid_argument(
                "El cursor, el orden temporal o el lote físico no es válido");
        }
        const auto observations = canonical_observations(cohort, limits);
        const auto emitted = observations.size() * definition.tasks.size();
        if (emitted > limits.max_pending - stored.pending.size() ||
            emitted > std::numeric_limits<std::size_t>::max() - stored.issued) {
            throw std::invalid_argument("La nueva cohorte supera la cola pendiente");
        }
        auto applied = resolve(labels, cohort.cutoff);
        const auto observations_sha = content_sha256(
            bounded_json(observation_json(cohort, observations), limits.max_record_bytes));
        busy = true;
        try {
            notify(fault, Boundary::before_predictions);
            auto predictions = predict(cohort, observations, batch_rows);
            notify(fault, Boundary::predictions_ready);
            Json forecast_rows = Json::array();
            Json feedback_rows = Json::array();
            for (const auto& prediction : predictions) {
                forecast_rows.push_back(prediction_json(prediction));
            }
            for (const auto& outcome : applied) {
                feedback_rows.push_back(outcome_json(outcome));
            }
            const auto generation = stored.cursor + 1;
            const Json record{{"schema_version", 1},
                              {"identity_sha256", identity},
                              {"generation", generation},
                              {"observations_sha256", observations_sha},
                              {"previous_sha256", stored.record_sha256},
                              {"predictions", std::move(forecast_rows)},
                              {"feedback", std::move(feedback_rows)},
                              {"state_before_sha256", content_sha256(stored.state.dump())},
                              {"cutoff", cohort.cutoff},
                              {"issued_total", stored.issued + predictions.size()},
                              {"applied_total", stored.applied + applied.size()}};
            const auto record_size = bounded_json(record, limits.max_record_bytes).size();
            if (record_size > limits.max_log_bytes - stored.log_bytes) {
                throw std::invalid_argument(
                    "El registro de decisiones agotó su presupuesto de disco");
            }
            const auto record_sha = seal(generation_path(output, "record", generation), record,
                                         limits.max_record_bytes);
            notify(fault, Boundary::record_written);
            auto next = stored;
            next.state = callbacks.update(stored.state, applied);
            check_state(next.state, limits);
            for (const auto& outcome : applied) {
                next.pending.erase(outcome.prediction.id);
            }
            for (const auto& prediction : predictions) {
                if (!next.pending.emplace(prediction.id, prediction).second) {
                    throw std::invalid_argument("La cohorte reintroduce una decisión pendiente");
                }
            }
            next.cursor = generation;
            next.last_at = cohort.cutoff;
            next.issued += predictions.size();
            next.applied += applied.size();
            next.log_bytes += record_size;
            next.record_sha256 = record_sha;
            notify(fault, Boundary::feedback_applied);
            const auto checksum = seal(generation_path(output, "checkpoint", generation),
                                       stored_json(next, identity), limits.max_checkpoint_bytes);
            notify(fault, Boundary::checkpoint_written);
            notify(fault, Boundary::before_commit);
            publish_head(next, checksum, checkpoint_sha256);
            stored = std::move(next);
            checkpoint_sha256 = checksum;
            notify(fault, Boundary::committed);
            prune();
            busy = false;
            return Commit{generation, std::move(predictions), std::move(applied), record_sha};
        } catch (...) {
            busy = false;
            failed = true;
            throw;
        }
    }
    std::filesystem::path output;
    Definition definition;
    Callbacks callbacks;
    std::unique_ptr<simulation::OutputLock> lock;
    std::string identity;
    std::string checkpoint_sha256;
    Stored stored;
    bool failed = false;
    bool busy = false;
};
Executor::Executor(const std::filesystem::path& output, Definition definition, Callbacks callbacks,
                   bool resume)
    : impl_(std::make_unique<Impl>(output, std::move(definition), std::move(callbacks), resume)) {}
Executor::~Executor() = default;
Commit Executor::step(const Cohort& cohort, std::span<const Feedback> labels,
                      std::size_t batch_rows, const std::function<void(Boundary)>& fault) {
    return impl_->step(cohort, labels, batch_rows, fault);
}
std::size_t Executor::cursor() const {
    impl_->healthy();
    return impl_->stored.cursor;
}
Json Executor::snapshot() const {
    impl_->healthy();
    return stored_json(impl_->stored, impl_->identity);
}
std::vector<Prediction> Executor::pending() const {
    impl_->healthy();
    std::vector<Prediction> result;
    result.reserve(impl_->stored.pending.size());
    for (const auto& [key, prediction] : impl_->stored.pending) {
        static_cast<void>(key);
        result.push_back(prediction);
    }
    return result;
}
Json Executor::record(std::size_t generation, std::size_t max_hops) const {
    impl_->healthy();
    if (generation == 0 || generation > impl_->stored.cursor || max_hops == 0 ||
        max_hops > maximum_cohorts || impl_->stored.cursor - generation >= max_hops) {
        throw std::invalid_argument(
            "La lectura histórica excede el prefijo confirmado o su presupuesto");
    }
    auto checksum = impl_->stored.record_sha256;
    for (auto current = impl_->stored.cursor; current >= generation; --current) {
        auto value = impl_->load_record(current, checksum);
        if (current == generation) {
            return value;
        }
        checksum = value.at("previous_sha256").get<std::string>();
    }
    throw std::logic_error("No se alcanzó la generación solicitada");
}
std::string replay_id(const ResolvedFeedback& feedback, std::uint64_t visit) {
    if (!digest(feedback.id) || feedback.id != feedback_id(feedback.prediction.id) ||
        feedback.label.revision != 0 || feedback.label.prediction_id != feedback.prediction.id) {
        throw std::invalid_argument("La exposición necesita un feedback identificado");
    }
    return content_sha256(
        Json{{"kind", "replay"}, {"feedback_id", feedback.id}, {"visit", visit}}.dump());
}
} // namespace mars_titan::cohorts
