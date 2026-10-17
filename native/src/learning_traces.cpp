#include "mars_titan/learning_traces.hpp"
#include "mars_titan/simulation_files.hpp"

#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mars_titan::learning {
namespace {
using Json = nlohmann::json;
constexpr std::size_t maximum_name_length = 96;
constexpr std::size_t maximum_parts = 65536;
constexpr std::size_t maximum_manifest_bytes = std::size_t{16} * 1024 * 1024;
constexpr std::array<std::string_view, 7> trace_stats{"mean", "std",          "min",
                                                      "max",  "abs_max",      "l2",
                                                      "finite_fraction"};
constexpr std::array<std::string_view, 6> trace_columns{"step",  "phase", "metric",
                                                        "group", "stat",  "value"};
constexpr std::array<std::string_view, 6> step_units{"optimizer_step", "write_event",
                                                     "boosting_round", "fit",
                                                     "transition",     "wave"};

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::invalid_argument(std::string(message));
    }
}

void check(const arrow::Status& status) {
    if (!status.ok()) {
        throw std::runtime_error("Arrow/Parquet: " + status.ToString());
    }
}

template <typename T> T arrow_value(arrow::Result<T> value) {
    check(value.status());
    return std::move(value).ValueOrDie();
}

// La misma expresión que `_NAME` en Python: [a-z][a-z0-9_.:/-]{0,95}.
bool valid_name(std::string_view name) {
    const auto allowed = [](char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '.' ||
               ch == ':' || ch == '/' || ch == '-';
    };
    return !name.empty() && name.size() <= maximum_name_length && name.front() >= 'a' &&
           name.front() <= 'z' && std::ranges::all_of(name, allowed);
}

std::string part_name(std::size_t sequence) {
    return std::format("part-{:08}.parquet", sequence);
}

Json commit_json(const TraceCommit& commit) { return Json::array({commit.major, commit.minor}); }

TraceCommit read_commit(const Json& value) {
    require(value.is_array() && value.size() == 2 && value[0].is_number_unsigned() &&
                value[1].is_number_unsigned(),
            "La marca de una parte de trazas no es válida");
    return {value[0].get<uint64_t>(), value[1].get<uint64_t>()};
}

struct Row {
    int64_t step = 0;
    std::string phase;
    std::string metric;
    std::string group;
    double value = 0;
};

std::string encode(const std::vector<Row>& rows) {
    arrow::Int64Builder steps;
    arrow::StringBuilder phases;
    arrow::StringBuilder metrics;
    arrow::StringBuilder groups;
    arrow::StringBuilder stats;
    arrow::DoubleBuilder values;
    for (const auto& row : rows) {
        check(steps.Append(row.step));
        check(phases.Append(row.phase));
        check(metrics.Append(row.metric));
        check(groups.Append(row.group));
        check(stats.Append("value"));
        check(values.Append(row.value));
    }
    // Campos anulables, como el esquema de pyarrow, aunque ninguna fila sea nula.
    const auto schema = arrow::schema({arrow::field("step", arrow::int64()),
                                       arrow::field("phase", arrow::utf8()),
                                       arrow::field("metric", arrow::utf8()),
                                       arrow::field("group", arrow::utf8()),
                                       arrow::field("stat", arrow::utf8()),
                                       arrow::field("value", arrow::float64())});
    const auto table = arrow::Table::Make(
        schema, {arrow_value(steps.Finish()), arrow_value(phases.Finish()),
                 arrow_value(metrics.Finish()), arrow_value(groups.Finish()),
                 arrow_value(stats.Finish()), arrow_value(values.Finish())});
    const auto output = arrow_value(arrow::io::BufferOutputStream::Create());
    parquet::WriterProperties::Builder properties;
    properties.compression(parquet::Compression::ZSTD);
    parquet::ArrowWriterProperties::Builder arrow_properties;
    arrow_properties.store_schema();
    check(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), output,
                                     static_cast<int64_t>(std::max<std::size_t>(rows.size(), 1)),
                                     properties.build(), arrow_properties.build()));
    return arrow_value(output->Finish())->ToString();
}
} // namespace

struct LearningTraceWriter::Impl {
    struct Part {
        std::size_t sequence = 0;
        TraceCommit commit;
        std::size_t bytes = 0;
    };
    std::filesystem::path directory;
    std::filesystem::path parts_directory;
    LearningTraceOptions options;
    Json identity;
    std::string unit;
    std::vector<Part> parts;
    std::vector<Row> pending;
    std::size_t total = 0;
    std::optional<uint64_t> exhausted;
    std::optional<TraceCommit> exhausted_commit;

    Json manifest() const {
        Json listed = Json::array();
        for (const auto& part : parts) {
            listed.push_back(Json{{"sequence", part.sequence},
                                  {"commit", commit_json(part.commit)},
                                  {"bytes", part.bytes}});
        }
        Json stats = Json::array();
        for (const auto name : trace_stats) {
            stats.push_back(name);
        }
        Json columns = Json::array();
        for (const auto name : trace_columns) {
            columns.push_back(name);
        }
        return Json{{"schema_version", 1},
                    {"kind", "learning_traces"},
                    {"identity", identity},
                    {"config", Json{{"every", options.every},
                                    {"max_bytes", options.max_bytes},
                                    {"enabled", true}}},
                    {"step_unit", unit},
                    {"stats", stats},
                    {"columns", columns},
                    {"exhausted_at_step", exhausted ? Json(*exhausted) : Json(nullptr)},
                    {"native", Json{{"parts", listed},
                                    {"exhausted_commit", exhausted_commit
                                                             ? commit_json(*exhausted_commit)
                                                             : Json(nullptr)}}}};
    }

    void publish() const {
        simulation::atomic_json_file(directory / "manifest.json", manifest(),
                                     maximum_manifest_bytes);
    }

    // Partes confirmadas del manifiesto anterior. El resto se borra.
    void recover(const TraceCommit& confirmed) {
        const auto saved = simulation::parse_bounded_json(
            simulation::read_bounded_file(directory / "manifest.json", maximum_manifest_bytes));
        auto expected = manifest();
        require(saved.is_object() && saved.value("kind", "") == "learning_traces" &&
                    saved.value("schema_version", 0) == 1 &&
                    saved.at("identity") == expected.at("identity") &&
                    saved.at("config") == expected.at("config") &&
                    saved.at("step_unit") == expected.at("step_unit"),
                "Las trazas existentes son de otra ejecución, cadencia o unidad");
        const auto& native = saved.at("native");
        require(native.is_object() && native.at("parts").is_array() &&
                    native.at("parts").size() <= maximum_parts,
                "El manifiesto de trazas no conserva sus partes nativas");
        TraceCommit previous;
        for (const auto& entry : native.at("parts")) {
            Part part{entry.at("sequence").get<std::size_t>(), read_commit(entry.at("commit")),
                      entry.at("bytes").get<std::size_t>()};
            require(part.sequence == parts.size() + 1 && part.commit >= previous,
                    "Las partes de trazas no conservan su orden");
            previous = part.commit;
            if (part.commit > confirmed) {
                break;
            }
            parts.push_back(part);
            total += part.bytes;
        }
        if (!native.at("exhausted_commit").is_null() &&
            read_commit(native.at("exhausted_commit")) <= confirmed) {
            exhausted_commit = read_commit(native.at("exhausted_commit"));
            exhausted = saved.at("exhausted_at_step").get<uint64_t>();
        }
        std::unordered_set<std::string> kept;
        for (const auto& part : parts) {
            kept.insert(part_name(part.sequence));
        }
        std::vector<std::filesystem::path> stale;
        std::size_t seen = 0;
        for (const auto& entry : std::filesystem::directory_iterator(parts_directory)) {
            require(++seen <= 2 * maximum_parts, "La carpeta de partes tiene demasiadas entradas");
            if (!kept.contains(entry.path().filename().string())) {
                stale.push_back(entry.path());
            }
        }
        // Partes sin confirmar y temporales de una escritura interrumpida. Un archivo ajeno
        // detiene la recuperación antes de borrar nada.
        require(std::ranges::all_of(stale, entry_is_part),
                "La carpeta de partes contiene un archivo ajeno");
        for (const auto& path : stale) {
            std::filesystem::remove(path);
        }
        for (const auto& part : parts) {
            require(std::filesystem::file_size(parts_directory / part_name(part.sequence)) ==
                        part.bytes,
                    "Falta una parte de trazas confirmada o cambió de tamaño");
        }
    }

    static bool entry_is_part(const std::filesystem::path& path) {
        const auto name = path.filename().string();
        return std::filesystem::is_regular_file(path) &&
               (name.starts_with("part-") || name.starts_with(".part-"));
    }
};

LearningTraceWriter::LearningTraceWriter(std::filesystem::path directory,
                                         LearningTraceOptions options, nlohmann::json identity,
                                         std::string step_unit, TraceCommit confirmed)
    : impl_(std::make_unique<Impl>()) {
    require(options.enabled() && options.max_bytes >= minimum_learning_trace_bytes &&
                options.max_bytes <= maximum_learning_trace_bytes,
            "Las trazas declaran una cadencia positiva y entre 1 KiB y 1 GiB");
    require(identity.is_object() && !identity.empty(), "Las trazas necesitan una identidad");
    require(std::ranges::find(step_units, step_unit) != step_units.end(),
            "Unidad de paso de trazas desconocida");
    impl_->directory = std::move(directory);
    impl_->parts_directory = impl_->directory / "parts";
    impl_->options = options;
    impl_->identity = std::move(identity);
    impl_->unit = std::move(step_unit);
    simulation::require_safe_path(impl_->directory);
    std::filesystem::create_directories(impl_->parts_directory);
    if (std::filesystem::exists(impl_->directory / "manifest.json")) {
        impl_->recover(confirmed);
    }
    impl_->publish();
}

LearningTraceWriter::~LearningTraceWriter() = default;

bool LearningTraceWriter::active() const noexcept { return !impl_->exhausted.has_value(); }

bool LearningTraceWriter::due(uint64_t previous, uint64_t current) const noexcept {
    const auto every = static_cast<uint64_t>(impl_->options.every);
    return active() && current / every > previous / every;
}

void LearningTraceWriter::scalar(uint64_t step, std::string_view phase, std::string_view metric,
                                 double value, std::string_view group) {
    if (!active()) {
        return;
    }
    require(step <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
                valid_name(phase) && valid_name(metric) && valid_name(group),
            "Paso o nombre de traza inválido");
    impl_->pending.push_back(Row{static_cast<int64_t>(step), std::string(phase),
                                 std::string(metric), std::string(group), value});
}

std::size_t LearningTraceWriter::pending() const noexcept { return impl_->pending.size(); }

void LearningTraceWriter::flush(TraceCommit commit) {
    auto rows = std::exchange(impl_->pending, {});
    if (!active() || rows.empty()) {
        return;
    }
    require(impl_->parts.empty() || commit >= impl_->parts.back().commit,
            "Las marcas de las partes de trazas deben crecer");
    require(impl_->parts.size() < maximum_parts, "Las trazas superan el número de partes");
    const auto bytes = encode(rows);
    if (impl_->total + bytes.size() > impl_->options.max_bytes) {
        // Se declara el primer paso que no cabe. Lo anterior ya está confirmado.
        const auto first = std::ranges::min_element(rows, {}, &Row::step)->step;
        impl_->exhausted = static_cast<uint64_t>(first);
        impl_->exhausted_commit = commit;
        impl_->publish();
        return;
    }
    const auto sequence = impl_->parts.size() + 1;
    simulation::atomic_binary_file(impl_->parts_directory / part_name(sequence), bytes,
                                   impl_->options.max_bytes);
    impl_->parts.push_back(Impl::Part{sequence, commit, bytes.size()});
    impl_->total += bytes.size();
    impl_->publish();
}

std::optional<uint64_t> LearningTraceWriter::exhausted_at() const noexcept {
    return impl_->exhausted;
}

std::size_t LearningTraceWriter::parts() const noexcept { return impl_->parts.size(); }

std::size_t LearningTraceWriter::bytes() const noexcept { return impl_->total; }
} // namespace mars_titan::learning
