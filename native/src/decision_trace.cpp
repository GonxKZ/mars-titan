#include "mars_titan/decision_trace.hpp"
#include "mars_titan/simulation_files.hpp"

#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <arrow/memory_pool.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mars_titan::learning {
namespace {
using Json = nlohmann::json;
using simulation::atomic_binary_file;
using simulation::content_sha256;
using simulation::parse_bounded_json;
using simulation::read_bounded_file;
constexpr std::size_t digest_width = 64;
constexpr std::size_t maximum_index_bytes = std::size_t{4} * 1024 * 1024;
constexpr std::size_t maximum_shard_bytes = std::size_t{1} * 1024 * 1024;
constexpr std::size_t arrow_memory_bytes = std::size_t{8} * 1024 * 1024;
constexpr std::size_t index_record_reserve = 512;
constexpr std::size_t maximum_directory_entries = 65536;
constexpr double probability_tolerance = 1e-5;

void require(bool value, std::string_view message) {
    if (!value) {
        throw std::invalid_argument(std::string(message));
    }
}

bool digest_valid(std::string_view value) {
    return value.size() == digest_width && std::ranges::all_of(value, [](char ch) {
               return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
           });
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

void validate_record(const DecisionRecord& record, uint64_t expected) {
    require(record.decision_id == expected && record.decision_id > 0 &&
                record.lane < simulation::maximum_assets && digest_valid(record.world_sha256) &&
                digest_valid(record.context_sha256) && record.decision_at >= 0 &&
                record.outcome_at > record.decision_at && record.action < trace_action_count &&
                record.retrieved_count <= trace_memory_count && std::isfinite(record.critic) &&
                std::isfinite(record.reward) &&
                (!record.costs_delta ||
                 (std::isfinite(*record.costs_delta) && *record.costs_delta >= 0)),
            "La decisión contiene identificadores, tiempos o valores inválidos");
    require(record.mode == "sampled" || record.mode == "greedy" || record.mode == "warmup",
            "El modo de la decisión no está admitido");
    require(record.mode != "warmup" || (!record.learning_allowed && record.action == 1),
            "El calentamiento debe conservar efectivo y excluir la decisión del aprendizaje");
    double total = 0;
    for (const auto probability : record.probabilities) {
        require(std::isfinite(probability) && probability >= 0 && probability <= 1,
                "La distribución de acciones contiene probabilidades inválidas");
        total += static_cast<double>(probability);
    }
    require(std::abs(total - 1) <= probability_tolerance,
            "Las probabilidades de acción no suman uno");
    for (std::size_t index = 0; index < record.retrieved_count; ++index) {
        require(std::isfinite(record.similarities.at(index)) && record.matured_at.at(index) >= 0 &&
                    record.matured_at.at(index) <= record.decision_at,
                "La recuperación contiene recuerdos no maduros o similitudes no finitas");
    }
    require(record.reward_valid || record.reward == 0,
            "Una recompensa sin valoración necesita el marcador cero");
    if (record.memory_sensitivity) {
        const auto& sensitivity = *record.memory_sensitivity;
        require(sensitivity.action < trace_action_count &&
                    sensitivity.action_changed == (sensitivity.action != record.action) &&
                    std::isfinite(sensitivity.probability_l1),
                "La sensibilidad contiene una acción o diferencia incoherente");
        double masked_total = 0;
        double distance = 0;
        for (std::size_t index = 0; index < trace_action_count; ++index) {
            const auto probability = sensitivity.probabilities.at(index);
            require(std::isfinite(probability) && probability >= 0 && probability <= 1,
                    "La política con recuerdos ocultos contiene probabilidades inválidas");
            masked_total += static_cast<double>(probability);
            distance += std::abs(static_cast<double>(probability) -
                                 static_cast<double>(record.probabilities.at(index)));
        }
        require(std::abs(masked_total - 1) <= probability_tolerance &&
                    std::abs(distance - sensitivity.probability_l1) <= probability_tolerance,
                "La sensibilidad no conserva las distribuciones observadas");
    }
}

template <typename Builder, typename Getter>
std::shared_ptr<arrow::Array> column(std::span<const DecisionRecord> records, Getter getter) {
    Builder builder;
    for (const auto& record : records) {
        check(builder.Append(getter(record)));
    }
    return arrow_value(builder.Finish());
}

std::shared_ptr<arrow::Table> table(std::span<const DecisionRecord> records,
                                    const std::string& identity) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    const auto add = [&](std::string name, std::shared_ptr<arrow::Array> values) {
        fields.push_back(arrow::field(std::move(name), values->type(), false));
        arrays.push_back(std::move(values));
    };
    add("decision_id",
        column<arrow::UInt64Builder>(records, [](const auto& r) { return r.decision_id; }));
    add("lane", column<arrow::UInt32Builder>(records, [](const auto& r) { return r.lane; }));
    add("world_sha256",
        column<arrow::StringBuilder>(records, [](const auto& r) { return r.world_sha256; }));
    add("context_sha256",
        column<arrow::StringBuilder>(records, [](const auto& r) { return r.context_sha256; }));
    add("episode", column<arrow::UInt64Builder>(records, [](const auto& r) { return r.episode; }));
    add("cursor", column<arrow::UInt64Builder>(records, [](const auto& r) { return r.cursor; }));
    add("optimizer_step",
        column<arrow::UInt64Builder>(records, [](const auto& r) { return r.optimizer_step; }));
    add("decision_at",
        column<arrow::Int64Builder>(records, [](const auto& r) { return r.decision_at; }));
    add("outcome_at",
        column<arrow::Int64Builder>(records, [](const auto& r) { return r.outcome_at; }));
    add("action", column<arrow::UInt8Builder>(records, [](const auto& r) { return r.action; }));
    add("mode", column<arrow::StringBuilder>(records, [](const auto& r) { return r.mode; }));
    add("learning_allowed",
        column<arrow::BooleanBuilder>(records, [](const auto& r) { return r.learning_allowed; }));
    for (std::size_t index = 0; index < trace_action_count; ++index) {
        add("probability_" + std::to_string(index),
            column<arrow::FloatBuilder>(
                records, [index](const auto& r) { return r.probabilities.at(index); }));
    }
    add("critic", column<arrow::DoubleBuilder>(records, [](const auto& r) { return r.critic; }));
    add("retrieved_count",
        column<arrow::UInt8Builder>(records, [](const auto& r) { return r.retrieved_count; }));
    for (std::size_t index = 0; index < trace_memory_count; ++index) {
        const auto suffix = std::to_string(index);
        add("memory_id_" + suffix, column<arrow::UInt64Builder>(records, [index](const auto& r) {
                return r.ids.at(index);
            }));
        add("similarity_" + suffix, column<arrow::DoubleBuilder>(records, [index](const auto& r) {
                return index < r.retrieved_count ? r.similarities.at(index) : 0.;
            }));
        add("matured_at_" + suffix, column<arrow::Int64Builder>(records, [index](const auto& r) {
                return index < r.retrieved_count ? r.matured_at.at(index) : int64_t{0};
            }));
    }
    add("reward", column<arrow::DoubleBuilder>(records, [](const auto& r) { return r.reward; }));
    add("reward_valid",
        column<arrow::BooleanBuilder>(records, [](const auto& r) { return r.reward_valid; }));
    add("terminated",
        column<arrow::BooleanBuilder>(records, [](const auto& r) { return r.terminated; }));
    add("truncated",
        column<arrow::BooleanBuilder>(records, [](const auto& r) { return r.truncated; }));
    add("costs_delta", column<arrow::DoubleBuilder>(
                           records, [](const auto& r) { return r.costs_delta.value_or(0); }));
    add("costs_present", column<arrow::BooleanBuilder>(
                             records, [](const auto& r) { return r.costs_delta.has_value(); }));
    for (std::size_t index = 0; index < trace_action_count; ++index) {
        add("memory_masked_probability_" + std::to_string(index),
            column<arrow::FloatBuilder>(records, [index](const auto& r) {
                return r.memory_sensitivity ? r.memory_sensitivity->probabilities.at(index) : 0.F;
            }));
    }
    add("memory_masked_action", column<arrow::UInt8Builder>(records, [](const auto& r) {
            return r.memory_sensitivity ? r.memory_sensitivity->action : uint8_t{0};
        }));
    add("memory_probability_l1", column<arrow::DoubleBuilder>(records, [](const auto& r) {
            return r.memory_sensitivity ? r.memory_sensitivity->probability_l1 : 0.;
        }));
    add("memory_action_changed", column<arrow::BooleanBuilder>(records, [](const auto& r) {
            return r.memory_sensitivity && r.memory_sensitivity->action_changed;
        }));
    add("memory_sensitivity_present", column<arrow::BooleanBuilder>(records, [](const auto& r) {
            return r.memory_sensitivity.has_value();
        }));
    return arrow::Table::Make(
        arrow::schema(fields, arrow::key_value_metadata({"trace_identity_sha256", "schema_version"},
                                                        {identity, "1"})),
        arrays);
}

std::string encode(const arrow::Table& rows) {
    arrow::ProxyMemoryPool accounting(arrow::default_memory_pool());
    arrow::CappedMemoryPool pool(&accounting, static_cast<int64_t>(arrow_memory_bytes));
    const auto output = arrow_value(arrow::io::BufferOutputStream::Create(0, &pool));
    parquet::WriterProperties::Builder properties;
    properties.compression(parquet::Compression::ZSTD);
    properties.enable_page_checksum();
    parquet::ArrowWriterProperties::Builder arrow_properties;
    arrow_properties.store_schema();
    check(parquet::arrow::WriteTable(rows, &pool, output, static_cast<int64_t>(trace_shard_records),
                                     properties.build(), arrow_properties.build()));
    const auto bytes = arrow_value(output->Finish())->ToString();
    if (bytes.size() > maximum_shard_bytes) {
        throw TraceCapacityError("El bloque de trazas supera su presupuesto de 1 MiB");
    }
    return bytes;
}

Json shard_record(std::string_view bytes, uint64_t first, uint64_t last) {
    const auto hash = content_sha256(bytes);
    return Json{{"path", "trace-" + hash + ".parquet"},
                {"sha256", hash},
                {"bytes", bytes.size()},
                {"first", first},
                {"last", last},
                {"rows", last - first + 1}};
}

std::string prefix(std::string bytes, const Json& record, const std::string& identity,
                   uint64_t retained) {
    const auto rows = record.at("rows").get<uint64_t>();
    const auto first = record.at("first").get<uint64_t>();
    constexpr std::size_t footer_bytes = 8;
    constexpr std::size_t magic_bytes = 4;
    constexpr int32_t maximum_metadata_items = 8192;
    require(bytes.size() >= footer_bytes && bytes.substr(0, magic_bytes) == "PAR1" &&
                bytes.substr(bytes.size() - magic_bytes) == "PAR1",
            "El bloque de trazas no es un Parquet válido");
    arrow::ProxyMemoryPool accounting(arrow::default_memory_pool());
    arrow::CappedMemoryPool pool(&accounting, static_cast<int64_t>(arrow_memory_bytes));
    parquet::ReaderProperties properties(&pool);
    properties.set_thrift_string_size_limit(static_cast<int32_t>(maximum_shard_bytes));
    properties.set_thrift_container_size_limit(maximum_metadata_items);
    properties.set_page_checksum_verification(true);
    parquet::ArrowReaderProperties arrow_properties(false);
    arrow_properties.set_pre_buffer(false);
    auto input =
        std::make_shared<arrow::io::BufferReader>(arrow::Buffer::FromString(std::move(bytes)));
    parquet::arrow::FileReaderBuilder builder;
    check(builder.Open(input, properties));
    builder.memory_pool(&pool)->properties(arrow_properties);
    const auto reader = arrow_value(builder.Build());
    const auto metadata = reader->parquet_reader()->metadata();
    const auto expected = table({}, identity)->schema();
    require(
        metadata->num_rows() == static_cast<int64_t>(rows) && rows <= trace_shard_records &&
            metadata->num_row_groups() == 1 && metadata->num_columns() == expected->num_fields() &&
            metadata->RowGroup(0)->total_byte_size() >= 0 &&
            metadata->RowGroup(0)->total_byte_size() <= static_cast<int64_t>(arrow_memory_bytes),
        "Los metadatos de la traza exceden las filas o la memoria admitidas");
    for (int index = 0; index < metadata->num_columns(); ++index) {
        require(metadata->RowGroup(0)->ColumnChunk(index)->file_path().empty(),
                "El bloque de trazas contiene una ruta externa");
    }
    std::shared_ptr<arrow::Schema> schema;
    check(reader->GetSchema(&schema));
    require(schema->Equals(*expected, true),
            "El bloque de trazas no conserva esquema e identidad: " + schema->ToString(true) +
                " frente a " + expected->ToString(true));
    const auto decoded = arrow_value(reader->ReadTable());
    check(decoded->ValidateFull());
    for (const auto& values : decoded->columns()) {
        require(values->null_count() == 0, "La traza contiene valores NULL no admitidos");
    }
    uint64_t offset = 0;
    for (const auto& chunk : decoded->GetColumnByName("decision_id")->chunks()) {
        const auto ids = std::static_pointer_cast<arrow::UInt64Array>(chunk);
        for (int64_t row = 0; row < ids->length(); ++row, ++offset) {
            require(ids->Value(row) == first + offset,
                    "El bloque de trazas contiene IDs incoherentes");
        }
    }
    return retained == 0 ? std::string{}
                         : encode(*decoded->Slice(0, static_cast<int64_t>(retained)));
}
} // namespace

struct TraceWriter::Impl {
    struct LedgerBytes {
        std::size_t parquet = 0;
        std::size_t entries = 0;
        std::size_t index = 0;
    };
    std::filesystem::path directory;
    std::string identity;
    std::size_t budget;
    simulation::OutputLock lock;
    std::string identity_bytes;
    std::string index_bytes;
    Json shards = Json::array();
    std::vector<DecisionRecord> pending;
    uint64_t accepted = 0;
    std::size_t reserved = 0;
    LedgerBytes ledger;
    bool unpublished = false;

    Impl(const std::filesystem::path& output, std::string digest, bool resume,
         std::size_t byte_budget)
        : directory(output), identity(std::move(digest)), budget(byte_budget),
          lock(output, resume) {
        require(digest_valid(identity) && budget > 0 && budget <= default_trace_bytes,
                "La identidad o el presupuesto de trazas no es válido");
        identity_bytes =
            Json{{"schema_version", 1}, {"identity_sha256", identity}, {"budget_bytes", budget}}
                .dump();
        if (!resume) {
            index_bytes = index(shards);
            capacity(measure(shards), false);
            atomic_binary_file(directory / "identity.json", identity_bytes, maximum_index_bytes,
                               false);
            atomic_binary_file(directory / "trace-index.json", index_bytes, maximum_index_bytes,
                               false);
        } else {
            require(read_bounded_file(directory / "identity.json", maximum_index_bytes) ==
                        identity_bytes,
                    "La traza pertenece a otra identidad o presupuesto");
            index_bytes = read_bounded_file(directory / "trace-index.json", maximum_index_bytes);
            const auto envelope = parse_bounded_json(index_bytes);
            const auto& payload = envelope.at("payload");
            require(content_sha256(payload.dump()) == envelope.at("sha256").get<std::string>() &&
                        payload.at("schema_version") == 1 &&
                        payload.at("identity_sha256") == identity &&
                        payload.at("shards").is_array(),
                    "El índice de trazas no conserva su integridad");
            shards = payload.at("shards");
            uint64_t next = 1;
            for (const auto& record : shards) {
                const auto first = number(record.at("first"));
                const auto last = number(record.at("last"));
                require(first == next && last >= first && last - first < trace_shard_records &&
                            number(record.at("rows")) == last - first + 1,
                        "El índice contiene huecos o bloques demasiado grandes");
                static_cast<void>(prefix(load(record), record, identity, 0));
                next = last + 1;
            }
            accepted = next - 1;
            require(number(payload.at("cursor")) == accepted,
                    "El cursor de trazas no concuerda con los bloques");
        }
        ledger = measure(shards);
        reserved = capacity(ledger, false);
        pending.reserve(trace_shard_records);
    }

    static uint64_t number(const Json& value) {
        const auto parsed = simulation::read_json_int64(value);
        require(parsed >= 0, "La traza contiene un contador negativo");
        return static_cast<uint64_t>(parsed);
    }

    std::string index(const Json& records, std::optional<uint64_t> through = std::nullopt) const {
        const auto cursor =
            through.value_or(records.empty() ? uint64_t{0} : number(records.back().at("last")));
        const Json payload{{"schema_version", 1},
                           {"identity_sha256", identity},
                           {"cursor", cursor},
                           {"shards", records}};
        return Json{{"payload", payload}, {"sha256", content_sha256(payload.dump())}}.dump();
    }

    static std::size_t entry_bytes(const Json& record) {
        const auto bytes = record.dump().size();
        constexpr std::size_t maximum_cursor_chars = 20;
        require(bytes <= index_record_reserve - maximum_cursor_chars - 1,
                "El registro de un bloque supera la reserva del índice");
        return bytes;
    }

    LedgerBytes measure(const Json& records) const {
        LedgerBytes result;
        for (const auto& record : records) {
            const auto bytes = number(record.at("bytes"));
            if (bytes > budget - result.parquet) {
                throw TraceCapacityError("Los bloques de trazas superan su presupuesto");
            }
            result.parquet += static_cast<std::size_t>(bytes);
            result.entries += entry_bytes(record);
        }
        const auto cursor = records.empty() ? uint64_t{0} : number(records.back().at("last"));
        // JSON mide el sobre y cada registro. Solo se añaden las comas del array.
        result.index = index(Json::array(), cursor).size() + result.entries +
                       (records.empty() ? 0 : records.size() - 1);
        return result;
    }

    LedgerBytes extend(const Json& record) const {
        auto result = ledger;
        result.parquet += static_cast<std::size_t>(number(record.at("bytes")));
        result.entries += entry_bytes(record);
        result.index =
            index(Json::array(), number(record.at("last"))).size() + result.entries + shards.size();
        return result;
    }

    std::size_t capacity(const LedgerBytes& candidate, bool pending_rows) const {
        const auto index_size = std::max(candidate.index, index_bytes.size());
        std::size_t total = identity_bytes.size() + index_size;
        if (index_size > maximum_index_bytes || total > budget) {
            throw TraceCapacityError("El índice de trazas supera su presupuesto");
        }
        if (candidate.parquet > budget - total) {
            throw TraceCapacityError(
                "La traza alcanzó su presupuesto. Se necesita una pausa recuperable");
        }
        total += candidate.parquet;
        if (pending_rows) {
            constexpr auto reservation = maximum_shard_bytes + index_record_reserve;
            if (reservation > budget - total ||
                index_size > maximum_index_bytes - index_record_reserve) {
                throw TraceCapacityError(
                    "La traza no tiene espacio reservado para el bloque pendiente");
            }
            total += reservation;
        }
        return total;
    }

    std::string load(const Json& record) const {
        const auto hash = record.at("sha256").get<std::string>();
        require(digest_valid(hash) && record.at("path") == "trace-" + hash + ".parquet" &&
                    number(record.at("bytes")) > 0 &&
                    number(record.at("bytes")) <= maximum_shard_bytes,
                "El bloque de trazas tiene una ruta o tamaño no admitidos");
        const auto bytes = read_bounded_file(directory / record.at("path").get<std::string>(),
                                             maximum_shard_bytes);
        require(bytes.size() == number(record.at("bytes")) && content_sha256(bytes) == hash,
                "El bloque de trazas no conserva tamaño y SHA256");
        return bytes;
    }

    bool write_shard(const Json& record, std::string_view bytes) const {
        const auto path = directory / record.at("path").get<std::string>();
        simulation::require_safe_path(path);
        if (std::filesystem::exists(path)) {
            require(load(record) == bytes, "El bloque previo no coincide con la traza");
            return false;
        }
        try {
            atomic_binary_file(path, bytes, maximum_shard_bytes, false);
        } catch (...) {
            // La publicación puede fallar después de crear el enlace. Solo se retira su contenido
            // exacto.
            if (std::filesystem::exists(path) && load(record) == bytes) {
                std::filesystem::remove(path);
            }
            throw;
        }
        return true;
    }

    void remove_orphans() const {
        std::unordered_set<std::string> retained;
        for (const auto& record : shards) {
            retained.insert(record.at("path").get<std::string>());
        }
        std::size_t visited = 0;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            require(++visited <= maximum_directory_entries,
                    "La carpeta de trazas contiene demasiados archivos para recuperarla");
            const auto name = entry.path().filename().string();
            constexpr std::size_t prefix_bytes = 6;
            constexpr std::size_t suffix_bytes = 8;
            if (retained.contains(name) ||
                name.size() != prefix_bytes + digest_width + suffix_bytes ||
                !name.starts_with("trace-") || !name.ends_with(".parquet") ||
                !digest_valid(std::string_view(name).substr(prefix_bytes, digest_width))) {
                continue;
            }
            auto bytes = read_bounded_file(entry.path(), maximum_shard_bytes);
            require(content_sha256(bytes) == name.substr(prefix_bytes, digest_width),
                    "Un bloque huérfano de trazas no conserva su SHA256");
            parquet::ReaderProperties properties;
            properties.set_thrift_string_size_limit(static_cast<int32_t>(maximum_shard_bytes));
            constexpr int32_t maximum_metadata_items = 8192;
            properties.set_thrift_container_size_limit(maximum_metadata_items);
            const auto input = std::make_shared<arrow::io::BufferReader>(
                arrow::Buffer::FromString(std::move(bytes)));
            const auto reader = parquet::ParquetFileReader::Open(input, properties);
            const auto metadata = reader->metadata()->key_value_metadata();
            const auto identity_key = metadata ? metadata->FindKey("trace_identity_sha256") : -1;
            if (identity_key >= 0 && metadata->value(identity_key) == identity) {
                // Solo se retiran bloques inmutables de esta identidad que el índice no usa.
                std::filesystem::remove(entry.path());
            }
        }
    }

    void publish(const Json& records, std::string_view bytes, const Json& added) {
        require(read_bounded_file(directory / "trace-index.json", maximum_index_bytes) ==
                    index_bytes,
                "El índice de trazas cambió durante la ejecución");
        auto next_index = index(records);
        std::optional<std::filesystem::path> created;
        try {
            if (!added.is_null()) {
                if (write_shard(added, bytes)) {
                    created = directory / added.at("path").get<std::string>();
                }
            }
            atomic_binary_file(directory / "trace-index.json", next_index, maximum_index_bytes);
        } catch (...) {
            if (read_bounded_file(directory / "trace-index.json", maximum_index_bytes) !=
                index_bytes) {
                atomic_binary_file(directory / "trace-index.json", index_bytes,
                                   maximum_index_bytes);
            }
            if (created) {
                std::filesystem::remove(*created);
            }
            throw;
        }
        index_bytes = std::move(next_index);
    }

    void append(std::span<const DecisionRecord> records) {
        require(!records.empty() && records.size() <= trace_shard_records &&
                    accepted <=
                        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - records.size(),
                "El lote de trazas está vacío o supera 128 decisiones");
        for (std::size_t offset = 0; offset < records.size(); ++offset) {
            validate_record(records[offset], accepted + offset + 1);
        }
        auto candidate = pending;
        candidate.insert(candidate.end(), records.begin(), records.end());
        if (candidate.size() < trace_shard_records) {
            auto size = reserved;
            if (pending.empty()) {
                constexpr auto reservation = maximum_shard_bytes + index_record_reserve;
                if (reservation > budget - size || std::max(ledger.index, index_bytes.size()) >
                                                       maximum_index_bytes - index_record_reserve) {
                    throw TraceCapacityError(
                        "La traza no tiene espacio reservado para el bloque pendiente");
                }
                size += reservation;
            }
            pending = std::move(candidate);
            accepted += records.size();
            reserved = size;
            return;
        }
        const auto full = encode(*table(
            std::span<const DecisionRecord>(candidate).first(trace_shard_records), identity));
        auto added = shard_record(full, candidate.front().decision_id,
                                  candidate[trace_shard_records - 1].decision_id);
        candidate.erase(candidate.begin(),
                        candidate.begin() + static_cast<std::ptrdiff_t>(trace_shard_records));
        const auto end = accepted + records.size();
        const auto next_ledger = extend(added);
        const auto size = capacity(next_ledger, !candidate.empty());
        auto& entries = shards.get_ref<Json::array_t&>();
        if (entries.size() == entries.capacity()) {
            entries.reserve(std::max(entries.size() + 1, entries.capacity() * 2));
        }
        static_cast<void>(write_shard(added, full));
        shards.push_back(std::move(added));
        ledger = next_ledger;
        pending = std::move(candidate);
        accepted = end;
        reserved = size;
        unpublished = true;
    }

    void flush() {
        if (pending.empty()) {
            if (unpublished) {
                publish(shards, {}, nullptr);
                unpublished = false;
                reserved = capacity(ledger, false);
            }
            return;
        }
        auto next = shards;
        const auto pending_bytes = encode(*table(pending, identity));
        const auto added =
            shard_record(pending_bytes, pending.front().decision_id, pending.back().decision_id);
        next.push_back(added);
        const auto next_ledger = extend(added);
        static_cast<void>(capacity(next_ledger, false));
        publish(next, pending_bytes, added);
        shards = std::move(next);
        ledger = next_ledger;
        pending.clear();
        unpublished = false;
        reserved = capacity(ledger, false);
    }

    void rewind(uint64_t confirmed) {
        require(confirmed <= accepted, "El checkpoint reclama decisiones que faltan en la traza");
        if (confirmed == accepted) {
            return;
        }
        auto kept = Json::array();
        std::vector<Json> retired;
        Json added = nullptr;
        std::string replacement;
        for (const auto& record : shards) {
            if (number(record.at("last")) <= confirmed) {
                kept.push_back(record);
                continue;
            }
            if (number(record.at("first")) <= confirmed) {
                const auto retained = confirmed - number(record.at("first")) + 1;
                replacement = prefix(load(record), record, identity, retained);
                added = shard_record(replacement, number(record.at("first")), confirmed);
                kept.push_back(added);
            }
            retired.push_back(record);
        }
        std::vector<DecisionRecord> tail;
        for (const auto& record : pending) {
            if (record.decision_id <= confirmed) {
                tail.push_back(record);
            }
        }
        const auto next_ledger = measure(kept);
        static_cast<void>(capacity(next_ledger, !tail.empty()));
        publish(kept, replacement, added);
        shards = std::move(kept);
        ledger = next_ledger;
        pending = std::move(tail);
        accepted = confirmed;
        unpublished = false;
        reserved = capacity(ledger, !pending.empty());
        for (const auto& record : retired) {
            static_cast<void>(load(record));
            std::filesystem::remove(directory / record.at("path").get<std::string>());
        }
    }
};

// El contrato público fija este orden. El cursor se contrasta con la traza y los bytes con el
// límite.
TraceWriter::TraceWriter(const std::filesystem::path& directory, std::string identity, bool resume,
                         // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                         uint64_t confirmed_cursor, std::size_t budget)
    : impl_(std::make_unique<Impl>(directory, std::move(identity), resume, budget)) {
    require(resume || confirmed_cursor == 0, "Una traza nueva necesita un cursor inicial cero");
    if (resume) {
        impl_->rewind(confirmed_cursor);
        impl_->remove_orphans();
    }
}
TraceWriter::~TraceWriter() = default;
void TraceWriter::append(std::span<const DecisionRecord> records) { impl_->append(records); }
void TraceWriter::flush() { impl_->flush(); }
uint64_t TraceWriter::cursor() const noexcept { return impl_->accepted; }
std::size_t TraceWriter::bytes() const noexcept { return impl_->reserved; }
void TraceWriter::rewind_to(uint64_t cursor) { impl_->rewind(cursor); }
} // namespace mars_titan::learning
