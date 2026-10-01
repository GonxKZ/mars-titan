#include "mars_titan/ppo_inputs.hpp"
#include "mars_titan/simulation_files.hpp"

#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <arrow/memory_pool.h>
#include <parquet/arrow/reader.h>
#include <parquet/metadata.h>
#include <parquet/properties.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mars_titan::learning {
namespace {
using Json = nlohmann::json;
using simulation::bytes_per_mebibyte;
constexpr std::size_t maximum_context_bytes = 64 * bytes_per_mebibyte;
constexpr std::size_t maximum_decoded_bytes = 128 * bytes_per_mebibyte;
constexpr std::size_t maximum_footer_bytes = 4 * bytes_per_mebibyte;
constexpr std::size_t maximum_identifier_bytes = 256;
constexpr std::size_t digest_characters = 64;
constexpr std::size_t footer_bytes = 8;
constexpr std::size_t magic_bytes = 4;
constexpr std::size_t bits_per_byte = 8;
constexpr int32_t maximum_metadata_items = 100'000;
constexpr int64_t reader_buffer_bytes = 64 * static_cast<int64_t>(simulation::bytes_per_kibibyte);
constexpr int64_t reader_batch_rows = 4096;
constexpr std::size_t session_column = 0;
constexpr std::size_t feature_column = 1;
constexpr std::size_t value_column = 2;
constexpr std::size_t present_column = 3;
constexpr std::size_t available_column = 4;
constexpr std::array<std::string_view, 5> columns{"session", "feature", "value", "present",
                                                  "available_at"};
constexpr std::array types{arrow::Type::INT32, arrow::Type::INT32, arrow::Type::FLOAT,
                           arrow::Type::BOOL, arrow::Type::INT64};

void require_arrow(const arrow::Status& status) {
    if (!status.ok()) {
        throw std::runtime_error("Arrow/Parquet: " + status.ToString());
    }
}

template <typename T> T arrow_value(arrow::Result<T> result) {
    require_arrow(result.status());
    return std::move(result).ValueOrDie();
}

void require_fields(const Json& object, std::initializer_list<std::string_view> fields) {
    if (!object.is_object() || object.size() != fields.size() ||
        std::ranges::any_of(fields, [&object](std::string_view field) {
            return !object.contains(std::string(field));
        })) {
        throw std::invalid_argument(
            "El manifiesto de contexto contiene campos distintos del contrato");
    }
}

bool valid_digest(const Json& value) {
    if (!value.is_string()) {
        return false;
    }
    const auto& digest = value.get_ref<const std::string&>();
    return digest.size() == digest_characters && std::ranges::all_of(digest, [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

std::vector<simulation::ContextField> read_fields(const Json& fields) {
    if (!fields.is_array() || fields.empty() ||
        fields.size() > simulation::maximum_context_fields) {
        throw std::invalid_argument("El contexto necesita entre uno y 512 campos");
    }
    std::vector<simulation::ContextField> result;
    result.reserve(fields.size());
    std::unordered_set<std::string> names;
    for (const auto& field : fields) {
        require_fields(field, {"name", "unit"});
        if (!field.at("name").is_string() || !field.at("unit").is_string()) {
            throw std::invalid_argument(
                "Los campos de contexto necesitan nombre y unidad de texto");
        }
        simulation::ContextField parsed{field.at("name").get<std::string>(),
                                        field.at("unit").get<std::string>()};
        if (parsed.name.empty() || parsed.unit.empty() ||
            parsed.name.size() > maximum_identifier_bytes ||
            parsed.unit.size() > maximum_identifier_bytes || !names.insert(parsed.name).second) {
            throw std::invalid_argument("El contexto necesita nombres únicos y unidades acotadas");
        }
        result.push_back(std::move(parsed));
    }
    return result;
}

void validate_footer(std::string_view bytes) {
    if (bytes.size() < footer_bytes || bytes.substr(0, magic_bytes) != "PAR1" ||
        bytes.substr(bytes.size() - magic_bytes) != "PAR1") {
        throw std::invalid_argument("El contexto no tiene cabecera y pie Parquet válidos");
    }
    uint32_t length = 0;
    for (std::size_t index = 0; index < magic_bytes; ++index) {
        const auto byte = static_cast<uint32_t>(
            static_cast<unsigned char>(bytes[bytes.size() - footer_bytes + index]));
        length |= byte << (index * bits_per_byte);
    }
    if (length == 0 || length > maximum_footer_bytes || length > bytes.size() - footer_bytes) {
        throw std::invalid_argument(
            "El pie Parquet del contexto supera el presupuesto de metadatos");
    }
}

void validate_metadata(const parquet::FileMetaData& metadata, std::size_t rows) {
    if (metadata.num_rows() != static_cast<int64_t>(rows) ||
        metadata.num_columns() != static_cast<int>(columns.size()) ||
        metadata.num_row_groups() <= 0 ||
        metadata.num_row_groups() > static_cast<int>(simulation::maximum_sessions)) {
        throw std::invalid_argument(
            "El contexto no conserva las filas, columnas o grupos previstos");
    }
    int64_t remaining = static_cast<int64_t>(rows);
    int64_t decoded = 0;
    for (int group_index = 0; group_index < metadata.num_row_groups(); ++group_index) {
        const auto group = metadata.RowGroup(group_index);
        if (group->num_rows() <= 0 || group->num_rows() > remaining) {
            throw std::invalid_argument("Los grupos del contexto no conservan sus filas");
        }
        remaining -= group->num_rows();
        int64_t group_bytes = 0;
        for (int column_index = 0; column_index < metadata.num_columns(); ++column_index) {
            const auto column = group->ColumnChunk(column_index);
            const auto size = column->total_uncompressed_size();
            if (!column->file_path().empty() || size < 0 ||
                size > static_cast<int64_t>(maximum_decoded_bytes) - decoded) {
                throw std::invalid_argument(
                    "El contexto usa rutas externas o supera 128 MiB decodificados");
            }
            decoded += size;
            group_bytes += size;
        }
        if (group->total_byte_size() != group_bytes) {
            throw std::invalid_argument("Los tamaños declarados de los grupos no concuerdan");
        }
    }
    if (remaining != 0) {
        throw std::invalid_argument("Los grupos del contexto no completan la matriz declarada");
    }
}

std::vector<int> column_indices(parquet::arrow::FileReader& reader) {
    std::shared_ptr<arrow::Schema> schema;
    require_arrow(reader.GetSchema(&schema));
    if (schema->num_fields() != static_cast<int>(columns.size())) {
        throw std::invalid_argument("El contexto no conserva las cinco columnas previstas");
    }
    std::vector<int> result;
    result.reserve(columns.size());
    for (std::size_t index = 0; index < columns.size(); ++index) {
        const int position = schema->GetFieldIndex(std::string(columns.at(index)));
        if (position < 0 || schema->field(position)->type()->id() != types.at(index)) {
            throw std::invalid_argument("La columna " + std::string(columns.at(index)) +
                                        " no conserva su nombre o tipo");
        }
        result.push_back(position);
    }
    return result;
}

void read_rows(parquet::arrow::FileReader& reader, simulation::ContextTape& context,
               std::size_t rows) {
    const auto indices = column_indices(reader);
    context.values.resize(rows);
    std::size_t position = 0;
    for (int group = 0; group < reader.num_row_groups(); ++group) {
        auto batches = arrow_value(reader.GetRecordBatchReader({group}, indices));
        while (const auto batch = arrow_value(batches->Next())) {
            require_arrow(batch->ValidateFull());
            if (batch->num_rows() <= 0 ||
                static_cast<uint64_t>(batch->num_rows()) > rows - position) {
                throw std::invalid_argument("El contexto decodificado supera las filas previstas");
            }
            for (const auto& column : batch->columns()) {
                if (column->null_count() != 0) {
                    throw std::invalid_argument(
                        "El contexto necesita máscaras explícitas, sin NULL");
                }
            }
            const auto sessions =
                std::static_pointer_cast<arrow::Int32Array>(batch->column(session_column));
            const auto features =
                std::static_pointer_cast<arrow::Int32Array>(batch->column(feature_column));
            const auto values =
                std::static_pointer_cast<arrow::FloatArray>(batch->column(value_column));
            const auto present =
                std::static_pointer_cast<arrow::BooleanArray>(batch->column(present_column));
            const auto available =
                std::static_pointer_cast<arrow::Int64Array>(batch->column(available_column));
            for (int64_t row = 0; row < batch->num_rows(); ++row, ++position) {
                if (sessions->Value(row) !=
                        static_cast<int32_t>(position / context.fields.size()) ||
                    features->Value(row) !=
                        static_cast<int32_t>(position % context.fields.size())) {
                    throw std::invalid_argument(
                        "El contexto no conserva el orden denso de sesiones y campos");
                }
                context.values[position] = {values->Value(row), present->Value(row),
                                            available->Value(row)};
            }
        }
    }
    if (position != rows) {
        throw std::invalid_argument("El contexto termina antes de completar su matriz");
    }
}

void read_parquet(std::string bytes, simulation::ContextTape& context, std::size_t rows) {
    validate_footer(bytes);
    arrow::ProxyMemoryPool accounting(arrow::default_memory_pool());
    arrow::CappedMemoryPool pool(&accounting, static_cast<int64_t>(maximum_decoded_bytes));
    parquet::ReaderProperties properties(&pool);
    properties.enable_buffered_stream();
    properties.set_buffer_size(reader_buffer_bytes);
    properties.set_thrift_string_size_limit(static_cast<int32_t>(maximum_footer_bytes));
    properties.set_thrift_container_size_limit(maximum_metadata_items);
    properties.set_page_checksum_verification(true);
    parquet::ArrowReaderProperties arrow_properties(false);
    arrow_properties.set_pre_buffer(false);
    arrow_properties.set_batch_size(reader_batch_rows);
    auto input =
        std::make_shared<arrow::io::BufferReader>(arrow::Buffer::FromString(std::move(bytes)));
    parquet::arrow::FileReaderBuilder builder;
    require_arrow(builder.Open(input, properties));
    builder.memory_pool(&pool)->properties(arrow_properties);
    const auto reader = arrow_value(builder.Build());
    validate_metadata(*reader->parquet_reader()->metadata(), rows);
    read_rows(*reader, context, rows);
}
} // namespace

std::optional<simulation::ContextTape> load_ppo_context(const std::filesystem::path& directory,
                                                        const simulation::MarketTape& market) {
    const auto path = directory / "context.json";
    simulation::require_safe_path(path);
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }
    const auto bytes = simulation::read_bounded_file(path, simulation::maximum_manifest_bytes);
    const auto manifest = simulation::parse_bounded_json(bytes);
    require_fields(manifest,
                   {"schema_version", "domain", "market_manifest_sha256", "fields", "file"});
    if (simulation::read_json_int64(manifest.at("schema_version")) != 1 ||
        manifest.at("domain") != market.domain ||
        manifest.at("market_manifest_sha256") != market.source_sha256 ||
        market.close_times.size() < 2 || market.close_times.size() > simulation::maximum_sessions) {
        throw std::invalid_argument(
            "El contexto no corresponde al formato o a la cinta de mercado");
    }
    simulation::ContextTape context;
    context.source_sha256 = simulation::content_sha256(bytes);
    context.fields = read_fields(manifest.at("fields"));
    const auto rows = market.close_times.size() * context.fields.size();
    if (rows > maximum_decoded_bytes / sizeof(simulation::ContextValue)) {
        throw std::invalid_argument("La matriz de contexto supera 128 MiB de estado");
    }
    const auto& file = manifest.at("file");
    require_fields(file, {"path", "sha256", "bytes"});
    const auto expected_bytes = simulation::read_json_int64(file.at("bytes"));
    if (file.at("path") != "context.parquet" || !valid_digest(file.at("sha256")) ||
        expected_bytes <= 0 || expected_bytes > static_cast<int64_t>(maximum_context_bytes)) {
        throw std::invalid_argument("El archivo de contexto no conserva ruta, tamaño o SHA256");
    }
    auto data = simulation::read_bounded_file(directory / "context.parquet", maximum_context_bytes);
    if (static_cast<int64_t>(data.size()) != expected_bytes ||
        simulation::content_sha256(data) != file.at("sha256").get_ref<const std::string&>()) {
        throw std::invalid_argument("El Parquet de contexto no conserva sus bytes verificados");
    }
    read_parquet(std::move(data), context, rows);
    context.validate(market);
    return context;
}

simulation::BatchInput load_ppo_input(const std::filesystem::path& directory) {
    auto tape = simulation::load_market_tape(directory);
    auto context = load_ppo_context(directory, *tape);
    return {std::move(tape), {}, std::move(context)};
}

} // namespace mars_titan::learning
