#include "mars_titan/simulation_files.hpp"

#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <arrow/memory_pool.h>
#include <arrow/util/thread_pool.h>
#include <parquet/arrow/reader.h>
#include <parquet/metadata.h>
#include <parquet/properties.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace mars_titan::simulation {
namespace {
using Json = nlohmann::json;
constexpr std::array<std::string_view, 10> columns{
    "open",  "high",       "low",       "close",           "volume",
    "score", "close_time", "open_time", "prediction_time", "asset"};
constexpr std::size_t price_columns = 5;
constexpr std::size_t numerical_columns = 6;
constexpr std::size_t calendar_columns = 3;
constexpr std::size_t footer_bytes = 8;
constexpr std::size_t maximum_footer_bytes = 4 * bytes_per_mebibyte;
constexpr std::size_t sha256_characters = 64;
constexpr std::size_t bits_per_byte = 8;
constexpr std::size_t maximum_asset_characters = 96;
constexpr int64_t reader_buffer_bytes = 64 * static_cast<int64_t>(bytes_per_kibibyte);
constexpr int32_t maximum_metadata_items = 100'000;
constexpr int64_t reader_batch_rows = 4096;

void require_arrow(const arrow::Status& status) {
    if (!status.ok()) {
        throw std::runtime_error("Arrow/Parquet: " + status.ToString());
    }
}

template <typename T> T arrow_value(arrow::Result<T> value) {
    require_arrow(value.status());
    return std::move(value).ValueOrDie();
}

std::size_t positive_count(const Json& value, std::size_t maximum) {
    if (!value.is_number_unsigned()) {
        throw std::invalid_argument("El manifiesto necesita recuentos enteros positivos");
    }
    const auto result = value.get<std::size_t>();
    if (result == 0 || result > maximum) {
        throw std::invalid_argument("El recuento excede el presupuesto del lector");
    }
    return result;
}

bool valid_digest(const Json& value) {
    if (!value.is_string()) {
        return false;
    }
    const auto& text = value.get_ref<const std::string&>();
    return text.size() == sha256_characters && std::ranges::all_of(text, [](char ch) {
               return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
           });
}

void validate_footer(std::string_view bytes) {
    if (bytes.size() < footer_bytes || bytes.substr(0, 4) != "PAR1" ||
        bytes.substr(bytes.size() - 4) != "PAR1") {
        throw std::invalid_argument("El archivo no tiene una cabecera y un pie Parquet válidos");
    }
    uint32_t length = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        const auto byte = static_cast<uint32_t>(
            static_cast<unsigned char>(bytes[bytes.size() - footer_bytes + i]));
        length |= byte << (i * bits_per_byte);
    }
    if (length == 0 || length > maximum_footer_bytes || length > bytes.size() - footer_bytes) {
        throw std::invalid_argument("El pie Parquet excede el presupuesto de metadatos");
    }
}

std::vector<CorporateAction> actions_from(const Json& values,
                                          const std::vector<std::string>& assets) {
    if (!values.is_array() || values.size() > maximum_actions) {
        throw std::invalid_argument("Las acciones corporativas exceden su contrato");
    }
    std::vector<CorporateAction> result;
    result.reserve(values.size());
    for (const auto& value : values) {
        const auto asset = value.at("asset").get<std::string>();
        const auto position = std::ranges::find(assets, asset);
        if (position == assets.end()) {
            throw std::invalid_argument("La acción corporativa no pertenece a los activos");
        }
        CorporateAction action;
        action.id = value.at("id").get<std::string>();
        action.asset = static_cast<std::size_t>(position - assets.begin());
        const auto kind = value.at("kind").get<std::string>();
        if (kind == "split") {
            action.kind = CorporateKind::split;
        } else if (kind == "dividend") {
            action.kind = CorporateKind::dividend;
        } else if (kind == "writeoff") {
            action.kind = CorporateKind::writeoff;
        } else {
            throw std::invalid_argument("La acción corporativa tiene un tipo desconocido");
        }
        action.effective_at = read_json_int64(value.at("effective_at"));
        action.value = value.at("value").get<double>();
        action.verified = value.at("verified").get<bool>();
        if (!value.at("pay_at").is_null()) {
            action.pay_at = read_json_int64(value.at("pay_at"));
        }
        result.push_back(std::move(action));
    }
    return result;
}

void read_batches(parquet::arrow::FileReader& reader, MarketTape& tape, std::size_t row_count) {
    std::shared_ptr<arrow::Schema> schema;
    require_arrow(reader.GetSchema(&schema));
    if (schema->num_fields() != static_cast<int>(columns.size())) {
        throw std::invalid_argument("El Parquet no conserva las diez columnas previstas");
    }
    std::array<int, columns.size()> indices{};
    for (std::size_t i = 0; i < columns.size(); ++i) {
        indices.at(i) = schema->GetFieldIndex(std::string(columns.at(i)));
        const auto wanted = i < numerical_columns                      ? arrow::Type::DOUBLE
                            : i < numerical_columns + calendar_columns ? arrow::Type::INT64
                                                                       : arrow::Type::STRING;
        if (indices.at(i) < 0 || schema->field(indices.at(i))->type()->id() != wanted) {
            throw std::invalid_argument("La columna " + std::string(columns.at(i)) +
                                        " no conserva su tipo");
        }
    }
    auto batches = arrow_value(reader.GetRecordBatchReader());
    std::size_t position = 0;
    std::size_t decoded_bytes = 0;
    while (auto batch = arrow_value(batches->Next())) {
        require_arrow(batch->ValidateFull());
        if (batch->num_rows() <= 0 ||
            static_cast<uint64_t>(batch->num_rows()) > row_count - position) {
            throw std::invalid_argument("Las filas decodificadas exceden la población declarada");
        }
        std::array<std::shared_ptr<arrow::DoubleArray>, numerical_columns> numeric{};
        std::array<std::shared_ptr<arrow::Int64Array>, calendar_columns> calendar{};
        for (std::size_t i = 0; i < numeric.size(); ++i) {
            numeric.at(i) =
                std::static_pointer_cast<arrow::DoubleArray>(batch->column(indices.at(i)));
        }
        for (std::size_t i = 0; i < calendar.size(); ++i) {
            calendar.at(i) = std::static_pointer_cast<arrow::Int64Array>(
                batch->column(indices.at(numerical_columns + i)));
            if (calendar.at(i)->null_count() != 0) {
                throw std::invalid_argument("El calendario contiene valores ausentes");
            }
        }
        const auto names =
            std::static_pointer_cast<arrow::StringArray>(batch->column(indices.back()));
        if (names->null_count() != 0) {
            throw std::invalid_argument("Falta un identificador de activo");
        }
        for (int64_t row = 0; row < batch->num_rows(); ++row, ++position) {
            const auto asset = position % tape.assets.size();
            const auto session = position / tape.assets.size();
            const auto name = names->GetView(row);
            if (name != tape.assets[asset]) {
                throw std::invalid_argument(
                    "La cohorte no conserva el orden ni la población de activos");
            }
            decoded_bytes += numerical_columns * sizeof(double) +
                             calendar_columns * sizeof(int64_t) + name.size() + sizeof(int32_t);
            if (decoded_bytes > maximum_market_bytes) {
                throw std::invalid_argument("Las columnas decodificadas exceden 256 MiB");
            }
            for (std::size_t i = 0; i < numeric.size(); ++i) {
                const double value = numeric.at(i)->IsNull(row)
                                         ? std::numeric_limits<double>::quiet_NaN()
                                         : numeric.at(i)->Value(row);
                if (std::isinf(value)) {
                    throw std::invalid_argument("El Parquet contiene valores infinitos");
                }
                if (i < price_columns) {
                    tape.prices[position * price_columns + i] = value;
                } else {
                    tape.scores[position] = value;
                }
            }
            const std::array times{calendar[0]->Value(row), calendar[1]->Value(row),
                                   calendar[2]->Value(row)};
            if (asset == 0) {
                tape.close_times[session] = times[0];
                tape.open_times[session] = times[1];
                tape.prediction_times[session] = times[2];
            } else if (tape.close_times[session] != times[0] ||
                       tape.open_times[session] != times[1] ||
                       tape.prediction_times[session] != times[2]) {
                throw std::invalid_argument("Las filas de una cohorte no comparten calendario");
            }
        }
    }
    if (position != row_count) {
        throw std::invalid_argument("El Parquet termina antes de completar la población");
    }
}
} // namespace

std::shared_ptr<const MarketTape> load_market_tape(const std::filesystem::path& directory) {
    const auto manifest_bytes =
        read_bounded_file(directory / "manifest.json", maximum_manifest_bytes);
    const auto manifest = parse_bounded_json(manifest_bytes);
    if (read_json_int64(manifest.at("schema_version")) != 1 ||
        manifest.at("final_test_opened") != false) {
        throw std::invalid_argument("El manifiesto no conserva el formato o el test cerrado");
    }
    const auto assets = positive_count(manifest.at("assets"), maximum_assets);
    const auto sessions = positive_count(manifest.at("sessions"), maximum_sessions);
    if (sessions < 2 || assets > maximum_rows / sessions ||
        !valid_digest(manifest.at("file_sha256")) || !valid_digest(manifest.at("tape_sha256"))) {
        throw std::invalid_argument("El volumen o las huellas del escenario no son válidos");
    }
    const auto& identity = manifest.at("identity");
    if (identity.at("domain") != "synthetic") {
        throw std::invalid_argument(
            "El ejecutable solo admite escenarios sintéticos, sin acreditar histórico real");
    }
    const auto partition = identity.at("partition").get<std::string>();
    if (partition != "train" && partition != "validation") {
        throw std::invalid_argument("La partición no está admitida. El test permanece cerrado");
    }
    auto tape = std::make_shared<MarketTape>();
    tape->assets = identity.at("assets").get<std::vector<std::string>>();
    if (tape->assets.size() != assets || !std::ranges::is_sorted(tape->assets) ||
        std::ranges::any_of(tape->assets,
                            [](const auto& name) {
                                return name.empty() || name.size() > maximum_asset_characters;
                            }) ||
        std::adjacent_find(tape->assets.begin(), tape->assets.end()) != tape->assets.end()) {
        throw std::invalid_argument("El manifiesto necesita activos únicos y ordenados");
    }
    tape->currency = identity.at("currency").get<std::string>();
    tape->domain = "synthetic";
    tape->partition = partition;
    tape->parent_id = identity.at("parent_id").get<std::string>();
    tape->source_sha256 = content_sha256(manifest_bytes);
    if (identity.at("actions") != manifest.at("actions")) {
        throw std::invalid_argument("Las acciones corporativas no conservan su identidad");
    }
    tape->actions = actions_from(manifest.at("actions"), tape->assets);
    auto parquet_bytes = read_bounded_file(directory / "market.parquet", maximum_market_bytes);
    if (positive_count(manifest.at("file_bytes"), maximum_market_bytes) != parquet_bytes.size() ||
        content_sha256(parquet_bytes) != manifest.at("file_sha256").get<std::string>()) {
        throw std::invalid_argument("El archivo Parquet no conserva su tamaño o SHA256");
    }
    validate_footer(parquet_bytes);
    require_arrow(arrow::SetCpuThreadPoolCapacity(1));
    arrow::ProxyMemoryPool accounting(arrow::default_memory_pool());
    arrow::CappedMemoryPool pool(&accounting, static_cast<int64_t>(maximum_market_bytes));
    parquet::ReaderProperties properties(&pool);
    properties.enable_buffered_stream();
    properties.set_buffer_size(reader_buffer_bytes);
    properties.set_thrift_string_size_limit(static_cast<int32_t>(maximum_footer_bytes));
    properties.set_thrift_container_size_limit(maximum_metadata_items);
    properties.set_page_checksum_verification(true);
    parquet::ArrowReaderProperties arrow_properties(false);
    arrow_properties.set_pre_buffer(false);
    arrow_properties.set_batch_size(reader_batch_rows);
    parquet::arrow::FileReaderBuilder builder;
    auto input = std::make_shared<arrow::io::BufferReader>(
        arrow::Buffer::FromString(std::move(parquet_bytes)));
    require_arrow(builder.Open(input, properties));
    builder.memory_pool(&pool)->properties(arrow_properties);
    auto reader = arrow_value(builder.Build());
    const auto metadata = reader->parquet_reader()->metadata();
    const auto rows = assets * sessions;
    if (metadata->num_rows() != static_cast<int64_t>(rows) ||
        metadata->num_columns() != static_cast<int>(columns.size()) ||
        metadata->num_row_groups() <= 0 ||
        metadata->num_row_groups() > static_cast<int>(maximum_sessions)) {
        throw std::invalid_argument("Los metadatos Parquet no conservan las filas y columnas");
    }
    int64_t decoded = 0;
    for (int group = 0; group < metadata->num_row_groups(); ++group) {
        const auto size = metadata->RowGroup(group)->total_byte_size();
        if (size < 0 || size > static_cast<int64_t>(maximum_market_bytes) - decoded) {
            throw std::invalid_argument("Los grupos Parquet exceden el presupuesto descomprimido");
        }
        decoded += size;
    }
    tape->prices.resize(rows * price_columns);
    tape->scores.resize(rows);
    tape->close_times.resize(sessions);
    tape->open_times.resize(sessions);
    tape->prediction_times.resize(sessions);
    read_batches(*reader, *tape, rows);
    tape->validate();
    return tape;
}
} // namespace mars_titan::simulation
