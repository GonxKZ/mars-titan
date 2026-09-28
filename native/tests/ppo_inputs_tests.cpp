#include "mars_titan/ppo_inputs.hpp"
#include "mars_titan/simulation_files.hpp"

#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <parquet/arrow/writer.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <utility>
#include <vector>

namespace {
using Json = nlohmann::json;
using namespace mars_titan::simulation;
using mars_titan::learning::load_ppo_context;
using mars_titan::learning::load_ppo_input;
constexpr std::size_t fixture_limit = bytes_per_mebibyte;
constexpr std::size_t context_file_limit = 64 * bytes_per_mebibyte;
constexpr std::size_t sessions = 3;
constexpr std::size_t feature_count = 2;
constexpr std::size_t fixture_rows = sessions * feature_count;
constexpr std::size_t digest_characters = 64;
constexpr std::size_t footer_bytes = 8;
constexpr std::size_t bits_per_byte = 8;
constexpr uint32_t byte_mask = 0xff;
constexpr int64_t final_close = 5;
constexpr double fixture_price = 10;
constexpr double fixture_volume = 1000;
constexpr double fixture_score = 0.01;
constexpr float initial_context = 0.1F;
constexpr float middle_context = 0.2F;
constexpr float final_context = 0.3F;
constexpr float changed_context = 0.15F;

void require(bool condition, std::string_view reason) {
    if (!condition) {
        throw std::runtime_error(std::string(reason));
    }
}

template <typename Function> void rejected(Function&& action, std::string_view reason) {
    try {
        std::forward<Function>(action)();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(std::string(reason));
}

template <typename Function> void rejected_with(Function&& action, std::string_view expected) {
    try {
        std::forward<Function>(action)();
    } catch (const std::exception& error) {
        require(std::string_view(error.what()).find(expected) != std::string_view::npos,
                "El rechazo debe ocurrir antes de descomprimir el grupo");
        return;
    }
    throw std::runtime_error("Se aceptaron metadatos que exceden el contrato");
}

void check(const arrow::Status& status) {
    if (!status.ok()) {
        throw std::runtime_error(status.ToString());
    }
}

template <typename T> T value(arrow::Result<T> result) {
    check(result.status());
    return std::move(result).ValueOrDie();
}

template <typename Builder, typename Values>
std::shared_ptr<arrow::Array> array(const Values& values) {
    Builder builder;
    check(builder.AppendValues(values));
    return value(builder.Finish());
}

class TemporaryDirectory {
  public:
    TemporaryDirectory() {
        auto pattern = (std::filesystem::temp_directory_path() / "mars-ppo-input-XXXXXX").string();
        const auto* created = ::mkdtemp(pattern.data());
        if (created == nullptr) {
            throw std::runtime_error("No se puede preparar la carpeta de prueba");
        }
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

MarketTape market() {
    MarketTape result;
    result.assets = {"FICTICIO"};
    result.open_times = {0, 2, 4};
    result.close_times = {1, 3, final_close};
    result.prediction_times = result.close_times;
    result.currency = "USD";
    result.domain = "synthetic";
    result.partition = "train";
    result.parent_id = "fixture-context-v1";
    result.source_sha256.assign(digest_characters, 'a');
    for (std::size_t at = 0; at < sessions; ++at) {
        result.prices.insert(result.prices.end(), {fixture_price, fixture_price, fixture_price,
                                                   fixture_price, fixture_volume});
        result.scores.push_back(fixture_score);
    }
    result.validate();
    return result;
}

std::shared_ptr<arrow::Table> context_table(std::string_view problem = {}) {
    std::vector<int32_t> session{0, 0, 1, 1, 2, 2};
    std::vector<int32_t> feature{0, 1, 0, 1, 0, 1};
    std::vector<float> values{initial_context, 0, middle_context, -1, final_context, 2};
    const std::vector<bool> present{true, false, true, true, true, true};
    std::vector<int64_t> available{1, 0, 3, 2, final_close, 4};
    if (problem == "duplicate") {
        session[2] = 0;
    } else if (problem == "order") {
        feature[0] = 1;
        feature[1] = 0;
    } else if (problem == "negative") {
        session[0] = -1;
    } else if (problem == "future") {
        available.back() = final_close + 1;
    } else if (problem == "absent") {
        values[1] = 1;
    } else if (problem == "nan") {
        values[0] = std::numeric_limits<float>::quiet_NaN();
    } else if (problem == "negative_time") {
        available[0] = -1;
    } else if (problem == "changed_value") {
        values[0] = changed_context;
    }
    std::vector<std::shared_ptr<arrow::Field>> fields{
        arrow::field("session", arrow::int32()), arrow::field("feature", arrow::int32()),
        arrow::field("value", arrow::float32()), arrow::field("present", arrow::boolean()),
        arrow::field("available_at", arrow::int64())};
    std::vector<std::shared_ptr<arrow::Array>> columns{
        array<arrow::Int32Builder>(session), array<arrow::Int32Builder>(feature),
        array<arrow::FloatBuilder>(values), array<arrow::BooleanBuilder>(present),
        array<arrow::Int64Builder>(available)};
    if (problem == "dtype") {
        fields[2] = arrow::field("value", arrow::float64());
        columns[2] = array<arrow::DoubleBuilder>(std::vector<double>(values.begin(), values.end()));
    } else if (problem == "missing_column") {
        fields.back() = arrow::field("unexpected", arrow::int64());
    } else if (problem == "extra_column") {
        fields.push_back(arrow::field("extra", arrow::int32()));
        columns.push_back(columns.front());
    } else if (problem == "null") {
        arrow::Int64Builder builder;
        check(builder.AppendNull());
        for (std::size_t index = 1; index < available.size(); ++index) {
            check(builder.Append(available[index]));
        }
        columns.back() = value(builder.Finish());
    }
    auto result = arrow::Table::Make(arrow::schema(fields), columns);
    if (problem == "short") {
        result = result->Slice(0, static_cast<int64_t>(fixture_rows - 1));
    }
    return result;
}

std::string parquet_bytes(const arrow::Table& table, int64_t group_rows = 3) {
    const auto sink = value(arrow::io::BufferOutputStream::Create());
    parquet::WriterProperties::Builder properties;
    properties.enable_page_checksum();
    check(parquet::arrow::WriteTable(table, arrow::default_memory_pool(), sink, group_rows,
                                     properties.build()));
    return value(sink->Finish())->ToString();
}

Json write_context(const std::filesystem::path& directory, const MarketTape& tape,
                   const std::shared_ptr<arrow::Table>& table = context_table()) {
    const auto bytes = parquet_bytes(*table);
    atomic_binary_file(directory / "context.parquet", bytes, context_file_limit);
    Json manifest{{"schema_version", 1},
                  {"domain", tape.domain},
                  {"market_manifest_sha256", tape.source_sha256},
                  {"fields", Json::array({{{"name", "macro"}, {"unit", "proporción"}},
                                          {{"name", "news"}, {"unit", "puntuación"}}})},
                  {"file",
                   {{"path", "context.parquet"},
                    {"sha256", content_sha256(bytes)},
                    {"bytes", bytes.size()}}}};
    atomic_json_file(directory / "context.json", manifest);
    return manifest;
}

void replace_context_bytes(const std::filesystem::path& directory, std::string_view bytes) {
    auto manifest =
        parse_bounded_json(read_bounded_file(directory / "context.json", fixture_limit));
    manifest["file"]["bytes"] = bytes.size();
    manifest["file"]["sha256"] = content_sha256(bytes);
    atomic_binary_file(directory / "context.parquet", bytes, context_file_limit);
    atomic_json_file(directory / "context.json", manifest);
}

void write_market(const std::filesystem::path& directory) {
    const auto tape = market();
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    for (const std::string name : {"open", "high", "low", "close", "volume", "score"}) {
        fields.push_back(arrow::field(name, arrow::float64()));
        const double number = name == "volume"  ? fixture_volume
                              : name == "score" ? fixture_score
                                                : fixture_price;
        columns.push_back(array<arrow::DoubleBuilder>(std::vector<double>(sessions, number)));
    }
    for (const auto& [name, times] : std::array<std::pair<std::string, std::vector<int64_t>>, 3>{
             {{"close_time", tape.close_times},
              {"open_time", tape.open_times},
              {"prediction_time", tape.prediction_times}}}) {
        fields.push_back(arrow::field(name, arrow::int64()));
        columns.push_back(array<arrow::Int64Builder>(times));
    }
    fields.push_back(arrow::field("asset", arrow::utf8()));
    columns.push_back(array<arrow::StringBuilder>(std::vector<std::string>(sessions, "FICTICIO")));
    const auto bytes = parquet_bytes(*arrow::Table::Make(arrow::schema(fields), columns));
    atomic_binary_file(directory / "market.parquet", bytes, fixture_limit);
    const Json manifest{{"schema_version", 1},
                        {"assets", 1},
                        {"sessions", sessions},
                        {"final_test_opened", false},
                        {"file_sha256", content_sha256(bytes)},
                        {"file_bytes", bytes.size()},
                        {"tape_sha256", tape.source_sha256},
                        {"actions", Json::array()},
                        {"identity",
                         {{"domain", "synthetic"},
                          {"partition", "train"},
                          {"currency", "USD"},
                          {"assets", tape.assets},
                          {"parent_id", tape.parent_id},
                          {"actions", Json::array()}}}};
    atomic_json_file(directory / "manifest.json", manifest);
}

void optional_context_and_verified_input_are_loaded() {
    TemporaryDirectory temporary;
    const auto tape = market();
    require(!load_ppo_context(temporary.path, tape), "Un contexto ausente debe ser opcional");
    write_market(temporary.path);
    const auto plain = load_ppo_input(temporary.path);
    require(plain.tape && !plain.context && plain.parameters == Parameters{},
            "La entrada sin contexto debe conservar la cinta y los parámetros iniciales");
    static_cast<void>(write_context(temporary.path, *plain.tape));
    const auto input = load_ppo_input(temporary.path);
    if (!input.context.has_value()) {
        throw std::runtime_error("Falta el contexto declarado en la entrada");
    }
    const auto& context = *input.context;
    require(context.fields.size() == feature_count && context.values.size() == fixture_rows,
            "El contexto debe conservar su matriz completa");
    require(context.fields.front().name == "macro" && context.fields.back().unit == "puntuación" &&
                context.values[0].value == initial_context && !context.values[1].present &&
                context.values.back().available_at == 4,
            "Las columnas deben conservar valores, máscara y disponibilidad");
    require(context.source_sha256 ==
                content_sha256(read_bounded_file(temporary.path / "context.json", fixture_limit)),
            "La identidad del contexto debe proceder del JSON realmente leído");
}

void manifest_identity_and_schema_are_rejected() {
    TemporaryDirectory temporary;
    const auto tape = market();
    const auto original = write_context(temporary.path, tape);
    const std::array changes{
        std::pair{Json::json_pointer("/schema_version"), Json(1.0)},
        std::pair{Json::json_pointer("/domain"), Json("real")},
        std::pair{Json::json_pointer("/market_manifest_sha256"), Json(std::string(64, 'b'))},
        std::pair{Json::json_pointer("/file/path"), Json("../context.parquet")},
        std::pair{Json::json_pointer("/file/path"), Json("/tmp/context.parquet")},
        std::pair{Json::json_pointer("/file/sha256"), Json(std::string(64, 'b'))},
        std::pair{Json::json_pointer("/file/bytes"), Json(1)},
        std::pair{Json::json_pointer("/file/bytes"), Json(-1)},
        std::pair{Json::json_pointer("/fields"), Json::array()},
        std::pair{Json::json_pointer("/fields/1/name"), Json("macro")},
        std::pair{Json::json_pointer("/fields/0/unit"), Json("")},
        std::pair{Json::json_pointer("/unexpected"), Json(true)}};
    for (const auto& [pointer, replacement] : changes) {
        auto changed = original;
        changed[pointer] = replacement;
        atomic_json_file(temporary.path / "context.json", changed);
        rejected([&] { static_cast<void>(load_ppo_context(temporary.path, tape)); },
                 "El cargador aceptó un manifiesto incompatible");
    }
    auto changed = original;
    changed["fields"] = Json::array();
    for (std::size_t field = 0; field <= maximum_context_fields; ++field) {
        changed["fields"].push_back({{"name", "f" + std::to_string(field)}, {"unit", "número"}});
    }
    atomic_json_file(temporary.path / "context.json", changed);
    rejected([&] { static_cast<void>(load_ppo_context(temporary.path, tape)); },
             "Se aceptaron más de 512 campos de contexto");
}

void dense_typed_rows_and_temporal_availability_are_required() {
    TemporaryDirectory temporary;
    const auto tape = market();
    for (const std::string_view problem :
         {"duplicate", "order", "negative", "future", "absent", "nan", "negative_time", "dtype",
          "missing_column", "extra_column", "null", "short"}) {
        static_cast<void>(write_context(temporary.path, tape, context_table(problem)));
        rejected([&] { static_cast<void>(load_ppo_context(temporary.path, tape)); },
                 "Se admitieron filas desordenadas, sin valorar o con información futura");
    }
    const auto table = context_table();
    for (int column = 0; column < table->num_columns(); ++column) {
        auto nulls = value(arrow::MakeArrayOfNull(table->field(column)->type(), table->num_rows()));
        const auto changed = value(table->SetColumn(column, table->field(column),
                                                    std::make_shared<arrow::ChunkedArray>(nulls)));
        static_cast<void>(write_context(temporary.path, tape, changed));
        rejected([&] { static_cast<void>(load_ppo_context(temporary.path, tape)); },
                 "Ninguna columna de contexto admite NULL");
    }
    const auto reversed = value(table->SelectColumns({4, 3, 2, 1, 0}));
    static_cast<void>(write_context(temporary.path, tape, reversed));
    const auto context = load_ppo_context(temporary.path, tape);
    require(context && context->values[0].value == initial_context &&
                context->values.back().available_at == 4,
            "La selección por nombre debe conservar la semántica si cambia el orden físico");
}

std::string metadata_only_parquet(bool external) {
    const auto bytes = parquet_bytes(*context_table());
    const auto input = std::make_shared<arrow::io::BufferReader>(arrow::Buffer::FromString(bytes));
    const auto reader = parquet::ParquetFileReader::Open(input);
    const auto builder = parquet::FileMetaDataBuilder::Make(
        reader->metadata()->schema(), parquet::WriterProperties::Builder().build());
    auto* group = builder->AppendRowGroup();
    constexpr int64_t rows = static_cast<int64_t>(sessions * feature_count);
    group->set_num_rows(rows);
    const int64_t decoded = external ? 4 : 129 * static_cast<int64_t>(bytes_per_mebibyte);
    for (int column = 0; column < reader->metadata()->num_columns(); ++column) {
        auto* chunk = group->NextColumnChunk();
        if (external) {
            chunk->set_file_path("../external.parquet");
        }
        chunk->Finish(rows, 0, 0, 4, 4, decoded, false, false, {}, {});
    }
    group->Finish(4 * static_cast<int64_t>(reader->metadata()->num_columns()));
    const auto footer = builder->Finish()->SerializeToString();
    std::string result = "PAR1" + footer;
    const auto size = static_cast<uint32_t>(footer.size());
    for (std::size_t byte = 0; byte < sizeof(size); ++byte) {
        result.push_back(static_cast<char>((size >> (byte * bits_per_byte)) & byte_mask));
    }
    return result + "PAR1";
}

void metadata_rejects_external_chunks_and_decompression_budgets_before_reading_pages() {
    TemporaryDirectory temporary;
    const auto tape = market();
    static_cast<void>(write_context(temporary.path, tape));
    replace_context_bytes(temporary.path, metadata_only_parquet(false));
    rejected_with([&] { static_cast<void>(load_ppo_context(temporary.path, tape)); }, "128 MiB");
    replace_context_bytes(temporary.path, metadata_only_parquet(true));
    rejected_with([&] { static_cast<void>(load_ppo_context(temporary.path, tape)); },
                  "rutas externas");
}

void content_hashes_footer_and_file_limits_are_verified() {
    TemporaryDirectory temporary;
    const auto tape = market();
    static_cast<void>(write_context(temporary.path, tape));
    const auto parquet = temporary.path / "context.parquet";
    auto bytes = read_bounded_file(parquet, fixture_limit);
    const auto changed = parquet_bytes(*context_table("changed_value"));
    require(changed.size() == bytes.size(),
            "La prueba de huella debe conservar el tamaño del archivo");
    atomic_binary_file(parquet, changed, fixture_limit);
    rejected_with([&] { static_cast<void>(load_ppo_context(temporary.path, tape)); },
                  "bytes verificados");
    bytes.front() = 'X';
    atomic_binary_file(parquet, bytes, fixture_limit);
    rejected([&] { static_cast<void>(load_ppo_context(temporary.path, tape)); },
             "Se aceptaron bytes que no conservan el SHA256");
    replace_context_bytes(temporary.path, bytes);
    rejected([&] { static_cast<void>(load_ppo_context(temporary.path, tape)); },
             "Se aceptó una cabecera Parquet inválida con hash actualizado");
    static_cast<void>(write_context(temporary.path, tape));
    bytes = read_bounded_file(parquet, fixture_limit);
    for (std::size_t index = bytes.size() - footer_bytes; index < bytes.size() - 4; ++index) {
        bytes[index] = static_cast<char>(byte_mask);
    }
    replace_context_bytes(temporary.path, bytes);
    rejected([&] { static_cast<void>(load_ppo_context(temporary.path, tape)); },
             "El pie Parquet debe comprobarse antes de leer sus metadatos");
    static_cast<void>(write_context(temporary.path, tape));
    std::filesystem::resize_file(parquet, context_file_limit + 1);
    rejected([&] { static_cast<void>(load_ppo_context(temporary.path, tape)); },
             "Se aceptó un Parquet de más de 64 MiB");
    std::filesystem::resize_file(temporary.path / "context.json", maximum_manifest_bytes + 1);
    rejected([&] { static_cast<void>(load_ppo_context(temporary.path, tape)); },
             "Se aceptó un manifiesto de más de 4 MiB");
}

void symlinks_and_non_regular_files_are_rejected() {
    TemporaryDirectory temporary;
    const auto tape = market();
    const auto json_path = temporary.path / "context.json";
    std::filesystem::create_symlink(temporary.path / "missing", json_path);
    rejected([&] { static_cast<void>(load_ppo_context(temporary.path, tape)); },
             "Un enlace roto no puede confundirse con contexto opcional ausente");
    std::filesystem::remove(json_path);
    static_cast<void>(write_context(temporary.path, tape));
    const auto parquet = temporary.path / "context.parquet";
    std::filesystem::rename(parquet, temporary.path / "original.parquet");
    std::filesystem::create_symlink(temporary.path / "original.parquet", parquet);
    rejected([&] { static_cast<void>(load_ppo_context(temporary.path, tape)); },
             "Se aceptó un enlace simbólico al contexto Parquet");
    std::filesystem::remove(json_path);
    require(::mkfifo(json_path.c_str(), S_IRUSR | S_IWUSR) == 0,
            "No se puede preparar el FIFO de prueba");
    rejected([&] { static_cast<void>(load_ppo_context(temporary.path, tape)); },
             "Se aceptó un manifiesto FIFO");
}

} // namespace

int main() {
    try {
        optional_context_and_verified_input_are_loaded();
        manifest_identity_and_schema_are_rejected();
        dense_typed_rows_and_temporal_availability_are_required();
        metadata_rejects_external_chunks_and_decompression_budgets_before_reading_pages();
        content_hashes_footer_and_file_limits_are_verified();
        symlinks_and_non_regular_files_are_rejected();
        std::cout << "Contexto Parquet, identidad y límites comprobados\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
