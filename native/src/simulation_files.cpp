#include "mars_titan/simulation_files.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <exception>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/file.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mars_titan::simulation {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr std::size_t sha_bytes = 32;
constexpr std::size_t sha_characters = 64;
constexpr std::size_t maximum_json_depth = 64;
constexpr std::size_t checkpoint_history = 2;
constexpr std::size_t stream_chunk_bytes = 64 * bytes_per_kibibyte;
constexpr std::size_t latest_bytes = 64 * bytes_per_kibibyte;
constexpr std::size_t utc_buffer_characters = 32;
constexpr std::size_t maximum_workers = 8;
constexpr unsigned int hexadecimal_digit_mask = 0x0f;
constexpr std::array reference_costs{0, 10, 25};

[[noreturn]] void io_error(std::string_view operation) {
    const int error = errno;
    throw std::runtime_error(std::string(operation) + ": " + std::strerror(error));
}

class FileDescriptor {
  public:
    explicit FileDescriptor(int value) : value_(value) {
        if (value_ < 0) {
            io_error("No se puede abrir el archivo");
        }
    }
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    FileDescriptor(FileDescriptor&&) = delete;
    FileDescriptor& operator=(FileDescriptor&&) = delete;
    ~FileDescriptor() { static_cast<void>(::close(value_)); }
    [[nodiscard]] int get() const noexcept { return value_; }

  private:
    int value_;
};

class OutputLock {
  public:
    OutputLock(const std::filesystem::path& directory, bool resume)
        : descriptor_(open_directory(directory, resume)) {
        if (::flock(descriptor_.get(), LOCK_EX | LOCK_NB) != 0) {
            io_error("La salida ya tiene otra ejecución activa");
        }
    }

  private:
    static int open_directory(const std::filesystem::path& directory, bool resume) {
        require_safe_path(directory);
        if (resume) {
            if (!std::filesystem::is_directory(directory) ||
                !std::filesystem::is_regular_file(directory / "identity.json")) {
                throw std::invalid_argument(
                    "La recuperación necesita una salida con identidad confirmada");
            }
        } else if (std::filesystem::exists(directory) ||
                   !std::filesystem::create_directories(directory)) {
            throw std::invalid_argument("La salida debe ser nueva o usar --resume explícito");
        }
        require_safe_path(directory / ".run.lock");
        // POSIX requiere open con su argumento mode_t al crear el archivo.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
        return ::open((directory / ".run.lock").c_str(), O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC,
                      S_IRUSR | S_IWUSR);
    }
    FileDescriptor descriptor_;
};

bool valid_digest(std::string_view value) {
    return value.size() == sha_characters && std::ranges::all_of(value, [](char ch) {
               return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
           });
}

std::size_t size_value(const Json& value, std::size_t maximum) {
    if (!value.is_number_unsigned() && !value.is_number_integer()) {
        throw std::invalid_argument("El estado necesita contadores enteros");
    }
    if (value.is_number_integer() && !value.is_number_unsigned() && value.get<int64_t>() < 0) {
        throw std::invalid_argument("El contador no puede ser negativo");
    }
    const auto result = value.get<uint64_t>();
    if (result > maximum) {
        throw std::invalid_argument("El contador excede el presupuesto");
    }
    return static_cast<std::size_t>(result);
}

double numeric_value(const Json& value, bool absent = false) {
    if (absent && value.is_null()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (!value.is_number()) {
        throw std::invalid_argument("El estado necesita un importe numérico");
    }
    const auto result = value.get<double>();
    if (!std::isfinite(result)) {
        throw std::invalid_argument("El importe no es finito");
    }
    return result;
}

Json nullable(double value) {
    if (std::isnan(value)) {
        return nullptr;
    }
    if (!std::isfinite(value)) {
        throw std::invalid_argument("No se serializan importes infinitos");
    }
    return value;
}

Json parameters_json(const Parameters& p) {
    return Json{{"capital", p.capital},
                {"cost_bps", p.cost_bps},
                {"participation", p.participation},
                {"score_scale", p.score_scale},
                {"ruin_penalty", p.ruin_penalty}};
}

Parameters parameters_from(const Json& value) {
    return Parameters{numeric_value(value.at("capital")), numeric_value(value.at("cost_bps")),
                      numeric_value(value.at("participation")),
                      numeric_value(value.at("score_scale")),
                      numeric_value(value.at("ruin_penalty"))};
}

std::vector<uint8_t> flags_from(const Json& value, std::size_t maximum) {
    if (!value.is_array() || value.size() > maximum) {
        throw std::invalid_argument("Las máscaras exceden el presupuesto");
    }
    std::vector<uint8_t> result;
    result.reserve(value.size());
    for (const auto& flag : value) {
        result.push_back(static_cast<uint8_t>(size_value(flag, 1)));
    }
    return result;
}

std::string utc_now() {
    const auto current = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    if (::gmtime_r(&current, &utc) == nullptr) {
        throw std::runtime_error("No se puede obtener la fecha UTC");
    }
    std::array<char, utc_buffer_characters> buffer{};
    const auto length = std::strftime(buffer.data(), buffer.size(), "%Y-%m-%dT%H:%M:%SZ", &utc);
    if (length == 0) {
        throw std::runtime_error("La fecha UTC no cabe en su formato");
    }
    return std::string(buffer.data(), length);
}

std::size_t process_peak_rss() {
    rusage usage{};
    if (::getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("No se puede medir el pico de memoria del proceso");
    }
    // glibc expone ru_maxrss mediante una unión de su ABI, escrita por getrusage.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
    const auto maximum = usage.ru_maxrss;
    if (maximum < 0) {
        throw std::runtime_error("El pico de memoria del proceso no puede ser negativo");
    }
#if defined(__APPLE__)
    return static_cast<std::size_t>(maximum);
#else
    if (static_cast<uintmax_t>(maximum) >
        std::numeric_limits<std::size_t>::max() / bytes_per_kibibyte) {
        throw std::overflow_error("El pico de memoria excede su representación en bytes");
    }
    return static_cast<std::size_t>(maximum) * bytes_per_kibibyte;
#endif
}

std::size_t executable_peak_rss() {
#if defined(__linux__)
    // El API POSIX conserva la firma variádica también en aperturas de lectura.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    const FileDescriptor file(::open("/proc/self/status", O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
    std::array<char, stream_chunk_bytes> buffer{};
    std::size_t used = 0;
    while (used < buffer.size()) {
        const auto remaining = std::span(buffer).subspan(used);
        const auto count = ::read(file.get(), remaining.data(), remaining.size());
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            io_error("No se puede leer VmHWM");
        }
        if (count == 0) {
            break;
        }
        used += static_cast<std::size_t>(count);
    }
    if (used == buffer.size()) {
        throw std::runtime_error("El estado de procfs alcanza el límite de 64 KiB");
    }
    std::string_view text(buffer.data(), used);
    constexpr std::string_view field = "VmHWM:";
    while (!text.empty()) {
        const auto end = text.find('\n');
        const auto line = text.substr(0, end);
        text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
        if (!line.starts_with(field)) {
            continue;
        }
        auto value = line.substr(field.size());
        const auto start = value.find_first_not_of(" \t");
        if (start == std::string_view::npos) {
            break;
        }
        value.remove_prefix(start);
        const auto split = value.find_first_not_of("0123456789");
        if (split == 0 || split == std::string_view::npos) {
            break;
        }
        const auto digits = value.substr(0, split);
        uint64_t kibibytes = 0;
        const auto converted = std::from_chars(digits.begin(), digits.end(), kibibytes);
        const auto unit = value.substr(split);
        const auto unit_start = unit.find_first_not_of(" \t");
        if (converted.ec != std::errc{} || converted.ptr != digits.end() ||
            unit_start == std::string_view::npos || unit.substr(unit_start) != "kB" ||
            kibibytes > std::numeric_limits<std::size_t>::max() / bytes_per_kibibyte) {
            break;
        }
        return static_cast<std::size_t>(kibibytes) * bytes_per_kibibyte;
    }
    throw std::runtime_error("VmHWM no conserva un valor y unidad reconocidos");
#else
    throw std::runtime_error("La medición de VmHWM solo está disponible en Linux");
#endif
}

void record_memory(Json& report) {
    const auto peak_rss = process_peak_rss();
    report["process_peak_rss_bytes"] = peak_rss;
    report["process_lifetime_peak_rss_bytes"] = peak_rss;
    report["process_lifetime_peak_rss_method"] = "getrusage_RUSAGE_SELF_including_pre_exec";
    report["executable_peak_rss_method"] = "linux_proc_self_status_VmHWM";
    try {
        report["executable_peak_rss_bytes"] = executable_peak_rss();
        report["executable_peak_rss_reason"] = nullptr;
    } catch (const std::exception& error) {
        report["executable_peak_rss_bytes"] = nullptr;
        report["executable_peak_rss_reason"] = error.what();
    }
}

Json metrics_json(const FinancialMetrics& metrics) {
    const std::unordered_set<std::string> reasons{"", "incomplete", "missing_close", "ruined"};
    if (!reasons.contains(metrics.invalid_reason)) {
        throw std::runtime_error("La sesión devuelve un motivo financiero desconocido");
    }
    return Json{
        {"net_return", metrics.net_return ? nullable(*metrics.net_return) : Json(nullptr)},
        {"max_drawdown", metrics.max_drawdown ? nullable(*metrics.max_drawdown) : Json(nullptr)},
        {"costs", metrics.costs},
        {"turnover", metrics.turnover},
        {"steps", metrics.steps},
        {"completed", metrics.completed},
        {"invalid_reason",
         metrics.invalid_reason.empty() ? Json(nullptr) : Json(metrics.invalid_reason)}};
}

std::string_view error_type(const std::exception& error) {
    if (dynamic_cast<const std::bad_alloc*>(&error) != nullptr) {
        return "bad_alloc";
    }
    if (dynamic_cast<const std::filesystem::filesystem_error*>(&error) != nullptr) {
        return "filesystem_error";
    }
    if (dynamic_cast<const Json::exception*>(&error) != nullptr) {
        return "json_error";
    }
    if (dynamic_cast<const std::invalid_argument*>(&error) != nullptr) {
        return "invalid_argument";
    }
    return "runtime_error";
}

Json run_identity(const MarketTape& tape, const RunOptions& options) {
    return Json{{"manifest_sha256", tape.source_sha256},
                {"parent_id", tape.parent_id},
                {"native_version", MARS_TITAN_NATIVE_VERSION},
                {"native_source_sha256", MARS_TITAN_NATIVE_SOURCE_SHA256},
                {"native_build_sha256", MARS_TITAN_NATIVE_BUILD_SHA256},
                {"compiler_id", MARS_TITAN_NATIVE_COMPILER_ID},
                {"compiler_version", MARS_TITAN_NATIVE_COMPILER_VERSION},
                {"build_type", MARS_TITAN_NATIVE_BUILD_TYPE},
                {"config", parameters_json(options.parameters)},
                {"policy", policy_name(options.policy)},
                {"diagnostic", options.diagnostic},
                {"partition", tape.partition},
                {"rng", "none_deterministic_policies"}};
}

Json receipt_contract(const MarketTape& tape, const RunOptions& options, const Json& identity) {
    return Json{{"schema_version", 1},
                {"activity", "simulation"},
                {"model", policy_name(options.policy)},
                {"domain", options.diagnostic ? "technical" : "synthetic"},
                {"partition", "validation"},
                {"identity", identity},
                {"total_steps", tape.close_times.size() - 1},
                {"final_test_opened", false},
                {"parent_frozen", true},
                {"currency", tape.currency},
                {"cost_bps", options.parameters.cost_bps}};
}

std::string receipt_time(const Json& value) {
    constexpr std::size_t timestamp_characters = 20;
    constexpr int calendar_year_offset = 1900;
    const auto text = value.get<std::string>();
    std::istringstream input(text);
    input.imbue(std::locale::classic());
    std::tm parsed{};
    input >> std::get_time(&parsed, "%Y-%m-%dT%H:%M:%SZ");
    const std::chrono::year_month_day date{
        std::chrono::year{parsed.tm_year + calendar_year_offset},
        std::chrono::month{static_cast<unsigned>(parsed.tm_mon + 1)},
        std::chrono::day{static_cast<unsigned>(parsed.tm_mday)}};
    if (text.size() != timestamp_characters || input.fail() || input.peek() != EOF || !date.ok()) {
        throw std::invalid_argument("El recibo necesita fechas UTC válidas");
    }
    return text;
}

void validate_receipt(const Json& report, const FinancialSession& session, const Json& expected) {
    for (const auto& [key, value] : expected.items()) {
        if (report.at(key) != value) {
            throw std::invalid_argument("El recibo no conserva el contrato de la ejecución");
        }
    }
    if (read_json_int64(report.at("schema_version")) != 1) {
        throw std::invalid_argument("El recibo no conserva su versión entera");
    }
    const auto step = size_value(report.at("global_step"), session.cursor());
    static_cast<void>(size_value(report.at("total_steps"), maximum_sessions));
    const std::unordered_set<std::string> statuses{"running", "paused", "failed", "completed"};
    if (!statuses.contains(report.at("status").get<std::string>())) {
        throw std::invalid_argument("El recibo contiene un estado de ejecución desconocido");
    }
    const auto total = numeric_value(report.at("total_seconds"));
    if (total < 0 || (report.contains("attempt_seconds") &&
                      (numeric_value(report.at("attempt_seconds")) < 0 ||
                       numeric_value(report.at("attempt_seconds")) > total))) {
        throw std::invalid_argument("Los tiempos del recibo deben ser no negativos y coherentes");
    }
    for (const auto* field : {"process_peak_rss_bytes", "process_lifetime_peak_rss_bytes",
                              "executable_peak_rss_bytes", "price_and_score_bytes"}) {
        if (report.contains(field) && !report.at(field).is_null()) {
            static_cast<void>(
                size_value(report.at(field), std::numeric_limits<std::size_t>::max()));
        }
    }
    if (report.contains("timing_scope") &&
        report.at("timing_scope") != "reference_with_checkpoints_excluding_shared_input") {
        throw std::invalid_argument("El tiempo del recibo tiene un alcance desconocido");
    }
    const auto started = receipt_time(report.at("started_at_utc"));
    const auto updated =
        report.contains("updated_at_utc") ? receipt_time(report.at("updated_at_utc")) : started;
    if (updated < started || updated > utc_now()) {
        throw std::invalid_argument("Las fechas del recibo no conservan el orden observado");
    }
    if (report.contains("checkpoint")) {
        const auto& checkpoint = report.at("checkpoint");
        const auto digest = checkpoint.at("sha256").get<std::string>();
        const auto saved = receipt_time(checkpoint.at("saved_at"));
        if (!valid_digest(digest) ||
            checkpoint.at("path") != "private/checkpoints/state-" + digest + ".json" ||
            size_value(checkpoint.at("step"), session.cursor()) != step ||
            checkpoint.at("resumable") != true || saved < started || saved > updated) {
            throw std::invalid_argument("El recibo no conserva sus metadatos de recuperación");
        }
    }
    if (step == session.cursor() &&
        report.at("financial_validation") != metrics_json(session.metrics())) {
        throw std::invalid_argument("El recibo no coincide con la contabilidad confirmada");
    }
}

void confirm_identity(const std::filesystem::path& output, const Json& identity, bool resume) {
    const auto path = output / "identity.json";
    if (resume) {
        if (parse_bounded_json(read_bounded_file(path, maximum_manifest_bytes)) != identity) {
            throw std::invalid_argument(
                "La salida pertenece a otra identidad, configuración o versión nativa");
        }
    } else {
        atomic_json_file(path, identity);
    }
}

Json read_checkpoint_index(const std::filesystem::path& folder) {
    if (!std::filesystem::exists(folder / "latest.json")) {
        return Json{{"schema_version", 1}, {"states", Json::array()}};
    }
    auto index = parse_bounded_json(read_bounded_file(folder / "latest.json", latest_bytes));
    const auto& states = index.at("states");
    if (read_json_int64(index.at("schema_version")) != 1 || !states.is_array() || states.empty() ||
        states.size() > checkpoint_history) {
        throw std::invalid_argument("El índice de checkpoints no conserva su contrato");
    }
    std::size_t previous = maximum_sessions;
    std::unordered_set<std::string> seen;
    for (const auto& item : states) {
        const auto digest = item.at("sha256").get<std::string>();
        const auto name = item.at("name").get<std::string>();
        const auto step = size_value(item.at("global_step"), maximum_sessions);
        if (!valid_digest(digest) || name != "state-" + digest + ".json" ||
            !seen.insert(name).second || step > previous) {
            throw std::invalid_argument(
                "El índice de checkpoints contiene una ruta o secuencia inválida");
        }
        previous = step;
        const auto bytes = read_bounded_file(folder / name, maximum_checkpoint_bytes);
        if (size_value(item.at("bytes"), maximum_checkpoint_bytes) != bytes.size() ||
            content_sha256(bytes) != digest) {
            throw std::invalid_argument("El checkpoint confirmado está corrupto");
        }
    }
    return index;
}

Json load_checkpoint(const std::filesystem::path& folder, const Json& identity) {
    const auto index = read_checkpoint_index(folder);
    if (index.at("states").empty()) {
        throw std::invalid_argument("Falta un checkpoint confirmado para recuperar");
    }
    const auto& record = index.at("states").front();
    const auto state = parse_bounded_json(
        read_bounded_file(folder / record.at("name").get<std::string>(), maximum_checkpoint_bytes));
    if (read_json_int64(state.at("schema_version")) != 1 || state.at("identity") != identity ||
        state.at("global_step") != record.at("global_step")) {
        throw std::invalid_argument("El checkpoint no corresponde a su identidad y cursor");
    }
    return state;
}

Json save_checkpoint(const std::filesystem::path& folder, const SessionSnapshot& snapshot,
                     const Json& identity, const Json& event) {
    require_safe_path(folder);
    std::filesystem::create_directories(folder);
    std::filesystem::permissions(folder.parent_path(), std::filesystem::perms::owner_all);
    std::filesystem::permissions(folder, std::filesystem::perms::owner_all);
    const auto previous = read_checkpoint_index(folder);
    const Json payload{{"schema_version", 1},
                       {"identity", identity},
                       {"global_step", snapshot.cursor},
                       {"snapshot", snapshot_json(snapshot)},
                       {"last_event", event}};
    const auto bytes = payload.dump();
    if (bytes.size() > maximum_checkpoint_bytes) {
        throw std::invalid_argument("El checkpoint supera 16 MiB");
    }
    const auto digest = content_sha256(bytes);
    const auto name = "state-" + digest + ".json";
    const auto destination = folder / name;
    if (std::filesystem::exists(destination)) {
        if (read_bounded_file(destination, maximum_checkpoint_bytes) != bytes) {
            throw std::invalid_argument("El checkpoint existente no conserva sus bytes");
        }
    } else {
        atomic_json_file(destination, payload, maximum_checkpoint_bytes);
    }
    Json states = Json::array({Json{{"name", name},
                                    {"sha256", digest},
                                    {"bytes", bytes.size()},
                                    {"global_step", snapshot.cursor}}});
    for (const auto& record : previous.at("states")) {
        if (record.at("global_step").get<std::size_t>() > snapshot.cursor) {
            throw std::invalid_argument("El checkpoint retrocede sobre un cursor confirmado");
        }
        if (record.at("name") != name && states.size() < checkpoint_history) {
            states.push_back(record);
        }
    }
    atomic_json_file(folder / "latest.json", Json{{"schema_version", 1}, {"states", states}},
                     latest_bytes);
    for (const auto& record : previous.at("states")) {
        if (std::ranges::none_of(states, [&record](const auto& kept) {
                return kept.at("name") == record.at("name");
            })) {
            const auto old = folder / record.at("name").get<std::string>();
            if (content_sha256(read_bounded_file(old, maximum_checkpoint_bytes)) !=
                record.at("sha256").get<std::string>()) {
                throw std::invalid_argument(
                    "El checkpoint anterior ha cambiado antes de su limpieza");
            }
            std::filesystem::remove(old);
        }
    }
    return states.front();
}

Json event_json(const StepOutcome& outcome) {
    Json trades = Json::array();
    for (std::size_t i = 0; i < outcome.trades.size(); ++i) {
        const auto& trade = outcome.trades[i];
        if (trade.quantity != 0 || trade.reason != MT_ORDER_COMPLETE) {
            trades.push_back(Json{{"asset", i},
                                  {"quantity", nullable(trade.quantity)},
                                  {"price", nullable(trade.price)},
                                  {"cost", nullable(trade.cost)},
                                  {"reason", trade.reason}});
        }
    }
    return Json{{"reward", outcome.reward_valid ? nullable(outcome.reward) : Json(nullptr)},
                {"terminated", outcome.terminated},
                {"truncated", outcome.truncated},
                {"trades", trades},
                {"unvalued", outcome.unvalued}};
}
} // namespace

void require_safe_path(const std::filesystem::path& path) {
    auto prefix = std::filesystem::absolute(path).root_path();
    for (const auto& component : std::filesystem::absolute(path).relative_path()) {
        prefix /= component;
        const auto status = std::filesystem::symlink_status(prefix);
        if (std::filesystem::is_symlink(status)) {
            throw std::invalid_argument("La ruta contiene un enlace simbólico");
        }
    }
}

std::string read_bounded_file(const std::filesystem::path& path, std::size_t maximum) {
    require_safe_path(path);
    // O_NONBLOCK permite rechazar un FIFO con fstat sin esperar a que otro proceso lo abra.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    const FileDescriptor file(::open(path.c_str(), O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC));
    struct stat before{};
    if (::fstat(file.get(), &before) != 0) {
        io_error("No se puede inspeccionar el archivo");
    }
    if (!S_ISREG(before.st_mode) || before.st_size <= 0 ||
        static_cast<uintmax_t>(before.st_size) > maximum) {
        throw std::invalid_argument("El archivo no es regular, está vacío o supera su límite");
    }
    std::string bytes;
    bytes.reserve(static_cast<std::size_t>(before.st_size));
    std::array<char, stream_chunk_bytes> buffer{};
    while (true) {
        const auto count = ::read(file.get(), buffer.data(), buffer.size());
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            io_error("No se puede leer el archivo");
        }
        if (count == 0) {
            break;
        }
        const auto size = static_cast<std::size_t>(count);
        if (size > maximum - bytes.size()) {
            throw std::invalid_argument("El archivo creció por encima de su presupuesto");
        }
        bytes.append(buffer.data(), size);
    }
    struct stat after{};
    if (::fstat(file.get(), &after) != 0) {
        io_error("No se puede comprobar la lectura");
    }
    if (before.st_size != after.st_size || static_cast<uintmax_t>(after.st_size) != bytes.size() ||
        before.st_mtime != after.st_mtime || before.st_ctime != after.st_ctime) {
        throw std::invalid_argument("El archivo ha cambiado durante la lectura");
    }
    return bytes;
}

std::string content_sha256(std::string_view bytes) {
    const std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),
                                                                          &EVP_MD_CTX_free);
    std::array<unsigned char, sha_bytes> result{};
    unsigned int count = 0;
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) != 1 ||
        EVP_DigestFinal_ex(context.get(), result.data(), &count) != 1 || count != result.size()) {
        throw std::runtime_error("OpenSSL no ha calculado una huella SHA256 válida");
    }
    constexpr std::string_view hexadecimal = "0123456789abcdef";
    std::string encoded;
    encoded.reserve(sha_characters);
    for (const auto byte : result) {
        encoded.push_back(hexadecimal[byte >> 4U]);
        encoded.push_back(hexadecimal[byte & hexadecimal_digit_mask]);
    }
    return encoded;
}

Json parse_bounded_json(std::string_view bytes) {
    std::vector<std::unordered_set<std::string>> keys;
    const auto callback = [&keys](int depth, Json::parse_event_t event, Json& parsed) {
        if (depth < 0 || static_cast<std::size_t>(depth) > maximum_json_depth) {
            throw std::invalid_argument("El JSON supera la profundidad admitida");
        }
        if (event == Json::parse_event_t::object_start) {
            keys.emplace_back();
        } else if (event == Json::parse_event_t::object_end) {
            keys.pop_back();
        } else if (event == Json::parse_event_t::key &&
                   !keys.back().insert(parsed.get<std::string>()).second) {
            throw std::invalid_argument("El JSON contiene una clave duplicada");
        }
        return true;
    };
    return Json::parse(bytes, callback);
}

int64_t read_json_int64(const Json& value) {
    if (!value.is_number_integer() ||
        (value.is_number_unsigned() &&
         value.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))) {
        throw std::invalid_argument("El JSON necesita un entero de 64 bits");
    }
    return value.get<int64_t>();
}

void atomic_json_file(const std::filesystem::path& path, const Json& value, std::size_t maximum) {
    require_safe_path(path);
    const auto bytes = value.dump();
    if (bytes.size() > maximum) {
        throw std::invalid_argument("El JSON excede el presupuesto de escritura");
    }
    std::string pattern =
        (path.parent_path() / ("." + path.filename().string() + ".pending-XXXXXX")).string();
    const FileDescriptor file(::mkstemp(pattern.data()));
    const std::filesystem::path pending(pattern);
    try {
        std::size_t written = 0;
        const std::string_view view(bytes);
        while (written < bytes.size()) {
            const auto remaining = view.substr(written);
            const auto count = ::write(file.get(), remaining.data(), remaining.size());
            if (count < 0) {
                if (errno == EINTR) {
                    continue;
                }
                io_error("No se puede escribir el JSON");
            }
            if (count == 0) {
                throw std::runtime_error("La escritura no avanza");
            }
            written += static_cast<std::size_t>(count);
        }
        if (::fsync(file.get()) != 0) {
            io_error("No se puede confirmar el JSON");
        }
        std::filesystem::rename(pending, path);
        // La sincronización del directorio necesita un descriptor POSIX con O_DIRECTORY.
        const FileDescriptor directory(
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
            ::open(path.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
        if (::fsync(directory.get()) != 0) {
            io_error("No se puede confirmar el directorio");
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(pending, ignored);
        throw;
    }
}

Json snapshot_json(const SessionSnapshot& snapshot) {
    Json positions = Json::array();
    for (const auto& p : snapshot.positions) {
        positions.push_back(Json{{"quantity", nullable(p.quantity)},
                                 {"target", nullable(p.target)},
                                 {"capacity", nullable(p.capacity)},
                                 {"decision_at", p.decision_at}});
    }
    Json receivables = Json::array();
    for (const auto& item : snapshot.receivables) {
        receivables.push_back(Json{
            {"action", item.action}, {"pay_at", item.pay_at}, {"amount", nullable(item.amount)}});
    }
    const auto& account = snapshot.account;
    return Json{{"source_sha256", snapshot.source_sha256},
                {"parameters", parameters_json(snapshot.parameters)},
                {"cursor", snapshot.cursor},
                {"done", snapshot.done},
                {"positions", positions},
                {"account", Json{{"cash", nullable(account.cash)},
                                 {"costs", nullable(account.costs)},
                                 {"turnover", nullable(account.turnover)},
                                 {"receivable", nullable(account.receivable)},
                                 {"nav", nullable(account.nav)}}},
                {"retired", snapshot.retired},
                {"applied_actions", snapshot.applied_actions},
                {"receivables", receivables},
                {"held_order", snapshot.held_order},
                {"peak_nav", nullable(snapshot.peak_nav)},
                {"max_drawdown", nullable(snapshot.max_drawdown)},
                {"unfilled_order_observations", snapshot.unfilled_order_observations}};
}

SessionSnapshot read_snapshot(const Json& value) {
    SessionSnapshot result;
    result.source_sha256 = value.at("source_sha256").get<std::string>();
    if (!valid_digest(result.source_sha256)) {
        throw std::invalid_argument("El estado no conserva su origen");
    }
    result.parameters = parameters_from(value.at("parameters"));
    result.cursor = size_value(value.at("cursor"), maximum_sessions);
    result.done = value.at("done").get<bool>();
    const auto& positions = value.at("positions");
    if (!positions.is_array() || positions.empty() || positions.size() > maximum_assets) {
        throw std::invalid_argument("Las posiciones exceden el presupuesto");
    }
    result.positions.reserve(positions.size());
    for (const auto& p : positions) {
        result.positions.push_back(mt_position_v1{
            numeric_value(p.at("quantity")), numeric_value(p.at("target"), true),
            numeric_value(p.at("capacity"), true), read_json_int64(p.at("decision_at"))});
    }
    const auto& account = value.at("account");
    result.account = mt_account_v1{
        numeric_value(account.at("cash")), numeric_value(account.at("costs")),
        numeric_value(account.at("turnover")), numeric_value(account.at("receivable")),
        numeric_value(account.at("nav"), true)};
    result.retired = flags_from(value.at("retired"), maximum_assets);
    result.applied_actions = flags_from(value.at("applied_actions"), maximum_actions);
    const auto& receivables = value.at("receivables");
    if (!receivables.is_array() || receivables.size() > maximum_actions) {
        throw std::invalid_argument("Los derechos de cobro exceden el presupuesto");
    }
    for (const auto& r : receivables) {
        result.receivables.push_back(Receivable{size_value(r.at("action"), maximum_actions),
                                                read_json_int64(r.at("pay_at")),
                                                numeric_value(r.at("amount"))});
    }
    const auto& held = value.at("held_order");
    if (!held.is_array() || held.size() > maximum_assets) {
        throw std::invalid_argument("El orden de posiciones excede el presupuesto");
    }
    for (const auto& index : held) {
        result.held_order.push_back(size_value(index, maximum_assets));
    }
    result.peak_nav = numeric_value(value.at("peak_nav"));
    result.max_drawdown = numeric_value(value.at("max_drawdown"));
    result.unfilled_order_observations =
        size_value(value.at("unfilled_order_observations"), maximum_rows);
    return result;
}

Json run_reference(std::shared_ptr<const MarketTape> tape, const RunOptions& options,
                   const std::function<bool()>& stop_requested) {
    const auto started = Clock::now();
    if (!tape || tape->domain != "synthetic" || tape->partition != "validation" ||
        options.checkpoint_steps == 0 || options.checkpoint_steps > maximum_sessions) {
        throw std::invalid_argument(
            "La referencia requiere validación sintética y checkpoints acotados");
    }
    FinancialSession session(tape, options.parameters);
    const auto identity = run_identity(*tape, options);
    const OutputLock lock(options.output, options.resume);
    confirm_identity(options.output, identity, options.resume);
    const auto folder = options.output / "private/checkpoints";
    Json event = nullptr;
    std::optional<StepOutcome> pending_event;
    Json report;
    if (options.resume) {
        const auto payload = load_checkpoint(folder, identity);
        const auto snapshot = read_snapshot(payload.at("snapshot"));
        if (snapshot.cursor != size_value(payload.at("global_step"), maximum_sessions)) {
            throw std::invalid_argument("El cursor no coincide con el checkpoint");
        }
        session.restore(snapshot);
        event = payload.at("last_event");
        report = parse_bounded_json(
            read_bounded_file(options.output / "run.json", maximum_manifest_bytes));
        validate_receipt(report, session, receipt_contract(*tape, options, identity));
        if (report.at("status") == "completed") {
            if (!session.done() || report.at("global_step") != session.cursor() ||
                report.at("financial_validation") != metrics_json(session.metrics())) {
                throw std::invalid_argument("La ejecución completa no coincide con su checkpoint");
            }
            return report;
        }
    } else {
        report = receipt_contract(*tape, options, identity);
        report["global_step"] = 0;
        report["started_at_utc"] = utc_now();
        report["total_seconds"] = 0.0;
    }
    const double previous_seconds = numeric_value(report.at("total_seconds"));
    report["status"] = "running";
    report.erase("error");
    report.erase("error_type");
    report["financial_validation"] = metrics_json(session.metrics());
    atomic_json_file(options.output / "run.json", report);
    const auto record_resources = [&] {
        const auto elapsed = std::chrono::duration<double>(Clock::now() - started).count();
        report["attempt_seconds"] = elapsed;
        report["total_seconds"] = previous_seconds + elapsed;
        report["timing_scope"] = "reference_with_checkpoints_excluding_shared_input";
        record_memory(report);
        report["price_and_score_bytes"] =
            (tape->prices.size() + tape->scores.size()) * sizeof(double);
        report["updated_at_utc"] = utc_now();
    };
    const auto publish = [&] {
        if (pending_event) {
            event = event_json(*pending_event);
            pending_event.reset();
        }
        const auto record = save_checkpoint(folder, session.snapshot(), identity, event);
        report["checkpoint"] =
            Json{{"path", "private/checkpoints/" + record.at("name").get<std::string>()},
                 {"sha256", record.at("sha256")},
                 {"step", session.cursor()},
                 {"saved_at", utc_now()},
                 {"resumable", true}};
        report["global_step"] = session.cursor();
        report["financial_validation"] = metrics_json(session.metrics());
        record_resources();
        atomic_json_file(options.output / "run.json", report);
    };
    std::size_t applied = 0;
    try {
        publish();
        while (!session.done() && !stop_requested() &&
               (!options.stop_after || applied < *options.stop_after)) {
            pending_event = session.step(policy_action(options.policy, session.cursor()));
            ++applied;
            if (session.cursor() % options.checkpoint_steps == 0) {
                publish();
            }
        }
        report["status"] = session.done() ? "completed" : "paused";
        publish();
    } catch (const std::exception& error) {
        report["status"] = "failed";
        report["error"] = error.what();
        report["error_type"] = error_type(error);
        record_resources();
        atomic_json_file(options.output / "run.json", report);
        throw;
    }
    record_resources();
    atomic_json_file(options.output / "run.json", report);
    return report;
}

Json run_comparison(std::shared_ptr<const MarketTape> tape, const ComparisonOptions& options,
                    const std::function<bool()>& stop_requested) {
    const auto started = Clock::now();
    if (options.workers != 1 && options.workers != 2 && options.workers != 4 &&
        options.workers != maximum_workers) {
        throw std::invalid_argument("La comparación admite 1, 2, 4 u 8 trabajadores");
    }
    if (!tape || tape->partition != "validation" || tape->domain != "synthetic") {
        throw std::invalid_argument("La comparación requiere una cinta sintética de validación");
    }
    const OutputLock lock(options.run.output, options.run.resume);
    auto identity = run_identity(*tape, options.run);
    identity.erase("policy");
    identity["config"].erase("cost_bps");
    identity["policies"] = {"cash", "hold_initial", "rebalance_50"};
    identity["cost_bps"] = reference_costs;
    confirm_identity(options.run.output, identity, options.run.resume);
    constexpr std::array policies{ReferencePolicy::cash, ReferencePolicy::hold_initial,
                                  ReferencePolicy::rebalance_50};
    constexpr auto costs = reference_costs;
    constexpr auto jobs = policies.size() * costs.size();
    std::vector<Json> results(jobs);
    std::vector<std::exception_ptr> failures(jobs);
    std::atomic<std::size_t> next{0};
    const auto worker = [&] {
        while (true) {
            const auto index = next.fetch_add(1, std::memory_order_relaxed);
            if (index >= jobs) {
                break;
            }
            try {
                const auto policy = policies.at(index / costs.size());
                const auto cost = costs.at(index % costs.size());
                const auto name = policy_name(policy) + "-cost-" + std::to_string(cost);
                auto job = options.run;
                job.output /= name;
                job.policy = policy;
                job.parameters.cost_bps = cost;
                job.resume = options.run.resume && std::filesystem::exists(job.output);
                const auto report = run_reference(tape, job, stop_requested);
                results[index] =
                    Json{{"name", name},
                         {"path", name + "/run.json"},
                         {"status", report.at("status")},
                         {"sha256", content_sha256(read_bounded_file(job.output / "run.json",
                                                                     maximum_manifest_bytes))}};
            } catch (...) {
                failures[index] = std::current_exception();
            }
        }
    };
    {
        std::vector<std::jthread> workers;
        workers.reserve(options.workers);
        for (std::size_t i = 0; i < options.workers; ++i) {
            workers.emplace_back(worker);
        }
    }
    const bool failed =
        std::ranges::any_of(failures, [](const auto& failure) { return failure != nullptr; });
    const bool completed = !failed && std::ranges::all_of(results, [](const auto& result) {
        return result.at("status") == "completed";
    });
    Json summary{{"schema_version", 1},
                 {"activity", "simulation_comparison"},
                 {"model", "simulator"},
                 {"domain", options.run.diagnostic ? "technical" : "synthetic"},
                 {"partition", "validation"},
                 {"identity", identity},
                 {"status", failed      ? "failed"
                            : completed ? "completed"
                                        : "paused"},
                 {"workers", options.workers},
                 {"runs", results},
                 {"final_test_opened", false},
                 {"parent_frozen", true},
                 {"updated_at_utc", utc_now()},
                 {"total_seconds", std::chrono::duration<double>(Clock::now() - started).count()},
                 {"timing_scope", "comparison_excluding_shared_input"}};
    record_memory(summary);
    atomic_json_file(options.run.output / "comparison.json", summary);
    for (const auto& failure : failures) {
        if (failure) {
            std::rethrow_exception(failure);
        }
    }
    return summary;
}
} // namespace mars_titan::simulation
