#include "mars_titan/memory_stress.hpp"

#include <ATen/Parallel.h>
#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {
using namespace mars_titan::stress;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr uint64_t bytes_per_kibibyte = 1024;
struct Options {
    Config config;
    std::filesystem::path output;
    std::filesystem::path checkpoint;
    std::filesystem::path resume;
    std::filesystem::path configuration;
    std::size_t stop_after = 0;
    bool help = false;
};
void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::invalid_argument(std::string(message));
    }
}

template <typename Number> Number parse_number(std::string_view text) {
    Number result{};
    const auto converted = std::from_chars(text.begin(), text.end(), result);
    require(converted.ec == std::errc{} && converted.ptr == text.end(),
            "Argumento numérico inválido");
    return result;
}
std::string read_file(const std::filesystem::path& path) {
    require(std::filesystem::is_regular_file(path), "La entrada debe ser un archivo regular");
    const auto size = std::filesystem::file_size(path);
    require(size > 0 && size <= maximum_stress_archive_bytes,
            "La entrada supera el límite o está vacía");
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "No se pudo abrir el archivo de entrada");
    std::string data(static_cast<std::size_t>(size), '\0');
    input.read(data.data(), static_cast<std::streamsize>(data.size()));
    require(input.gcount() == static_cast<std::streamsize>(data.size()) && input.peek() == EOF,
            "La entrada cambió durante la lectura o no se leyó completa");
    return data;
}

// Publica solo archivos completos. La sincronización incluye el directorio de destino.
void write_atomic(const std::filesystem::path& path, std::string_view data) {
    require(data.size() <= maximum_stress_archive_bytes, "La salida supera el límite");
    auto temporary = path;
    temporary += ".pending." + std::to_string(::getpid());
    constexpr int flags = O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW;
    constexpr mode_t private_mode = 0600;
    // POSIX exige modo explícito al crear el archivo.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    const int descriptor = ::open(temporary.c_str(), flags, private_mode);
    if (descriptor < 0) {
        throw std::system_error(errno, std::generic_category(),
                                "No se pudo crear el archivo provisional");
    }
    bool closed = false;
    try {
        std::size_t offset = 0;
        while (offset < data.size()) {
            const auto written =
                ::write(descriptor, data.substr(offset).data(), data.size() - offset);
            if (written < 0 && errno == EINTR) {
                continue;
            }
            if (written <= 0) {
                throw std::system_error(errno, std::generic_category(),
                                        "No se pudo escribir la salida");
            }
            offset += static_cast<std::size_t>(written);
        }
        if (::fsync(descriptor) != 0) {
            throw std::system_error(errno, std::generic_category(),
                                    "No se pudo sincronizar la salida");
        }
        const int status = ::close(descriptor);
        closed = true;
        if (status != 0) {
            throw std::system_error(errno, std::generic_category(), "No se pudo cerrar la salida");
        }
        std::filesystem::rename(temporary, path);
        const auto directory =
            path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
        // Esta llamada POSIX no crea archivos ni necesita un modo.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
        const int parent = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (parent < 0) {
            throw std::system_error(errno, std::generic_category(),
                                    "No se pudo abrir el directorio");
        }
        const int synced = ::fsync(parent);
        const int sync_errno = errno;
        const int close_status = ::close(parent);
        if (synced != 0) {
            throw std::system_error(sync_errno, std::generic_category(),
                                    "No se pudo sincronizar el directorio");
        }
        if (close_status != 0) {
            throw std::system_error(errno, std::generic_category(),
                                    "No se pudo cerrar el directorio");
        }
    } catch (...) {
        if (!closed) {
            static_cast<void>(::close(descriptor));
        }
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

Options parse_options(std::span<char*> arguments) {
    Options options;
    bool device_declared = false;
    bool overrides = false;
    std::set<std::string_view> used;
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const std::string_view name(arguments[index]);
        require(used.insert(name).second, "Argumento repetido");
        if (name == "--help") {
            options.help = true;
            continue;
        }
        require(++index < arguments.size(), "Falta el valor de un argumento");
        const std::string_view value(arguments[index]);
        if (name == "--output") {
            options.output = value;
        } else if (name == "--checkpoint") {
            options.checkpoint = value;
        } else if (name == "--resume") {
            options.resume = value;
        } else if (name == "--config") {
            options.configuration = value;
        } else if (name == "--stop-after") {
            options.stop_after = parse_number<std::size_t>(value);
        } else if (name == "--device") {
            require(value == "cpu", "Este control nativo requiere --device cpu");
            device_declared = true;
        } else {
            overrides = true;
            if (name == "--steps") {
                options.config.steps = parse_number<std::size_t>(value);
            } else if (name == "--capacity") {
                options.config.capacity = parse_number<std::size_t>(value);
            } else if (name == "--regime-length") {
                options.config.regime_length = parse_number<std::size_t>(value);
            } else if (name == "--delay") {
                options.config.delay = parse_number<std::size_t>(value);
            } else if (name == "--data-seed") {
                options.config.data_seed = parse_number<uint64_t>(value);
            } else if (name == "--retention-seed") {
                options.config.retention_seed = parse_number<uint64_t>(value);
            } else if (name == "--noise") {
                options.config.noise = parse_number<double>(value);
            } else if (name == "--invalid-every") {
                options.config.invalid_every = parse_number<std::size_t>(value);
            } else if (name == "--scenario") {
                options.config.scenario = scenario_from_name(value);
            } else {
                throw std::invalid_argument("Argumento desconocido: " + std::string(name));
            }
        }
    }
    if (options.help) {
        return options;
    }
    require(device_declared, "Declare explícitamente --device cpu");
    require(options.stop_after == 0 || !options.checkpoint.empty(),
            "Una parada parcial necesita --checkpoint");
    require((options.resume.empty() && options.configuration.empty()) || !overrides,
            "La configuración y la recuperación no admiten cambios de parámetros");
    require(options.resume.empty() || options.configuration.empty(), "Elija --config o --resume");
    const auto same = [](const auto& left, const auto& right) {
        return !left.empty() && !right.empty() &&
               std::filesystem::weakly_canonical(left) == std::filesystem::weakly_canonical(right);
    };
    require(!same(options.output, options.checkpoint) && !same(options.output, options.resume) &&
                !same(options.output, options.configuration) &&
                !same(options.checkpoint, options.configuration),
            "Las salidas deben conservar los archivos de entrada y el checkpoint");
    if (!options.configuration.empty()) {
        options.config = read_config(read_file(options.configuration));
    }
    return options;
}

int run(const Options& options, Clock::time_point started) {
    std::string recovery;
    Config config = options.config;
    if (!options.resume.empty()) {
        recovery = read_file(options.resume);
        config = read_config(recovery);
    }
    Experiment experiment(config);
    if (!recovery.empty()) {
        experiment.restore(recovery);
    }
    experiment.run_until(options.stop_after == 0 ? config.steps : options.stop_after);
    std::size_t checkpoint_bytes = 0;
    double checkpoint_seconds = 0;
    if (!options.checkpoint.empty()) {
        const auto before = Clock::now();
        const auto checkpoint = experiment.checkpoint();
        checkpoint_bytes = checkpoint.size();
        write_atomic(options.checkpoint, checkpoint);
        checkpoint_seconds = std::chrono::duration<double>(Clock::now() - before).count();
    }
    auto report = Json::parse(experiment.report());
    struct rusage usage{};
    if (::getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::system_error(errno, std::generic_category(), "No se pudo medir la RAM");
    }
    // ru_maxrss es el campo documentado de rusage en Linux.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
    const auto peak_kibibytes = usage.ru_maxrss;
    report["measurements"]["peak_rss_bytes"] =
        static_cast<uint64_t>(peak_kibibytes) * bytes_per_kibibyte;
    report["measurements"]["checkpoint_bytes"] = checkpoint_bytes;
    report["measurements"]["checkpoint_seconds"] = checkpoint_seconds;
    report["measurements"]["main_seconds_before_report_write"] =
        std::chrono::duration<double>(Clock::now() - started).count();
    const auto text = report.dump(2) + '\n';
    if (options.output.empty()) {
        std::cout << text;
        if (!std::cout) {
            throw std::runtime_error("No se pudo publicar el informe");
        }
    } else {
        write_atomic(options.output, text);
    }
    return 0;
}
} // namespace
int main(int argc, char** argv) {
    const auto started = Clock::now();
    try {
        const auto options = parse_options({argv, static_cast<std::size_t>(argc)});
        if (options.help) {
            std::cout << "Escenarios controlados de memoria y ruido, C++20 y CPU\n"
                         "--device cpu [--config archivo | --resume checkpoint | parámetros]\n"
                         "--scenario recurrence|persistent|noise|outliers\n"
                         "--steps N --capacity N --regime-length N --delay N\n"
                         "--data-seed N --retention-seed N --noise X --invalid-every N\n"
                         "--output informe --checkpoint archivo --stop-after N\n";
            return 0;
        }
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        return run(options, started);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
