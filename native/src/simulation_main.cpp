#include "mars_titan/simulation_files.hpp"

#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

namespace {
using namespace mars_titan::simulation;
// Las señales solo publican una bandera atómica sin bloqueos ni asignaciones.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<bool> stop_signal{false};
static_assert(std::atomic<bool>::is_always_lock_free);

void signal_stop(int) { stop_signal.store(true, std::memory_order_relaxed); }

struct Options {
    std::filesystem::path input;
    RunOptions run;
    std::size_t workers = 1;
    bool compare = false;
    bool help = false;
};

template <typename T> T number(std::string_view text) {
    T value{};
    const auto result = std::from_chars(text.begin(), text.end(), value);
    if (result.ec != std::errc{} || result.ptr != text.end()) {
        throw std::invalid_argument("El argumento numérico no tiene un formato válido");
    }
    return value;
}

Options parse(int argc, char** argv) {
    Options result;
    const std::span<char*> arguments(argv, static_cast<std::size_t>(argc));
    std::unordered_set<std::string_view> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string_view name(arguments[static_cast<std::size_t>(i)]);
        if (!seen.insert(name).second) {
            throw std::invalid_argument("Hay un argumento duplicado");
        }
        if (name == "--help") {
            result.help = true;
        } else if (name == "--resume") {
            result.run.resume = true;
        } else if (name == "--diagnostic") {
            result.run.diagnostic = true;
        } else if (name == "--compare") {
            result.compare = true;
        } else {
            if (++i >= argc) {
                throw std::invalid_argument("Falta el valor de un argumento");
            }
            const std::string_view value(arguments[static_cast<std::size_t>(i)]);
            if (name == "--input") {
                result.input = value;
            } else if (name == "--output") {
                result.run.output = value;
            } else if (name == "--policy") {
                result.run.policy = parse_policy(std::string(value));
            } else if (name == "--cost-bps") {
                result.run.parameters.cost_bps = number<double>(value);
            } else if (name == "--capital") {
                result.run.parameters.capital = number<double>(value);
            } else if (name == "--participation") {
                result.run.parameters.participation = number<double>(value);
            } else if (name == "--workers") {
                result.workers = number<std::size_t>(value);
            } else if (name == "--stop-after") {
                result.run.stop_after = number<std::size_t>(value);
            } else if (name == "--checkpoint-steps") {
                result.run.checkpoint_steps = number<std::size_t>(value);
            } else {
                throw std::invalid_argument("El argumento no está reconocido: " +
                                            std::string(name));
            }
        }
    }
    if (result.help) {
        return result;
    }
    if (result.input.empty() || result.run.output.empty()) {
        throw std::invalid_argument("Se requieren --input y --output");
    }
    if (!result.compare && result.workers != 1) {
        throw std::invalid_argument("--workers requiere --compare");
    }
    if (result.compare && (seen.contains("--policy") || seen.contains("--cost-bps"))) {
        throw std::invalid_argument("--compare fija las tres políticas y los tres costes");
    }
    if (!std::isfinite(result.run.parameters.cost_bps) ||
        !std::isfinite(result.run.parameters.capital) ||
        !std::isfinite(result.run.parameters.participation)) {
        throw std::invalid_argument("Los parámetros numéricos deben ser finitos");
    }
    return result;
}

bool contains(const std::filesystem::path& parent, const std::filesystem::path& child) {
    const auto prefix = std::filesystem::weakly_canonical(parent);
    const auto path = std::filesystem::weakly_canonical(child);
    auto a = prefix.begin();
    auto b = path.begin();
    while (a != prefix.end() && b != path.end() && *a == *b) {
        ++a;
        ++b;
    }
    return a == prefix.end();
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (options.help) {
            std::cout << "mars-titan-sim --input DIR --output DIR [--policy "
                         "cash|hold_initial|rebalance_25|rebalance_50|rebalance_75|rebalance_100]\n"
                         "  [--cost-bps N] [--capital N] [--participation N] [--resume]\n"
                         "  [--stop-after N] [--checkpoint-steps N] [--diagnostic]\n"
                         "  [--compare --workers 1|2|4|8]\n"
                         "Ejecuta referencias deterministas sobre validación sintética, sin abrir "
                         "el test.\n";
            return 0;
        }
        if (contains(options.input, options.run.output) ||
            contains(options.run.output, options.input)) {
            throw std::invalid_argument("La salida debe permanecer separada de la fuente");
        }
        if (std::signal(SIGINT, signal_stop) == SIG_ERR ||
            std::signal(SIGTERM, signal_stop) == SIG_ERR) {
            throw std::runtime_error("No se pueden instalar las señales de parada");
        }
        const auto started = std::chrono::steady_clock::now();
        const auto tape = mars_titan::simulation::load_market_tape(options.input);
        const auto read_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const auto stop = [] { return stop_signal.load(std::memory_order_relaxed); };
        const auto result =
            options.compare
                ? mars_titan::simulation::run_comparison(tape, {options.run, options.workers}, stop)
                : mars_titan::simulation::run_reference(tape, options.run, stop);
        std::cout << "Estado: " << result.at("status").get<std::string>()
                  << ". Lectura compartida: " << read_seconds << " s.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
