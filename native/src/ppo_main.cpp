#include "mars_titan/ppo_experiment.hpp"

#include <atomic>
#include <charconv>
#include <csignal>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

namespace {
// La señal solo publica una bandera sin bloqueos ni asignaciones.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<bool> stop_signal{false};
static_assert(std::atomic<bool>::is_always_lock_free);
void signal_stop(int) { stop_signal.store(true, std::memory_order_relaxed); }

template<class Number> Number number(std::string_view value) {
    Number result{};
    const auto parsed = std::from_chars(value.begin(), value.end(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.end()) {
        throw std::invalid_argument("El argumento numérico PPO no tiene un formato válido");
    }
    return result;
}

struct Arguments {
    mars_titan::learning::PpoExperimentOptions experiment;
    bool help = false;
};

Arguments parse(int argc, char** argv) {
    Arguments result;
    const std::span<char*> arguments(argv, static_cast<std::size_t>(argc));
    std::unordered_set<std::string_view> seen;
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const std::string_view name(arguments[index]);
        if (name != "--train-tape" && name != "--validation-tape" && !seen.insert(name).second) {
            throw std::invalid_argument("Hay un argumento PPO duplicado");
        }
        if (name == "--help") {
            result.help = true;
        } else if (name == "--resume") {
            result.experiment.resume = true;
        } else if (name == "--diagnostic") {
            result.experiment.diagnostic = true;
        } else {
            if (++index >= arguments.size()) {
                throw std::invalid_argument("Falta el valor de un argumento PPO");
            }
            const std::string_view value(arguments[index]);
            auto& options = result.experiment;
            if (name == "--config") {
                options.config = value;
            } else if (name == "--output") {
                options.output = value;
            } else if (name == "--train-tape") {
                options.train_tapes.emplace_back(value);
            } else if (name == "--validation-tape") {
                options.validation_tapes.emplace_back(value);
            } else if (name == "--device") {
                options.device = value;
            } else if (name == "--stop-after") {
                options.stop_after = number<std::size_t>(value);
            } else if (name == "--gpu-lease-fd") {
                options.gpu_lease_fd = number<int>(value);
            } else if (name == "--vram-budget-bytes") {
                options.vram_budget_bytes = number<std::size_t>(value);
            } else if (name == "--vram-total-bytes") {
                options.vram_total_bytes = number<std::size_t>(value);
            } else {
                throw std::invalid_argument("El argumento PPO no está reconocido: " + std::string(name));
            }
        }
    }
    if (!result.help && (result.experiment.config.empty() || result.experiment.output.empty())) {
        throw std::invalid_argument("Se requieren --config y --output para PPO");
    }
    return result;
}
}

int main(int argc, char** argv) {
    try {
        const auto arguments = parse(argc, argv);
        if (arguments.help) {
            std::cout << "mars-titan-ppo --config ARCHIVO --output DIR --train-tape DIR "
                         "--validation-tape DIR [--resume] [--stop-after N]\n"
                         "  Diagnóstico: --device cpu --diagnostic\n"
                         "  CUDA: --device cuda:0 --gpu-lease-fd FD "
                         "--vram-budget-bytes N --vram-total-bytes N\n"
                         "Entrena y valida PPO en escenarios sintéticos, sin abrir el test.\n";
            return 0;
        }
        if (std::signal(SIGINT, signal_stop) == SIG_ERR || std::signal(SIGTERM, signal_stop) == SIG_ERR) {
            throw std::runtime_error("No se pueden instalar las señales de pausa PPO");
        }
        const auto result = mars_titan::learning::run_ppo_experiment(arguments.experiment, [] {
            return stop_signal.load(std::memory_order_relaxed);
        });
        std::cout << "Estado: " << result.at("status").get<std::string>()
                  << ". Transiciones: " << result.at("transitions") << ".\n";
        return result.at("status") == "paused" ? 2 : 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
