#include "mars_titan/policy_command.hpp"

#include <atomic>
#include <charconv>
#include <csignal>
#include <exception>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

#if defined(__linux__)
#include <sys/prctl.h>
#include <unistd.h>
#endif

namespace mars_titan::learning {
namespace {
// La señal solo publica una bandera sin bloqueos ni asignaciones.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<bool> stop_signal{false};
static_assert(std::atomic<bool>::is_always_lock_free);
void signal_stop(int) { stop_signal.store(true, std::memory_order_relaxed); }

template <class Number> Number number(std::string_view value) {
    Number result{};
    const auto parsed = std::from_chars(value.begin(), value.end(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.end()) {
        throw std::invalid_argument("El argumento numérico PPO no tiene un formato válido");
    }
    return result;
}

struct Arguments {
    PpoExperimentOptions experiment;
    std::optional<int> parent_pid;
    bool help = false;
    bool capabilities = false;
};

void bind_to_parent(const std::optional<int>& parent_pid) {
    if (!parent_pid) {
        return;
    }
    if (*parent_pid <= 0) {
        throw std::invalid_argument("El PID del padre supervisor debe ser positivo");
    }
#if defined(__linux__)
    // prctl recibe argumentos variables definidos por la ABI de Linux.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    if (::prctl(PR_SET_PDEATHSIG, static_cast<unsigned long>(SIGKILL), 0UL, 0UL, 0UL) != 0) {
        throw std::runtime_error("No se pudo vincular la ejecución a la vida del padre supervisor");
    }
    if (::getppid() != static_cast<pid_t>(*parent_pid)) {
        throw std::runtime_error("El padre supervisor cambió antes de iniciar la ejecución");
    }
#else
    throw std::runtime_error("La vigilancia de muerte del padre necesita Linux");
#endif
}

Arguments parse(int argc, char** argv) {
    Arguments result;
    const std::span<char*> arguments(argv, static_cast<std::size_t>(argc));
    std::unordered_set<std::string_view> seen;
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const std::string_view name(arguments[index]);
        if (name != "--train-tape" && name != "--validation-tape" && name != "--audit-tape" &&
            name != "--evaluation-cost" && !seen.insert(name).second) {
            throw std::invalid_argument("Hay un argumento PPO duplicado");
        }
        if (name == "--help") {
            result.help = true;
        } else if (name == "--capabilities") {
            result.capabilities = true;
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
            } else if (name == "--audit-run") {
                options.audit_run = value;
            } else if (name == "--audit-tape") {
                options.audit_tapes.emplace_back(value);
            } else if (name == "--evaluation-cost") {
                options.evaluation_costs.push_back(number<double>(value));
            } else if (name == "--device") {
                options.device = value;
            } else if (name == "--stop-after") {
                options.stop_after = number<std::size_t>(value);
            } else if (name == "--parent-pid") {
                result.parent_pid = number<int>(value);
            } else if (name == "--gpu-lease-fd") {
                options.gpu_lease_fd = number<int>(value);
            } else if (name == "--vram-budget-bytes") {
                options.vram_budget_bytes = number<std::size_t>(value);
            } else if (name == "--vram-total-bytes") {
                options.vram_total_bytes = number<std::size_t>(value);
            } else {
                throw std::invalid_argument("El argumento PPO no está reconocido: " +
                                            std::string(name));
            }
        }
    }
    if (result.capabilities && arguments.size() != 2) {
        throw std::invalid_argument("--capabilities no admite otros argumentos");
    }
    if (!result.help && !result.capabilities &&
        (result.experiment.config.empty() || result.experiment.output.empty())) {
        throw std::invalid_argument("Se requieren --config y --output para PPO");
    }
    if (!result.experiment.evaluation_costs.empty() && !result.experiment.audit_run) {
        throw std::invalid_argument("--evaluation-cost solo se admite con --audit-run");
    }
    return result;
}
} // namespace

nlohmann::json policy_capabilities(const PolicyCommand& command) {
    nlohmann::json names = nlohmann::json::array();
    for (const auto name : command.capabilities) {
        names.push_back(name);
    }
    return nlohmann::json{{"schema_version", 1},
                          {"binary", command.name},
                          {"capabilities", names},
                          {"native_source_sha256", MARS_TITAN_NATIVE_SOURCE_SHA256},
                          {"native_build_sha256", MARS_TITAN_NATIVE_BUILD_SHA256}};
}

int run_policy_command(int argc, char** argv, const PolicyCommand& command) {
    try {
        const auto arguments = parse(argc, argv);
        if (arguments.help) {
            std::cout << command.help;
            return 0;
        }
        if (arguments.capabilities) {
            std::cout << policy_capabilities(command).dump() << '\n';
            return 0;
        }
        bind_to_parent(arguments.parent_pid);
        if (std::signal(SIGINT, signal_stop) == SIG_ERR ||
            std::signal(SIGTERM, signal_stop) == SIG_ERR) {
            throw std::runtime_error("No se pueden instalar las señales de pausa PPO");
        }
        const auto result = command.run(
            arguments.experiment, [] { return stop_signal.load(std::memory_order_relaxed); });
        std::cout << "Estado: " << result.at("status").get<std::string>();
        if (result.contains("confirmed_episodes")) {
            std::cout << ". Episodios auditados: " << result.at("confirmed_episodes");
        } else {
            std::cout << ". Transiciones: " << result.at("transitions");
        }
        std::cout << ".\n";
        return result.at("status") == "paused" ? 2 : 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}

} // namespace mars_titan::learning
