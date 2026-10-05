#include "mars_titan/financial_controls.hpp"

#include <atomic>
#include <csignal>
#include <exception>
#include <iostream>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<bool> stopped{false};
static_assert(std::atomic<bool>::is_always_lock_free);
void stop(int) { stopped.store(true, std::memory_order_relaxed); }
} // namespace

int main(int argc, char** argv) {
    try {
        mars_titan::simulation::FinancialControlsOptions options;
        std::set<std::string_view> seen;
        const std::span<char*> arguments(argv, static_cast<std::size_t>(argc));
        for (std::size_t index = 1; index < arguments.size(); ++index) {
            const std::string_view name(arguments[index]);
            if (name == "--help" && arguments.size() == 2) {
                std::cout << "mars-titan-financial-controls --campaign DIR --campaign-sha256 SHA256\n"
                             "  --freeze-sha256 SHA256 --identity-sha256 SHA256 --scenarios INDEX --output DIR\n"
                             "Las huellas corresponden a los archivos brutos campaign.json, freeze.json e identity.json.\n"
                             "Evalúa seis referencias sin aprendizaje sobre la auditoría ya congelada.\n"
                             "Usa efectivo hasta el cierre 64 y costes de 0, 10 y 25 puntos básicos.\n"
                             "La salida debe ser nueva. Una interrupción conserva los mundos terminados,\n"
                             "pero no admite recuperación ni abre el test final.\n";
                return 0;
            }
            if (!seen.insert(name).second || ++index >= arguments.size()) {
                throw std::invalid_argument("Hay un argumento repetido o sin valor");
            }
            const std::string_view value(arguments[index]);
            if (name == "--campaign") {
                options.campaign = value;
            } else if (name == "--campaign-sha256") {
                options.campaign_sha256 = value;
            } else if (name == "--freeze-sha256") {
                options.freeze_sha256 = value;
            } else if (name == "--identity-sha256") {
                options.identity_sha256 = value;
            } else if (name == "--scenarios") {
                options.scenarios = value;
            } else if (name == "--output") {
                options.output = value;
            } else {
                throw std::invalid_argument("El argumento no está reconocido");
            }
        }
        if (std::signal(SIGINT, stop) == SIG_ERR || std::signal(SIGTERM, stop) == SIG_ERR) {
            throw std::runtime_error("No se pueden registrar las señales de interrupción");
        }
        const auto result = mars_titan::simulation::run_financial_controls(
            options, [] { return stopped.load(std::memory_order_relaxed); });
        std::cout << "Estado: " << result.at("status").get<std::string>() << '\n';
        return result.at("status") == "completed" ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
