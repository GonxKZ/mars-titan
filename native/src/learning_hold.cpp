#include "mars_titan/learning_hold.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

namespace mars_titan::learning {
namespace {
constexpr std::uintmax_t maximum_hold_bytes = std::uintmax_t{1} << 20U;
} // namespace

std::filesystem::path learning_hold_path() {
    if (const char* selected = std::getenv("MARS_TITAN_TRAINING_HOLD"); selected != nullptr) {
        // Una ruta vacía no equivale a una protección ausente.
        if (*selected == '\0') {
            throw std::runtime_error("MARS_TITAN_TRAINING_HOLD no puede estar vacía");
        }
        return selected;
    }
    const char* home = std::getenv("HOME");
    if (home == nullptr || *home == '\0') {
        throw std::runtime_error("No se puede localizar la protección del aprendizaje sin HOME");
    }
    return std::filesystem::path(home) / ".local/state/mars-titan/training-hold-2000.json";
}

bool learning_blocked(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        return false;
    }
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path) > maximum_hold_bytes) {
        throw std::runtime_error("La protección del aprendizaje no es un archivo regular acotado");
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("No se puede leer la protección del aprendizaje");
    }
    const auto document = nlohmann::json::parse(stream);
    if (!document.is_object() || !document.contains("training_allowed") ||
        !document.at("training_allowed").is_boolean()) {
        throw std::runtime_error(
            "La protección del aprendizaje no declara training_allowed como booleano");
    }
    return !document.at("training_allowed").get<bool>();
}

void require_learning_allowed(std::string_view action) {
    const auto path = learning_hold_path();
    if (learning_blocked(path)) {
        throw std::runtime_error("Bloqueo de aprendizaje vigente: " + std::string(action) +
                                 " no se ejecuta mientras " + path.string() +
                                 " no declare training_allowed verdadero");
    }
}
} // namespace mars_titan::learning
