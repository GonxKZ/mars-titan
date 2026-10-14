#ifndef MARS_TITAN_PPO_EXPERIMENT_HPP
#define MARS_TITAN_PPO_EXPERIMENT_HPP

#include "mars_titan/learning_traces.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace mars_titan::learning {
struct PpoExperimentOptions {
    std::filesystem::path config;
    std::filesystem::path output;
    std::vector<std::filesystem::path> train_tapes;
    std::vector<std::filesystem::path> validation_tapes;
    std::optional<std::filesystem::path> audit_run;
    std::vector<std::filesystem::path> audit_tapes;
    // Costes de la evaluación congelada sobre cintas reconstruidas. Si la lista está vacía,
    // la evaluación usa los costes por omisión.
    std::vector<double> evaluation_costs;
    // Registrar acción y logits de cada decisión de la evaluación congelada (decisions.json).
    // Solo con --audit-run sobre cintas reconstruidas. Sirve para comparar CPU y CUDA.
    bool decisions = false;
    // Modos sin aprendizaje que la protección admite sin rebajarse. Ambos activan el cerrojo de
    // solo evaluación antes de leer nada y no crean entrenador, controlador ni Adam terminal.
    // La evaluación inicial evalúa con argmax la política inicial de la configuración sobre
    // cintas de evaluación. La comprobación de fuentes lee y valida las cintas de ajuste y
    // validación como el ajuste y publica su identidad.
    bool initial_evaluation = false;
    bool source_check = false;
    std::string device = "cuda:0";
    bool diagnostic = false;
    bool resume = false;
    std::optional<std::size_t> stop_after;
    std::optional<std::size_t> vram_budget_bytes;
    std::optional<std::size_t> vram_total_bytes;
    std::optional<int> gpu_lease_fd;
    // Trazas de aprendizaje de #448 en `output/traces`. Solo observan y no entran en la
    // identidad de la ejecución.
    LearningTraceOptions learning_traces;
};

[[nodiscard]] nlohmann::json run_ppo_experiment(const PpoExperimentOptions& options,
                                                const std::function<bool()>& stop_requested = {});

// Admisión compartida por PPO y KLPO: CPU solo en el diagnóstico explícito de hasta 32
// transiciones y CUDA con el bloqueo GPU heredado y su presupuesto de VRAM.
void require_policy_device(const PpoExperimentOptions& options, std::size_t total_transitions);
void admit_policy_gpu(const PpoExperimentOptions& options);
// Hilos, algoritmos deterministas y FP32 IEEE antes de crear tensores.
void configure_policy_runtime(const PpoExperimentOptions& options);
} // namespace mars_titan::learning
#endif
