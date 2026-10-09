#ifndef MARS_TITAN_PPO_EXPERIMENT_HPP
#define MARS_TITAN_PPO_EXPERIMENT_HPP

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
    // Costes de la evaluación congelada sobre cintas reconstruidas. Vacío usa los de omisión.
    std::vector<double> evaluation_costs;
    std::string device = "cuda:0";
    bool diagnostic = false;
    bool resume = false;
    std::optional<std::size_t> stop_after;
    std::optional<std::size_t> vram_budget_bytes;
    std::optional<std::size_t> vram_total_bytes;
    std::optional<int> gpu_lease_fd;
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
