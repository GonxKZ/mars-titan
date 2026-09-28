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
    std::string device = "cuda:0";
    bool diagnostic = false;
    bool resume = false;
    std::optional<std::size_t> stop_after;
    std::optional<std::size_t> vram_budget_bytes;
    std::optional<std::size_t> vram_total_bytes;
    std::optional<int> gpu_lease_fd;
};

[[nodiscard]] nlohmann::json run_ppo_experiment(
    const PpoExperimentOptions& options, const std::function<bool()>& stop_requested = {});
}
#endif
