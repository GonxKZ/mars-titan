#ifndef MARS_TITAN_PPO_INPUTS_HPP
#define MARS_TITAN_PPO_INPUTS_HPP

#include "mars_titan/financial_batch.hpp"

#include <filesystem>
#include <optional>

namespace mars_titan::learning {

[[nodiscard]] std::optional<simulation::ContextTape>
load_ppo_context(const std::filesystem::path& directory, const simulation::MarketTape& market);
[[nodiscard]] simulation::BatchInput load_ppo_input(const std::filesystem::path& directory);

} // namespace mars_titan::learning
#endif
