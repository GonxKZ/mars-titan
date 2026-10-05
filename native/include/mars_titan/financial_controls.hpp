#ifndef MARS_TITAN_FINANCIAL_CONTROLS_HPP
#define MARS_TITAN_FINANCIAL_CONTROLS_HPP

#include "mars_titan/simulation_files.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace mars_titan::simulation {

struct FinancialControlsOptions {
    std::filesystem::path campaign;
    std::filesystem::path scenarios;
    std::filesystem::path output;
    // Las huellas corresponden al archivo bruto, incluidos espacios y salto final.
    std::string campaign_sha256;
    std::string freeze_sha256;
    std::string identity_sha256;
};

[[nodiscard]] uint8_t warmed_reference_action(ReferencePolicy policy, std::size_t cursor,
                                             std::size_t decision_start);
[[nodiscard]] FinancialMetrics evaluate_financial_control(
    std::shared_ptr<const MarketTape> tape, ReferencePolicy policy, Parameters parameters,
    std::size_t decision_start);
[[nodiscard]] nlohmann::json run_financial_controls(
    const FinancialControlsOptions& options, const std::function<bool()>& stop_requested = {});

} // namespace mars_titan::simulation

#endif
