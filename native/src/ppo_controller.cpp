#include "mars_titan/ppo_controller.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mars_titan::learning {
namespace {
constexpr std::array<std::string_view, 4> identities{
    "ppo_clip_legacy", "ppo_clip_full_kl_v1", "ppo_kl_penalty_adaptive_v1", "ppo_clip_kl_epoch_stop_v1"};
constexpr int64_t maximum_epochs = 64;
constexpr int64_t maximum_rows = 1 << 20;
constexpr double negative_kl_tolerance = 1e-12;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

double checked_kl(int64_t rows, std::optional<double> value) {
    require(rows >= 0 && rows <= maximum_rows && (rows == 0) == !value.has_value(),
            "La medición KL no corresponde al número de filas válidas");
    if (!value) { return 0; }
    require(std::isfinite(*value) && *value >= -negative_kl_tolerance,
            "La medición KL no es finita o excede el error negativo de redondeo");
    return std::max(0., *value);
}
} // namespace

std::string_view objective_id(PpoObjectiveKind kind) {
    const auto index = static_cast<std::size_t>(kind);
    require(index < identities.size(), "El objetivo PPO no está admitido");
    return identities.at(index);
}

PpoObjectiveKind objective_kind(std::string_view id) {
    for (std::size_t index = 0; index < identities.size(); ++index) {
        if (identities.at(index) == id) { return static_cast<PpoObjectiveKind>(index); }
    }
    throw std::invalid_argument("La identidad del objetivo PPO no está admitida");
}

void PpoObjectiveConfig::validate() const {
    static_cast<void>(objective_id(kind));
    require(std::isfinite(target_kl) && target_kl > 0 &&
                target_kl <= std::numeric_limits<double>::max() / ppo_kl_band &&
                std::isfinite(beta_initial) && std::isfinite(beta_min) && std::isfinite(beta_max) &&
                beta_min > 0 && beta_min <= beta_initial && beta_initial <= beta_max,
            "El controlador PPO necesita un umbral y un intervalo positivo de beta");
    require(kind == PpoObjectiveKind::kl_penalty_adaptive ||
                (beta_initial == 1 && beta_min == ppo_default_beta_min && beta_max == ppo_default_beta_max),
            "Solo el objetivo penalizado admite configurar beta");
    require(kind == PpoObjectiveKind::kl_penalty_adaptive || kind == PpoObjectiveKind::clip_kl_epoch_stop ||
                target_kl == ppo_default_target_kl,
            "El diagnóstico sin controlador no admite configurar umbral");
}

PpoEpochDecision epoch_decision(const PpoObjectiveConfig& config, int64_t completed_epochs,
    int64_t configured_epochs, int64_t valid_rows, std::optional<double> mean_kl) {
    config.validate();
    require(config.enabled() && configured_epochs > 0 && configured_epochs <= maximum_epochs &&
                completed_epochs >= 0 && completed_epochs <= configured_epochs &&
                (valid_rows == 0 ? completed_epochs == 0 : completed_epochs > 0),
            "La decisión KL necesita una época completa y un objetivo explícito");
    const auto value = checked_kl(valid_rows, mean_kl);
    const bool exceeded = valid_rows > 0 && value > config.target_kl * ppo_kl_band;
    return {exceeded, config.kind == PpoObjectiveKind::clip_kl_epoch_stop && exceeded &&
                         completed_epochs < configured_epochs};
}

double next_beta(const PpoObjectiveConfig& config, double current_beta,
                 std::optional<double> final_kl, int64_t valid_rows) {
    config.validate();
    require(config.kind == PpoObjectiveKind::kl_penalty_adaptive && std::isfinite(current_beta) &&
                current_beta >= config.beta_min && current_beta <= config.beta_max,
            "La adaptación de beta necesita el objetivo penalizado y su intervalo");
    const auto value = checked_kl(valid_rows, final_kl);
    if (valid_rows == 0) { return current_beta; }
    if (value < config.target_kl / ppo_kl_band) { return std::max(config.beta_min, current_beta / 2); }
    if (value > config.target_kl * ppo_kl_band) {
        return current_beta >= config.beta_max / 2 ? config.beta_max : current_beta * 2;
    }
    return current_beta;
}

void PpoControllerState::validate(const PpoObjectiveConfig& config, int64_t adam_steps, int64_t epochs) const {
    config.validate();
    require(config.enabled() && epochs > 0 && epochs <= maximum_epochs &&
                completed_rollouts >= 0 && completed_rollouts <= maximum_rows &&
                optimizer_steps == adam_steps && adam_steps >= 0 &&
                completed_epochs >= 0 && skipped_epochs >= 0,
            "El estado del controlador no concuerda con Adam o su presupuesto");
    static_cast<void>(checked_kl(valid_rows, full_kl));
    require(std::isfinite(beta) && std::isfinite(last_beta) &&
                beta >= config.beta_min && beta <= config.beta_max &&
                last_beta >= config.beta_min && last_beta <= config.beta_max,
            "El checkpoint contiene una beta incompatible");
    if (completed_rollouts == 0) {
        require(adam_steps == 0 && valid_rows == 0 && completed_epochs == 0 && skipped_epochs == 0 &&
                    !threshold_exceeded && beta == config.beta_initial && last_beta == beta,
                "El estado inicial del controlador no es coherente");
        return;
    }
    if (valid_rows == 0) {
        require(completed_epochs == 0 && skipped_epochs == 0 && !threshold_exceeded && beta == last_beta,
                "Un rollout vacío no cambia el controlador");
        return;
    }
    require(completed_epochs > 0 && completed_epochs <= epochs && skipped_epochs == epochs - completed_epochs &&
                adam_steps >= completed_epochs,
            "Las épocas confirmadas no corresponden al controlador");
    const auto decision = epoch_decision(config, completed_epochs, epochs, valid_rows, full_kl);
    require(threshold_exceeded == decision.threshold_exceeded &&
                (skipped_epochs == 0 || decision.stop_remaining_epochs),
            "El checkpoint altera la regla de parada KL");
    const auto expected = config.kind == PpoObjectiveKind::kl_penalty_adaptive
        ? next_beta(config, last_beta, full_kl, valid_rows) : config.beta_initial;
    require(beta == expected && (config.kind == PpoObjectiveKind::kl_penalty_adaptive || last_beta == expected),
            "La beta guardada no corresponde a la medición confirmada");
}
} // namespace mars_titan::learning
