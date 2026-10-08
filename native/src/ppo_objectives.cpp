#include "mars_titan/ppo_objectives.hpp"
#include "mars_titan/ppo_policy.hpp"

#include <ATen/ATen.h>

#include <limits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mars_titan::learning {
namespace {
constexpr double normalization_tolerance = 1e-6;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::invalid_argument(std::string(message));
    }
}

at::Tensor normalized_logs(const at::Tensor& values) {
    require((at::isfinite(values) | at::isneginf(values)).all().item<bool>(),
            "Los logaritmos contienen NaN o infinito positivo");
    const auto promoted = values.to(at::kDouble);
    const auto normalization = promoted.logsumexp(1, true);
    require((normalization.abs() <= normalization_tolerance).all().item<bool>(),
            "Las probabilidades categóricas no están normalizadas");
    return promoted - normalization;
}
} // namespace

at::Tensor ppo_behavior_log_probabilities(const at::Tensor& weights) {
    require(weights.defined() && weights.layout() == at::kStrided && weights.dim() == 2 &&
                weights.size(0) > 0 && weights.size(0) <= maximum_ppo_kl_rows &&
                weights.size(1) == ppo_action_count && weights.scalar_type() == at::kFloat &&
                (weights.device().is_cpu() || weights.device().is_cuda()),
            "El muestreador necesita pesos FP32 de seis acciones en CPU o CUDA");
    require(at::isfinite(weights).all().item<bool>() && (weights >= 0).all().item<bool>(),
            "Los pesos del muestreador contienen valores no finitos o negativos");
    const auto promoted = weights.to(at::kDouble);
    const auto total = promoted.sum(1, true);
    require(((total - 1).abs() <= normalization_tolerance).all().item<bool>(),
            "Los pesos del muestreador no están normalizados");
    const auto present = promoted > 0;
    // Enmascarar antes de log evita derivar log(0) en masas ausentes.
    const auto safe = at::where(present, promoted / total, 1.);
    return at::where(present, safe.log(), -std::numeric_limits<double>::infinity());
}

void validate_ppo_behavior(const at::Tensor& weights, const at::Tensor& actions,
                           const at::Tensor& selected_logp, const at::Tensor& valid) {
    static_cast<void>(ppo_behavior_log_probabilities(weights));
    const auto shape = weights.sizes().slice(0, 1);
    require(actions.defined() && actions.layout() == at::kStrided && actions.sizes() == shape &&
                actions.scalar_type() == at::kLong && actions.device() == weights.device() &&
                selected_logp.defined() && selected_logp.layout() == at::kStrided &&
                selected_logp.sizes() == shape && selected_logp.device() == weights.device() &&
                selected_logp.scalar_type() == at::kDouble && valid.defined() &&
                valid.layout() == at::kStrided && valid.sizes() == shape &&
                valid.scalar_type() == at::kBool && valid.device() == weights.device(),
            "Las acciones, probabilidades y máscara no concuerdan con el muestreador");
    require(((actions >= 0) & (actions < ppo_action_count)).all().item<bool>() &&
                at::isfinite(selected_logp).all().item<bool>() &&
                (selected_logp <= normalization_tolerance).all().item<bool>(),
            "La acción o su logaritmo histórico no son válidos");
    const auto selected = weights.gather(1, actions.unsqueeze(1)).squeeze(1).to(at::kDouble);
    const auto expected = selected_logp.exp();
    constexpr double relative_rounding = 2e-6;
    constexpr double absolute_rounding = 2. * static_cast<double>(std::numeric_limits<float>::denorm_min());
    const auto matches = (selected > 0) &
        ((selected - expected).abs() <= expected.abs() * relative_rounding + absolute_rounding);
    require((matches | ~valid).all().item<bool>(),
            "La acción válida no corresponde a los pesos históricos registrados");
}

at::Tensor ppo_penalized_objective(const at::Tensor& logp, const at::Tensor& old_weights,
    const at::Tensor& actions, const at::Tensor& old_logp, const at::Tensor& advantages, double beta) {
    require(std::isfinite(beta) && beta > 0 && logp.defined() && logp.scalar_type() == at::kFloat &&
                logp.sizes() == old_weights.sizes() && logp.device() == old_weights.device(),
            "El actor penalizado necesita beta positiva y logits FP32 compatibles");
    const auto current = ppo_behavior_log_probabilities(logp.exp());
    const auto previous = ppo_behavior_log_probabilities(old_weights.detach());
    validate_ppo_behavior(old_weights, actions, old_logp,
                          at::ones({logp.size(0)}, logp.options().dtype(at::kBool)));
    require(advantages.defined() && advantages.layout() == at::kStrided &&
                advantages.sizes() == old_logp.sizes() && advantages.device() == logp.device() &&
                (advantages.scalar_type() == at::kFloat || advantages.scalar_type() == at::kDouble) &&
                at::isfinite(advantages).all().item<bool>(),
            "La ventaja del actor penalizado no corresponde a sus acciones");
    const auto selected = logp.gather(1, actions.unsqueeze(1)).squeeze(1).to(at::kDouble);
    const auto ratio = (selected - old_logp.detach()).exp();
    const auto result = -ratio * advantages.detach() + beta * ppo_categorical_kl(current, previous);
    require(at::isfinite(ratio).all().item<bool>() && at::isfinite(result).all().item<bool>(),
            "El objetivo penalizado desborda su precisión");
    return result;
}

at::Tensor ppo_categorical_kl(const at::Tensor& log_probabilities,
                            const at::Tensor& old_log_probabilities) {
    require(log_probabilities.defined() && old_log_probabilities.defined() &&
                log_probabilities.layout() == at::kStrided &&
                old_log_probabilities.layout() == at::kStrided &&
                log_probabilities.dim() == 2 && log_probabilities.size(0) > 0 &&
                log_probabilities.size(0) <= maximum_ppo_kl_rows &&
                log_probabilities.size(1) == ppo_action_count &&
                old_log_probabilities.sizes() == log_probabilities.sizes(),
            "La KL necesita dos matrices de hasta 65536 filas y seis acciones");
    require(log_probabilities.device() == old_log_probabilities.device() &&
                (log_probabilities.device().is_cpu() || log_probabilities.device().is_cuda()) &&
                (log_probabilities.scalar_type() == at::kFloat ||
                 log_probabilities.scalar_type() == at::kDouble) &&
                (old_log_probabilities.scalar_type() == at::kFloat ||
                 old_log_probabilities.scalar_type() == at::kDouble),
            "La KL necesita FP32 o FP64 en un mismo dispositivo CPU o CUDA");
    const auto current = normalized_logs(log_probabilities);
    const auto previous = normalized_logs(old_log_probabilities.detach());
    const auto q = previous.exp();
    const auto support = q > 0;
    require((support | at::isneginf(previous)).all().item<bool>(),
            "La política histórica pierde soporte al representar sus probabilidades");
    require(!(support & at::isneginf(current)).any().item<bool>(),
            "La política actual asigna masa cero a una acción histórica observada");
    // Evitar 0*(-inf-(-inf)) sin cambiar el soporte ni suavizar probabilidades.
    const auto difference = at::where(support, previous, 0.0) -
                            at::where(support, current, 0.0);
    const auto result = (q * difference).sum(1);
    require(at::isfinite(result).all().item<bool>(), "La divergencia categórica no es finita");
    return result;
}
} // namespace mars_titan::learning
