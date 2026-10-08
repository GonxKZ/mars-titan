#include "mars_titan/ppo_objectives.hpp"
#include "mars_titan/ppo_policy.hpp"

#include <ATen/ATen.h>

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
