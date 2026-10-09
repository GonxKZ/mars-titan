#include "mars_titan/quantile_dqn.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mars_titan::learning {
namespace {
constexpr int64_t actions_count = 6;
constexpr double atom_tolerance = 1e-9;

void require(bool value, std::string_view message) {
    if (!value) {
        throw std::invalid_argument(std::string(message));
    }
}

bool floating(const at::Tensor& tensor) {
    return tensor.defined() && tensor.layout() == at::kStrided &&
           (tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kDouble);
}

void require_quantiles(const at::Tensor& quantiles) {
    require(floating(quantiles) && quantiles.dim() == 3 && quantiles.size(0) > 0 &&
                quantiles.size(0) <= maximum_quantile_batch && quantiles.size(1) == actions_count &&
                quantiles.size(2) >= 2 && quantiles.size(2) <= maximum_quantiles &&
                (quantiles.device().is_cpu() || quantiles.device().is_cuda()),
            "Los cuantiles necesitan la forma [B,6,N] en FP32 o FP64");
    require(at::isfinite(quantiles.detach()).all().item<bool>(),
            "Los cuantiles contienen NaN o infinito");
}

// Número de niveles inferiores que promedia la medida de riesgo. Se exige un entero para no
// interpolar dentro de un intervalo, que sería una aproximación distinta a la declarada.
int64_t risk_atoms(int64_t quantiles, double alpha) {
    require(std::isfinite(alpha) && alpha > 0 && alpha <= 1,
            "La medida de riesgo necesita alpha en (0, 1]");
    const double atoms = alpha * static_cast<double>(quantiles);
    const auto count = static_cast<int64_t>(std::llround(atoms));
    require(count >= 1 && count <= quantiles &&
                std::abs(atoms - static_cast<double>(count)) <= atom_tolerance,
            "alpha por el número de cuantiles debe ser un entero positivo");
    return count;
}
} // namespace

at::Tensor quantile_midpoints(std::int64_t quantiles, const at::Device& device) {
    require(quantiles >= 2 && quantiles <= maximum_quantiles,
            "El número de cuantiles está fuera de sus límites");
    return ((at::arange(quantiles, at::kDouble) * 2 + 1) / (2 * static_cast<double>(quantiles)))
        .to(device);
}

at::Tensor quantile_action_scores(const at::Tensor& quantiles, double risk_alpha) {
    require_quantiles(quantiles);
    const auto atoms = risk_atoms(quantiles.size(2), risk_alpha);
    return quantiles.narrow(2, 0, atoms).mean(2);
}

at::Tensor quantile_double_targets(const at::Tensor& rewards, const at::Tensor& terminated,
                                   const at::Tensor& online_next, const at::Tensor& target_next,
                                   QuantileTargetOptions options) {
    const auto [gamma, risk_alpha] = options;
    require_quantiles(online_next);
    require_quantiles(target_next);
    const auto batch = online_next.size(0);
    const auto count = online_next.size(2);
    const auto device = online_next.device();
    require(std::isfinite(gamma) && gamma >= 0 && gamma <= 1 && floating(rewards) &&
                rewards.sizes() == at::IntArrayRef({batch}) && rewards.device() == device &&
                rewards.scalar_type() == online_next.scalar_type() && terminated.defined() &&
                terminated.layout() == at::kStrided && terminated.scalar_type() == at::kBool &&
                terminated.sizes() == rewards.sizes() && terminated.device() == device &&
                target_next.sizes() == online_next.sizes() && target_next.device() == device &&
                target_next.scalar_type() == online_next.scalar_type(),
            "Los objetivos cuantílicos necesitan recompensas, terminaciones y redes comunes");
    require(at::isfinite(rewards).all().item<bool>(), "Las recompensas QR-DQN no son finitas");
    const at::NoGradGuard no_grad;
    // Selección doble. La red online elige la acción siguiente con la misma medida de riesgo con
    // la que actúa, como la política pi_beta de la ecuación 2 de IQN.
    const auto chosen = quantile_action_scores(online_next, risk_alpha).argmax(1);
    const auto index = chosen.view({batch, 1, 1}).expand({batch, 1, count});
    const auto selected = target_next.gather(1, index).squeeze(1);
    const auto alive = (~terminated).to(rewards.scalar_type()).unsqueeze(1);
    const auto targets = rewards.unsqueeze(1) + gamma * alive * selected;
    require(at::isfinite(targets).all().item<bool>(), "Los objetivos QR-DQN no son finitos");
    return targets;
}

at::Tensor quantile_huber_loss(const at::Tensor& predicted, const at::Tensor& targets,
                               double kappa) {
    require(floating(predicted) && floating(targets) && predicted.dim() == 2 &&
                predicted.size(0) > 0 && predicted.size(0) <= maximum_quantile_batch &&
                predicted.size(1) >= 2 && predicted.size(1) <= maximum_quantiles &&
                targets.sizes() == predicted.sizes() && targets.device() == predicted.device() &&
                std::isfinite(kappa) && kappa > 0,
            "La pérdida cuantílica necesita cuantiles [B,N] comunes y kappa positivo");
    require(at::isfinite(predicted.detach()).all().item<bool>() &&
                at::isfinite(targets).all().item<bool>(),
            "La pérdida cuantílica recibe valores no finitos");
    const auto count = predicted.size(1);
    const auto current = predicted.to(at::kDouble);
    const auto target = targets.detach().to(at::kDouble);
    // u_ij = T_j - theta_i, con una fila por cuantil predicho y una columna por cada átomo del
    // objetivo, que el algoritmo 1 trata como muestras de la distribución de destino.
    const auto error = target.unsqueeze(1) - current.unsqueeze(2);
    const auto magnitude = error.abs();
    const auto huber =
        at::where(magnitude <= kappa, 0.5 * error * error, kappa * (magnitude - 0.5 * kappa));
    const auto tau = quantile_midpoints(count, predicted.device()).view({1, count, 1});
    const auto weight = (tau - (error.detach() < 0).to(at::kDouble)).abs();
    // QR-DQN, ecuación 10, no divide por kappa (la división aparece después en IQN).
    const auto result = (weight * huber).mean(2).sum(1);
    require(at::isfinite(result).all().item<bool>(), "La pérdida cuantílica ha desbordado FP64");
    return result;
}
} // namespace mars_titan::learning
