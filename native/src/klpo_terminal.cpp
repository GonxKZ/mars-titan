#include "mars_titan/klpo_terminal.hpp"

#include <ATen/ATen.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mars_titan::learning {
namespace {
constexpr int64_t actions_count = 6;
constexpr double normalization_tolerance = 1e-6;
constexpr std::size_t fixed_buffer_bytes = 65536;
constexpr std::size_t bytes_per_action = 192;
constexpr std::size_t bytes_per_decision = 128;
constexpr std::size_t bytes_per_trajectory = 128;

void require(bool value, std::string_view message) {
    if (!value) {
        throw std::invalid_argument(std::string(message));
    }
}

bool floating(const at::Tensor& tensor) {
    return tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kDouble;
}

at::Tensor normalized(const at::Tensor& logs, const at::Tensor& mask) {
    // El padding se neutraliza antes de exp y de cualquier reducción.
    const auto safe = at::where(mask.unsqueeze(-1), logs, -std::log(actions_count));
    require(at::isfinite(safe).all().item<bool>(),
            "Los logaritmos activos necesitan valores finitos");
    require((safe.detach().exp() > 0).all().item<bool>(),
            "Una probabilidad pierde soporte en la precisión de entrada");
    const auto promoted = safe.to(at::kDouble);
    const auto total = promoted.logsumexp(-1, true);
    require((total.abs() <= normalization_tolerance).all().item<bool>(),
            "Las seis probabilidades deben estar normalizadas");
    return promoted - total;
}
} // namespace

at::Tensor klpo_terminal_full_loss(const at::Tensor& logp, const at::Tensor& logq,
                                   const at::Tensor& actions, const at::Tensor& terminal_returns,
                                   const at::Tensor& policy_mask, double beta,
                                   KlpoTerminalBudget budget) {
    require(logp.defined() && logq.defined() && actions.defined() && terminal_returns.defined() &&
                policy_mask.defined() && logp.layout() == at::kStrided &&
                logq.layout() == at::kStrided && actions.layout() == at::kStrided &&
                terminal_returns.layout() == at::kStrided && policy_mask.layout() == at::kStrided &&
                logp.dim() == 3 && logp.size(0) > 0 &&
                logp.size(0) <= maximum_klpo_terminal_batch && logp.size(1) > 0 &&
                logp.size(1) <= maximum_klpo_terminal_length && logp.size(2) == actions_count &&
                logq.sizes() == logp.sizes(),
            "KLPO terminal necesita lotes acotados [B,T,6]");
    const auto batch = logp.size(0), length = logp.size(1);
    require(actions.sizes() == at::IntArrayRef({batch, length}) &&
                policy_mask.sizes() == actions.sizes() &&
                terminal_returns.sizes() == at::IntArrayRef({batch}) &&
                actions.scalar_type() == at::kLong && policy_mask.scalar_type() == at::kBool &&
                floating(logp) && floating(logq) && floating(terminal_returns),
            "Las acciones, máscara o retornos no conservan forma y tipo");
    const auto device = logp.device();
    require((device.is_cpu() || device.is_cuda()) && logq.device() == device &&
                actions.device() == device && terminal_returns.device() == device &&
                policy_mask.device() == device && std::isfinite(beta) && beta > 0,
            "KLPO terminal requiere un dispositivo común y beta positiva");
    const auto decisions = static_cast<std::size_t>(batch * length);
    const auto estimated = fixed_buffer_bytes +
                           decisions * (static_cast<std::size_t>(actions_count) * bytes_per_action +
                                        bytes_per_decision) +
                           static_cast<std::size_t>(batch) * bytes_per_trajectory;
    require(budget.max_working_bytes <= maximum_klpo_terminal_bytes &&
                estimated <= budget.max_working_bytes,
            "Los buffers propios de KLPO terminal superan el presupuesto");
    require(
        policy_mask.select(1, 0).all().item<bool>() &&
            !(policy_mask.slice(1, 1) & ~policy_mask.slice(1, 0, length - 1)).any().item<bool>(),
        "Cada trayectoria necesita decisiones seguidas de padding, sin huecos");
    const auto selected = at::where(policy_mask, actions, 0);
    require(((selected >= 0) & (selected < actions_count)).all().item<bool>() &&
                at::isfinite(terminal_returns).all().item<bool>(),
            "Las acciones activas o el retorno terminal no son válidos");
    const auto current = normalized(logp, policy_mask);
    const auto historical = normalized(logq.detach(), policy_mask);
    const auto q = historical.exp();
    const auto indices = selected.unsqueeze(-1);
    const auto chosen = current.gather(-1, indices).squeeze(-1);
    const auto old = historical.gather(-1, indices).squeeze(-1);
    const auto coefficient =
        terminal_returns.detach().to(at::kDouble).unsqueeze(1) - beta * (chosen.detach() - old);
    const auto centered = chosen - (q * current).sum(-1);
    const auto result = -at::where(policy_mask, coefficient * centered, 0.).sum(1);
    require(at::isfinite(result).all().item<bool>(), "El sustituto terminal ha desbordado FP64");
    return result;
}
} // namespace mars_titan::learning
