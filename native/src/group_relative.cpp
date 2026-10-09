#include "mars_titan/group_relative.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mars_titan::learning {
namespace {
constexpr int64_t actions_count = 6;
constexpr double normalization_tolerance = 1e-6;
constexpr std::size_t fixed_buffer_bytes = 65536;
constexpr std::size_t bytes_per_action = 192;
constexpr std::size_t bytes_per_decision = 256;
constexpr std::size_t bytes_per_episode = 256;
constexpr double maximum_clip = 10;
constexpr double maximum_kl_beta = 1e3;
constexpr double maximum_advantage_epsilon = 1;

void require(bool value, std::string_view message) {
    if (!value) {
        throw std::invalid_argument(std::string(message));
    }
}

bool floating(const at::Tensor& tensor) {
    return tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kDouble;
}

// Mismo tratamiento que el núcleo KLPO. El padding se sustituye antes de exp para que un NaN
// o un infinito fuera de la máscara no llegue a ningún valor ni gradiente, y las seis
// probabilidades se renormalizan en FP64 para absorber el redondeo de FP32.
at::Tensor normalized(const at::Tensor& logs, const at::Tensor& mask) {
    const auto safe =
        at::where(mask.unsqueeze(-1), logs, -std::log(static_cast<double>(actions_count)));
    require(at::isfinite(safe).all().item<bool>(),
            "Los logaritmos activos del grupo necesitan valores finitos");
    require((safe.detach().exp() > 0).all().item<bool>(),
            "Una probabilidad del grupo pierde soporte en la precisión de entrada");
    const auto promoted = safe.to(at::kDouble);
    const auto total = promoted.logsumexp(-1, true);
    require((total.abs() <= normalization_tolerance).all().item<bool>(),
            "Las seis probabilidades del grupo deben estar normalizadas");
    return promoted - total;
}

double scalar(const at::Tensor& value) { return value.to(at::kCPU).item<double>(); }

// Tensores ya calculados por la pérdida que la traza lee sin gradiente. Copiar un at::Tensor solo
// copia su manejador, y la estructura únicamente se construye cuando se pide la traza.
struct TraceInputs {
    at::Tensor current;
    at::Tensor historical;
    at::Tensor chosen;
    at::Tensor old;
    at::Tensor ratio;
    at::Tensor clipped;
    at::Tensor advantages;
    at::Tensor mask;
    bool sequence = false;
};

void fill_trace(GroupObjectiveTrace& trace, const TraceInputs& inputs) {
    const auto& [current, historical, chosen, old, ratio, clipped, advantages, mask, sequence] =
        inputs;
    const at::NoGradGuard no_grad;
    const auto active = mask.to(at::kDouble);
    const auto decisions = active.sum();
    trace.episodes = static_cast<std::size_t>(mask.size(0));
    trace.decisions = static_cast<std::size_t>(scalar(decisions));
    const auto p = current.detach().exp();
    const auto per_decision = [&](const at::Tensor& values) {
        return scalar((values * active).sum() / decisions);
    };
    trace.entropy = per_decision(-(p * current.detach()).sum(-1));
    trace.full_kl = per_decision((p * (current.detach() - historical)).sum(-1));
    const auto delta = at::where(mask, old - chosen.detach(), 0.);
    trace.k3_kl = per_decision(delta.exp() - delta - 1);
    if (sequence) {
        trace.ratio_mean = scalar(ratio.mean());
        trace.ratio_min = scalar(ratio.min());
        trace.ratio_max = scalar(ratio.max());
        trace.clip_fraction = scalar(clipped.to(at::kDouble).mean());
    } else {
        const auto inside = ratio.masked_select(mask);
        trace.ratio_mean = scalar(inside.mean());
        trace.ratio_min = scalar(inside.min());
        trace.ratio_max = scalar(inside.max());
        trace.clip_fraction = per_decision(clipped.to(at::kDouble));
    }
    const auto values = advantages.to(at::kDouble);
    trace.advantage_mean = scalar(values.mean());
    trace.advantage_std = scalar(values.std(/*unbiased=*/false));
    trace.advantage_min = scalar(values.min());
    trace.advantage_max = scalar(values.max());
}
} // namespace

GroupObjectiveKind group_objective_kind(std::string_view id) {
    if (id == "grpo_outcome_v1") {
        return GroupObjectiveKind::grpo;
    }
    if (id == "dr_grpo_outcome_v1") {
        return GroupObjectiveKind::dr_grpo;
    }
    if (id == "dapo_outcome_static_v1") {
        return GroupObjectiveKind::dapo;
    }
    if (id == "gspo_outcome_v1") {
        return GroupObjectiveKind::gspo;
    }
    throw std::invalid_argument("El objetivo de grupo no está admitido");
}

std::string GroupObjectiveConfig::id() const {
    switch (kind) {
    case GroupObjectiveKind::grpo:
        return "grpo_outcome_v1";
    case GroupObjectiveKind::dr_grpo:
        return "dr_grpo_outcome_v1";
    case GroupObjectiveKind::dapo:
        return "dapo_outcome_static_v1";
    case GroupObjectiveKind::gspo:
        return "gspo_outcome_v1";
    }
    throw std::invalid_argument("El objetivo de grupo no está admitido");
}

GroupAdvantage GroupObjectiveConfig::advantage() const noexcept {
    return kind == GroupObjectiveKind::dr_grpo ? GroupAdvantage::mean : GroupAdvantage::mean_std;
}

GroupRatio GroupObjectiveConfig::ratio() const noexcept {
    return kind == GroupObjectiveKind::gspo ? GroupRatio::sequence : GroupRatio::token;
}

GroupAggregation GroupObjectiveConfig::aggregation() const noexcept {
    switch (kind) {
    case GroupObjectiveKind::grpo:
        return GroupAggregation::sequence_mean_token_mean;
    case GroupObjectiveKind::dr_grpo:
        return GroupAggregation::sequence_mean_token_sum_constant;
    case GroupObjectiveKind::dapo:
        return GroupAggregation::token_mean;
    case GroupObjectiveKind::gspo:
        return GroupAggregation::sequence_mean;
    }
    return GroupAggregation::sequence_mean_token_mean;
}

void GroupObjectiveConfig::validate() const {
    static_cast<void>(group_objective_kind(id()));
    require(group_size >= 2 && group_size <= maximum_group_size && std::isfinite(clip_low) &&
                std::isfinite(clip_high) && clip_low > 0 && clip_low < 1 && clip_high > 0 &&
                clip_high <= maximum_clip && std::isfinite(kl_beta) && kl_beta >= 0 &&
                kl_beta <= maximum_kl_beta && std::isfinite(advantage_epsilon) &&
                advantage_epsilon >= 0 && advantage_epsilon <= maximum_advantage_epsilon &&
                length_normalizer > 0 &&
                length_normalizer <= static_cast<std::size_t>(maximum_group_length),
            "El objetivo de grupo declara parámetros fuera de sus límites");
    // Cada identidad conserva la forma de su fuente. Solo GRPO usa el término KL. DAPO recorta
    // más por arriba que por abajo (clip-higher), GSPO admite los dos rangos que declara su
    // artículo y GRPO y Dr. GRPO recortan de forma simétrica.
    require(kind == GroupObjectiveKind::grpo ? kl_beta > 0 : kl_beta == 0,
            "Solo GRPO declara el término KL k3 de su fuente");
    const bool shape = kind == GroupObjectiveKind::dapo   ? clip_high > clip_low
                       : kind == GroupObjectiveKind::gspo ? clip_high >= clip_low
                                                          : clip_high == clip_low;
    require(shape, "El recorte no conserva la forma de la identidad de grupo");
    require(kind == GroupObjectiveKind::dr_grpo ? advantage_epsilon == 0 : advantage_epsilon > 0,
            "Dr. GRPO no normaliza por desviación y el resto necesita su suelo positivo");
}

at::Tensor group_advantages(const at::Tensor& returns, const at::Tensor& groups,
                            GroupAdvantage advantage, double epsilon) {
    require(returns.defined() && groups.defined() && returns.layout() == at::kStrided &&
                groups.layout() == at::kStrided && returns.device().is_cpu() &&
                groups.device().is_cpu() && returns.scalar_type() == at::kDouble &&
                groups.scalar_type() == at::kLong && returns.dim() == 1 &&
                returns.sizes() == groups.sizes() && returns.numel() > 0 &&
                returns.numel() <= maximum_group_batch && std::isfinite(epsilon) && epsilon >= 0,
            "Las ventajas de grupo necesitan retornos FP64 y grupos int64 en CPU");
    const auto values = returns.contiguous();
    const auto labels = groups.contiguous();
    const auto count = static_cast<std::size_t>(values.numel());
    const std::span<const double> data(values.const_data_ptr<double>(), count);
    const std::span<const int64_t> group(labels.const_data_ptr<int64_t>(), count);
    require(std::ranges::all_of(data, [](double value) { return std::isfinite(value); }) &&
                std::ranges::all_of(group, [](int64_t value) { return value >= 0; }),
            "Los retornos de grupo deben ser finitos y los grupos no negativos");
    // Suma en orden de episodio y en dos pasadas (media y luego dispersión), que es más estable
    // que acumular cuadrados y da el mismo resultado sea cual sea el resto del lote.
    std::map<int64_t, std::pair<double, std::size_t>> totals;
    for (std::size_t index = 0; index < count; ++index) {
        auto& [sum, members] = totals[group[index]];
        sum += data[index];
        ++members;
    }
    std::map<int64_t, double> means;
    std::map<int64_t, double> spread;
    for (const auto& [label, entry] : totals) {
        require(entry.second >= 2, "Un grupo necesita al menos dos episodios del mismo estado");
        means[label] = entry.first / static_cast<double>(entry.second);
        spread[label] = 0;
    }
    for (std::size_t index = 0; index < count; ++index) {
        const auto centered = data[index] - means[group[index]];
        spread[group[index]] += centered * centered;
    }
    auto result = at::empty({static_cast<int64_t>(count)}, at::kDouble);
    const std::span output(result.data_ptr<double>(), count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto label = group[index];
        const auto centered = data[index] - means[label];
        if (advantage == GroupAdvantage::mean) {
            output[index] = centered;
            continue;
        }
        const auto members = static_cast<double>(totals[label].second);
        const auto deviation = std::sqrt(spread[label] / (members - 1));
        output[index] = centered / (deviation + epsilon);
    }
    require(std::ranges::all_of(output, [](double value) { return std::isfinite(value); }),
            "Una ventaja de grupo no es finita");
    return result;
}

at::Tensor group_episode_weights(const at::Tensor& decisions, const GroupObjectiveConfig& config) {
    config.validate();
    require(decisions.defined() && decisions.layout() == at::kStrided &&
                decisions.device().is_cpu() && decisions.scalar_type() == at::kLong &&
                decisions.dim() == 1 && decisions.numel() > 0 &&
                decisions.numel() <= maximum_group_batch && (decisions >= 0).all().item<bool>() &&
                (decisions <= maximum_group_length).all().item<bool>(),
            "Los pesos de grupo necesitan decisiones int64 acotadas por episodio en CPU");
    const auto episodes = static_cast<double>(decisions.numel());
    const auto counts = decisions.to(at::kDouble);
    const auto present = decisions > 0;
    // Un episodio sin decisiones conserva su puesto en el denominador y aporta peso cero.
    const auto constant = [&](double value) {
        return at::full_like(counts, value).masked_fill(~present, 0.);
    };
    switch (config.aggregation()) {
    case GroupAggregation::sequence_mean_token_mean:
        return at::where(present, 1. / (episodes * counts.clamp_min(1)), 0.);
    case GroupAggregation::sequence_mean_token_sum_constant:
        return constant(1. / (episodes * static_cast<double>(config.length_normalizer)));
    case GroupAggregation::token_mean: {
        const auto total = counts.sum().item<double>();
        require(total > 0, "La agregación por decisión necesita al menos una decisión");
        return constant(1. / total);
    }
    case GroupAggregation::sequence_mean:
        return constant(1. / episodes);
    }
    throw std::invalid_argument("La agregación de grupo no está admitida");
}

at::Tensor group_relative_loss(const at::Tensor& logp, const at::Tensor& logq,
                               const at::Tensor& actions, const at::Tensor& advantages,
                               const at::Tensor& weights, const at::Tensor& policy_mask,
                               const GroupObjectiveConfig& config, GroupObjectiveTrace* trace) {
    config.validate();
    require(logp.defined() && logq.defined() && actions.defined() && advantages.defined() &&
                weights.defined() && policy_mask.defined() && logp.layout() == at::kStrided &&
                logq.layout() == at::kStrided && actions.layout() == at::kStrided &&
                advantages.layout() == at::kStrided && weights.layout() == at::kStrided &&
                policy_mask.layout() == at::kStrided && logp.dim() == 3 && logp.size(0) > 0 &&
                logp.size(0) <= maximum_group_batch && logp.size(1) > 0 &&
                logp.size(1) <= maximum_group_length && logp.size(2) == actions_count &&
                logq.sizes() == logp.sizes(),
            "El objetivo de grupo necesita lotes acotados [B,T,6]");
    const auto batch = logp.size(0);
    const auto length = logp.size(1);
    require(actions.sizes() == at::IntArrayRef({batch, length}) &&
                policy_mask.sizes() == actions.sizes() &&
                advantages.sizes() == at::IntArrayRef({batch}) &&
                weights.sizes() == advantages.sizes() && actions.scalar_type() == at::kLong &&
                policy_mask.scalar_type() == at::kBool && floating(logp) && floating(logq) &&
                advantages.scalar_type() == at::kDouble && weights.scalar_type() == at::kDouble,
            "Las acciones, máscara, ventajas o pesos de grupo no conservan forma y tipo");
    const auto device = logp.device();
    require((device.is_cpu() || device.is_cuda()) && logq.device() == device &&
                actions.device() == device && advantages.device() == device &&
                weights.device() == device && policy_mask.device() == device,
            "El objetivo de grupo requiere un dispositivo común");
    const auto decisions = static_cast<std::size_t>(batch * length);
    const auto estimated = fixed_buffer_bytes +
                           decisions * (static_cast<std::size_t>(actions_count) * bytes_per_action +
                                        bytes_per_decision) +
                           static_cast<std::size_t>(batch) * bytes_per_episode;
    require(estimated <= maximum_group_bytes,
            "Los buffers propios del objetivo de grupo superan el presupuesto");
    require(
        policy_mask.select(1, 0).all().item<bool>() &&
            !(policy_mask.slice(1, 1) & ~policy_mask.slice(1, 0, length - 1)).any().item<bool>(),
        "Cada episodio del grupo necesita decisiones seguidas de padding, sin huecos");
    const auto selected = at::where(policy_mask, actions, 0);
    require(((selected >= 0) & (selected < actions_count)).all().item<bool>() &&
                at::isfinite(advantages).all().item<bool>() &&
                at::isfinite(weights).all().item<bool>() && (weights >= 0).all().item<bool>(),
            "Las acciones activas, ventajas o pesos de grupo no son válidos");
    const auto current = normalized(logp, policy_mask);
    const auto historical = normalized(logq.detach(), policy_mask);
    const auto indices = selected.unsqueeze(-1);
    const auto chosen = current.gather(-1, indices).squeeze(-1);
    const auto old = historical.gather(-1, indices).squeeze(-1);
    // Logaritmo del cociente pi_theta/q en cada decisión, cero en el padding.
    const auto log_ratio = at::where(policy_mask, chosen - old, 0.);
    const auto advantage = advantages.detach();
    const auto lower = 1 - config.clip_low;
    const auto upper = 1 + config.clip_high;
    at::Tensor result;
    at::Tensor ratio;
    at::Tensor clipped_active;
    if (config.ratio() == GroupRatio::sequence) {
        // GSPO, ecuación 7: el cociente de la secuencia es la media geométrica de los cocientes
        // de sus decisiones, de modo que su escala no depende de la longitud del episodio.
        const auto count = policy_mask.sum(1).to(at::kDouble);
        ratio = (log_ratio.sum(1) / count).exp();
        const auto unclipped = ratio * advantage;
        const auto clipped = ratio.clamp(lower, upper) * advantage;
        result = -weights.detach() * at::minimum(unclipped, clipped);
        clipped_active = ((advantage > 0) & (ratio > upper)) | ((advantage < 0) & (ratio < lower));
    } else {
        ratio = log_ratio.exp();
        const auto token_advantage = advantage.unsqueeze(1);
        const auto unclipped = ratio * token_advantage;
        const auto clipped = ratio.clamp(lower, upper) * token_advantage;
        auto term = at::minimum(unclipped, clipped);
        if (config.kl_beta > 0) {
            // Estimador k3 de la ecuación 4 de DeepSeekMath con la referencia q, q/p - log(q/p)
            // - 1. Se evalúa en acciones de q sin peso de importancia, como en la fuente, aunque
            // Zhang et al. (2025) muestran que ese gradiente no es el del KL fuera de política.
            const auto delta = at::where(policy_mask, old - chosen, 0.);
            term = term - config.kl_beta * (delta.exp() - delta - 1);
        }
        result = -weights.detach() * at::where(policy_mask, term, 0.).sum(1);
        clipped_active = policy_mask & (((token_advantage > 0) & (ratio > upper)) |
                                        ((token_advantage < 0) & (ratio < lower)));
    }
    require(at::isfinite(result).all().item<bool>(), "El objetivo de grupo ha desbordado FP64");
    if (trace != nullptr) {
        fill_trace(*trace, {current, historical, chosen, old, ratio.detach(), clipped_active,
                            advantage, policy_mask, config.ratio() == GroupRatio::sequence});
    }
    return result;
}
} // namespace mars_titan::learning
