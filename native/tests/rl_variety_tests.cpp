#include "mars_titan/group_relative.hpp"
#include "mars_titan/ppo_policy.hpp"
#include "mars_titan/quantile_dqn.hpp"

#include <ATen/ATen.h>
#include <torch/csrc/autograd/autograd.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Constantes de la referencia FP64 independiente y tensores escritos en la prueba, sin ajuste.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using Json = nlohmann::json;
using mars_titan::learning::group_advantages;
using mars_titan::learning::group_episode_weights;
using mars_titan::learning::group_objective_kind;
using mars_titan::learning::group_relative_loss;
using mars_titan::learning::GroupObjectiveConfig;
using mars_titan::learning::GroupObjectiveKind;
using mars_titan::learning::GroupObjectiveTrace;

constexpr double relative_tolerance = 1e-12;
constexpr double absolute_tolerance = 1e-15;

void require(bool value, std::string_view message) {
    if (!value) {
        throw std::runtime_error(std::string(message));
    }
}

template <class F> void rejected(F&& action, std::string_view message) {
    bool failed = false;
    try {
        std::forward<F>(action)();
    } catch (const std::invalid_argument&) {
        failed = true;
    }
    require(failed, message);
}

at::Tensor doubles(const Json& values) {
    std::vector<double> flat;
    std::vector<int64_t> shape;
    const Json* level = &values;
    while (level->is_array()) {
        shape.push_back(static_cast<int64_t>(level->size()));
        if (level->empty()) {
            break;
        }
        level = &level->front();
    }
    const auto flatten = [&](const auto& self, const Json& node) -> void {
        if (node.is_array()) {
            for (const auto& child : node) {
                self(self, child);
            }
        } else if (node.is_boolean()) {
            flat.push_back(node.get<bool>() ? 1. : 0.);
        } else {
            flat.push_back(node.get<double>());
        }
    };
    flatten(flatten, values);
    return at::tensor(flat, at::kDouble).reshape(shape);
}

at::Tensor longs(const Json& values) { return doubles(values).to(at::kLong); }

bool close(const at::Tensor& actual, const at::Tensor& expected) {
    return actual.sizes() == expected.sizes() &&
           at::allclose(actual, expected, relative_tolerance, absolute_tolerance);
}

GroupObjectiveConfig config_of(const Json& value) {
    GroupObjectiveConfig config;
    const auto kind = value.at("kind").get<std::string>();
    config.kind = kind == "grpo"      ? GroupObjectiveKind::grpo
                  : kind == "dr_grpo" ? GroupObjectiveKind::dr_grpo
                  : kind == "dapo"    ? GroupObjectiveKind::dapo
                                      : GroupObjectiveKind::gspo;
    config.group_size = value.at("group_size").get<std::size_t>();
    config.clip_low = value.at("clip_low").get<double>();
    config.clip_high = value.at("clip_high").get<double>();
    config.kl_beta = value.at("kl_beta").get<double>();
    config.advantage_epsilon = value.at("advantage_epsilon").get<double>();
    config.length_normalizer = value.at("length_normalizer").get<std::size_t>();
    config.validate();
    require(group_objective_kind(config.id()) == config.kind, "La identidad no es reversible");
    return config;
}

struct GroupInputs {
    at::Tensor logits;
    at::Tensor logp;
    at::Tensor logq;
    at::Tensor actions;
    at::Tensor mask;
    at::Tensor lengths;
};

GroupInputs group_inputs(const Json& item) {
    GroupInputs result;
    result.lengths = longs(item.at("lengths"));
    result.logits = doubles(item.at("logits")).requires_grad_(true);
    const auto batch = result.logits.size(0);
    const auto horizon = result.logits.size(1);
    result.mask = at::arange(horizon, at::kLong).unsqueeze(0) < result.lengths.unsqueeze(1);
    // El padding lleva NaN, infinito y acciones centinela: no debe intervenir.
    result.logp = at::where(result.mask.unsqueeze(-1), result.logits.log_softmax(-1),
                            std::numeric_limits<double>::quiet_NaN());
    result.logq = at::where(result.mask.unsqueeze(-1), doubles(item.at("behavior")).log(),
                            std::numeric_limits<double>::infinity());
    result.actions = at::where(result.mask, longs(item.at("actions")), 999);
    require(result.mask.sizes() == at::IntArrayRef({batch, horizon}), "Máscara mal formada");
    return result;
}

void reference_case(const Json& item) {
    const auto name = item.at("name").get<std::string>();
    const auto config = config_of(item.at("config"));
    const auto inputs = group_inputs(item);
    const auto advantages = group_advantages(doubles(item.at("returns")), longs(item.at("groups")),
                                             config.advantage(), config.advantage_epsilon);
    require(close(advantages, doubles(item.at("advantages"))), name + ": ventajas");
    const auto weights = group_episode_weights(inputs.lengths, config);
    require(close(weights, doubles(item.at("weights"))), name + ": pesos");
    const auto loss = group_relative_loss(inputs.logp, inputs.logq, inputs.actions, advantages,
                                          weights, inputs.mask, config);
    require(loss.scalar_type() == at::kDouble && close(loss, doubles(item.at("loss"))),
            name + ": pérdida por episodio");
    const auto gradient =
        torch::autograd::grad({loss.sum()}, {inputs.logits}, {}, true, false, true)[0];
    require(close(gradient, doubles(item.at("gradient"))), name + ": gradiente o padding");

    // Las trazas se calculan sin gradiente y no cambian ni un bit del resultado.
    GroupObjectiveTrace trace;
    const auto traced = group_relative_loss(inputs.logp, inputs.logq, inputs.actions, advantages,
                                            weights, inputs.mask, config, &trace);
    const auto traced_gradient =
        torch::autograd::grad({traced.sum()}, {inputs.logits}, {}, false, false, true)[0];
    require(at::equal(loss, traced) && at::equal(gradient, traced_gradient),
            name + ": las trazas alteran el cálculo");
    const auto& expected = item.at("trace");
    require(trace.episodes == expected.at("episodes").get<std::size_t>() &&
                trace.decisions == expected.at("decisions").get<std::size_t>(),
            name + ": recuentos de la traza");
    const std::vector<std::pair<std::string_view, double>> values{
        {"entropy", trace.entropy},
        {"full_kl", trace.full_kl},
        {"k3_kl", trace.k3_kl},
        {"ratio_mean", trace.ratio_mean},
        {"ratio_min", trace.ratio_min},
        {"ratio_max", trace.ratio_max},
        {"clip_fraction", trace.clip_fraction},
        {"advantage_mean", trace.advantage_mean},
        {"advantage_std", trace.advantage_std},
        {"advantage_min", trace.advantage_min},
        {"advantage_max", trace.advantage_max}};
    for (const auto& [key, value] : values) {
        const auto reference = expected.at(std::string(key)).get<double>();
        require(std::abs(value - reference) <= 1e-12 * std::max(1., std::abs(reference)),
                name + ": traza " + std::string(key));
    }
}

GroupObjectiveConfig grpo() {
    GroupObjectiveConfig config;
    config.kl_beta = 0.04;
    return config;
}

// Con q igual al actor (cadencia uno) los cocientes valen uno y el KL k3 no tiene gradiente:
// GRPO y GSPO dan el mismo gradiente. Solo se separan en oleadas fuera de política.
void on_policy_grpo_equals_gspo() {
    const auto logits =
        (at::arange(60, at::kDouble).reshape({2, 5, 6}).cos() / 3).requires_grad_(true);
    const auto logp = logits.log_softmax(-1);
    const auto logq = logp.detach();
    const auto actions = at::tensor({0, 1, 2, 3, 4, 5, 4, 3, 2, 1}, at::kLong).reshape({2, 5});
    const auto mask = at::ones({2, 5}, at::kBool);
    const auto advantages = at::tensor({0.7, -0.7}, at::kDouble);
    auto gspo = grpo();
    gspo.kind = GroupObjectiveKind::gspo;
    gspo.kl_beta = 0;
    gspo.clip_low = gspo.clip_high = 0.0003;
    const auto lengths = at::full({2}, 5, at::kLong);
    const auto first = group_relative_loss(logp, logq, actions, advantages,
                                           group_episode_weights(lengths, grpo()), mask, grpo());
    const auto second = group_relative_loss(logp, logq, actions, advantages,
                                            group_episode_weights(lengths, gspo), mask, gspo);
    const auto left = torch::autograd::grad({first.sum()}, {logits}, {}, true)[0];
    const auto right = torch::autograd::grad({second.sum()}, {logits}, {}, false)[0];
    require(at::allclose(left, right, 1e-14, 1e-16) && at::allclose(first, second, 1e-14, 1e-16),
            "GRPO y GSPO deben coincidir en política");
}

// A2C es PPO con una época y un minilote: en cociente uno el recorte no actúa y el gradiente
// recortado es el de política -A * grad log pi. Sostiene el descarte de A2C como brazo propio.
void ppo_at_unit_ratio_is_policy_gradient() {
    const auto logits =
        (at::arange(18, at::kDouble).reshape({3, 6}).sin() / 2).requires_grad_(true);
    const auto actions = at::tensor({1, 4, 2}, at::kLong);
    const auto logp = logits.log_softmax(-1).gather(1, actions.unsqueeze(1)).squeeze(1);
    const auto advantages = at::tensor({0.3, -1.1, 0.8}, at::kDouble);
    const auto clipped =
        -mars_titan::learning::ppo_clipped_objective(logp, logp.detach(), advantages, 0.2).mean();
    const auto policy_gradient = -(advantages * logp).mean();
    const auto left = torch::autograd::grad({clipped}, {logits}, {}, true)[0];
    const auto right = torch::autograd::grad({policy_gradient}, {logits}, {}, false)[0];
    require(at::allclose(left, right, 1e-14, 1e-16), "PPO en cociente uno no es A2C");
}

// Las ventajas de un grupo solo dependen de los retornos de ese grupo y del orden de sus
// episodios: cambiar otro grupo o permutar el lote no las altera.
void advantages_stay_within_their_group() {
    const auto returns = at::tensor({0.1, 0.3, -0.2, 0.05, 1.0, 2.0, 3.0, 4.0}, at::kDouble);
    const auto groups = at::tensor({0, 0, 0, 0, 1, 1, 1, 1}, at::kLong);
    for (const auto mode : {mars_titan::learning::GroupAdvantage::mean_std,
                            mars_titan::learning::GroupAdvantage::mean}) {
        const auto base = group_advantages(returns, groups, mode, 1e-6);
        auto changed = returns.clone();
        changed.slice(0, 4).mul_(-7).add_(0.5);
        const auto other = group_advantages(changed, groups, mode, 1e-6);
        require(at::equal(base.slice(0, 0, 4), other.slice(0, 0, 4)),
                "Un grupo usa los retornos de otro grupo");
        const auto order = at::tensor({5, 0, 7, 2, 1, 4, 3, 6}, at::kLong);
        const auto permuted = group_advantages(returns.index_select(0, order),
                                               groups.index_select(0, order), mode, 1e-6);
        require(at::allclose(permuted, base.index_select(0, order), 1e-15, 1e-15),
                "Las ventajas dependen del orden del lote");
    }
    const auto equal = group_advantages(at::full({4}, 0.25, at::kDouble), at::zeros({4}, at::kLong),
                                        mars_titan::learning::GroupAdvantage::mean_std, 1e-6);
    require(at::equal(equal, at::zeros({4}, at::kDouble)),
            "Un grupo sin dispersión debe dar ventaja cero sin dividir por cero");
}

// QR-DQN: puntos medios, puntuación de riesgo por nivel tau, objetivo con selección doble,
// pérdida cuantílica de Huber y su gradiente frente a la derivación analítica FP64.
void quantile_case(const Json& item) {
    using mars_titan::learning::quantile_action_scores;
    const auto name = item.at("name").get<std::string>();
    const auto count = item.at("quantiles").get<int64_t>();
    const auto alpha = item.at("alpha").get<double>();
    const auto kappa = item.at("kappa").get<double>();
    const auto gamma = item.at("gamma").get<double>();
    require(close(mars_titan::learning::quantile_midpoints(count, at::Device(at::kCPU)),
                  doubles(item.at("midpoints"))),
            name + ": puntos medios");
    require(close(quantile_action_scores(doubles(item.at("current")), alpha),
                  doubles(item.at("scores"))),
            name + ": puntuación de riesgo");
    const auto terminated = longs(item.at("terminated")).to(at::kBool);
    const auto rewards = doubles(item.at("rewards"));
    const auto targets = mars_titan::learning::quantile_double_targets(
        rewards, terminated, doubles(item.at("online")), doubles(item.at("target")),
        {.gamma = gamma, .risk_alpha = alpha});
    require(close(targets, doubles(item.at("targets"))), name + ": objetivo con selección doble");
    require(at::equal(targets.select(0, 1), rewards[1].expand({count})),
            name + ": una terminación no debe tener bootstrap");
    const auto predicted = doubles(item.at("predicted")).requires_grad_(true);
    const auto loss = mars_titan::learning::quantile_huber_loss(predicted, targets, kappa);
    require(loss.scalar_type() == at::kDouble && close(loss, doubles(item.at("loss"))),
            name + ": pérdida cuantílica");
    const auto gradient = torch::autograd::grad({loss.sum()}, {predicted}, {}, false)[0];
    require(close(gradient, doubles(item.at("gradient"))), name + ": gradiente cuantílico");
    // La red trabaja en FP32: la pérdida se promueve a FP64 y conserva el valor con ese redondeo.
    const auto single = mars_titan::learning::quantile_huber_loss(predicted.detach().to(at::kFloat),
                                                                  targets.to(at::kFloat), kappa);
    require(at::allclose(single, loss.detach(), 1e-5, 1e-9), name + ": ruta FP32");
}

void quantile_rejections() {
    using mars_titan::learning::quantile_action_scores;
    const auto quantiles = at::zeros({2, 6, 8}, at::kDouble);
    rejected([&] { static_cast<void>(quantile_action_scores(quantiles, 0.3)); },
             "Se aceptó un alpha que no corresponde a un número entero de niveles");
    rejected([&] { static_cast<void>(quantile_action_scores(quantiles, 0.)); },
             "Se aceptó alpha nulo");
    rejected([&] { static_cast<void>(quantile_action_scores(at::zeros({2, 5, 8}), 1.)); },
             "Se aceptó una cabeza sin seis acciones");
    auto broken = quantiles.clone();
    broken[0][0][0] = std::numeric_limits<double>::quiet_NaN();
    rejected([&] { static_cast<void>(quantile_action_scores(broken, 1.)); },
             "Se aceptaron cuantiles no finitos");
    const auto targets = at::zeros({2, 8}, at::kDouble);
    rejected(
        [&] { static_cast<void>(mars_titan::learning::quantile_huber_loss(targets, targets, 0.)); },
        "Se aceptó kappa nulo");
    rejected(
        [&] {
            static_cast<void>(mars_titan::learning::quantile_double_targets(
                at::zeros({2}, at::kDouble), at::zeros({2}, at::kBool), quantiles, quantiles,
                {.gamma = 1.5, .risk_alpha = 1.}));
        },
        "Se aceptó un descuento mayor que uno");
}

void rejections() {
    const auto returns = at::tensor({0.1, 0.2, 0.3}, at::kDouble);
    rejected(
        [&] {
            static_cast<void>(group_advantages(returns, at::tensor({0, 0, 1}, at::kLong),
                                               mars_titan::learning::GroupAdvantage::mean_std,
                                               1e-6));
        },
        "Se aceptó un grupo de un solo episodio");
    rejected(
        [&] {
            static_cast<void>(group_advantages(
                at::tensor({0.1, std::numeric_limits<double>::quiet_NaN()}, at::kDouble),
                at::zeros({2}, at::kLong), mars_titan::learning::GroupAdvantage::mean, 0));
        },
        "Se aceptó un retorno no finito");
    auto config = grpo();
    config.kl_beta = 0;
    rejected([&] { config.validate(); }, "GRPO sin su término KL");
    config = grpo();
    config.kind = GroupObjectiveKind::dapo;
    config.kl_beta = 0;
    rejected([&] { config.validate(); }, "DAPO con recorte simétrico");
    config.kind = GroupObjectiveKind::gspo;
    config.clip_low = 0.0004;
    config.clip_high = 0.0003;
    rejected([&] { config.validate(); }, "GSPO con el rango superior menor que el inferior");
    config.kind = GroupObjectiveKind::dapo;
    config.clip_low = 0.2;
    config.clip_high = 0.28;
    config.validate();
    config.kind = GroupObjectiveKind::dr_grpo;
    config.clip_high = 0.2;
    rejected([&] { config.validate(); }, "Dr. GRPO con suelo de desviación");
    config.advantage_epsilon = 0;
    config.validate();
    config.group_size = 1;
    rejected([&] { config.validate(); }, "Se aceptó un grupo de tamaño uno");
    rejected([] { static_cast<void>(group_objective_kind("grpo_plus_plus")); },
             "Se aceptó una identidad sin fuente");

    const auto logits = at::zeros({2, 3, 6}, at::kDouble).requires_grad_(true);
    const auto logp = logits.log_softmax(-1);
    const auto actions = at::zeros({2, 3}, at::kLong);
    const auto advantages = at::tensor({1., -1.}, at::kDouble);
    const auto weights = at::full({2}, 0.5, at::kDouble);
    const auto gap = at::tensor({1, 0, 1, 1, 1, 1}, at::kLong).to(at::kBool).reshape({2, 3});
    rejected(
        [&] {
            static_cast<void>(group_relative_loss(logp, logp.detach(), actions, advantages, weights,
                                                  gap, grpo()));
        },
        "Se aceptó una máscara con huecos");
    const auto mask = at::ones({2, 3}, at::kBool);
    rejected(
        [&] {
            static_cast<void>(group_relative_loss(logp, logp.detach(),
                                                  at::full({2, 3}, 6, at::kLong), advantages,
                                                  weights, mask, grpo()));
        },
        "Se aceptó una acción fuera de las seis");
    rejected(
        [&] {
            static_cast<void>(group_relative_loss(logp, logp.detach(), actions, advantages,
                                                  at::tensor({0.5, -0.5}, at::kDouble), mask,
                                                  grpo()));
        },
        "Se aceptó un peso negativo");
    rejected(
        [&] {
            static_cast<void>(group_relative_loss(logp, (logp.detach() + 0.1), actions, advantages,
                                                  weights, mask, grpo()));
        },
        "Se aceptó un muestreador sin normalizar");
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Uso: rl_variety_tests REFERENCIA.json");
        std::ifstream stream(std::span(argv, static_cast<std::size_t>(argc))[1]);
        require(stream.good(), "Falta la referencia FP64");
        const auto reference = Json::parse(stream);
        require(reference.at("schema_version") == 1 &&
                    reference.at("kind") == "rl_variety_fp64_reference",
                "La referencia no corresponde a este contrato");
        for (const auto& item : reference.at("group")) {
            reference_case(item);
        }
        for (const auto& item : reference.at("quantile")) {
            quantile_case(item);
        }
        quantile_rejections();
        on_policy_grpo_equals_gspo();
        ppo_at_unit_ratio_is_policy_gradient();
        advantages_stay_within_their_group();
        rejections();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
