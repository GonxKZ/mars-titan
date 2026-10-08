#include "mars_titan/ppo_controller.hpp"
#include "mars_titan/ppo_objectives.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

// Constantes de fixtures, sin política ni optimizador.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using namespace mars_titan::learning;
void require(bool value, std::string_view message) {
    if (!value) { throw std::runtime_error(std::string(message)); }
}
template<class F> void rejected(F&& action) {
    bool failed = false;
    try { std::forward<F>(action)(); } catch (const std::invalid_argument&) { failed = true; }
    require(failed, "Se aceptó un contrato incompatible");
}

void beta_changes_only_outside_the_band() {
    PpoObjectiveConfig config;
    config.kind = PpoObjectiveKind::kl_penalty_adaptive;
    config.validate();
    require(next_beta(config, 1., config.target_kl / 1.5, 10) == 1., "Frontera inferior");
    require(next_beta(config, 1., config.target_kl * 1.5, 10) == 1., "Frontera superior");
    require(next_beta(config, 1., 0., 10) == .5, "Falta reducir beta");
    require(next_beta(config, 1., .1, 10) == 2., "Falta aumentar beta");
    require(next_beta(config, config.beta_min, 0., 10) == config.beta_min, "Límite inferior");
    require(next_beta(config, config.beta_max, .1, 10) == config.beta_max, "Límite superior");
    require(next_beta(config, 1., std::nullopt, 0) == 1., "Se adaptó un rollout vacío");
    rejected([&] { (void)next_beta(config, 1., .1, 0); });
    rejected([&] { (void)next_beta(config, 1., std::nullopt, 1); });
    rejected([&] { (void)next_beta(config, 0., .1, 1); });
    rejected([&] { (void)next_beta(config, 1., -.1, 1); });
    rejected([&] { (void)next_beta(config, 1., std::numeric_limits<double>::infinity(), 1); });
    require(next_beta(config, 1., -1e-13, 1) == .5, "Redondeo KL cercano a cero");
    config.kind = PpoObjectiveKind::clip_full_kl;
    rejected([&] { (void)next_beta(config, 1., .1, 1); });
}

void epoch_stop_preserves_the_completed_epoch() {
    PpoObjectiveConfig config;
    config.kind = PpoObjectiveKind::clip_kl_epoch_stop;
    const auto before = config;
    const auto equal = epoch_decision(config, 1, 4, 10, .015);
    require(!equal.threshold_exceeded && !equal.stop_remaining_epochs, "La igualdad detiene");
    const auto crossing = epoch_decision(config, 1, 4, 10, .02);
    require(crossing.threshold_exceeded && crossing.stop_remaining_epochs, "No para las restantes");
    const auto last = epoch_decision(config, 4, 4, 10, .02);
    require(last.threshold_exceeded && !last.stop_remaining_epochs, "La última no omite épocas");
    const auto empty = epoch_decision(config, 0, 4, 0, std::nullopt);
    require(!empty.threshold_exceeded && !empty.stop_remaining_epochs, "No hay medición vacía");
    rejected([&] { (void)epoch_decision(config, 0, 4, 1, .01); });
    rejected([&] { (void)epoch_decision(config, 5, 4, 1, .01); });
    rejected([&] { (void)epoch_decision(config, 1, 4, -1, .01); });
    require(config == before, "Se modificó la configuración");
    config.kind = PpoObjectiveKind::clip_full_kl;
    require(!epoch_decision(config, 1, 4, 1, .5).stop_remaining_epochs, "El diagnóstico para");
}

void weights_preserve_sampler_mass_and_support() {
    auto weights = at::tensor({.5F,.1F,.1F,.1F,.1F,.1F},at::kFloat).reshape({1,6}).requires_grad_(true);
    const auto before = weights.detach().clone();
    const auto logs = ppo_behavior_log_probabilities(weights);
    require(logs.scalar_type()==at::kDouble, "No se promovió a FP64");
    const auto expected = weights.to(at::kDouble) / weights.to(at::kDouble).sum(1,true);
    require(at::allclose(logs.exp(),expected,1e-15,1e-15), "Cambió la masa del sampler");
    const auto delta = at::tensor({1.F,0.F,0.F,0.F,0.F,0.F}).reshape({1,6});
    const auto sparse = ppo_behavior_log_probabilities(delta);
    require(sparse[0][0].item<double>()==0 && at::isneginf(sparse.narrow(1,1,5)).all().item<bool>(), "Se inventó soporte");
    const auto objective = ppo_categorical_kl(logs, at::full({1,6},-std::log(6.),at::kDouble));
    const auto gradient = torch::autograd::grad({objective.sum()},{weights}).front();
    require(at::isfinite(gradient).all().item<bool>(), "Gradiente de pesos no finito");
    require(at::equal(weights,before), "Se alteraron los pesos");
    rejected([&] { (void)ppo_behavior_log_probabilities(weights.to(at::kDouble)); });
    rejected([&] { (void)ppo_behavior_log_probabilities(weights.to(at::kHalf)); });
    rejected([&] { (void)ppo_behavior_log_probabilities(at::zeros({1,6})); });
    rejected([&] { (void)ppo_behavior_log_probabilities(weights * 2); });
    rejected([&] { (void)ppo_behavior_log_probabilities(at::full({1,6},-.1F)); });
}

void observed_subnormal_mass_needs_a_representable_logged_probability() {
    const auto weights = at::tensor({1.F,std::numeric_limits<float>::denorm_min(),0.F,0.F,0.F,0.F}).reshape({1,6});
    const auto actions = at::tensor({1},at::kLong);
    const auto logp = weights.select(1,1).to(at::kDouble).log();
    const auto valid = at::ones({1},at::kBool);
    validate_ppo_behavior(weights, actions, logp, valid);
    const auto impossible = at::full({1},-200.,at::kDouble);
    require(impossible.to(at::kFloat).exp().item<float>() == 0, "El fixture debe perder soporte FP32");
    rejected([&] { validate_ppo_behavior(weights, actions, impossible, valid); });
    validate_ppo_behavior(weights, actions, impossible, at::zeros_like(valid));
}

void stored_row_counts_respect_the_rollout_limit() {
    PpoObjectiveConfig config;
    config.kind = PpoObjectiveKind::clip_full_kl;
    PpoControllerState state;
    state.completed_rollouts = 1;
    state.optimizer_steps = 4;
    state.completed_epochs = 4;
    state.full_kl = 0.;
    state.valid_rows = 16384;
    state.validate(config,4,4);
    state.valid_rows = 16385;
    rejected([&] { state.validate(config,4,4); });
    state.valid_rows = 1024;
    state.validate(config,4,4,1024);
    state.valid_rows = 1025;
    rejected([&] { state.validate(config,4,4,1024); });
    state.valid_rows = 0;
    state.optimizer_steps = 0;
    state.completed_epochs = 0;
    state.full_kl.reset();
    state.completed_rollouts = (int64_t{1} << 20) + 1;
    state.validate(config,0,4);
}

void configurations_have_distinct_contracts() {
    PpoObjectiveConfig legacy;
    require(!legacy.enabled(), "Se activó el camino nuevo por defecto");
    for (const auto kind : {PpoObjectiveKind::clip_full_kl, PpoObjectiveKind::kl_penalty_adaptive,
                            PpoObjectiveKind::clip_kl_epoch_stop}) {
        PpoObjectiveConfig config;
        config.kind=kind;
        require(objective_kind(objective_id(kind))==kind && config.enabled(), "Identidad no reversible");
    }
    rejected([] { (void)objective_kind("ppo"); });
    PpoObjectiveConfig bad;
    bad.kind=static_cast<PpoObjectiveKind>(99);
    rejected([&] { bad.validate(); });
    bad.kind=PpoObjectiveKind::kl_penalty_adaptive;
    bad.beta_initial=0;
    rejected([&] { bad.validate(); });
    bad.beta_initial=1;
    bad.target_kl=std::numeric_limits<double>::quiet_NaN();
    rejected([&] { bad.validate(); });
}
}

int main() {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        const auto rng=at::detail::getDefaultCPUGenerator().get_state().clone();
        beta_changes_only_outside_the_band();
        epoch_stop_preserves_the_completed_epoch();
        weights_preserve_sampler_mass_and_support();
        stored_row_counts_respect_the_rollout_limit();
        observed_subnormal_mass_needs_a_representable_logged_probability();
        configurations_have_distinct_contracts();
        require(at::equal(rng,at::detail::getDefaultCPUGenerator().get_state()), "Se consumió RNG");
        std::cout << "Controladores PPO comprobados sin actualizaciones\n";
    } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
