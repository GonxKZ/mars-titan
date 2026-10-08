#include "mars_titan/ppo_objectives.hpp"
#include "mars_titan/ppo_policy.hpp"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>

// Álgebra e inferencia de fixtures sin actualizar parámetros.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using namespace mars_titan::learning;
void require(bool value, std::string_view message) {
    if (!value) { throw std::runtime_error(std::string(message)); }
}
double present(std::optional<double> value) {
    if (!value) { throw std::runtime_error("Falta la medición del fixture válido"); }
    return *value;
}

void penalized_actor_matches_literal_gradient() {
    auto logits = at::tensor({.3F,-.2F,.7F,.1F,.5F,-.4F}).reshape({1,6}).requires_grad_(true);
    auto old = at::full({1,6},1.F/6).requires_grad_(true);
    auto advantages = at::tensor({.8F}).requires_grad_(true);
    const auto actions = at::tensor({2},at::kLong);
    const auto selected_old = at::full({1},-std::log(6.),at::kDouble);
    const auto logp = logits.log_softmax(1);
    const auto actual = ppo_penalized_objective(logp, old, actions, selected_old, advantages, .3);
    const auto q = old.detach().to(at::kDouble) / old.detach().to(at::kDouble).sum(1,true);
    const auto lp = logp.to(at::kDouble);
    const auto ratio = (lp[0][2] + std::log(6.)).exp();
    const auto expected = -ratio * .8F + .3 * (q * (q.log() - lp)).sum();
    require(at::allclose(actual.sum(),expected,1e-6,1e-7),"La penalización no coincide con la suma literal");
    const auto gradients = torch::autograd::grad({actual.sum()},{logits,old,advantages},{},false,false,true);
    require(!gradients[1].defined() && !gradients[2].defined(),"El histórico o la ventaja reciben gradiente");
    const auto p = logits.detach().softmax(1).to(at::kDouble);
    const auto one_hot = at::one_hot(actions,6).to(at::kDouble);
    const auto expected_gradient = -ratio.detach() * .8F * (one_hot-p) + .3 * (p-q);
    require(at::allclose(gradients[0].to(at::kDouble),expected_gradient,1e-6,1e-7),"Gradiente del actor incorrecto");
}

PpoRollout fixture(const PpoPolicy& policy) {
    PpoRollout r;
    r.observations=at::arange(24,at::kFloat).reshape({3,2,4})/20;
    r.actions=at::zeros({3,2},at::kLong);
    r.old_action_weights=policy.forward(r.observations.flatten(0,1)).logits.log_softmax(1).exp().reshape({3,2,6});
    r.old_log_probabilities=r.old_action_weights.select(2,0).log().to(at::kDouble);
    r.old_values=at::zeros({3,2},at::kDouble);
    r.next_values=at::zeros({3,2},at::kDouble);
    r.rewards=at::zeros({3,2},at::kDouble);
    r.reward_valid=at::ones({3,2},at::kBool);
    r.terminated=at::zeros({3,2},at::kBool);
    r.truncated=at::zeros({3,2},at::kBool);
    return r;
}

void measured_kl_uses_all_valid_rows_without_rng_or_updates() {
    PpoObjectiveConfig config;
    config.kind=PpoObjectiveKind::clip_full_kl;
    PpoHyperparameters parameters;
    parameters.minibatch_size=2;
    PpoPolicy policy(4,parameters,42,"cpu",default_ppo_memory_bytes,{},config);
    auto r=fixture(policy);
    const auto before=policy.random_state();
    const auto state=policy.controller_state();
    require(std::abs(present(policy.full_kl(r)))<1e-14,"La política idéntica no tiene KL cero");
    r.old_action_weights.fill_(1.F/6);
    r.old_log_probabilities.fill_(-std::log(6.));
    r.reward_valid[0][0]=false;
    const auto logp=ppo_behavior_log_probabilities(policy.forward(r.observations.flatten(0,1)).logits.log_softmax(1).exp());
    const auto expected=ppo_categorical_kl(logp,ppo_behavior_log_probabilities(r.old_action_weights.flatten(0,1)))
                            .masked_select(r.reward_valid.flatten()).mean().item<double>();
    const auto measured = present(policy.full_kl(r));
    require(std::abs(measured-expected)<1e-8,"La media excede el redondeo de los GEMM FP32");
    const auto valid = r.reward_valid.flatten().nonzero().squeeze(1);
    double sum = 0;
    for (int64_t offset = 0; offset < valid.numel(); offset += 2) {
        const auto rows = valid.narrow(0, offset, std::min(int64_t{2}, valid.numel()-offset));
        const auto current = ppo_behavior_log_probabilities(policy.forward(
            r.observations.flatten(0,1).index_select(0,rows)).logits.log_softmax(1).exp());
        const auto historical = ppo_behavior_log_probabilities(r.old_action_weights.flatten(0,1).index_select(0,rows));
        sum += ppo_categorical_kl(current,historical).sum().item<double>();
    }
    require(std::abs(measured-sum/static_cast<double>(valid.numel()))<1e-14,
            "El bloque final corto tiene un denominador incorrecto");
    r.reward_valid.zero_();
    require(!policy.full_kl(r),"El vacío genera una medición");
    const auto after=policy.random_state();
    require(at::equal(before.sampling,after.sampling) && at::equal(before.shuffle,after.shuffle),"La medición consume RNG");
    require(policy.optimizer_steps()==0 && policy.controller_state()==state,"La medición cambia estado");
}

void recurrent_measurement_reconstructs_prefix_and_resets() {
    PpoObjectiveConfig objective;
    objective.kind=PpoObjectiveKind::clip_full_kl;
    PpoArchitecture architecture;
    architecture.kind=PpoNetworkKind::gru;
    PpoHyperparameters hyper;
    hyper.minibatch_size=16;
    PpoPolicy policy(4,hyper,42,"cpu",default_ppo_memory_bytes,architecture,objective);
    PpoRollout r;
    r.observations=at::arange(24,at::kFloat).reshape({3,2,4})/20;
    r.actions=at::zeros({3,2},at::kLong);
    r.old_action_weights=at::empty({3,2,6});
    r.old_log_probabilities=at::empty({3,2},at::kDouble);
    r.old_values=at::zeros({3,2},at::kDouble);
    r.next_values=at::zeros({3,2},at::kDouble);
    r.rewards=at::zeros({3,2},at::kDouble);
    r.reward_valid=at::ones({3,2},at::kBool);
    r.terminated=at::zeros({3,2},at::kBool);
    r.truncated=at::zeros({3,2},at::kBool);
    r.episode_starts=at::zeros({3,2},at::kBool);
    r.terminated[0][1]=true;
    r.episode_starts[1][1]=true;
    r.reward_valid[0][1]=false;
    r.prefix_observations=at::full({2,2,4},.2F);
    r.prefix_lengths=at::tensor({2,1},at::kLong);
    auto hidden=policy.state_from_history(r.prefix_observations,r.prefix_lengths);
    for (int64_t time=0;time<3;++time) {
        const auto output=policy.infer(r.observations[time],hidden,r.episode_starts[time]);
        const auto logp=output.logits.log_softmax(1);
        r.old_action_weights[time].copy_(logp.exp());
        r.old_log_probabilities[time].copy_(logp.select(1,0).to(at::kDouble));
        hidden=output.next_state;
    }
    const auto rng=policy.random_state();
    require(std::abs(present(policy.full_kl(r)))<1e-12,"La KL GRU pierde el prefijo o los reinicios");
    r.prefix_observations.fill_(2);
    require(present(policy.full_kl(r))>1e-8,"La medición no utiliza la historia anterior");
    require(policy.optimizer_steps()==0 && at::equal(rng.shuffle,policy.random_state().shuffle),
            "El diagnóstico recurrente altera su estado");
}
}

int main() {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        penalized_actor_matches_literal_gradient();
        measured_kl_uses_all_valid_rows_without_rng_or_updates();
        recurrent_measurement_reconstructs_prefix_and_resets();
        std::cout << "Objetivos PPO contrastados sin pasos de optimizador\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
