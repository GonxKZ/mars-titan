#include "mars_titan/ppo_training.hpp"
#include "mars_titan/ppo_objectives.hpp"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <functional>
#include <utility>
#include <span>

// Fixtures pequeños sin llamar al optimizador ni avanzar un entorno.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using namespace mars_titan::learning;
void require(bool value, std::string_view message) {
    if (!value) { throw std::runtime_error(std::string(message)); }
}
template<class F> void rejected(F&& action) {
    bool failed = false;
    try { std::forward<F>(action)(); } catch(const std::exception&) { failed = true; }
    require(failed, "Se aceptó un estado incompatible");
}
int64_t version(const std::string& bytes) {
    torch::serialize::InputArchive archive;
    std::istringstream input(bytes);
    archive.load_from(input, at::Device(at::kCPU));
    c10::IValue value;
    archive.read("schema_version", value);
    return value.toInt();
}

std::string altered(const std::string& bytes, const std::function<void(torch::serialize::InputArchive&)>& change) {
    torch::serialize::InputArchive input;
    std::istringstream source(bytes);
    input.load_from(source, at::Device(at::kCPU));
    change(input);
    torch::serialize::OutputArchive output;
    for (const auto& key : input.keys()) {
        c10::IValue value;
        input.read(key,value);
        output.write(key,value);
    }
    std::ostringstream destination;
    output.save_to(destination);
    return destination.str();
}

void incompatible_checkpoint_fields_are_rejected() {
    PpoObjectiveConfig objective;
    objective.kind=PpoObjectiveKind::kl_penalty_adaptive;
    PpoPolicy policy(4,{},42,"cpu",default_ppo_memory_bytes,{},objective);
    std::ostringstream initial;
    policy.save(initial);
    for (const auto key : {"beta", "last_beta", "completed_rollouts", "optimizer_steps", "threshold_exceeded"}) {
        const auto bytes=altered(initial.str(),[&](torch::serialize::InputArchive& input) {
            c10::IValue controller;
            input.read("controller",controller);
            const std::string_view field(key);
            if (field=="beta" || field=="last_beta") {
                controller.toObject()->setAttr(key,c10::IValue(2.));
            } else if(field=="threshold_exceeded") {
                controller.toObject()->setAttr(key,c10::IValue(true));
            } else {
                controller.toObject()->setAttr(key,c10::IValue(field=="completed_rollouts" ? int64_t{-1} : int64_t{1}));
            }
        });
        rejected([&] { std::istringstream source(bytes); (void)PpoPolicy::load(source); });
    }
    const auto bytes=altered(initial.str(),[](torch::serialize::InputArchive& input) {
        c10::IValue config;
        input.read("objective",config);
        config.toObject()->setAttr("sampler",c10::IValue("other"));
    });
    rejected([&] { std::istringstream source(bytes); (void)PpoPolicy::load(source); });
    require(policy.optimizer_steps()==0,"La corrupción modificó la política original");
}

void policy_roundtrip_keeps_controller_and_legacy_version() {
    for (const auto kind : {PpoObjectiveKind::legacy_clip, PpoObjectiveKind::clip_full_kl,
                            PpoObjectiveKind::kl_penalty_adaptive, PpoObjectiveKind::clip_kl_epoch_stop}) {
        PpoObjectiveConfig objective;
        objective.kind=kind;
        PpoPolicy policy(4, {}, 42, "cpu", default_ppo_memory_bytes, {}, objective);
        std::ostringstream buffer;
        policy.save(buffer);
        require(version(buffer.str())==(objective.enabled()?3:2), "Versión de la política incorrecta");
        std::istringstream input(buffer.str());
        auto restored=PpoPolicy::load(input);
        require(restored.objective()==objective && restored.controller_state()==policy.controller_state(),
                "Se perdió identidad o beta");
        require(policy.optimizer_steps()==0 && restored.optimizer_steps()==0, "Se ejecutó Adam");
        const auto observations=at::zeros({2,4});
        require(at::equal(policy.forward(observations).logits,restored.forward(observations).logits),
                "Se alteraron pesos al recuperar");
        const auto before=policy.random_state();
        const auto after=restored.random_state();
        require(at::equal(before.sampling,after.sampling) && at::equal(before.shuffle,after.shuffle),
                "Se perdió RNG al recuperar");
    }
    PpoObjectiveConfig objective;
    objective.kind=PpoObjectiveKind::clip_full_kl;
    PpoArchitecture auxiliary;
    auxiliary.auxiliary=true;
    rejected([&] { PpoPolicy policy(4,{},42,"cpu",default_ppo_memory_bytes,auxiliary,objective); });
    auxiliary.auxiliary=false;
    auxiliary.double_dqn=true;
    rejected([&] { PpoPolicy policy(4,{},42,"cpu",default_ppo_memory_bytes,auxiliary,objective); });
}

PpoRollout rollout(bool complete_distribution, int64_t ticks=2, int64_t lanes=2) {
    PpoRollout result;
    result.observations=at::zeros({ticks,lanes,4});
    result.actions=at::zeros({ticks,lanes},at::kLong);
    result.old_log_probabilities=at::full({ticks,lanes},-std::log(6.),at::kDouble);
    result.old_values=at::zeros({ticks,lanes},at::kDouble);
    result.rewards=at::zeros({ticks,lanes},at::kDouble);
    result.next_values=at::zeros({ticks,lanes},at::kDouble);
    result.reward_valid=at::zeros({ticks,lanes},at::kBool);
    result.terminated=at::zeros({ticks,lanes},at::kBool);
    result.truncated=at::zeros({ticks,lanes},at::kBool);
    if (complete_distribution) { result.old_action_weights=at::full({ticks,lanes,6},1.F/6); }
    return result;
}

void rollout_preserves_weights_without_alias_or_migration() {
    for (const bool complete : {false,true}) {
        auto original=rollout(complete);
        const auto archive=serialize_rollout(original);
        auto restored=deserialize_rollout(archive);
        require(restored.old_action_weights.defined()==complete,"Se migró o perdió la distribución");
        require(at::equal(restored.old_log_probabilities,original.old_log_probabilities),"Se alteró el log elegido");
        if (complete) {
            require(at::equal(restored.old_action_weights,original.old_action_weights),"Cambió q");
            restored.old_action_weights.zero_();
            require(original.old_action_weights.sum().item<float>()>0,"Alias con el original");
        }
    }
    auto invalid=rollout(true);
    invalid.old_action_weights=invalid.old_action_weights.to(at::kDouble);
    rejected([&] { (void)serialize_rollout(invalid); });
    invalid=rollout(true);
    invalid.old_action_weights.zero_();
    rejected([&] { (void)serialize_rollout(invalid); });
}

void empty_update_checks_q_without_executing_optimizer() {
    PpoObjectiveConfig config;
    config.kind=PpoObjectiveKind::kl_penalty_adaptive;
    PpoPolicy policy(4,{},42,"cpu",default_ppo_memory_bytes,{},config);
    const auto before=policy.random_state();
    const auto state=policy.controller_state();
    const auto result=policy.update_from_cpu(rollout(true));
    require(result.minibatches==0 && !result.full_kl && policy.optimizer_steps()==0,
            "El rollout vacío produjo una actualización");
    require(policy.controller_state().beta==state.beta,"El vacío cambió beta");
    const auto after=policy.random_state();
    require(at::equal(before.shuffle,after.shuffle),"El vacío consumió RNG");
    rejected([&] { (void)policy.update_from_cpu(rollout(false)); });
    auto invalid=rollout(true);
    invalid.old_action_weights[0][0][0]=1;
    rejected([&] { (void)policy.update_from_cpu(invalid); });
}

void trainer_reconciles_controller_with_observed_rollouts() {
    auto tape=std::make_shared<mars_titan::simulation::MarketTape>();
    tape->assets={"FIC0"};
    tape->currency="USD";
    tape->domain="synthetic";
    tape->partition="train";
    tape->source_sha256.assign(64,'a');
    tape->parent_id="fixed-fixture";
    for (int64_t index=0;index<3;++index) {
        tape->open_times.push_back(2*index);
        tape->close_times.push_back(2*index+1);
        tape->prediction_times.push_back(2*index+1);
        tape->prices.insert(tape->prices.end(),{10.,10.,10.,10.,10000.});
        tape->scores.push_back(.01);
    }
    PpoTrainingConfig config;
    config.total_transitions=2;
    config.rollout_transitions=2;
    PpoObjectiveConfig objective;
    objective.kind=PpoObjectiveKind::clip_full_kl;
    PpoTrainer trainer({{tape,{},{}}},config,{},"cpu",true,{},objective);
    auto state=trainer.snapshot();
    trainer.restore(state);
    state.controller.completed_rollouts=1;
    state.policy_archive=altered(state.policy_archive,[](torch::serialize::InputArchive& input) {
        c10::IValue controller;
        input.read("controller",controller);
        controller.toObject()->setAttr("completed_rollouts",c10::IValue(int64_t{1}));
    });
    rejected([&] { trainer.restore(state); });
    require(trainer.transitions()==0 && trainer.optimizer_steps()==0,"La restauración alteró los cursores");
}

void sampler_temporaries_count_toward_the_policy_budget() {
    PpoObjectiveConfig objective;
    objective.kind=PpoObjectiveKind::clip_full_kl;
    PpoPolicy policy(4,{},42,"cpu",std::size_t{5}*1024*1024,{},objective);
    rejected([&] { (void)policy.full_kl(rollout(true,1,1000)); });
    require(policy.optimizer_steps()==0,"La comprobación de presupuesto actualizó Adam");
}
}

int main(int argc, char** argv) {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        if (argc==2) {
            const std::string_view selected(std::span(argv,static_cast<std::size_t>(argc)).subspan(1).front());
            if (selected=="budget") { sampler_temporaries_count_toward_the_policy_budget(); }
            else if(selected=="rollout-count") { trainer_reconciles_controller_with_observed_rollouts(); }
            else { throw std::invalid_argument("Sonda desconocida"); }
            return 0;
        }
        policy_roundtrip_keeps_controller_and_legacy_version();
        rollout_preserves_weights_without_alias_or_migration();
        empty_update_checks_q_without_executing_optimizer();
        incompatible_checkpoint_fields_are_rejected();
        trainer_reconciles_controller_with_observed_rollouts();
        sampler_temporaries_count_toward_the_policy_budget();
        std::cout << "Persistencia PPO comprobada sin actualizaciones\n";
    } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
