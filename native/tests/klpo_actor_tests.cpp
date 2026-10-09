#include "mars_titan/klpo_terminal.hpp"
#include "mars_titan/ppo_policy.hpp"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <torch/optim/adam.h>
#include <torch/serialize/archive.h>
#include <type_traits>
#include <utility>

// Actores fijados y archivos de prueba. Ninguna prueba llama a terminal_step.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using namespace mars_titan::learning;
static_assert(!std::is_reference_v<decltype(std::declval<PpoPolicy>().terminal_adam_options())>);
void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
template <class F> void rejected(F f) {
    try {
        f();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("Se aceptó un contrato incompatible");
}

PpoPolicy actor(PpoNetworkKind kind) {
    PpoArchitecture architecture;
    architecture.kind = kind;
    PpoHyperparameters parameters;
    parameters.learning_rate = .002;
    PpoPolicy result(3, parameters, 42, "cpu", default_ppo_memory_bytes, architecture);
    result.enable_terminal_adam({.learning_rate = .002, .gradient_norm = 0.});
    return result;
}

at::Tensor loss(const PpoPolicy& policy, int64_t begin, int64_t count) {
    const auto all = (at::arange(18 * 4 * 3, at::kFloat).reshape({18, 4, 3}) / 100);
    const auto values = all.narrow(1, begin, count);
    const auto logits = policy.terminal_forward(values, at::full({count}, 18, at::kLong)).logits;
    const auto logp = logits.log_softmax(-1).transpose(0, 1);
    const auto logq = at::full_like(logp, -std::log(6.));
    const auto actions = at::full({count, 18}, 2, at::kLong);
    const auto mask = at::ones({count, 18}, at::kBool);
    const auto rewards = at::arange(begin + 1, begin + count + 1, at::kDouble) / 10;
    return klpo_terminal_full_loss(logp, logq, actions, rewards, mask, .2).sum() / 4;
}

void accumulation_keeps_gradients_and_critic_invariant(PpoNetworkKind kind) {
    auto full = actor(kind);
    auto blocked = actor(kind);
    const auto before = full.parameter_fingerprint();
    full.terminal_zero_grad();
    loss(full, 0, 4).backward();
    const auto first = full.terminal_gradients();
    require(full.validate_terminal_gradients() > 0, "Falta un gradiente finito");
    blocked.terminal_zero_grad();
    for (int64_t begin = 0; begin < 4; ++begin) {
        loss(blocked, begin, 1).backward();
    }
    const auto second = blocked.terminal_gradients();
    require(first.size() == second.size() && first.size() == 6, "Faltan parámetros del actor");
    for (std::size_t index = 0; index < first.size(); ++index) {
        require(at::allclose(first[index], second[index], 1e-5, 1e-6),
                "La acumulación cambió el gradiente global");
    }
    require(full.parameter_fingerprint() == before && blocked.parameter_fingerprint() == before &&
                full.optimizer_steps() == 0 && blocked.optimizer_steps() == 0,
            "El cálculo de gradientes alteró pesos o ejecutó Adam");
    full.terminal_zero_grad();
    full.terminal_forward(at::ones({2, 1, 3}), at::full({1}, 2, at::kLong)).values.sum().backward();
    rejected([&] { static_cast<void>(full.validate_terminal_gradients()); });
}

void terminal_role_roundtrips_and_old_role_is_rejected() {
    auto original = actor(PpoNetworkKind::gru);
    std::stringstream bytes;
    original.save(bytes);
    std::istringstream wrong(bytes.str());
    rejected([&] { static_cast<void>(PpoPolicy::load(wrong)); });
    std::istringstream input(bytes.str());
    auto restored = PpoPolicy::load_terminal(input);
    require(restored.terminal_adam_enabled() &&
                restored.terminal_adam_options() == original.terminal_adam_options(),
            "Se perdió el contrato de Adam terminal");
    require(restored.parameter_fingerprint() == original.parameter_fingerprint() &&
                restored.optimizer_steps() == 0,
            "La recuperación cambió el actor fijado");
    PpoPolicy legacy(3, {}, 42);
    std::stringstream legacy_bytes;
    legacy.save(legacy_bytes);
    rejected([&] { static_cast<void>(PpoPolicy::load_terminal(legacy_bytes)); });
}

void reference_copy_has_its_own_rng_and_no_optimizer_role() {
    auto learner = actor(PpoNetworkKind::gru);
    const auto rng = learner.random_state();
    auto reference = learner.frozen_reference(rng);
    require(!reference.terminal_adam_enabled() && reference.optimizer_steps() == 0 &&
                reference.parameter_fingerprint() == learner.parameter_fingerprint(),
            "La referencia no copió solamente parámetros y RNG");
    static_cast<void>(reference.act_recurrent(at::zeros({2, 3})));
    require(at::equal(learner.random_state().sampling, rng.sampling) &&
                !at::equal(reference.random_state().sampling, rng.sampling),
            "El sampler comparte su estado aleatorio con el actor");
}

// El contador y los momentos se escriben como datos de fixture. No se llama a Adam::step.
std::string optimizer_fixture(bool terminal, bool corrupt_critic) {
    PpoHyperparameters hyper;
    hyper.learning_rate = .002;
    PpoPolicy policy(3, hyper, 42);
    if (terminal) {
        policy.enable_terminal_adam({.learning_rate = .002, .gradient_norm = 0});
    }
    std::stringstream saved;
    policy.save(saved);
    torch::serialize::InputArchive source;
    source.load_from(saved, at::Device(at::kCPU));
    c10::IValue network;
    source.read("network", network);
    std::vector<at::Tensor> parameters;
    for (const auto* name : {"first_weight", "first_bias", "second_weight", "second_bias",
                             "output_weight", "output_bias"}) {
        parameters.push_back(network.toObject()->getAttr(name).toTensor().detach().clone());
    }
    torch::optim::Adam fixture(parameters, torch::optim::AdamOptions(hyper.learning_rate));
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        auto state = std::make_unique<torch::optim::AdamParamState>();
        state->step(1);
        state->exp_avg(at::zeros_like(parameters[index]));
        state->exp_avg_sq(at::zeros_like(parameters[index]));
        if (corrupt_critic && index + 2 >= parameters.size()) {
            state->exp_avg().select(0, ppo_action_count).fill_(.25);
        }
        fixture.state()[parameters[index].unsafeGetTensorImpl()] = std::move(state);
    }
    torch::serialize::OutputArchive output, optimizer;
    fixture.save(optimizer);
    for (const auto& key : source.keys()) {
        if (key == "optimizer") {
            output.write(key, optimizer);
        } else {
            c10::IValue value;
            source.read(key, value);
            output.write(key, value);
        }
    }
    std::ostringstream stream;
    output.save_to(stream);
    return stream.str();
}

void moment_fixtures_do_not_import_ppo_or_accept_critic_updates() {
    std::istringstream valid(optimizer_fixture(true, false));
    const auto recovered = PpoPolicy::load_terminal(valid);
    require(recovered.optimizer_steps() == 1, "El fixture no conservó su contador declarado");
    std::istringstream invalid(optimizer_fixture(true, true));
    rejected([&] { static_cast<void>(PpoPolicy::load_terminal(invalid)); });
    std::istringstream old(optimizer_fixture(false, false));
    auto legacy = PpoPolicy::load(old);
    require(legacy.optimizer_steps() == 1, "El fixture PPO no conservó su contador");
    rejected([&] { legacy.enable_terminal_adam({.learning_rate = .002, .gradient_norm = 0}); });
}

void changed_adam_contract_is_rejected() {
    auto policy = actor(PpoNetworkKind::mlp);
    std::ostringstream bytes;
    policy.save(bytes);
    for (const auto* name : {"weight_decay", "epsilon", "learning_rate", "beta1"}) {
        torch::serialize::InputArchive source;
        std::istringstream input(bytes.str());
        source.load_from(input, at::Device(at::kCPU));
        c10::IValue options;
        source.read("terminal_adam", options);
        options.toObject()->setAttr(name, c10::IValue(.25));
        torch::serialize::OutputArchive output;
        for (const auto& key : source.keys()) {
            c10::IValue value;
            source.read(key, value);
            output.write(key, value);
        }
        std::stringstream altered;
        output.save_to(altered);
        rejected([&] { static_cast<void>(PpoPolicy::load_terminal(altered)); });
    }
}

void terminal_archive_rejects_precision_conversion_and_keeps_legacy_acceptance() {
    for (const bool terminal : {false, true}) {
        PpoHyperparameters hyper;
        hyper.learning_rate = .002;
        PpoPolicy policy(3, hyper, 42);
        if (terminal) {
            policy.enable_terminal_adam({.learning_rate = .002, .gradient_norm = 0});
        }
        std::stringstream bytes;
        policy.save(bytes);
        torch::serialize::InputArchive input;
        input.load_from(bytes, at::Device(at::kCPU));
        c10::IValue network;
        input.read("network", network);
        network.toObject()->setAttr(
            "output_weight",
            network.toObject()->getAttr("output_weight").toTensor().to(at::kDouble));
        torch::serialize::OutputArchive output;
        for (const auto& key : input.keys()) {
            c10::IValue value;
            input.read(key, value);
            output.write(key, value);
        }
        std::stringstream changed;
        output.save_to(changed);
        if (terminal) {
            rejected([&] { static_cast<void>(PpoPolicy::load_terminal(changed)); });
        } else {
            auto recovered = PpoPolicy::load(changed);
            require(!recovered.terminal_adam_enabled() && recovered.observation_width() == 3,
                    "Se cambió la aceptación legacy");
        }
    }
}
} // namespace

int main() {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        accumulation_keeps_gradients_and_critic_invariant(PpoNetworkKind::mlp);
        accumulation_keeps_gradients_and_critic_invariant(PpoNetworkKind::gru);
        terminal_role_roundtrips_and_old_role_is_rejected();
        reference_copy_has_its_own_rng_and_no_optimizer_role();
        moment_fixtures_do_not_import_ppo_or_accept_critic_updates();
        changed_adam_contract_is_rejected();
        terminal_archive_rejects_precision_conversion_and_keeps_legacy_acceptance();
        std::cout << "Contratos del actor terminal comprobados sin pasos de Adam\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
