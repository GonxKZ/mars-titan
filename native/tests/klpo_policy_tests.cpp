#include "mars_titan/ppo_policy.hpp"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

// Gradientes de tensores fijos, sin PpoTrainer ni pasos de optimizador.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using namespace mars_titan::learning;
void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

void full_history_reaches_the_forced_prefix(PpoNetworkKind kind) {
    PpoArchitecture architecture;
    architecture.kind = kind;
    PpoHyperparameters hyper;
    hyper.minibatch_size = 16;
    PpoPolicy policy(3, hyper, 42, "cpu", default_ppo_memory_bytes, architecture);
    auto history =
        (at::arange(18 * 2 * 3, at::kFloat).reshape({18, 2, 3}) / 100).requires_grad_(true);
    const auto lengths = at::tensor({18, 12}, at::kLong);
    const auto rng = policy.random_state();
    std::ostringstream before;
    policy.save(before);
    const auto result = policy.terminal_forward(history, lengths);
    require(result.logits.sizes() == at::IntArrayRef({18, 2, 6}),
            "Faltan logits de la historia completa");
    auto state = policy.initial_state(2);
    for (int64_t time = 0; time < 18; ++time) {
        const auto reference = policy.infer(history[time].detach(), state);
        state = reference.next_state;
        if (time < 12) {
            require(at::allclose(result.logits[time].detach(), reference.logits, 1e-5, 1e-6),
                    "La ruta diferenciable cambia la política de referencia");
        }
    }
    const auto loss = result.logits[17][0][2];
    const auto gradients =
        torch::autograd::grad({loss}, {history, result.values}, {}, false, false, true);
    require(gradients[0].defined() && at::isfinite(gradients[0]).all().item<bool>(),
            "Falta un gradiente finito");
    require(!gradients[1].defined(), "El consumidor del actor utiliza la cabeza de valor");
    if (kind == PpoNetworkKind::gru) {
        require(gradients[0][0][0].abs().sum().item<double>() > 0,
                "Se desacopló el prefijo anterior al paso 16");
    }
    require(gradients[0].select(1, 1).abs().sum().item<double>() == 0,
            "Se mezclaron historias de episodios");
    require(at::equal(rng.sampling, policy.random_state().sampling) &&
                at::equal(rng.shuffle, policy.random_state().shuffle),
            "El consumidor consume RNG");
    std::ostringstream after;
    policy.save(after);
    require(before.str() == after.str() && policy.optimizer_steps() == 0,
            "El cálculo del gradiente cambió parámetros o estado recuperable");
}

void reference_identity_ignores_rng_but_survives_restore() {
    PpoPolicy policy(3, {}, 42);
    const auto reference = policy.parameter_fingerprint();
    require(reference.size() == 64, "Falta la huella de los parámetros del sampler");
    const auto rng = policy.random_state();
    static_cast<void>(policy.act_recurrent(at::zeros({2, 3}, at::kFloat)));
    require(!at::equal(rng.sampling, policy.random_state().sampling),
            "No avanzó el muestreador del fixture");
    require(policy.parameter_fingerprint() == reference, "El RNG cambió la identidad de los pesos");
    std::ostringstream archive;
    policy.save(archive);
    std::istringstream saved(archive.str());
    const auto recovered = PpoPolicy::load(saved);
    require(recovered.parameter_fingerprint() == reference, "La recuperación alteró la referencia");
    PpoPolicy other(3, {}, 43);
    require(other.parameter_fingerprint() != reference, "Se confundieron pesos diferentes");
}
} // namespace

int main() {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        full_history_reaches_the_forced_prefix(PpoNetworkKind::mlp);
        full_history_reaches_the_forced_prefix(PpoNetworkKind::gru);
        reference_identity_ignores_rng_but_survives_restore();
        std::cout << "Historia diferenciable contrastada sin actualizaciones\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
