#include "mars_titan/ppo_objectives.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

// Las constantes describen distribuciones pequeñas con soluciones conocidas.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using mars_titan::learning::ppo_categorical_kl;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <class F> void rejected(F&& action) {
    bool failed = false;
    try {
        std::forward<F>(action)();
    } catch (const std::invalid_argument&) {
        failed = true;
    }
    require(failed, "Se aceptó una distribución incompatible");
}

at::Tensor uniform() { return at::full({2, 6}, -std::log(6.0), at::kDouble); }

void literal_divergence_and_direction() {
    const auto q = uniform();
    const auto p = at::tensor({0.5, 0.1, 0.1, 0.1, 0.1, 0.1}, at::kDouble)
                       .log().unsqueeze(0).repeat({2, 1});
    const double expected = std::log(1.0 / 3.0) / 6.0 + 5.0 * std::log(5.0 / 3.0) / 6.0;
    const auto actual = ppo_categorical_kl(p, q);
    require(actual.scalar_type() == at::kDouble && actual.sizes() == at::IntArrayRef({2}),
            "La divergencia necesita un resultado FP64 por fila");
    require(at::allclose(actual, at::full({2}, expected, at::kDouble), 1e-12, 1e-12),
            "La dirección o el valor de la KL no coincide con el cálculo manual");
    require(!at::allclose(actual, ppo_categorical_kl(q, p), 1e-4, 1e-4),
            "Se confundieron las dos direcciones de KL");
    require(at::equal(ppo_categorical_kl(q, q), at::zeros({2}, at::kDouble)),
            "La distribución idéntica tiene divergencia distinta de cero");
    require(at::allclose(ppo_categorical_kl(p.to(at::kFloat), q.to(at::kFloat)), actual, 1e-6, 1e-7),
            "La promoción FP32 no conserva el resultado dentro de su precisión");
}

void gradients_and_finite_differences() {
    auto logits = at::tensor({0.2, -0.3, 0.7, -0.1, 0.4, 0.0}, at::kDouble)
                      .reshape({1, 6}).requires_grad_(true);
    auto history = uniform().narrow(0, 0, 1).clone().requires_grad_(true);
    const auto before = logits.detach().clone();
    const auto old = history.detach().clone();
    const auto loss = ppo_categorical_kl(logits.log_softmax(1), history).sum();
    const auto gradient = torch::autograd::grad({loss}, {logits, history}, {}, false, false, true);
    require(!gradient.at(1).defined(), "La política histórica recibió gradiente");
    require(at::allclose(gradient.at(0), logits.detach().softmax(1) - 1.0 / 6.0, 1e-12, 1e-12),
            "El gradiente no coincide con p menos q");
    for (int64_t index = 0; index < 6; ++index) {
        const auto delta = at::one_hot(at::tensor(index, at::kLong), 6).to(at::kDouble)
                               .reshape({1, 6}) * 1e-6;
        const auto upper = ppo_categorical_kl((before + delta).log_softmax(1), old);
        const auto lower = ppo_categorical_kl((before - delta).log_softmax(1), old);
        const auto finite = ((upper - lower) / 2e-6).item<double>();
        require(std::abs(finite - gradient.at(0)[0][index].item<double>()) < 1e-8,
                "La diferencia finita no coincide con el gradiente");
    }
    require(at::equal(logits, before) && at::equal(history, old), "Se modificaron las entradas");
}

void support_and_invalid_inputs() {
    const double missing = -std::numeric_limits<double>::infinity();
    const auto delta = at::tensor({0.0, missing, missing, missing, missing, missing}, at::kDouble)
                           .reshape({1, 6});
    require(ppo_categorical_kl(delta, delta).item<double>() == 0, "Falló el soporte degenerado");
    auto differentiable = delta.clone().requires_grad_(true);
    const auto zero = ppo_categorical_kl(differentiable, delta).sum();
    const auto gradient = torch::autograd::grad({zero}, {differentiable}).front();
    require(at::isfinite(gradient).all().item<bool>() && gradient.abs().sum().item<double>() == 0,
            "El soporte degenerado produjo un gradiente inválido");
    rejected([&] { (void)ppo_categorical_kl(delta, uniform().narrow(0, 0, 1)); });
    rejected([&] { (void)ppo_categorical_kl(uniform() + 1, uniform()); });
    rejected([&] { (void)ppo_categorical_kl(uniform(), uniform() + 1); });
    rejected([&] { (void)ppo_categorical_kl(at::zeros({2, 5}, at::kDouble), uniform()); });
    const auto wrong_width = at::full({2, 5}, -std::log(5.0), at::kDouble);
    rejected([&] { (void)ppo_categorical_kl(wrong_width, wrong_width); });
    rejected([&] { (void)ppo_categorical_kl(uniform().to(at::kHalf), uniform()); });
    rejected([&] { (void)ppo_categorical_kl(delta.to(at::kHalf), delta); });
    rejected([&] { (void)ppo_categorical_kl(at::Tensor{}, uniform()); });
    rejected([&] { (void)ppo_categorical_kl(uniform().narrow(0, 0, 0), uniform().narrow(0, 0, 0)); });
    rejected([&] { (void)ppo_categorical_kl(at::full({2, 6}, std::numeric_limits<double>::quiet_NaN()), uniform()); });
    rejected([&] { (void)ppo_categorical_kl(at::full({2, 6}, std::numeric_limits<double>::infinity()), uniform()); });
    const auto unrepresentable = at::tensor({0.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0}, at::kDouble)
                                    .reshape({1, 6});
    rejected([&] { (void)ppo_categorical_kl(delta, unrepresentable); });
    const auto oversized = at::full({1, 6}, -std::log(6.0), at::kDouble).expand({65537, 6});
    rejected([&] { (void)ppo_categorical_kl(oversized, oversized); });
}

void strided_rows_and_partition() {
    const auto source = at::arange(36, at::kDouble).reshape({6, 6}).sin().log_softmax(1);
    const auto p = source.transpose(0, 1).contiguous().transpose(0, 1);
    const auto q = at::full({6, 6}, -std::log(6.0), at::kDouble);
    const auto before = p.clone();
    const auto actual = ppo_categorical_kl(p, q);
    require(!p.is_contiguous(), "El fixture necesita strides distintos");
    const auto indices = at::tensor({5, 2, 0}, at::kLong);
    require(at::allclose(ppo_categorical_kl(p.index_select(0, indices), q.index_select(0, indices)),
                        actual.index_select(0, indices), 1e-14, 1e-14),
            "La selección cambia las divergencias por fila");
    require(at::equal(p, before), "La entrada con strides cambió");
    const auto boundary = q.narrow(0, 0, 1).expand({65536, 6});
    require(at::isfinite(ppo_categorical_kl(boundary, boundary)).all().item<bool>(),
            "Se rechazó el límite de filas admitido");
}
} // namespace

int main() {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        const auto rng = at::detail::getDefaultCPUGenerator().get_state().clone();
        literal_divergence_and_direction();
        gradients_and_finite_differences();
        support_and_invalid_inputs();
        strided_rows_and_partition();
        require(at::equal(rng, at::detail::getDefaultCPUGenerator().get_state()), "Cambió el RNG");
        std::cout << "Contratos de KL categórica correctos, sin actualizaciones\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
