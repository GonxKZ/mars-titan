#include "mars_titan/candidate.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

// Los valores literales son casos pequeños y expectativas independientes.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using namespace mars_titan::candidate;
void require(bool value, std::string_view reason) {
    if (!value) { throw std::runtime_error(std::string(reason)); }
}
template<class F> void rejected(F&& action) {
    bool failed = false;
    try { std::forward<F>(action)(); } catch (const std::exception&) { failed = true; }
    require(failed, "Se aceptó una entrada incompatible");
}
Config small_config() {
    Config config;
    config.dimensions = {2, 3, 4, 2, 3};
    config.normalization_id = "synthetic-v1";
    config.max_batch = 8;
    config.max_episodes = 16;
    return config;
}
Inputs inputs(int64_t batch = 2) {
    const auto options = at::TensorOptions().dtype(at::kDouble);
    return {at::linspace(-1., 1., batch * 64 * 2, options).reshape({batch, 64, 2}),
            at::ones({batch, 3}, options), at::ones({batch, 4}, options) * 2,
            at::ones({batch, 2}, options) * 3, at::ones({batch, 3}, options) * 4,
            at::ones({batch, 5}, at::kBool)};
}
void empty_memory_and_ordered_quantiles() {
    const Candidate model(small_config(), at::kDouble);
    const auto output = model.forward(inputs(), model.empty_memory());
    require(output.quantiles.sizes() == at::IntArrayRef({2, 5}), "Forma de cuantiles incorrecta");
    require(at::isfinite(output.quantiles).all().item<bool>(), "Cuantiles no finitos");
    require((output.quantiles.slice(1, 1) >= output.quantiles.slice(1, 0, 4)).all().item<bool>(),
            "Los cuantiles se cruzan");
    require(output.read.values.abs().sum().item<double>() == 0, "La memoria vacía aporta un valor");
    require(!output.read.presence.any().item<bool>(), "La memoria vacía aparece presente");
    require(at::equal(output.quantiles, model.forward(inputs(), model.empty_memory()).quantiles),
            "El cálculo guarda estado entre ventanas");
}
void incomplete_or_nonfinite_rows_are_rejected() {
    const Candidate model(small_config(), at::kDouble);
    auto data = inputs();
    data.presence.select(1, 4).fill_(false);
    rejected([&] { (void)model.forward(data, model.empty_memory()); });
    data = inputs();
    data.prices.select(0, 0).fill_(std::numeric_limits<double>::quiet_NaN());
    rejected([&] { (void)model.forward(data, model.empty_memory()); });
    data = inputs();
    data.macro = data.macro.to(at::kFloat);
    rejected([&] { (void)model.forward(data, model.empty_memory()); });
}
void zero_head_has_hand_computed_intervals() {
    Candidate model(small_config(), at::kDouble);
    {
        const at::NoGradGuard guard;
        for (auto& parameter : model.parameters()) { parameter.zero_(); }
    }
    const auto output = model.forward(inputs(1), model.empty_memory());
    const double gap = std::log(2.);
    const auto expected = at::tensor({-2 * gap, -gap, 0., gap, 2 * gap}, at::kDouble).unsqueeze(0);
    require(at::allclose(output.quantiles, expected, 1e-12, 1e-12), "Cabeza distinta de la fórmula");
}
}
int main() {
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    try {
        empty_memory_and_ordered_quantiles();
        incomplete_or_nonfinite_rows_are_rejected();
        zero_head_has_hand_computed_intervals();
        std::cout << "Pruebas del cálculo candidato correctas\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
