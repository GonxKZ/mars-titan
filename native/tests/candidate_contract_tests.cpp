#include "mars_titan/candidate.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

// Fixtures con valores numéricos explícitos para contrastar fórmulas y errores.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using namespace mars_titan::candidate;
void require(bool value, std::string_view reason) {
    if (!value) {
        throw std::runtime_error(std::string(reason));
    }
}
template <class F> void rejected(F&& action) {
    bool failed = false;
    try {
        std::forward<F>(action)();
    } catch (const std::exception&) {
        failed = true;
    }
    require(failed, "Se aceptó una entrada incompatible");
}
void same(const at::Tensor& a, const at::Tensor& b, std::string_view reason) {
    require(at::allclose(a, b, 1e-10, 1e-10), reason);
}
Config config() {
    Config value;
    value.dimensions = {2, 3, 4, 2, 3};
    value.normalization_id = "synthetic-v1";
    value.max_batch = 8;
    value.max_episodes = 16;
    return value;
}
Inputs inputs(int64_t batch = 2) {
    return {at::linspace(-1., 1., batch * 64 * 2, at::kDouble).reshape({batch, 64, 2}),
            at::ones({batch, 3}, at::kDouble),
            at::ones({batch, 4}, at::kDouble) * 2,
            at::ones({batch, 2}, at::kDouble) * 3,
            at::ones({batch, 3}, at::kDouble) * 4,
            at::ones({batch, 5}, at::kBool)};
}
MemorySnapshot memory(const Candidate& model, int64_t count = 2) {
    const auto keys = at::zeros({count, 128}, at::kDouble);
    keys.select(1, 0).fill_(1);
    const auto features = at::zeros({count, 256}, at::kDouble);
    return model.snapshot(keys, features, at::arange(count, at::kDouble),
                          at::arange(count, at::kLong), model.representation_id());
}
void stable_ties_and_snapshot_ownership() {
    const Candidate model(config(), at::kDouble);
    const auto keys = at::zeros({12, 128}, at::kDouble);
    keys.select(1, 0).fill_(1);
    const auto features = at::ones({12, 256}, at::kDouble).set_requires_grad(true);
    const auto outcomes = at::arange(12, at::kDouble).set_requires_grad(true);
    const auto ids = at::arange(12, at::kLong) * 3;
    const auto snapshot = model.snapshot(keys, features, outcomes, ids, model.representation_id());
    const auto state = at::ones({2, 128}, at::kDouble).set_requires_grad(true);
    const auto before = model.read(state, snapshot);
    require(at::equal(before.ids.select(0, 0), at::arange(8, at::kLong) * 3),
            "Los empates no conservan el orden canónico");
    same(before.weights, at::full({2, 8}, 0.125, at::kDouble), "Pesos de empate incorrectos");
    {
        const at::NoGradGuard guard;
        keys.zero_();
        features.zero_();
        outcomes.zero_();
        ids.fill_(99);
    }
    same(before.values, model.read(state, snapshot).values,
         "La instantánea conserva alias del llamador");
    before.values.sum().backward();
    require(!features.grad().defined() && !outcomes.grad().defined(),
            "Se retuvo el grafo de episodios");
    rejected(
        [&] { (void)model.snapshot(keys, features, outcomes, ids, model.representation_id()); });
    rejected([&] {
        (void)model.snapshot(keys, features, outcomes, at::arange(12, at::kLong), "other");
    });
}
void attention_gradient_matches_finite_differences() {
    Candidate model(config(), at::kDouble);
    {
        const at::NoGradGuard guard;
        for (auto& parameter : model.parameters()) {
            parameter.zero_();
        }
        model.named_parameters()["query_weight"].copy_(at::eye(128, at::kDouble));
        model.named_parameters()["value_weight"].select(0, 0).select(0, 256).fill_(1);
    }
    const auto keys = at::zeros({2, 128}, at::kDouble);
    keys.select(0, 0).select(0, 0).fill_(1);
    keys.select(0, 1).select(0, 0).fill_(-1);
    const auto snapshot =
        model.snapshot(keys, at::zeros({2, 256}, at::kDouble), at::tensor({-1., 2.}, at::kDouble),
                       at::tensor({10, 20}, at::kLong), model.representation_id());
    const auto state = at::zeros({1, 128}, at::kDouble);
    state.select(1, 0).fill_(1);
    state.select(1, 1).fill_(1);
    state.set_requires_grad(true);
    const auto read = model.read(state, snapshot);
    const double weight = 1 / (1 + std::exp(-std::sqrt(2.)));
    require(std::abs(read.values.select(1, 0).item<double>() - (2 - 3 * weight)) < 1e-12,
            "La atención difiere del caso analítico");
    read.values.sum().backward();
    require(model.named_parameters()["query_weight"].grad().abs().sum().item<double>() > 0,
            "La consulta no recibe gradiente");
    require(model.named_parameters()["value_weight"].grad().abs().sum().item<double>() > 0,
            "La proyección de valores no recibe gradiente");
    constexpr double epsilon = 1e-6;
    for (const int64_t coordinate : {0, 1, 2}) {
        const auto plus = state.detach().clone();
        const auto minus = state.detach().clone();
        plus.select(1, coordinate).add_(epsilon);
        minus.select(1, coordinate).sub_(epsilon);
        const double numerical = (model.read(plus, snapshot).values.sum().item<double>() -
                                  model.read(minus, snapshot).values.sum().item<double>()) /
                                 (2 * epsilon);
        require(std::abs(numerical - state.grad().select(1, coordinate).item<double>()) < 1e-7,
                "El gradiente de consulta no coincide con diferencias finitas");
    }
    model.zero_grad();
    const auto one = model.read(state, memory(model, 1));
    require(one.weights.item<double>() == 1, "Un vecino no tiene peso uno");
    one.values.sum().backward();
    require(model.named_parameters()["query_weight"].grad().abs().sum().item<double>() == 0,
            "Un vecino produce un gradiente de selección ficticio");
}
void seeds_are_independent_and_do_not_change_global_rng() {
    const auto before = at::detail::getDefaultCPUGenerator().get_state().clone();
    const Candidate first(config(), at::kDouble);
    auto alternate = config();
    alternate.feature_seed += 1;
    const Candidate second(alternate, at::kDouble);
    require(at::equal(before, at::detail::getDefaultCPUGenerator().get_state()),
            "Cambió el RNG global");
    const auto left = first.named_parameters();
    const auto right = second.named_parameters();
    for (const auto& parameter : left) {
        require(at::equal(parameter.value(), right[parameter.key()]),
                "Los buffers alteran la inicialización aprendida");
    }
    require(!at::equal(first.encode(inputs()).episode_features,
                       second.encode(inputs()).episode_features),
            "La semilla fija no altera la representación");
}
void fused_gru_matches_independent_recurrence() {
    const Candidate model(config(), at::kDouble);
    const auto data = inputs();
    const auto parameters = model.named_parameters();
    auto hidden = at::zeros({2, 128}, at::kDouble);
    for (int64_t step = 0; step < 64; ++step) {
        const auto input_gates =
            at::linear(data.prices.select(1, step), parameters["price_weight_ih"],
                       parameters["price_bias_ih"])
                .chunk(3, -1);
        const auto hidden_gates =
            at::linear(hidden, parameters["price_weight_hh"], parameters["price_bias_hh"])
                .chunk(3, -1);
        const auto reset = (input_gates.at(0) + hidden_gates.at(0)).sigmoid();
        const auto update = (input_gates.at(1) + hidden_gates.at(1)).sigmoid();
        const auto candidate = (input_gates.at(2) + reset * hidden_gates.at(2)).tanh();
        hidden = (1 - update) * candidate + update * hidden;
    }
    const std::array<std::pair<std::string, at::Tensor>, 4> blocks{
        {{"news", data.news},
         {"charts", data.charts},
         {"fundamentals", data.fundamentals},
         {"macro", data.macro}}};
    std::vector<at::Tensor> projected{hidden};
    for (const auto& [name, block] : blocks) {
        projected.push_back(
            at::silu(at::linear(block, parameters[name + "_weight"], parameters[name + "_bias"])));
    }
    const auto expected = at::silu(
        at::linear(at::cat(projected, -1), parameters["fusion_weight"], parameters["fusion_bias"]));
    same(model.encode(data).fused, expected,
         "La GRU o la fusión difieren de la recurrencia explícita");
}
void refinement_includes_state_fusion_memory_and_presence() {
    Candidate model(config(), at::kDouble);
    {
        const at::NoGradGuard guard;
        model.named_parameters()["update_weight"].fill_(0.01);
        model.named_parameters()["update_bias"].fill_(-0.2);
    }
    const auto state = at::ones({1, 128}, at::kDouble);
    const auto fused = at::ones({1, 256}, at::kDouble) * 2;
    Read read{at::ones({1, 128}, at::kDouble) * 3, {}, {}, at::ones({1, 1}, at::kBool)};
    const double expected = 1 + 0.1 * std::tanh(0.01 * (128 + 512 + 384 + 1) - 0.2);
    same(model.refine(state, fused, read), at::full({1, 128}, expected, at::kDouble),
         "El refinamiento omite un término o aplica una escala diferente");
    const auto snapshot = memory(model);
    for (const int64_t steps : {1, 2, 4}) {
        const auto encoded = model.encode(inputs());
        auto current = model.initial_state(encoded.fused);
        for (int64_t step = 0; step < steps; ++step) {
            current = model.refine(current, encoded.fused, model.read(current, snapshot));
        }
        same(model.forward(inputs(), snapshot, steps).state, current,
             "K cambia la instantánea o la transición");
    }
}
void configuration_and_boundaries_fail_fast() {
    auto invalid = config();
    invalid.dimensions.at(0) = 0;
    rejected([&] { (void)Candidate(invalid); });
    invalid = config();
    invalid.temperature = 0;
    rejected([&] { (void)Candidate(invalid); });
    invalid = config();
    invalid.max_episodes = 1000000;
    rejected([&] { (void)Candidate(invalid); });
    invalid = config();
    invalid.normalization_id.clear();
    rejected([&] { (void)Candidate(invalid); });
    const Candidate model(config(), at::kDouble);
    for (const int64_t rows : {0, 9}) {
        const auto state = at::zeros({rows, 128}, at::kDouble);
        const auto fused = at::zeros({rows, 256}, at::kDouble);
        const Read read{at::zeros_like(state), {}, {}, at::zeros({rows, 1}, at::kBool)};
        rejected([&] { (void)model.initial_state(fused); });
        rejected([&] { (void)model.refine(state, fused, read); });
        rejected([&] { (void)model.quantiles(state); });
    }
    rejected([&] { (void)model.forward(inputs(), model.empty_memory(), 3); });
    auto data = inputs();
    data.news = at::zeros({2, 0}, at::kDouble);
    rejected([&] { (void)model.encode(data); });
    data = inputs();
    data.macro.fill_(std::numeric_limits<double>::infinity());
    rejected([&] { (void)model.encode(data); });
    rejected([&] { (void)model.forward(inputs(9), model.empty_memory()); });
    const auto keys = at::ones({1, 128}, at::kDouble);
    rejected([&] {
        (void)model.snapshot(keys, at::zeros({1, 256}, at::kDouble), at::zeros({1}, at::kDouble),
                             at::zeros({1}, at::kLong), model.representation_id());
    });
    rejected([&] {
        (void)model.snapshot(at::zeros({1, 128}, at::kDouble), at::zeros({1, 256}, at::kDouble),
                             at::zeros({1}, at::kLong), at::zeros({1}, at::kDouble),
                             model.representation_id());
    });
}
void rows_are_independent_and_fp32_matches_fp64() {
    const Candidate precise(config(), at::kDouble);
    Candidate compact(config(), at::kDouble);
    compact.to(at::kFloat);
    const auto data = inputs();
    const auto order = at::tensor({1, 0}, at::kLong);
    const Inputs permuted{
        data.prices.index_select(0, order), data.news.index_select(0, order),
        data.charts.index_select(0, order), data.fundamentals.index_select(0, order),
        data.macro.index_select(0, order),  data.presence.index_select(0, order)};
    const auto snapshot = memory(precise);
    const auto before = precise.forward(data, snapshot);
    same(precise.forward(permuted, snapshot).quantiles, before.quantiles.index_select(0, order),
         "El orden de filas modifica la predicción");
    const Inputs fp32{data.prices.to(at::kFloat), data.news.to(at::kFloat),
                      data.charts.to(at::kFloat), data.fundamentals.to(at::kFloat),
                      data.macro.to(at::kFloat),  data.presence};
    require(at::allclose(compact.forward(fp32, compact.empty_memory()).quantiles.to(at::kDouble),
                         precise.forward(data, precise.empty_memory()).quantiles, 1e-5, 1e-6),
            "FP32 difiere de la referencia FP64");
}
void serialization_restores_changed_parameters_and_configuration() {
    Candidate model(config(), at::kDouble);
    {
        const at::NoGradGuard guard;
        model.named_parameters()["head_bias"].select(0, 2).add_(0.7);
    }
    const auto snapshot = memory(model);
    const auto before = model.forward(inputs(), snapshot, 2);
    std::stringstream archive;
    model.save_state(archive);
    const auto restored = Candidate::load_state(archive);
    require(restored->config() == model.config(), "La recuperación cambió la configuración");
    const auto after = restored->forward(inputs(), snapshot, 2);
    require(at::equal(before.quantiles, after.quantiles), "La recuperación cambia la predicción");
    require(at::equal(before.encoded.episode_keys, after.encoded.episode_keys),
            "No se conservaron los buffers");
    before.quantiles.sum().backward();
    after.quantiles.sum().backward();
    for (const auto& parameter : model.named_parameters()) {
        require(at::equal(parameter.value().grad(),
                          restored->named_parameters()[parameter.key()].grad()),
                "La recuperación cambia el gradiente");
    }
    std::stringstream corrupted("not an archive");
    rejected([&] { (void)Candidate::load_state(corrupted); });
}
void public_aliases_cannot_change_fixed_projections() {
    Candidate model(config(), at::kDouble);
    const auto before = model.encode(inputs());
    const auto identity = model.representation_id();
    const auto snapshot =
        model.snapshot(before.episode_keys, before.episode_features, at::ones({2}, at::kDouble),
                       at::arange(2, at::kLong), identity);
    const auto prediction = model.forward(inputs(), snapshot).quantiles;
    {
        const at::NoGradGuard guard;
        for (const auto& buffer : model.named_buffers()) {
            buffer.value().data().zero_();
        }
    }
    require(at::equal(before.episode_features, model.encode(inputs()).episode_features),
            "Un alias público modificó las proyecciones fijas");
    require(identity == model.representation_id(),
            "La identidad fija cambió mediante un alias público");
    require(at::equal(prediction, model.forward(inputs(), snapshot).quantiles),
            "Un alias público invalidó una instantánea de la misma representación");
}
void large_finite_query_keeps_its_direction() {
    Candidate model(config(), at::kFloat);
    {
        const at::NoGradGuard guard;
        for (auto& parameter : model.parameters()) {
            parameter.zero_();
        }
        model.named_parameters()["query_weight"].copy_(at::eye(128));
        model.named_parameters()["value_weight"].select(0, 0).select(0, 256).fill_(1);
    }
    const auto keys = at::zeros({2, 128});
    keys.select(0, 0).select(0, 0).fill_(1);
    keys.select(0, 1).select(0, 0).fill_(-1);
    const auto snapshot =
        model.snapshot(keys, at::zeros({2, 256}), at::tensor({-1.F, 2.F}),
                       at::tensor({10, 20}, at::kLong), model.representation_id());
    const auto state = at::zeros({1, 128});
    state.select(1, 0).fill_(1e20);
    state.select(1, 1).fill_(1e20);
    const double expected = 2 - 3 / (1 + std::exp(-std::sqrt(2.)));
    require(std::abs(model.read(state, snapshot).values.select(1, 0).item<double>() - expected) <
                1e-6,
            "Una norma desbordada ocultó la dirección de la consulta");
    same(model.read(at::zeros_like(state), snapshot).weights.to(at::kDouble),
         at::full({1, 2}, 0.5, at::kDouble), "La consulta nula no conserva pesos uniformes");
}
} // namespace
int main() {
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    const std::array<std::pair<std::string_view, void (*)()>, 10> tests{
        {{"empates y copia", stable_ties_and_snapshot_ownership},
         {"gradientes de lectura", attention_gradient_matches_finite_differences},
         {"RNG independiente", seeds_are_independent_and_do_not_change_global_rng},
         {"referencia GRU", fused_gru_matches_independent_recurrence},
         {"refinamiento completo", refinement_includes_state_fusion_memory_and_presence},
         {"límites", configuration_and_boundaries_fail_fast},
         {"serialización", serialization_restores_changed_parameters_and_configuration},
         {"permutación y precisión", rows_are_independent_and_fp32_matches_fp64},
         {"propiedad de proyecciones", public_aliases_cannot_change_fixed_projections},
         {"consulta finita grande", large_finite_query_keeps_its_direction}}};
    int failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "OK " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << name << ": " << error.what() << '\n';
        }
    }
    return failures == 0 ? 0 : 1;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
