#include "mars_titan/klpo_terminal.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

// Distribuciones y resultados literales del oráculo independiente, sin ajuste.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using mars_titan::learning::klpo_terminal_full_loss;
void require(bool value, std::string_view message) {
    if (!value)
        throw std::runtime_error(std::string(message));
}
template <class F> void rejected(F&& action) {
    bool failed = false;
    try {
        std::forward<F>(action)();
    } catch (const std::invalid_argument&) {
        failed = true;
    }
    require(failed, "Se aceptó una entrada terminal incompatible");
}

void values_gradients_and_padding() {
    auto logits = (at::arange(36, at::kDouble).reshape({2, 3, 6}).sin() / 2).requires_grad_(true);
    auto historical = at::full({2, 3, 6}, -std::log(6.), at::kDouble).requires_grad_(true);
    auto returns = at::tensor({0.7, -0.2}, at::kDouble).requires_grad_(true);
    const auto mask = at::tensor({1, 1, 1, 1, 0, 0}, at::kLong).to(at::kBool).reshape({2, 3});
    const auto actions = at::tensor({0, 2, 4, 3, -1, 999}, at::kLong).reshape({2, 3});
    const auto logp = at::where(mask.unsqueeze(-1), logits.log_softmax(-1),
                                std::numeric_limits<double>::quiet_NaN());
    const auto logq =
        at::where(mask.unsqueeze(-1), historical, std::numeric_limits<double>::infinity());
    const auto actual = klpo_terminal_full_loss(logp, logq, actions, returns, mask, 0.3);
    require(actual.scalar_type() == at::kDouble && actual.sizes() == at::IntArrayRef({2}),
            "La pérdida no conserva una suma FP64 por trayectoria");
    auto expected = at::zeros({2}, at::kDouble);
    auto gradient_expected = at::zeros_like(logits);
    const auto clean = logits.detach().log_softmax(-1);
    for (int64_t row = 0; row < 2; ++row) {
        for (int64_t time = 0; time < (row == 0 ? 3 : 1); ++time) {
            const auto action = actions[row][time].item<int64_t>();
            const double selected = clean[row][time][action].item<double>();
            const double coefficient =
                returns[row].item<double>() - 0.3 * (selected + std::log(6.));
            expected[row] -= coefficient * (selected - clean[row][time].mean().item<double>());
            gradient_expected[row][time].fill_(coefficient / 6.);
            gradient_expected[row][time][action] -= coefficient;
        }
    }
    require(at::allclose(actual, expected, 1e-13, 1e-13),
            "El sustituto no coincide con la suma escalar");
    const auto gradients = torch::autograd::grad({actual.sum()}, {logits, historical, returns}, {},
                                                 false, false, true);
    require(at::allclose(gradients[0], gradient_expected, 1e-13, 1e-13),
            "Gradiente o padding incorrectos");
    require(!gradients[1].defined() && !gradients[2].defined(),
            "La historia o retorno recibe gradiente");
}

struct Population {
    std::vector<double> features, offsets, logs, weights, returns;
    std::vector<int64_t> actions, lengths;
};

struct PathCase {
    int first_action, following_state, second_action;
    double terminal, probability;
};

Population population() {
    Population data;
    constexpr std::array<double, 6> first{-0.5, 0.15, 0.4, -0.8, 0.6, -0.1};
    constexpr std::array<int, 6> following{6, 4, 2, 5, 3, 1};
    const auto append = [&](const PathCase& path) {
        const auto [a, state, b, terminal, probability] = path;
        data.returns.push_back(terminal);
        data.weights.push_back(probability);
        data.lengths.push_back(state == 0 ? 1 : 2);
        for (int step = 0; step < 2; ++step) {
            const int s = step == 0 ? 0 : state;
            data.actions.push_back(step == 0 ? a : b);
            for (int action = 0; action < 6; ++action) {
                data.features.push_back(static_cast<double>(((s + 2) * (action + 1)) % 11 - 5) /
                                        4.);
                data.features.push_back(static_cast<double>(((s + 3) * (action + 2)) % 13 - 6) /
                                        5.);
                data.offsets.push_back(static_cast<double>((s + 3 * action) % 7 - 3) / 10.);
                data.logs.push_back(std::log(
                    static_cast<double>(
                        s == 0 ? action + 1
                               : following.at(static_cast<std::size_t>((action + s) % 6))) /
                    21.));
            }
        }
    };
    for (int a = 0; a < 6; ++a) {
        const double initial = static_cast<double>(a + 1) / 21.;
        if (a < 2) {
            append({a, 0, 0, first.at(static_cast<std::size_t>(a)) - 0.2, initial / 3.});
            append({a, 0, 0, first.at(static_cast<std::size_t>(a)) + 0.1, initial * 2. / 3.});
            continue;
        }
        for (int event = 0; event < 2; ++event) {
            const int state = 1 + 2 * (a - 2) + event;
            const double branch = initial * static_cast<double>(event + 1) / 3.;
            for (int b = 0; b < 6; ++b) {
                const double immediate =
                    static_cast<double>((3 * a + 5 * b + 7 * event) % 17 - 8) / 5. +
                    static_cast<double>(a * b) / 20.;
                const double probability =
                    branch *
                    static_cast<double>(following.at(static_cast<std::size_t>((b + state) % 6))) /
                    21.;
                append({a, state, b,
                        first.at(static_cast<std::size_t>(a)) + 0.83 * (immediate - 0.3),
                        probability / 4.});
                append({a, state, b,
                        first.at(static_cast<std::size_t>(a)) + 0.83 * (immediate + 0.1),
                        probability * 3. / 4.});
            }
        }
    }
    return data;
}

void independent_population_and_microbatches() {
    const auto data = population();
    require(data.weights.size() == 100, "La tabla no conserva 100 trayectorias");
    const auto features = at::tensor(data.features, at::kDouble).reshape({100, 2, 6, 2});
    const auto offsets = at::tensor(data.offsets, at::kDouble).reshape({100, 2, 6});
    const auto logq = at::tensor(data.logs, at::kDouble).reshape({100, 2, 6});
    const auto actions = at::tensor(data.actions, at::kLong).reshape({100, 2});
    const auto returns = at::tensor(data.returns, at::kDouble);
    const auto weights = at::tensor(data.weights, at::kDouble);
    const auto mask =
        at::arange(2, at::kLong).unsqueeze(0) < at::tensor(data.lengths, at::kLong).unsqueeze(1);
    constexpr std::array<std::array<double, 4>, 4> oracle{
        {{-0.4, 0.7, -0.030967886318368134, 0.015907125478634522},
         {0., 0., 0.19981448059665183, -0.2082208520615879},
         {0.37, -0.2, 0.353297028377669, -0.30653806942472367},
         {1.1, 0.3, 0.5361860117487952, -0.28261170618101344}}};
    for (const auto& row : oracle) {
        auto theta = at::tensor({row[0], row[1]}, at::kDouble).requires_grad_(true);
        const auto logp = ((features * theta).sum(-1) + offsets).log_softmax(-1);
        const auto full = klpo_terminal_full_loss(logp, logq, actions, returns, mask, 0.3);
        const auto gradient =
            torch::autograd::grad({(full * weights).sum()}, {theta}, {}, true).front();
        require(at::allclose(gradient, at::tensor({row[2], row[3]}, at::kDouble), 1e-12, 1e-12),
                "El gradiente no coincide con el oráculo Decimal independiente");
        const auto first = klpo_terminal_full_loss(
            logp.narrow(0, 0, 37), logq.narrow(0, 0, 37), actions.narrow(0, 0, 37),
            returns.narrow(0, 0, 37), mask.narrow(0, 0, 37), 0.3);
        const auto last = klpo_terminal_full_loss(
            logp.narrow(0, 37, 63), logq.narrow(0, 37, 63), actions.narrow(0, 37, 63),
            returns.narrow(0, 37, 63), mask.narrow(0, 37, 63), 0.3);
        require(at::allclose(at::cat({first, last}), full, 1e-13, 1e-13),
                "Cambió el resultado al dividir el lote");
        const auto split_gradient =
            torch::autograd::grad({(first.sum() + last.sum()) / 100.}, {theta}, {}, true).front();
        const auto whole_gradient = torch::autograd::grad({full.mean()}, {theta}).front();
        require(at::allclose(split_gradient, whole_gradient, 1e-13, 1e-13),
                "Cambió el denominador global");
    }
}

void invalid_inputs() {
    const auto q = at::full({2, 3, 6}, -std::log(6.), at::kDouble);
    const auto a = at::zeros({2, 3}, at::kLong);
    const auto r = at::ones({2}, at::kDouble);
    const auto m = at::ones({2, 3}, at::kBool);
    for (const double beta : {0., -1., std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()})
        rejected([&] { (void)klpo_terminal_full_loss(q, q, a, r, m, beta); });
    rejected([&] { (void)klpo_terminal_full_loss(q, q, a, r, m, 0.3, {1}); });
    rejected([&] {
        (void)klpo_terminal_full_loss(q, q, a, r, m, 0.3, {std::size_t{65} * 1024 * 1024});
    });
    rejected([&] { (void)klpo_terminal_full_loss(q + 1, q, a, r, m, 0.3); });
    rejected([&] { (void)klpo_terminal_full_loss(q, q + 1, a, r, m, 0.3); });
    rejected([&] { (void)klpo_terminal_full_loss(q, q, a.to(at::kFloat), r, m, 0.3); });
    rejected([&] { (void)klpo_terminal_full_loss(q, q, a + 6, r, m, 0.3); });
    rejected([&] { (void)klpo_terminal_full_loss(q, q, a, r, m.to(at::kLong), 0.3); });
    rejected([&] { (void)klpo_terminal_full_loss(q, q, a, r, at::zeros_like(m), 0.3); });
    auto hole = m.clone();
    hole[0][1] = false;
    rejected([&] { (void)klpo_terminal_full_loss(q, q, a, r, hole, 0.3); });
    rejected([&] { (void)klpo_terminal_full_loss(q.to(at::kHalf), q, a, r, m, 0.3); });
    rejected([&] { (void)klpo_terminal_full_loss(q, q.to(at::kHalf), a, r, m, 0.3); });
    rejected([&] { (void)klpo_terminal_full_loss(q, q, a, r.to(at::kHalf), m, 0.3); });
    for (const double bad :
         {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(),
          -std::numeric_limits<double>::infinity()}) {
        auto changed = q.clone();
        changed[0][0][0] = bad;
        rejected([&] { (void)klpo_terminal_full_loss(changed, q, a, r, m, 0.3); });
        rejected([&] { (void)klpo_terminal_full_loss(q, changed, a, r, m, 0.3); });
        rejected([&] { (void)klpo_terminal_full_loss(q, q, a, at::full_like(r, bad), m, 0.3); });
    }
    const auto lost =
        at::tensor({0., -1000., -1000., -1000., -1000., -1000.}, at::kDouble).expand({2, 3, 6});
    rejected([&] { (void)klpo_terminal_full_loss(q, lost, a, r, m, 0.3); });
    const auto rounded =
        at::tensor({0., -200., -200., -200., -200., -200.}, at::kFloat).expand({2, 3, 6});
    rejected([&] { (void)klpo_terminal_full_loss(rounded, q, a, r, m, 0.3); });
    const auto large = q.narrow(0, 0, 1).narrow(1, 0, 1).expand({129, 257, 6});
    rejected([&] {
        (void)klpo_terminal_full_loss(large, large, at::zeros({129, 257}, at::kLong),
                                      at::ones({129}, at::kDouble), at::ones({129, 257}, at::kBool),
                                      0.3);
    });
}

void precision_strides_and_capacity() {
    const auto logits = at::arange(48, at::kDouble).reshape({2, 4, 6}).cos();
    const auto logs = logits.log_softmax(-1);
    const auto history = (-logits).log_softmax(-1);
    const auto actions = at::arange(8, at::kLong).reshape({2, 4}).remainder(6);
    const auto returns = at::tensor({0.4, -0.7}, at::kDouble);
    const auto mask = at::ones({2, 4}, at::kBool);
    const auto expected = klpo_terminal_full_loss(logs, history, actions, returns, mask, 0.3);
    const auto strided = logs.transpose(0, 1).contiguous().transpose(0, 1);
    require(!strided.is_contiguous(), "Faltan strides en el fixture");
    require(at::allclose(klpo_terminal_full_loss(strided, history, actions, returns, mask, 0.3),
                         expected, 1e-13, 1e-13),
            "La disposición cambia la pérdida");
    const auto indices = at::tensor({1, 0}, at::kLong);
    require(at::allclose(klpo_terminal_full_loss(logs.index_select(0, indices),
                                                 history.index_select(0, indices),
                                                 actions.index_select(0, indices),
                                                 returns.index_select(0, indices), mask, 0.3),
                         expected.index_select(0, indices), 1e-13, 1e-13),
            "La permutación cambia la pérdida");
    auto single = logs.to(at::kFloat).detach().requires_grad_(true);
    const auto promoted = single.detach().to(at::kDouble).requires_grad_(true);
    const auto a = klpo_terminal_full_loss(single, history.to(at::kFloat), actions,
                                           returns.to(at::kFloat), mask, 0.3);
    const auto b = klpo_terminal_full_loss(promoted, history.to(at::kFloat), actions,
                                           returns.to(at::kFloat), mask, 0.3);
    require(at::equal(a, b), "La promoción de FP32 cambia el valor");
    const auto ga = torch::autograd::grad({a.sum()}, {single}).front();
    const auto gb = torch::autograd::grad({b.sum()}, {promoted}).front();
    require(at::allclose(ga.to(at::kDouble), gb, 2e-6, 2e-7),
            "FP32 perdió el gradiente al promover");
    auto boundary = at::full({128, 256, 6}, -std::log(6.), at::kDouble).requires_grad_(true);
    const auto result =
        klpo_terminal_full_loss(boundary, boundary.detach(), at::zeros({128, 256}, at::kLong),
                                at::ones({128}, at::kDouble), at::ones({128, 256}, at::kBool), 0.3);
    require(at::isfinite(result).all().item<bool>() &&
                at::isfinite(torch::autograd::grad({result.sum()}, {boundary}).front())
                    .all()
                    .item<bool>(),
            "El máximo admitido no conserva valores y gradientes finitos");
    const auto one = logs.narrow(0, 0, 1).narrow(1, 0, 1);
    require(at::isfinite(klpo_terminal_full_loss(one, one, actions.narrow(0, 0, 1).narrow(1, 0, 1),
                                                 returns.narrow(0, 0, 1),
                                                 mask.narrow(0, 0, 1).narrow(1, 0, 1), 0.3))
                .all()
                .item<bool>(),
            "No se admite una trayectoria de una decisión");
}
} // namespace

int main() {
    try {
        at::set_num_threads(2);
        at::set_num_interop_threads(2);
        const auto rng = at::detail::getDefaultCPUGenerator().get_state().clone();
        values_gradients_and_padding();
        independent_population_and_microbatches();
        invalid_inputs();
        precision_strides_and_capacity();
        require(at::equal(rng, at::detail::getDefaultCPUGenerator().get_state()), "Cambió el RNG");
        std::cout
            << "KLPO terminal: valores, gradientes, oráculo, máscaras y presupuestos correctos\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
