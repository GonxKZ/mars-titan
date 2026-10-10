#include "mars_titan/policy_context.hpp"
#include "mars_titan/ppo_policy.hpp"
#include "mars_titan/quantile_dqn.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/serialize/archive.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

// Comprueba la cabeza QR-DQN dentro de PpoPolicy sin ejecutar ningún paso del optimizador. Las
// pérdidas se calculan con double_dqn_loss sobre redes cuyos pesos fija la prueba, y los valores
// esperados salen de la ecuación escrita aquí con bucles escalares, no del núcleo que se prueba.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using namespace mars_titan::learning;

constexpr int64_t observation_width = 4;
constexpr uint64_t seed = 719;

void require(bool value, std::string_view message) {
    if (!value) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Function> void rejected(Function&& function, std::string_view message) {
    try {
        std::forward<Function>(function)();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
PpoArchitecture quantile_architecture(int64_t quantiles, double alpha) {
    PpoArchitecture architecture{PpoNetworkKind::mlp, default_ppo_hidden_width, false, true};
    architecture.quantiles = quantiles;
    architecture.risk_alpha = alpha;
    return architecture;
}

PpoPolicy make_policy(const PpoArchitecture& architecture, double gamma = default_ppo_gamma) {
    PpoHyperparameters parameters;
    parameters.gamma = gamma;
    return PpoPolicy(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes,
                     architecture);
}

void architecture_rules() {
    auto ppo = quantile_architecture(8, 1);
    ppo.double_dqn = false;
    rejected([&] { ppo.validate(); }, "Se aceptó una cabeza cuantílica sin Double DQN");
    rejected([] { quantile_architecture(1, 1).validate(); }, "Se aceptó un único cuantil");
    rejected([] { quantile_architecture(maximum_quantiles + 1, 1).validate(); },
             "Se aceptaron demasiados cuantiles");
    rejected([] { quantile_architecture(8, 0.3).validate(); },
             "Se aceptó un alpha que no corresponde a un número entero de niveles");
    rejected([] { quantile_architecture(0, 0.5).validate(); },
             "Se aceptó un alpha distinto de uno sin cabeza cuantílica");
    quantile_architecture(qr_dqn_quantiles, qr_dqn_cvar_alpha).validate();
    quantile_architecture(0, 1).validate();
    require(value_variant("double_dqn") && value_variant("qr_dqn") &&
                value_variant("qr_dqn_cvar") && !value_variant("ppo") && !value_variant("iqn"),
            "Las variantes de valor no coinciden con las identidades declaradas");
}

// Media de las puntuaciones de riesgo y elección voraz con la misma medida con la que aprende.
void head_scores_follow_the_quantiles() {
    constexpr int64_t quantiles = 8;
    constexpr double alpha = 0.25;
    auto policy = make_policy(quantile_architecture(quantiles, alpha));
    const auto observations =
        at::linspace(-1, 1, 3 * observation_width, at::kFloat).reshape({3, observation_width});
    const auto values = policy.action_quantiles(observations);
    require(values.sizes() == at::IntArrayRef({3, ppo_action_count, quantiles}),
            "La cabeza cuantílica no tiene la forma [B,6,N]");
    const auto scores = quantile_action_scores(values, alpha);
    const auto forward = policy.forward(observations);
    require(at::equal(forward.logits, scores) &&
                at::equal(forward.values, std::get<0>(scores.max(1))),
            "forward no expone la puntuación de riesgo de la cabeza cuantílica");
    require(at::equal(policy.act_double_dqn(observations, 0).packed.select(1, 0).to(at::kLong),
                      scores.argmax(1)),
            "La acción voraz no sigue la puntuación de riesgo");
    const auto hidden = static_cast<std::size_t>(default_ppo_hidden_width);
    const auto width = static_cast<std::size_t>(observation_width);
    const auto expected = (width + 1) * hidden + (hidden + 1) * hidden +
                          (hidden + 1) * static_cast<std::size_t>(ppo_action_count * quantiles);
    require(policy.parameter_count() == expected,
            "La cabeza cuantílica no tiene N salidas por acción y nada más");
    rejected(
        [&] {
            static_cast<void>(
                make_policy(quantile_architecture(0, 1)).action_quantiles(observations));
        },
        "Una política sin cuantiles devolvió cuantiles");
}

void checkpoint_keeps_the_head() {
    auto policy = make_policy(quantile_architecture(8, 0.25));
    std::stringstream checkpoint;
    policy.save(checkpoint);
    auto restored = PpoPolicy::load(checkpoint);
    const auto observations = at::ones({2, observation_width}, at::kFloat);
    require(restored.architecture() == policy.architecture() &&
                restored.parameter_fingerprint() == policy.parameter_fingerprint() &&
                at::equal(restored.action_quantiles(observations),
                          policy.action_quantiles(observations)),
            "El checkpoint QR-DQN pierde la cabeza o su medida de riesgo");
    // Con la misma semilla las redes coinciden, así que solo la medida de riesgo cambia la huella.
    require(make_policy(quantile_architecture(8, 1)).parameter_fingerprint() !=
                policy.parameter_fingerprint(),
            "La huella no distingue la medida de riesgo");
    // Double DQN no escribe las claves nuevas, de modo que sus archivos conservan los bytes.
    auto scalar = make_policy(quantile_architecture(0, 1));
    std::stringstream scalar_checkpoint;
    scalar.save(scalar_checkpoint);
    torch::serialize::InputArchive archive;
    archive.load_from(scalar_checkpoint, at::Device(at::kCPU));
    const auto keys = archive.keys();
    require(std::ranges::find(keys, "quantiles") == keys.end() &&
                std::ranges::find(keys, "risk_alpha") == keys.end(),
            "Double DQN escribe claves de QR-DQN en su checkpoint");
}

std::string rewrite_checkpoint(torch::serialize::InputArchive& input) {
    torch::serialize::OutputArchive output;
    for (const auto& key : input.keys()) {
        c10::IValue value;
        input.read(key, value);
        output.write(key, value);
    }
    std::ostringstream destination;
    output.save_to(destination);
    return std::move(destination).str();
}

// Política con todos los pesos a cero salvo el sesgo final, de modo que con observaciones nulas
// la salida de cada red es exactamente el sesgo que fija la prueba.
PpoPolicy controlled(const PpoArchitecture& architecture, const std::vector<float>& online,
                     const std::vector<float>& target, double gamma) {
    auto policy = make_policy(architecture, gamma);
    std::stringstream checkpoint;
    policy.save(checkpoint);
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint, at::Device(at::kCPU));
    {
        const at::NoGradGuard no_grad;
        for (const auto name : {"network", "target_network"}) {
            c10::IValue network;
            archive.read(name, network);
            const auto object = network.toObject();
            for (std::size_t index = 0; index < object->type()->numAttributes(); ++index) {
                object->getSlot(index).toTensor().zero_();
            }
            const auto& values = std::string_view(name) == "network" ? online : target;
            object->getAttr("output_bias")
                .toTensor()
                .copy_(at::tensor(at::ArrayRef<float>(values), at::kFloat));
        }
    }
    std::istringstream fixture(rewrite_checkpoint(archive));
    return PpoPolicy::load(fixture);
}

DqnBatch analytic_batch(int64_t rows) {
    return {at::zeros({rows, observation_width}, at::kFloat),
            at::zeros({rows, observation_width}, at::kFloat),
            at::tensor({0, 2, 1}, at::kLong).narrow(0, 0, rows),
            at::ones({rows}, at::kFloat),
            at::zeros({rows}, at::kBool),
            at::ones({rows}, at::kBool)};
}

// Ecuación 10 de QR-DQN con kappa = 1, escrita con escalares: media de lote de la suma sobre i
// de la media sobre j de |tau_i - 1{u_ij < 0}| H(u_ij), con u_ij = T_j - theta_i.
double scalar_quantile_loss(const std::vector<std::vector<double>>& predicted,
                            const std::vector<double>& target) {
    double total = 0;
    for (const auto& row : predicted) {
        const auto count = row.size();
        for (std::size_t i = 0; i < count; ++i) {
            const double tau =
                (2. * static_cast<double>(i) + 1.) / (2. * static_cast<double>(count));
            double inner = 0;
            for (const double value : target) {
                const double error = value - row[i];
                const double huber =
                    std::abs(error) <= 1 ? 0.5 * error * error : std::abs(error) - 0.5;
                inner += std::abs(tau - (error < 0 ? 1. : 0.)) * huber;
            }
            total += inner / static_cast<double>(target.size());
        }
    }
    return total / static_cast<double>(predicted.size());
}

void losses_match_their_equations_without_optimizer_steps() {
    // Double DQN escalar: mismo caso que la prueba analítica de update_double_dqn, que da 0,5625.
    auto scalar = controlled(quantile_architecture(0, 1), {1.F, 4.F, 2.F, 3.F, 0.F, -1.F},
                             {10.F, 3.F, 8.F, 6.F, 4.F, 5.F}, 0.5);
    const auto scalar_loss = scalar.double_dqn_loss(analytic_batch(2));
    require(scalar_loss.valid_transitions == 2 && scalar_loss.loss.item<double>() == 0.5625,
            "La pérdida SmoothL1 extraída no conserva el valor de la actualización");

    // Dos cuantiles por acción en orden acción mayor. La media prefiere la acción 3 (4,5) y el
    // nivel inferior prefiere la acción 1 (4 frente a 1), así que alpha cambia el objetivo.
    const std::vector<float> online{1, 3, 4, 4, 0, 2, 1, 8, 0, 0, -1, -1};
    const std::vector<float> target{10, 12, 3, 5, 8, 9, 6, 7, 4, 4, 5, 5};
    const std::vector<std::vector<double>> predicted{{1, 3}, {0, 2}};
    const auto zero = at::zeros({1, observation_width}, at::kFloat);
    for (const auto& [alpha, action, following] :
         {std::tuple{1., int64_t{3}, std::vector<double>{6, 7}},
          std::tuple{0.5, int64_t{1}, std::vector<double>{3, 5}}}) {
        auto policy = controlled(quantile_architecture(2, alpha), online, target, 0.5);
        require(policy.act_double_dqn(zero, 0).packed[0][0].item<double>() ==
                    static_cast<double>(action),
                "La acción voraz no corresponde a la medida de riesgo declarada");
        std::vector<double> targets;
        for (const double value : following) {
            targets.push_back(1 + 0.5 * value);
        }
        const auto expected = scalar_quantile_loss(predicted, targets);
        const auto computed = policy.double_dqn_loss(analytic_batch(2));
        require(computed.valid_transitions == 2 &&
                    std::abs(computed.loss.item<double>() - expected) <= 1e-12,
                "La pérdida QR-DQN no coincide con la ecuación 10");
        // Una fila inválida con recompensa enorme no puede cambiar la pérdida.
        auto masked = analytic_batch(3);
        masked.rewards[2] = 1e20F;
        masked.reward_valid[2] = false;
        require(policy.double_dqn_loss(masked).loss.item<double>() == computed.loss.item<double>(),
                "Una transición inválida participa en la pérdida QR-DQN");
        // El grafo existe y su gradiente es finito, pero no se ejecuta ningún paso de Adam.
        const auto before = policy.action_quantiles(zero);
        computed.loss.backward();
        require(policy.optimizer_steps() == 0 && at::equal(before, policy.action_quantiles(zero)),
                "Calcular la pérdida QR-DQN modificó la política");
    }
    // Los valores esperados escritos a mano: 1,75 con la media y 0,7421875 con alpha = 0,5.
    require(scalar_quantile_loss({{1, 3}, {0, 2}}, {4, 4.5}) == 1.75 &&
                scalar_quantile_loss({{1, 3}, {0, 2}}, {2.5, 3.5}) == 0.7421875,
            "La ecuación escalar de la prueba no reproduce el cálculo a mano");
}
} // namespace

int main() {
    try {
        at::manual_seed(static_cast<int64_t>(seed));
        architecture_rules();
        head_scores_follow_the_quantiles();
        checkpoint_keeps_the_head();
        losses_match_their_equations_without_optimizer_steps();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
