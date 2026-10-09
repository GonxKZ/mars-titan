#include "mars_titan/ppo_training.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

// Evaluación greedy de una política fija. Estas pruebas no crean entrenadores ni optimizadores.
namespace {
using namespace mars_titan::learning;
using namespace mars_titan::simulation;
constexpr std::size_t observation_width = 8;
constexpr std::size_t full_exposure = 5;
constexpr std::size_t digest_size = 64;
constexpr std::size_t price_columns = 5;
constexpr std::size_t close_column = 3;
constexpr double price = 10;
constexpr double volume = 1e6;
constexpr double prediction = 0.01;
constexpr double capital = 10'000;
constexpr double cost_bps = 10;
constexpr double tolerance = 1e-12;

void require(bool condition, const char* reason) {
    if (!condition) {
        throw std::runtime_error(reason);
    }
}

std::shared_ptr<MarketTape> tape(std::size_t sessions) {
    auto data = std::make_shared<MarketTape>();
    data->assets = {"FIC0"};
    data->currency = "USD";
    data->domain = "synthetic";
    data->partition = "validation";
    data->source_sha256.assign(digest_size, 'e');
    data->parent_id = "analytic-frozen";
    for (std::size_t at = 0; at < sessions; ++at) {
        data->open_times.push_back(static_cast<int64_t>(2 * at));
        data->close_times.push_back(static_cast<int64_t>(2 * at + 1));
        data->prediction_times.push_back(static_cast<int64_t>(2 * at + 1));
        data->prices.insert(data->prices.end(), {price, price, price, price, volume});
        data->scores.push_back(prediction);
    }
    return data;
}

// Política que elige siempre la exposición completa, con pesos editados a mano.
PpoPolicy buyer() {
    PpoPolicy policy(observation_width, {}, ppo_default_training_seed);
    std::ostringstream serialized;
    policy.save(serialized);
    std::istringstream source(serialized.str());
    torch::serialize::InputArchive input;
    input.load_from(source, at::Device(at::kCPU));
    c10::IValue network;
    input.read("network", network);
    {
        const at::NoGradGuard guard;
        const auto object = network.toObject();
        object->getAttr("output_weight").toTensor().zero_();
        auto bias = object->getAttr("output_bias").toTensor();
        bias.zero_();
        bias[full_exposure].fill_(1);
    }
    torch::serialize::OutputArchive output;
    for (const auto& key : input.keys()) {
        c10::IValue value;
        input.read(key, value);
        output.write(key, value);
    }
    std::ostringstream edited;
    output.save_to(edited);
    std::istringstream bytes(edited.str());
    return PpoPolicy::load(bytes);
}

Parameters parameters() {
    Parameters result;
    result.capital = capital;
    result.cost_bps = cost_bps;
    result.participation = 1;
    return result;
}

void incomplete_evaluation_has_no_score() {
    constexpr std::size_t sessions = 4;
    auto market = tape(sessions);
    market->prices[2 * price_columns + close_column] = std::numeric_limits<double>::quiet_NaN();
    const auto result = evaluate_policy(buyer(), {{market, parameters(), {}}}, 1);
    require(result.episodes == 1 && result.incomplete == 1,
            "El cierre ausente de la posición debe dejar la evaluación incompleta");
    require(std::isnan(result.mean_log_growth) && std::isnan(result.mean_liquidated_log_growth),
            "Una evaluación incompleta no puede devolver una puntuación finita");
}

void liquidated_growth_pays_the_final_exit() {
    constexpr std::size_t sessions = 4;
    const auto market = tape(sessions);
    const auto result = evaluate_policy(buyer(), {{market, parameters(), {}}, {market, parameters(), {}}}, 1);
    require(result.incomplete == 0 && result.ruined == 0 && result.episodes == 2,
            "La evaluación completa debe valorar ambos episodios");
    FinancialSession session(market, parameters());
    while (!session.done()) {
        static_cast<void>(session.step(static_cast<uint8_t>(full_exposure)));
    }
    const auto state = session.snapshot();
    const double held = state.positions.front().quantity * price;
    require(held > 0, "La política compradora debe terminar invertida");
    const double expected_nav = state.account.nav - held * cost_bps / 10'000;
    require(std::abs(liquidated_nav(state, *market) - expected_nav) <= tolerance * capital,
            "La liquidación final debe restar el coste de vender al último cierre");
    const auto metrics = session.metrics();
    require(metrics.liquidated_net_return &&
                std::abs(*metrics.liquidated_net_return - (expected_nav / capital - 1)) <= tolerance,
            "Las métricas deben publicar el retorno liquidado");
    require(std::abs(result.mean_log_growth - std::log(state.account.nav / capital)) <= tolerance &&
                std::abs(result.mean_liquidated_log_growth - std::log(expected_nav / capital)) <=
                    tolerance,
            "La evaluación debe distinguir el patrimonio marcado del liquidado");
    require(result.mean_log_growth - result.mean_liquidated_log_growth > 9e-4,
            "Con 10 pb y exposición completa la venta final cuesta cerca de 0,001");
}

void cash_has_no_exit_cost() {
    auto market = tape(3);
    FinancialSession session(market, parameters());
    while (!session.done()) {
        static_cast<void>(session.step(1));
    }
    require(liquidated_nav(session.snapshot(), *market) == capital,
            "El efectivo no paga coste de salida");
}
} // namespace

int main() {
    try {
        incomplete_evaluation_has_no_score();
        liquidated_growth_pays_the_final_exit();
        cash_has_no_exit_cost();
        std::cout << "Evaluación incompleta y liquidación final comprobadas sin optimizador\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
