#include "mars_titan/policy_context.hpp"
#include "mars_titan/policy_evaluation.hpp"
#include "mars_titan/ppo_training.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <nlohmann/json.hpp>
#include <torch/serialize/archive.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

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
nlohmann::json sealed_payload(const std::filesystem::path& path) {
    std::ifstream stream(path);
    const auto envelope = nlohmann::json::parse(stream);
    require(envelope.size() == 2 && envelope.contains("sha256"), "El registro no está sellado");
    return envelope.at("payload");
}

FrozenEvaluationRequest frozen_request(const std::filesystem::path& output, bool decisions,
                                       const PolicyTape& evaluated, bool resume = false) {
    FrozenEvaluationRequest request;
    request.output = output;
    request.resume = resume;
    request.costs = {0.0, cost_bps};
    request.decisions = decisions;
    request.identity = nlohmann::json{{"final_test_opened", false},
                                      {"cost_bps", request.costs},
                                      {"decisions", decisions}};
    request.tapes = {evaluated};
    return request;
}

// El registro de decisiones de la evaluación congelada (--decisions de --audit-run) guarda, por
// coste y cinta, la acción, el modo y las salidas de cada decisión. La política conserva sus
// pesos iniciales y no pasa por ningún optimizador. Cada acción debe ser el primer máximo de sus
// salidas, como en std::max_element, y pedir el registro no puede cambiar las métricas.
void decisions_record_the_first_maximum_without_changing_metrics() {
    const auto root = std::filesystem::temp_directory_path() /
                      ("mars-titan-decisions-" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    constexpr std::size_t sessions = 7;
    auto market = tape(sessions);
    // Puntuaciones distintas en cada sesión para que las salidas cambien entre decisiones.
    for (std::size_t at = 0; at < sessions; ++at) {
        market->scores[at] = prediction * (static_cast<double>(at % 3) - 1);
    }
    const PolicyTape evaluated{{market, parameters(), {}},
                               nlohmann::json{{"manifest_sha256", market->source_sha256}}};
    const PpoPolicy policy(observation_width, {}, ppo_default_training_seed);
    const auto plain = run_frozen_evaluation(policy, frozen_request(root / "plain", false, evaluated));
    const auto recorded =
        run_frozen_evaluation(policy, frozen_request(root / "recorded", true, evaluated));
    require(plain.at("status") == "completed" && recorded.at("status") == "completed" &&
                plain.at("metrics") == recorded.at("metrics"),
            "El registro de decisiones cambió las métricas de la evaluación");
    require(!std::filesystem::exists(root / "plain" / "decisions.json"),
            "Sin --decisions no debe escribirse el registro");
    const auto record = sealed_payload(root / "recorded" / "decisions.json");
    require(record.at("kind") == "native_policy_decisions" && record.at("outputs") == "logits" &&
                record.at("device") == "cpu" &&
                record.at("parameter_fingerprint") == policy.parameter_fingerprint() &&
                record.at("costs").size() == 2,
            "El registro no identifica la política, el dispositivo o los costes");
    for (std::size_t index = 0; index < 2; ++index) {
        const auto& cost = record.at("costs").at(index);
        require(cost.at("cost_bps") == recorded.at("metrics").at(index).at("cost_bps") &&
                    cost.at("tapes").size() == 1 &&
                    cost.at("tapes").at(0).at("manifest_sha256") == market->source_sha256,
                "El registro no sigue los costes y las cintas de la evaluación");
        const auto& decisions = cost.at("tapes").at(0).at("decisions");
        require(decisions.size() == sessions - 1, "Falta alguna decisión del episodio");
        for (std::size_t step = 0; step < decisions.size(); ++step) {
            const auto& decision = decisions.at(step);
            const auto logits = decision.at("logits").get<std::vector<double>>();
            const auto first = std::distance(logits.begin(), std::ranges::max_element(logits));
            require(decision.at("cursor") == step && decision.at("mode") == "greedy" &&
                        logits.size() == full_exposure + 1 &&
                        decision.at("action").get<std::ptrdiff_t>() == first,
                    "Una decisión no es el primer máximo de sus salidas");
        }
    }
    // La identidad declara el registro: una evaluación con él no se reanuda sin él.
    bool refused = false;
    try {
        static_cast<void>(
            run_frozen_evaluation(policy, frozen_request(root / "recorded", false, evaluated, true)));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "Una reanudación sin el registro aceptó una evaluación que lo declaraba");
    // Una petición cuya identidad no declara el registro que pide se rechaza antes de escribir.
    auto mismatched = frozen_request(root / "mismatched", true, evaluated);
    mismatched.identity["decisions"] = false;
    refused = false;
    try {
        static_cast<void>(run_frozen_evaluation(policy, mismatched));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused && !std::filesystem::exists(root / "mismatched"),
            "Una identidad sin el registro pedido no debe admitirse");
    require(policy.optimizer_steps() == 0, "La evaluación congelada no puede actualizar la política");
    std::filesystem::remove_all(root);
}
// Una sesión sin negociación toma la acción fija del calentamiento, que no sale de las salidas
// de la red. El registro la marca con su modo para que la paridad no la compare como argmax.
void decisions_mark_the_fixed_action_of_a_session_without_trading() {
    const auto root = std::filesystem::temp_directory_path() /
                      ("mars-titan-decisions-mask-" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    constexpr std::size_t sessions = 5;
    constexpr std::size_t closed = 2;
    const auto market = tape(sessions);
    ContextTape context;
    context.source_sha256.assign(digest_size, 'c');
    context.fields = {{"trade", "1"}};
    for (std::size_t at = 0; at < sessions; ++at) {
        context.values.push_back({at < closed ? 0.0F : 1.0F, true, market->prediction_times[at]});
    }
    PpoLearningOptions learning;
    learning.enabled = true;
    learning.variant = "ppo";
    learning.environments = 1;
    learning.trading_field = 0;
    const BatchInput input{market, parameters(), context};
    const std::vector<BatchInput> inputs{input};
    FinancialBatch batch(inputs, 1);
    const PolicyContext shape(inputs, learning, ppo_default_training_seed, batch.observations());
    const PpoPolicy policy(shape.observation_width(), {}, ppo_default_training_seed);
    auto request = frozen_request(root, true,
                                  PolicyTape{input, nlohmann::json{{"manifest_sha256",
                                                                    market->source_sha256}}});
    request.learning = learning;
    require(run_frozen_evaluation(policy, request).at("status") == "completed",
            "La evaluación con sesiones sin negociación no terminó");
    const auto record = sealed_payload(root / "decisions.json");
    const auto& decisions = record.at("costs").at(0).at("tapes").at(0).at("decisions");
    require(decisions.size() == sessions - 1, "Falta alguna decisión del episodio");
    for (std::size_t step = 0; step < decisions.size(); ++step) {
        const auto& decision = decisions.at(step);
        const auto logits = decision.at("logits").get<std::vector<double>>();
        const auto first = std::distance(logits.begin(), std::ranges::max_element(logits));
        const bool fixed = step < closed;
        require(decision.at("mode") == (fixed ? "warmup" : "greedy") &&
                    decision.at("action").get<std::ptrdiff_t>() == (fixed ? 1 : first),
                "El registro no distingue la acción fija de una sesión sin negociación");
    }
    std::filesystem::remove_all(root);
}
} // namespace

int main() {
    try {
        incomplete_evaluation_has_no_score();
        liquidated_growth_pays_the_final_exit();
        cash_has_no_exit_cost();
        decisions_record_the_first_maximum_without_changing_metrics();
        decisions_mark_the_fixed_action_of_a_session_without_trading();
        std::cout << "Evaluación incompleta y liquidación final comprobadas sin optimizador\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
