#include "mars_titan/policy_evaluation.hpp"
#include "mars_titan/simulation_files.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mars_titan::learning {
namespace {
using Json = nlohmann::json;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

Json optional_number(const std::optional<double>& value) {
    return value ? Json(*value) : Json(nullptr);
}

// Tolerancia relativa del patrimonio final reconstruido frente a la contabilidad. Cada paso
// suma unos pocos ulp al exponer y deshacer el logaritmo de la recompensa.
constexpr double equity_tolerance = 1e-9;

// Patrimonio en cada cierre de una cinta, desde el capital inicial.
struct Equity {
    std::vector<double> nav;
};

Json equity_json(const PolicyTape& tape, const Equity& equity) {
    Json times = Json::array();
    Json values = Json::array();
    for (std::size_t session = 0; session < equity.nav.size(); ++session) {
        times.push_back(tape.input.tape->close_times.at(session));
        values.push_back(std::isfinite(equity.nav[session]) ? Json(equity.nav[session])
                                                            : Json(nullptr));
    }
    return Json{{"basis", "close_valuation_from_log_rewards"},
                {"close_times", times},
                {"nav", values}};
}

// Mismo registro que un episodio de la etapa en Python: estado, motivo, métricas finales y
// patrimonio por sesión.
Json episode(const PolicyTape& tape, double cost, const simulation::FinancialMetrics& metrics,
             const Equity& equity) {
    const bool ruined = metrics.invalid_reason == "ruined";
    require(equity.nav.size() == metrics.steps + 1,
            "El patrimonio por sesión no cubre los pasos del episodio");
    if (metrics.completed) {
        if (!metrics.net_return) {
            throw std::invalid_argument("Un episodio completo necesita su retorno neto");
        }
        const auto expected = tape.input.parameters.capital * (1 + *metrics.net_return);
        require(std::abs(equity.nav.back() - expected) <=
                    equity_tolerance * std::max(1.0, std::abs(expected)),
                "El patrimonio reconstruido no concilia con la contabilidad");
    }
    return Json{{"manifest_sha256", tape.input.tape->source_sha256},
                {"cost_bps", cost},
                {"status", ruined ? "ruined" : metrics.completed ? "completed" : "failed"},
                {"reason", metrics.invalid_reason.empty() ? Json(nullptr)
                                                          : Json(metrics.invalid_reason)},
                {"net_return", optional_number(metrics.net_return)},
                {"liquidated_net_return", optional_number(metrics.liquidated_net_return)},
                {"max_drawdown", optional_number(metrics.max_drawdown)},
                {"costs", metrics.costs},
                {"turnover", metrics.turnover},
                {"steps", metrics.steps},
                {"equity", equity_json(tape, equity)}};
}

Json seal(const Json& value) {
    return Json{{"payload", value}, {"sha256", simulation::content_sha256(value.dump())}};
}

Json unseal(const std::filesystem::path& path) {
    const auto envelope = simulation::parse_bounded_json(
        simulation::read_bounded_file(path, simulation::maximum_manifest_bytes));
    require(envelope.is_object() && envelope.size() == 2 &&
                simulation::content_sha256(envelope.at("payload").dump()) ==
                    envelope.at("sha256").get<std::string>(),
            "El registro de la evaluación perdió su integridad");
    return envelope.at("payload");
}
} // namespace

std::vector<double> frozen_costs(const std::vector<double>& declared) {
    if (declared.empty()) {
        return {frozen_evaluation_costs.begin(), frozen_evaluation_costs.end()};
    }
    require(declared.size() <= maximum_evaluation_costs &&
                std::ranges::all_of(declared,
                                    [](double cost) {
                                        return std::isfinite(cost) && cost >= 0 &&
                                               cost <= maximum_evaluation_cost_bps;
                                    }) &&
                std::ranges::adjacent_find(declared, std::greater_equal<>()) == declared.end(),
            "Los costes de evaluación deben ser finitos, crecientes y estar entre 0 y 1000 pb");
    return declared;
}

Json run_frozen_evaluation(const PpoPolicy& policy, const FrozenEvaluationRequest& request,
                           const std::function<bool()>& stop) {
    require(!request.tapes.empty() && request.identity.is_object() &&
                request.identity.at("final_test_opened") == false &&
                frozen_costs(request.costs) == request.costs &&
                request.identity.at("cost_bps") == Json(request.costs) &&
                request.identity.at("decisions") == request.decisions,
            "La evaluación necesita su identidad sellada, sus costes y al menos una cinta");
    // Cada cinta de una política es única, así que su huella identifica su carril.
    std::unordered_map<std::string, std::size_t> lanes;
    for (std::size_t index = 0; index < request.tapes.size(); ++index) {
        require(lanes.emplace(request.tapes[index].input.tape->source_sha256, index).second,
                "La evaluación repite una cinta");
    }
    const auto identity_sha256 = simulation::content_sha256(request.identity.dump());
    const simulation::OutputLock lock(request.output, request.resume);
    const auto identity_path = request.output / "identity.json";
    const auto report_path = request.output / "evaluation.json";
    if (request.resume) {
        require(unseal(identity_path) == request.identity,
                "La salida pertenece a otra evaluación, política o cinta");
        if (std::filesystem::exists(report_path)) {
            auto previous = unseal(report_path);
            if (previous.at("status") == "completed") {
                return previous;
            }
        }
    } else {
        simulation::atomic_json_file(identity_path, seal(request.identity));
    }
    Json report{{"schema_version", 1},
                {"kind", "native_policy_evaluation"},
                {"activity", "evaluation"},
                {"partition", "evaluation"},
                {"domain", "real"},
                {"policy", "greedy_argmax"},
                {"identity_sha256", identity_sha256},
                {"identity", request.identity},
                {"cost_bps", request.costs},
                {"status", "paused"},
                {"confirmed_episodes", 0},
                {"metrics", Json::array()},
                {"final_test_opened", false}};
    // Acción y logits de cada decisión por coste y cinta, solo si se piden.
    Json decisions = Json::array();
    for (const auto cost : request.costs) {
        std::vector<simulation::BatchInput> inputs;
        std::vector<Equity> equities(request.tapes.size());
        std::vector<Json> chosen(request.tapes.size(), Json::array());
        inputs.reserve(request.tapes.size());
        for (std::size_t index = 0; index < request.tapes.size(); ++index) {
            inputs.push_back(request.tapes[index].input);
            inputs.back().parameters.cost_bps = cost;
            equities[index].nav.push_back(inputs.back().parameters.capital);
        }
        // El receptor ve cada transición confirmada en orden. El patrimonio de la sesión
        // siguiente sale de la recompensa logarítmica, la ruina y la valoración ausente.
        const PpoDecisionObserver observer = [&](std::span<const DecisionRecord> records) {
            for (const auto& record : records) {
                const auto lane = lanes.at(record.world_sha256);
                if (request.decisions) {
                    // Un float se convierte a double sin pérdida y el JSON lo conserva exacto.
                    Json logits = Json::array();
                    for (const auto value : record.logits) {
                        logits.push_back(static_cast<double>(value));
                    }
                    // El modo distingue la elección argmax de la acción fija del calentamiento
                    // de la memoria, que no sale de los logits.
                    chosen.at(lane).push_back(Json{{"cursor", record.cursor},
                                                   {"action", record.action},
                                                   {"mode", record.mode},
                                                   {"logits", std::move(logits)}});
                }
                auto& nav = equities.at(lane).nav;
                require(nav.size() == record.cursor + 1 && std::isfinite(nav.back()),
                        "El patrimonio por sesión recibe una transición fuera de orden");
                if (!record.reward_valid) {
                    nav.push_back(std::numeric_limits<double>::quiet_NaN());
                } else if (record.terminated) {
                    nav.push_back(0);
                } else {
                    nav.push_back(nav.back() * std::exp(record.reward));
                }
            }
        };
        const auto evaluation = evaluate_policy(policy, std::move(inputs), request.workers, stop,
                                                request.learning, observer);
        if (evaluation.paused) {
            // Los episodios de los costes anteriores se repiten al reanudar.
            report["metrics"] = Json::array();
            report["confirmed_episodes"] = 0;
            simulation::atomic_json_file(report_path, seal(report));
            return report;
        }
        require(evaluation.metrics.size() == request.tapes.size(),
                "La evaluación no conserva un episodio por cinta");
        Json tapes = Json::array();
        for (std::size_t index = 0; index < request.tapes.size(); ++index) {
            report["metrics"].push_back(episode(request.tapes[index], cost,
                                                evaluation.metrics[index], equities[index]));
            tapes.push_back(Json{{"manifest_sha256", request.tapes[index].input.tape->source_sha256},
                                 {"decisions", std::move(chosen[index])}});
        }
        decisions.push_back(Json{{"cost_bps", cost}, {"tapes", std::move(tapes)}});
    }
    if (request.decisions) {
        // Se escribe antes de confirmar la evaluación: una evaluación completa siempre tiene
        // su registro y una interrumpida lo repite entero al reanudar.
        simulation::atomic_json_file(
            request.output / "decisions.json",
            seal(Json{{"schema_version", 1},
                      {"kind", "native_policy_decisions"},
                      {"identity_sha256", identity_sha256},
                      {"device", policy.device()},
                      {"parameter_fingerprint", policy.parameter_fingerprint()},
                      {"outputs", policy.architecture().double_dqn ? "q_values" : "logits"},
                      {"costs", std::move(decisions)}}));
    }
    report["status"] = "completed";
    report["confirmed_episodes"] = report["metrics"].size();
    simulation::atomic_json_file(report_path, seal(report));
    return report;
}

} // namespace mars_titan::learning
