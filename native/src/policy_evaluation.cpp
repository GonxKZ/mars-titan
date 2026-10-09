#include "mars_titan/policy_evaluation.hpp"
#include "mars_titan/simulation_files.hpp"

#include <stdexcept>
#include <string>

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

// Mismo registro que un episodio de la etapa en Python: estado, motivo y métricas finales.
Json episode(const PolicyTape& tape, double cost, const simulation::FinancialMetrics& metrics) {
    const bool ruined = metrics.invalid_reason == "ruined";
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
                {"steps", metrics.steps}};
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

Json run_frozen_evaluation(const PpoPolicy& policy, const FrozenEvaluationRequest& request,
                           const std::function<bool()>& stop) {
    require(!request.tapes.empty() && request.identity.is_object() &&
                request.identity.at("final_test_opened") == false,
            "La evaluación necesita su identidad sellada y al menos una cinta");
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
                {"cost_bps", frozen_evaluation_costs},
                {"status", "paused"},
                {"confirmed_episodes", 0},
                {"metrics", Json::array()},
                {"final_test_opened", false}};
    for (const auto cost : frozen_evaluation_costs) {
        std::vector<simulation::BatchInput> inputs;
        inputs.reserve(request.tapes.size());
        for (const auto& tape : request.tapes) {
            inputs.push_back(tape.input);
            inputs.back().parameters.cost_bps = cost;
        }
        const auto evaluation =
            evaluate_policy(policy, std::move(inputs), request.workers, stop, request.learning);
        if (evaluation.paused) {
            // Los episodios de los costes anteriores se repiten al reanudar.
            report["metrics"] = Json::array();
            report["confirmed_episodes"] = 0;
            simulation::atomic_json_file(report_path, seal(report));
            return report;
        }
        require(evaluation.metrics.size() == request.tapes.size(),
                "La evaluación no conserva un episodio por cinta");
        for (std::size_t index = 0; index < request.tapes.size(); ++index) {
            report["metrics"].push_back(
                episode(request.tapes[index], cost, evaluation.metrics[index]));
        }
    }
    report["status"] = "completed";
    report["confirmed_episodes"] = report["metrics"].size();
    simulation::atomic_json_file(report_path, seal(report));
    return report;
}

} // namespace mars_titan::learning
