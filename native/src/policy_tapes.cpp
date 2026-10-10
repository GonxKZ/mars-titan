#include "mars_titan/policy_tapes.hpp"
#include "mars_titan/ppo_inputs.hpp"
#include "mars_titan/simulation_files.hpp"

#include <algorithm>
#include <set>
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

Json periods_json(const std::vector<simulation::RulePeriod>& periods) {
    Json result = Json::array();
    for (const auto& period : periods) {
        result.push_back(Json::array(
            {period.start, period.end, period.band, period.buy, period.sell}));
    }
    return result;
}

// Huella de lo que la política observa y negocia: activos, moneda y reglas de cada activo.
std::string observation_sha256(const simulation::MarketTape& tape) {
    Json rules = Json::array();
    for (const auto& instrument : tape.instruments) {
        rules.push_back(Json{{"rules", instrument.rules},
                             {"lot", instrument.lot},
                             {"minimum_order", instrument.minimum_order},
                             {"odd_lot_exit", instrument.odd_lot_exit},
                             {"price_limits", periods_json(instrument.price_limits)},
                             {"taxes", periods_json(instrument.taxes)}});
    }
    return simulation::content_sha256(
        Json{{"assets", tape.assets}, {"currency", tape.currency}, {"instruments", rules}}.dump());
}

int64_t integer(const Json& value) { return simulation::read_json_int64(value); }
} // namespace

std::string_view policy_tape_role_name(PolicyTapeRole role) noexcept {
    switch (role) {
    case PolicyTapeRole::train:
        return "train";
    case PolicyTapeRole::validation:
        return "validation";
    case PolicyTapeRole::evaluation:
        return "evaluation";
    }
    return "train";
}

void require_policy_tape_manifest(const Json& manifest, PolicyTapeRole role) {
    const auto partition = role == PolicyTapeRole::train ? "train" : "validation";
    const auto version = integer(manifest.at("schema_version"));
    const auto& identity = manifest.at("identity");
    require((version == 1 || version == 2) && manifest.at("final_test_opened") == false &&
                identity.at("domain") == "real" && identity.at("partition") == partition,
            "La cinta reconstruida no corresponde a su papel o pretende abrir el test");
}

PolicyTape load_policy_tape(const std::filesystem::path& directory, PolicyTapeRole role,
                            const simulation::Parameters& parameters) {
    const auto manifest_bytes = simulation::read_bounded_file(directory / "manifest.json",
                                                              simulation::maximum_manifest_bytes);
    const auto manifest = simulation::parse_bounded_json(manifest_bytes);
    require_policy_tape_manifest(manifest, role);
    PolicyTape result{load_ppo_input(directory), Json::object()};
    result.input.parameters = parameters;
    const auto& tape = *result.input.tape;
    require(tape.domain == "real" && tape.historical_audit_verified &&
                tape.source_sha256 == simulation::content_sha256(manifest_bytes),
            "La cinta reconstruida cambió durante la lectura o no conserva su auditoría");
    require(!result.input.context,
            "Las cintas reconstruidas de la política no admiten un contexto observable");
    result.identity = Json{{"role", policy_tape_role_name(role)},
                           {"manifest_sha256", tape.source_sha256},
                           {"market_sha256", manifest.at("file_sha256")},
                           {"tape_sha256", manifest.at("tape_sha256")},
                           {"historical_basis", tape.historical_basis},
                           {"observation_sha256", observation_sha256(tape)},
                           {"first_open", tape.open_times.front()},
                           {"last_close", tape.close_times.back()},
                           {"sessions", tape.close_times.size()}};
    return result;
}

void require_policy_sequence(std::span<const PolicyTape> tapes) {
    require(!tapes.empty(), "La política necesita al menos una cinta");
    std::set<std::string> manifests;
    const auto& first = tapes.front().identity;
    for (std::size_t index = 0; index < tapes.size(); ++index) {
        const auto& identity = tapes[index].identity;
        require(manifests.insert(identity.at("manifest_sha256").get<std::string>()).second,
                "La política repite una cinta reconstruida");
        require(identity.at("historical_basis") == first.at("historical_basis") &&
                    identity.at("observation_sha256") == first.at("observation_sha256") &&
                    simulation::same_policy_origin(*tapes[index].input.tape,
                                                   *tapes.front().input.tape),
                "Las cintas de la política no comparten activos, moneda, reglas o base histórica");
        if (index != 0) {
            require(integer(tapes[index - 1].identity.at("last_close")) <
                        integer(identity.at("first_open")),
                    "Una cinta de la política empieza antes de que termine la anterior");
        }
    }
}

void require_after_selection(const Json& selection, const PolicyTape& evaluation) {
    const auto& identity = evaluation.identity;
    std::size_t sources = 0;
    for (const auto* partition : {"train", "validation"}) {
        for (const auto& source : selection.at(partition)) {
            require(source.at("historical_basis") == identity.at("historical_basis") &&
                        source.at("observation_sha256") == identity.at("observation_sha256") &&
                        source.at("manifest_sha256") != identity.at("manifest_sha256") &&
                        integer(source.at("last_close")) < integer(identity.at("first_open")),
                    "La evaluación no es posterior a la selección o cambia activos, reglas o "
                    "base histórica");
            ++sources;
        }
    }
    require(sources != 0, "La selección no registra sus cintas de ajuste y validación");
}

} // namespace mars_titan::learning
