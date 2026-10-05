#include "mars_titan/financial_controls.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace mars_titan::simulation {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr std::size_t maximum_cases = 75;
constexpr std::size_t maximum_selections = 27;
constexpr std::size_t maximum_worlds = 896;
constexpr std::size_t maximum_audit_worlds = 512;
constexpr std::size_t journal_bytes = 32 * bytes_per_mebibyte;
constexpr std::size_t result_bytes = 16 * bytes_per_mebibyte;
constexpr std::size_t decision_start = 64;
constexpr std::size_t digest_characters = 64;
constexpr std::size_t expected_sessions = 256;
constexpr std::size_t expected_assets = 16;
constexpr std::array costs{0., 10., 25.};
constexpr std::array policies{ReferencePolicy::cash, ReferencePolicy::hold_initial,
                              ReferencePolicy::rebalance_25, ReferencePolicy::rebalance_50,
                              ReferencePolicy::rebalance_75, ReferencePolicy::rebalance_100};

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::invalid_argument(std::string(message));
    }
}

bool valid_digest(std::string_view value) {
    return value.size() == digest_characters && std::ranges::all_of(value, [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

bool contains(const std::filesystem::path& parent, const std::filesystem::path& child) {
    const auto prefix = std::filesystem::weakly_canonical(
        std::filesystem::absolute(parent.empty() ? std::filesystem::path{"."} : parent));
    const auto path = std::filesystem::weakly_canonical(
        std::filesystem::absolute(child.empty() ? std::filesystem::path{"."} : child));
    auto a = prefix.begin();
    auto b = path.begin();
    while (a != prefix.end() && b != path.end() && *a == *b) {
        ++a;
        ++b;
    }
    return a == prefix.end();
}

std::filesystem::path relative_path(const std::filesystem::path& root, const Json& value) {
    const std::filesystem::path relative(value.get<std::string>());
    require(!relative.empty() && !relative.is_absolute(), "Se necesita una ruta relativa");
    for (const auto& component : relative) {
        require(component != "." && component != "..", "La ruta sale de la fuente confirmada");
    }
    const auto path = root / relative;
    require_safe_path(path);
    require(contains(root, path), "La ruta sale de la fuente confirmada");
    return path;
}

Json document(const std::filesystem::path& path, std::size_t maximum = journal_bytes) {
    return parse_bounded_json(read_bounded_file(path, maximum));
}

void closed_test(const Json& value) {
    require(value.at("final_test_opened") == false, "El test final debe permanecer cerrado");
}

struct AuthorizedWorlds {
    Json sources;
    std::string catalog_sha256;
    std::string identity_sha256;
};

AuthorizedWorlds authorize(const FinancialControlsOptions& options) {
    require(valid_digest(options.campaign_sha256) && valid_digest(options.freeze_sha256) &&
                valid_digest(options.identity_sha256),
            "Se requieren las tres huellas SHA256 de los archivos brutos revisados");
    const auto campaign_bytes = read_bounded_file(options.campaign / "campaign.json", journal_bytes);
    const auto freeze_bytes = read_bounded_file(options.campaign / "freeze.json", journal_bytes);
    const auto identity_bytes = read_bounded_file(options.campaign / "identity.json", journal_bytes);
    require(content_sha256(campaign_bytes) == options.campaign_sha256 &&
                content_sha256(freeze_bytes) == options.freeze_sha256 &&
                content_sha256(identity_bytes) == options.identity_sha256,
            "La campaña, la congelación o la identidad han cambiado desde su revisión");
    const auto envelope = parse_bounded_json(campaign_bytes);
    const auto& state = envelope.at("payload");
    const auto freeze = parse_bounded_json(freeze_bytes);
    const auto report = document(options.campaign / "run.json");
    const auto identity = parse_bounded_json(identity_bytes);
    closed_test(freeze);
    closed_test(report);
    closed_test(identity);
    closed_test(identity.at("settings"));
    closed_test(identity.at("base_configuration"));
    require(state.at("status") == "completed" && state.at("phase") == "completed" &&
                state.at("budget_complete") == true && state.at("audit_opened") == true &&
                state.at("active_case").is_null() &&
                state.at("freeze_sha256") == options.freeze_sha256 &&
                report.at("status") == "completed" && report.at("phase") == "completed" &&
                report.at("selection_frozen") == true && report.at("audit_opened") == true &&
                report.at("budget_complete") == true && report.at("parent_frozen") == true &&
                report.at("domain") == "synthetic" && report.at("analysis_domain") == "technical" &&
                freeze.at("selection_uses") == "validation_only",
            "Las referencias necesitan una campaña cerrada con auditoría y selección congeladas");
    const auto identity_hash = state.at("identity_sha256").get<std::string>();
    require(valid_digest(identity_hash) && freeze.at("identity_sha256") == identity_hash &&
                report.at("identity_sha256") == identity_hash,
            "La congelación y el cierre pertenecen a otra campaña");
    const auto& environment = identity.at("base_configuration").at("environment");
    require(environment.at("capital") == default_capital &&
                environment.at("participation") == default_participation &&
                environment.at("score_scale") == default_score_scale &&
                environment.at("ruin_penalty") == default_ruin_penalty,
            "Los parámetros financieros no corresponden al contrato de referencia");
    const auto& selections = freeze.at("selections");
    const auto& cases = state.at("cases");
    require(selections.is_array() && !selections.empty() && selections.size() <= maximum_selections &&
                cases.is_array() && !cases.empty() && cases.size() <= maximum_cases &&
                report.at("completed_cases") == cases.size(),
            "Los recuentos de la campaña no conservan su plan confirmado");
    std::map<std::string, const Json*> selected;
    for (const auto& selection : selections) {
        require(selected.emplace(selection.at("output").get<std::string>(), &selection).second,
                "La congelación repite una selección");
    }
    std::set<std::string> identifiers;
    std::set<std::string> trained;
    std::set<std::string> audited;
    for (const auto& entry : cases) {
        require(identifiers.insert(entry.at("id").get<std::string>()).second &&
                    entry.at("status") == "completed", "Hay casos repetidos o sin terminar");
        const auto stage = entry.at("stage").get<std::string>();
        require(stage == "pilot" || stage == "main" || stage == "auxiliary" || stage == "audit",
                "La etapa no está admitida");
        const auto path = relative_path(options.campaign, entry.at("output"));
        const auto receipt_bytes = read_bounded_file(path / (stage == "audit" ? "audit.json" : "run.json"),
                                                     maximum_manifest_bytes);
        require(content_sha256(receipt_bytes) == entry.at("receipt_sha256").get<std::string>(),
                "Un recibo ha cambiado desde el cierre confirmado");
        const auto receipt = parse_bounded_json(receipt_bytes);
        closed_test(receipt);
        require(receipt.at("status") == "completed" && receipt.at("domain") == "synthetic",
                "Un recibo no acredita una ejecución sintética completa");
        if (stage == "main" || stage == "auxiliary") {
            const auto output = entry.at("output").get<std::string>();
            require(selected.contains(output) && trained.insert(output).second &&
                        selected.at(output)->at("receipt_sha256") == entry.at("receipt_sha256"),
                    "El entrenamiento no corresponde a la selección congelada");
        } else if (stage == "audit") {
            const auto source = entry.at("training_output").get<std::string>();
            require(selected.contains(source) && audited.insert(source).second,
                    "La auditoría no conserva una selección única");
            const auto& chosen = *selected.at(source);
            require(receipt.at("identity").at("policy_sha256") == chosen.at("policy_sha256") &&
                        receipt.at("identity").at("selected_identity_sha256") ==
                            chosen.at("training_identity_sha256"),
                    "La auditoría no evaluó el estado congelado");
        }
    }
    require(trained.size() == selections.size() && audited == trained,
            "Falta una selección entrenada o su auditoría terminada");
    const auto catalog_bytes = read_bounded_file(options.scenarios, journal_bytes);
    const auto catalog = parse_bounded_json(catalog_bytes);
    const auto catalog_hash = content_sha256(catalog_bytes);
    closed_test(catalog);
    require(catalog_hash == identity.at("index_sha256").get<std::string>() &&
                catalog.at("domain") == "synthetic" && catalog.at("analysis_domain") == "technical" &&
                catalog.at("status") == "completed" && catalog.at("real_corpus_compatible") == false,
            "El catálogo no corresponde a los mundos sintéticos de la campaña");
    const auto& records = catalog.at("records");
    const auto& frozen_sources = freeze.at("audit_sources");
    require(records.is_array() && records.size() <= maximum_worlds && frozen_sources.is_array() &&
                !frozen_sources.empty() && frozen_sources.size() <= maximum_audit_worlds,
            "El catálogo supera sus límites o no tiene auditoría congelada");
    std::map<std::string, const Json*> by_hash;
    std::set<int64_t> seeds;
    for (const auto& record : records) {
        require(seeds.insert(read_json_int64(record.at("seed"))).second &&
                    by_hash.emplace(record.at("manifest_sha256").get<std::string>(), &record).second,
                "El catálogo reutiliza una semilla o manifiesto");
    }
    Json sources = Json::array();
    std::set<std::string> used;
    for (const auto& frozen : frozen_sources) {
        const auto hash = frozen.at("manifest_sha256").get<std::string>();
        require(by_hash.contains(hash) && used.insert(hash).second,
                "La congelación repite o pierde un mundo");
        const auto& record = *by_hash.at(hash);
        require(record.at("split") == "audit" && frozen.at("split") == "audit" &&
                    record.at("evaluator_only") == true && record.at("name") == frozen.at("name") &&
                    record.at("context_sha256") == frozen.at("context_sha256") &&
                    record.at("warmup_sessions") == decision_start &&
                    record.at("decision_start") == decision_start,
                "La fuente no conserva la reserva de auditoría y su calentamiento");
        sources.push_back(record);
    }
    require(sources.size() == static_cast<std::size_t>(std::count_if(records.begin(), records.end(),
                [](const Json& value) { return value.at("split") == "audit"; })),
            "La congelación no cubre toda la auditoría del catálogo");
    return {std::move(sources), catalog_hash, identity_hash};
}

Json metrics_json(const FinancialMetrics& metrics) {
    return Json{{"net_return", metrics.net_return ? Json(*metrics.net_return) : Json(nullptr)},
                {"max_drawdown", metrics.max_drawdown ? Json(*metrics.max_drawdown) : Json(nullptr)},
                {"costs", metrics.costs}, {"turnover", metrics.turnover}, {"steps", metrics.steps},
                {"completed", metrics.completed}, {"invalid_reason", metrics.invalid_reason}};
}
} // namespace

uint8_t warmed_reference_action(ReferencePolicy policy, std::size_t cursor,
                                std::size_t first_decision) {
    const auto action = policy_action(policy, cursor >= first_decision ? cursor - first_decision : 0);
    return cursor < first_decision ? uint8_t{1} : action;
}

FinancialMetrics evaluate_financial_control(std::shared_ptr<const MarketTape> tape,
                                            ReferencePolicy policy, Parameters parameters,
                                            std::size_t first_decision) {
    require(tape && tape->domain == "synthetic" && tape->close_times.size() >= 2 &&
                first_decision < tape->close_times.size() - 1,
            "La referencia necesita una cinta sintética con decisiones tras el calentamiento");
    FinancialSession session(std::move(tape), parameters);
    while (!session.done()) {
        static_cast<void>(session.step(warmed_reference_action(policy, session.cursor(), first_decision)));
    }
    return session.metrics();
}

Json run_financial_controls(const FinancialControlsOptions& options,
                            const std::function<bool()>& stop_requested) {
    const auto started = Clock::now();
    require(!options.campaign.empty() && !options.scenarios.empty() && !options.output.empty(),
            "Se requieren campaña, catálogo y salida");
    require_safe_path(options.output);
    for (const auto& source : {options.campaign, options.scenarios.parent_path()}) {
        require(!contains(source, options.output) && !contains(options.output, source),
                "La salida debe estar separada de la campaña y del catálogo");
    }
    const auto authorized = authorize(options);
    OutputLock lock(options.output, false);
    Json report{{"schema_version", 1}, {"kind", "frozen_audit_financial_controls"},
                {"domain", "synthetic"}, {"analysis_domain", "technical"}, {"split", "audit"},
                {"final_test_opened", false}, {"learning", false}, {"device", "cpu"},
                {"native_source_sha256", MARS_TITAN_NATIVE_SOURCE_SHA256},
                {"native_version", MARS_TITAN_NATIVE_VERSION},
                {"campaign_file_sha256", options.campaign_sha256},
                {"freeze_file_sha256", options.freeze_sha256},
                {"identity_file_sha256", options.identity_sha256},
                {"catalog_file_sha256", authorized.catalog_sha256},
                {"campaign_identity_sha256", authorized.identity_sha256},
                {"decision_start", decision_start}, {"capital", default_capital},
                {"participation", default_participation}, {"score_scale", default_score_scale},
                {"ruin_penalty", default_ruin_penalty}, {"cost_bps", costs},
                {"sources", authorized.sources}, {"status", "running"},
                {"expected_episodes", authorized.sources.size() * policies.size() * costs.size()},
                {"completed_episodes", 0}, {"invalid_episodes", 0}, {"metrics", Json::array()},
                {"workers", 1}};
    const auto publish = [&] {
        report["wall_seconds"] = std::chrono::duration<double>(Clock::now() - started).count();
        report["ram_peak_bytes"] = process_memory_high_water();
        atomic_json_file(options.output / "controls.json", report, result_bytes);
    };
    publish();
    std::size_t completed = 0;
    std::size_t invalid = 0;
    try {
        for (const auto& source : authorized.sources) {
            if (stop_requested && stop_requested()) {
                report["status"] = "interrupted";
                publish();
                return report;
            }
            const auto path = relative_path(options.scenarios.parent_path(), source.at("path"));
            const auto manifest_bytes = read_bounded_file(path / "manifest.json", maximum_manifest_bytes);
            require(content_sha256(manifest_bytes) == source.at("manifest_sha256").get<std::string>(),
                    "El manifiesto ya no corresponde al mundo congelado");
            const auto manifest = parse_bounded_json(manifest_bytes);
            const auto& provenance = manifest.at("identity").at("source");
            require(provenance.at("generator").at("split") == "audit" &&
                        provenance.at("generator").at("family") == source.at("family") &&
                        provenance.at("generator").at("seed") == source.at("seed") &&
                        provenance.at("decision_start") == decision_start &&
                        provenance.at("warmup_action") == 1 &&
                        manifest.at("sessions") == expected_sessions &&
                        manifest.at("assets") == expected_assets,
                    "El mundo no conserva el contrato sintético de auditoría");
            const auto tape = load_market_tape(path);
            require(tape->source_sha256 == source.at("manifest_sha256").get<std::string>(),
                    "El manifiesto cambió durante la lectura de la cinta");
            for (const auto policy : policies) {
                for (const double cost : costs) {
                    Parameters parameters;
                    parameters.cost_bps = cost;
                    const auto metrics = evaluate_financial_control(tape, policy, parameters, decision_start);
                    auto row = metrics_json(metrics);
                    row["policy"] = policy_name(policy);
                    row["family"] = source.at("family");
                    row["generator_seed"] = source.at("seed");
                    row["manifest_sha256"] = source.at("manifest_sha256");
                    row["cost_bps"] = cost;
                    report["metrics"].push_back(std::move(row));
                    ++completed;
                    invalid += metrics.completed ? 0U : 1U;
                }
            }
            report["completed_episodes"] = completed;
            report["invalid_episodes"] = invalid;
        }
        report["status"] = invalid == 0 ? "completed" : "incomplete";
        publish();
        return report;
    } catch (const std::exception& error) {
        report["completed_episodes"] = completed;
        report["invalid_episodes"] = invalid;
        report["status"] = "failed";
        report["error"] = error.what();
        publish();
        throw;
    }
}
} // namespace mars_titan::simulation
