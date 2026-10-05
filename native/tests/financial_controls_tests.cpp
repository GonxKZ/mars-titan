#include "mars_titan/financial_controls.hpp"

#include <cmath>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {
using namespace mars_titan::simulation;
using Json = nlohmann::json;
constexpr double tolerance = 1e-10;
constexpr std::size_t digest_characters = 64;
constexpr std::size_t first_decision = 64;
constexpr std::size_t fixture_sessions = 67;
constexpr std::size_t final_cursor = fixture_sessions - 1;
constexpr std::size_t price_width = 5;
constexpr double fixture_volume = 1'000'000;
constexpr double fixture_capital = 10'000;
constexpr double fixture_participation = .01;
constexpr double fixture_score_scale = .01;
constexpr double fixture_ruin_penalty = -20;
constexpr uint8_t full_exposure_action = 5;
constexpr std::size_t reference_combinations = 18;
constexpr std::size_t excessive_cases = 76;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void near(double actual, double expected, std::string_view message) {
    require(std::isfinite(actual) && std::abs(actual - expected) < tolerance, message);
}

void near(std::optional<double> actual, double expected, std::string_view message) {
    if (!actual.has_value()) {
        throw std::runtime_error(std::string(message));
    }
    near(*actual, expected, message);
}

template <typename Function> void rejected(Function&& function) {
    try {
        std::forward<Function>(function)();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Se aceptó una entrada inválida");
}

std::shared_ptr<MarketTape> fixture() {
    auto tape = std::make_shared<MarketTape>();
    tape->assets = {"A"};
    tape->currency = "USD";
    tape->domain = "synthetic";
    tape->partition = "validation";
    tape->parent_id = "referencia-de-prueba";
    tape->source_sha256.assign(digest_characters, 'a');
    for (std::size_t at = 0; at < fixture_sessions; ++at) {
        tape->open_times.push_back(static_cast<int64_t>(2 * at));
        tape->close_times.push_back(static_cast<int64_t>(2 * at + 1));
        tape->prediction_times.push_back(static_cast<int64_t>(2 * at + 1));
        const double price = at == final_cursor ? 12 : 10;
        tape->prices.insert(tape->prices.end(), {price, price, price, price, fixture_volume});
        tape->scores.push_back(1);
    }
    return tape;
}

void warmup_does_not_buy_before_the_allowed_close() {
    auto tape = fixture();
    constexpr double low_close = 5;
    constexpr double final_return = .2;
    tape->prices[first_decision * price_width + 3] = low_close;
    Parameters parameters;
    parameters.cost_bps = 0;
    const auto result = evaluate_financial_control(tape, ReferencePolicy::hold_initial, parameters, first_decision);
    near(result.net_return, final_return, "La compra debe ejecutarse después del cierre 64");
    near(result.turnover, 1, "El salto de apertura limita la compra al efectivo disponible");
    near(result.max_drawdown, 0, "El calentamiento no debe crear posiciones");
    require(result.completed && result.steps == final_cursor, "Deben evaluarse todas las transiciones");
    require(warmed_reference_action(ReferencePolicy::hold_initial, first_decision - 1, first_decision) == 1 &&
                warmed_reference_action(ReferencePolicy::hold_initial, first_decision, first_decision) == full_exposure_action &&
                warmed_reference_action(ReferencePolicy::hold_initial, first_decision + 1, first_decision) == 0,
            "La asignación inicial debe desplazarse al primer cierre permitido");
}

void controls_match_the_explicit_native_action_sequence() {
    const auto tape = fixture();
    for (const auto policy : {ReferencePolicy::cash, ReferencePolicy::hold_initial,
                              ReferencePolicy::rebalance_25, ReferencePolicy::rebalance_50,
                              ReferencePolicy::rebalance_75, ReferencePolicy::rebalance_100}) {
        for (const double cost : {0., 10., 25.}) {
            Parameters parameters;
            parameters.cost_bps = cost;
            FinancialSession reference(tape, parameters);
            for (std::size_t at = 0; at < first_decision; ++at) {
                static_cast<void>(reference.step(1));
                near(reference.snapshot().account.turnover, 0, "El calentamiento debe ser efectivo");
            }
            static_cast<void>(reference.step(policy_action(policy, 0)));
            static_cast<void>(reference.step(policy_action(policy, 1)));
            const auto actual = evaluate_financial_control(tape, policy, parameters, first_decision);
            const auto expected = reference.metrics();
            require(actual.completed == expected.completed && actual.steps == expected.steps,
                    "La referencia y el control deben terminar en la misma sesión");
            require(actual.net_return == expected.net_return, "Cambió el retorno nativo");
            require(actual.max_drawdown == expected.max_drawdown, "Cambió el drawdown");
            near(actual.costs, expected.costs, "Cambió la contabilidad de costes");
            near(actual.turnover, expected.turnover, "Cambió la rotación");
        }
    }
}

void cash_is_zero_and_hold_pays_only_its_purchase() {
    const auto tape = fixture();
    constexpr double cost_bps = 10;
    constexpr double purchase_cost = 9.99;
    constexpr double turnover = .999;
    constexpr double net_return = .198801;
    Parameters parameters;
    parameters.cost_bps = cost_bps;
    const auto cash = evaluate_financial_control(tape, ReferencePolicy::cash, parameters, first_decision);
    near(cash.net_return, 0, "El efectivo no debe generar rentabilidad ficticia");
    near(cash.turnover, 0, "El efectivo no debe negociar");
    const auto hold = evaluate_financial_control(tape, ReferencePolicy::hold_initial, parameters, first_decision);
    near(hold.costs, purchase_cost, "La compra de 999 unidades debe pagar 9,99 USD");
    near(hold.turnover, turnover, "La posición no se debe liquidar al final");
    near(hold.net_return, net_return, "El retorno debe incluir los costes de compra");
}

void missing_close_remains_an_incomplete_episode() {
    auto tape = fixture();
    tape->prices[(first_decision + 1) * price_width + 3] = std::numeric_limits<double>::quiet_NaN();
    const auto result = evaluate_financial_control(tape, ReferencePolicy::hold_initial, Parameters{}, first_decision);
    require(!result.completed && !result.net_return && !result.max_drawdown &&
                result.invalid_reason == "missing_close" && result.steps == first_decision + 1,
            "Un cierre ausente no puede publicarse como retorno cero");
}

void invalid_inputs_are_rejected() {
    rejected([] { static_cast<void>(evaluate_financial_control(nullptr, ReferencePolicy::cash,
                                                              Parameters{}, first_decision)); });
    rejected([] { static_cast<void>(evaluate_financial_control(fixture(), ReferencePolicy::cash,
                                                              Parameters{}, final_cursor)); });
    auto real = fixture();
    real->domain = "real";
    rejected([&] { static_cast<void>(evaluate_financial_control(real, ReferencePolicy::cash,
                                                               Parameters{}, first_decision)); });
    auto test = fixture();
    test->partition = "test";
    rejected([&] { static_cast<void>(evaluate_financial_control(test, ReferencePolicy::cash,
                                                               Parameters{}, first_decision)); });
    rejected([] { static_cast<void>(parse_policy("unknown")); });
}

struct ClosedCampaign {
    std::filesystem::path root;
    FinancialControlsOptions options;
    Json state;
    Json report;
    Json identity;
    Json freeze;

    ClosedCampaign() {
        const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
        root = std::filesystem::temp_directory_path() /
               ("mars-titan-financial-controls-" + std::to_string(unique));
        require(std::filesystem::create_directory(root), "La carpeta de prueba ya existe");
        options.campaign = root / "campaign";
        options.scenarios = root / "worlds" / "index.json";
        options.output = root / "output";
        std::filesystem::create_directories(options.campaign / "main" / "control-42");
        std::filesystem::create_directories(options.campaign / "audit" / "control-42");
        std::filesystem::create_directories(options.scenarios.parent_path());
        const std::string hash(digest_characters, 'a');
        const Json source{{"name", "world"}, {"path", "world"}, {"family", "no_signal"},
                          {"split", "audit"}, {"seed", 1}, {"manifest_sha256", hash},
                          {"context_sha256", hash}, {"evaluator_only", true},
                          {"warmup_sessions", first_decision}, {"decision_start", first_decision}};
        const Json catalog{{"domain", "synthetic"}, {"analysis_domain", "technical"},
                           {"status", "completed"}, {"real_corpus_compatible", false},
                           {"final_test_opened", false}, {"records", Json::array({source})}};
        atomic_json_file(options.scenarios, catalog);
        const Json receipt{{"status", "completed"}, {"domain", "synthetic"},
                           {"final_test_opened", false},
                           {"identity", {{"policy_sha256", hash}, {"selected_identity_sha256", hash}}}};
        atomic_json_file(options.campaign / "main/control-42/run.json", receipt);
        atomic_json_file(options.campaign / "audit/control-42/audit.json", receipt);
        const auto receipt_hash = content_sha256(receipt.dump());
        freeze = {{"final_test_opened", false}, {"identity_sha256", hash},
                  {"selection_uses", "validation_only"}, {"audit_sources", Json::array({source})},
                  {"selections", Json::array({{{"output", "main/control-42"},
                      {"receipt_sha256", receipt_hash}, {"policy_sha256", hash},
                      {"training_identity_sha256", hash}}})}};
        state = {{"status", "completed"}, {"phase", "completed"}, {"budget_complete", true},
                 {"audit_opened", true}, {"active_case", nullptr}, {"identity_sha256", hash},
                 {"cases", Json::array({
                     {{"id", "main-control-42"}, {"stage", "main"}, {"status", "completed"},
                      {"output", "main/control-42"}, {"receipt_sha256", receipt_hash}},
                     {{"id", "audit-control-42"}, {"stage", "audit"}, {"status", "completed"},
                      {"output", "audit/control-42"}, {"training_output", "main/control-42"},
                      {"receipt_sha256", receipt_hash}}})}};
        report = {{"status", "completed"}, {"phase", "completed"}, {"budget_complete", true},
                  {"audit_opened", true}, {"selection_frozen", true}, {"parent_frozen", true},
                  {"final_test_opened", false}, {"identity_sha256", hash}, {"domain", "synthetic"},
                  {"analysis_domain", "technical"}, {"completed_cases", 2}};
        identity = {{"final_test_opened", false},
                    {"index_sha256", content_sha256(catalog.dump())},
                    {"settings", {{"final_test_opened", false}}},
                    {"base_configuration", {{"final_test_opened", false},
                        {"environment", {{"capital", fixture_capital}, {"participation", fixture_participation},
                         {"score_scale", fixture_score_scale}, {"ruin_penalty", fixture_ruin_penalty}}}}}};
        publish();
    }

    ClosedCampaign(const ClosedCampaign&) = delete;
    ClosedCampaign& operator=(const ClosedCampaign&) = delete;
    ClosedCampaign(ClosedCampaign&&) = delete;
    ClosedCampaign& operator=(ClosedCampaign&&) = delete;

    ~ClosedCampaign() {
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
    }

    void publish() {
        atomic_json_file(options.campaign / "freeze.json", freeze);
        options.freeze_sha256 = content_sha256(freeze.dump());
        state["freeze_sha256"] = options.freeze_sha256;
        const Json envelope{{"payload", state}, {"sha256", content_sha256(state.dump())}};
        atomic_json_file(options.campaign / "campaign.json", envelope);
        options.campaign_sha256 = content_sha256(envelope.dump());
        atomic_json_file(options.campaign / "run.json", report);
        atomic_json_file(options.campaign / "identity.json", identity);
        options.identity_sha256 = content_sha256(identity.dump());
    }

    void rejects_before_creating_output() {
        rejected([&] { static_cast<void>(run_financial_controls(options)); });
        require(!std::filesystem::exists(options.output), "El rechazo no debe abrir la salida ni el Parquet");
    }
};

void audit_requires_the_reviewed_hashes_and_closed_selection() {
    {
        ClosedCampaign fixture;
        fixture.identity["changed_after_review"] = true;
        atomic_json_file(fixture.options.campaign / "identity.json", fixture.identity);
        rejected([&] { static_cast<void>(run_financial_controls(fixture.options, [] { return true; })); });
        require(!std::filesystem::exists(fixture.options.output),
                "La identidad modificada debe rechazarse antes de crear la salida");
    }
    {
        ClosedCampaign fixture;
        const auto result = run_financial_controls(fixture.options, [] { return true; });
        require(result.at("status") == "interrupted" && result.at("expected_episodes") == reference_combinations &&
                    result.at("completed_episodes") == 0 && result.at("metrics").empty(),
                "La autorización válida deriva el plan y respeta una interrupción antes de leer precios");
    }
    {
        ClosedCampaign fixture;
        fixture.options.campaign_sha256.assign(digest_characters, 'b');
        fixture.rejects_before_creating_output();
    }
    {
        ClosedCampaign fixture;
        fixture.options.freeze_sha256.assign(digest_characters, 'b');
        fixture.rejects_before_creating_output();
    }
    {
        ClosedCampaign fixture;
        fixture.state["cases"][0]["status"] = "running";
        fixture.publish();
        fixture.rejects_before_creating_output();
    }
    {
        ClosedCampaign fixture;
        fixture.report["selection_frozen"] = false;
        fixture.publish();
        fixture.rejects_before_creating_output();
    }
    {
        ClosedCampaign fixture;
        fixture.report["final_test_opened"] = true;
        fixture.publish();
        fixture.rejects_before_creating_output();
    }
    {
        ClosedCampaign fixture;
        fixture.identity["index_sha256"] = std::string(digest_characters, 'b');
        fixture.publish();
        fixture.rejects_before_creating_output();
    }
    {
        ClosedCampaign fixture;
        fixture.options.output = fixture.options.campaign / "new-results";
        fixture.rejects_before_creating_output();
    }
    {
        ClosedCampaign fixture;
        fixture.state["cases"] = Json::array();
        for (std::size_t index = 0; index < excessive_cases; ++index) {
            fixture.state["cases"].push_back(Json::object());
        }
        fixture.report["completed_cases"] = excessive_cases;
        fixture.publish();
        fixture.rejects_before_creating_output();
    }
}

void catalog_filename_uses_the_current_directory() {
    ClosedCampaign fixture;
    const auto previous = std::filesystem::current_path();
    std::filesystem::current_path(fixture.options.scenarios.parent_path());
    fixture.options.scenarios = fixture.options.scenarios.filename();
    try {
        const auto result = run_financial_controls(fixture.options, [] { return true; });
        require(result.at("status") == "interrupted" &&
                    result.at("expected_episodes") == reference_combinations,
                "El nombre relativo del catálogo debe autorizar una salida externa");
        fixture.options.output = "new-results";
        fixture.rejects_before_creating_output();
    } catch (...) {
        std::filesystem::current_path(previous);
        throw;
    }
    std::filesystem::current_path(previous);
}

void relative_output_cannot_write_inside_the_catalog() {
    ClosedCampaign fixture;
    const auto previous = std::filesystem::current_path();
    std::filesystem::current_path(fixture.options.scenarios.parent_path());
    fixture.options.output = "new-results";
    try {
        rejected([&] { static_cast<void>(run_financial_controls(fixture.options, [] { return true; })); });
        require(!std::filesystem::exists(fixture.options.output),
                "La salida relativa no debe crearse dentro del catálogo");
    } catch (...) {
        std::filesystem::current_path(previous);
        throw;
    }
    std::filesystem::current_path(previous);
}
} // namespace

int main() {
    try {
        warmup_does_not_buy_before_the_allowed_close();
        controls_match_the_explicit_native_action_sequence();
        cash_is_zero_and_hold_pays_only_its_purchase();
        missing_close_remains_an_incomplete_episode();
        invalid_inputs_are_rejected();
        audit_requires_the_reviewed_hashes_and_closed_selection();
        relative_output_cannot_write_inside_the_catalog();
        catalog_filename_uses_the_current_directory();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Referencias, calentamiento, costes y errores comprobados\n";
    return 0;
}
