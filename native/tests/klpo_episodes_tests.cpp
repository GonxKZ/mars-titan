#include "mars_titan/klpo_episodes.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Registros fijos, sin actor, mercado simulado ni optimizador.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using namespace mars_titan::learning;
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
template <class F> void rejected(F operation, const char* message) {
    try {
        operation();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

KlpoEpisodeStep step(std::size_t cursor, bool sampled, double reward, bool final) {
    KlpoEpisodeStep result;
    result.cursor = cursor;
    result.decision_at = static_cast<int64_t>(10 + cursor * 10);
    result.outcome_at = result.decision_at + 10;
    result.observation = {.25F, -.5F};
    result.action = sampled ? uint8_t{2} : uint8_t{1};
    result.sampled = sampled;
    if (sampled) {
        result.behavior.fill(1.F / 6);
    }
    result.reward = reward;
    result.valuation_valid = true;
    result.truncated = final;
    return result;
}

KlpoEpisodeBatch batch() {
    KlpoEpisodeBatch result;
    result.fold = "fixture-fold";
    result.reference_sha256 = std::string(64, 'a');
    result.beta = .2;
    result.gamma = .25;
    result.observation_width = 2;
    KlpoEpisodeRecord record;
    record.spec.id = "episode-0";
    record.spec.source_sha256 = std::string(64, 'b');
    record.spec.context_sha256 = std::string(64, 'c');
    record.spec.close_times = {10, 20, 30};
    record.spec.forced_prefix = 1;
    record.steps = {step(0, false, 3., false), step(1, true, 4., true)};
    result.episodes.push_back(record);
    return result;
}

void forced_steps_keep_real_reward_and_clock() {
    const auto original = batch();
    validate_klpo_batch(original, true);
    const auto returns = klpo_terminal_returns(original);
    require(returns.size() == 1 && returns[0] == 4.,
            "Se perdió el retorno del paso forzado o su reloj");
    require(klpo_episode_status(original.episodes[0]) == KlpoEpisodeStatus::horizon,
            "Un calentamiento valorado se confundió con falta de valoración");
    auto forced = original;
    forced.episodes[0].spec.forced_prefix = 2;
    forced.episodes[0].steps[1] = step(1, false, 4., true);
    validate_klpo_batch(forced, true);
    require(klpo_terminal_returns(forced)[0] == 4., "Se descartó el episodio enteramente forzado");
}

void incomplete_and_unvalued_batches_are_not_objectives() {
    auto partial = batch();
    partial.episodes[0].steps.pop_back();
    validate_klpo_batch(partial, false);
    rejected([&] { static_cast<void>(klpo_terminal_returns(partial)); },
             "Se consumió un episodio parcial");
    auto invalid = batch();
    invalid.episodes[0].steps.back().valuation_valid = false;
    invalid.episodes[0].steps.back().reward = 0.;
    validate_klpo_batch(invalid, false);
    require(klpo_episode_status(invalid.episodes[0]) == KlpoEpisodeStatus::missing_valuation,
            "Falta la causa de valoración inválida");
    rejected([&] { static_cast<void>(klpo_terminal_returns(invalid)); },
             "Se convirtió la falta de valoración en retorno");
    auto ruined = batch();
    ruined.episodes[0].steps.back().terminated = true;
    ruined.episodes[0].steps.back().truncated = false;
    ruined.episodes[0].steps.back().reward = -20.;
    require(klpo_terminal_returns(ruined)[0] == -2., "La ruina perdió su recompensa contractual");
}

void invalid_records_and_budgets_fail() {
    for (int mutation = 0; mutation < 9; ++mutation) {
        auto invalid = batch();
        auto& last = invalid.episodes[0].steps.back();
        switch (mutation) {
        case 0:
            last.behavior.fill(.2F);
            last.behavior[3] = 0;
            break;
        case 1:
            last.outcome_at += 1;
            break;
        case 2:
            last.cursor = 0;
            break;
        case 3:
            last.reward = std::numeric_limits<double>::quiet_NaN();
            break;
        case 4:
            last.action = 6;
            break;
        case 5:
            last.truncated = false;
            break;
        case 6:
            invalid.episodes.push_back(invalid.episodes.front());
            break;
        case 7:
            invalid.episodes[0].steps.front().behavior[0] = 1;
            break;
        case 8:
            for (auto& value : invalid.episodes[0].spec.close_times) {
                value += 1'704'067'200'000'000;
            }
            for (auto& value : invalid.episodes[0].steps) {
                value.decision_at += 1'704'067'200'000'000;
                value.outcome_at += 1'704'067'200'000'000;
            }
            break;
        default:
            break;
        }
        rejected([&] { validate_klpo_batch(invalid, true); }, "Se aceptó un registro corrupto");
    }
    auto invalid = batch();
    invalid.max_bytes = 1;
    rejected([&] { validate_klpo_batch(invalid, false); }, "Se ignoró el presupuesto previo");
}

void codec_roundtrips_exactly_and_rejects_damage() {
    const auto value = batch();
    const auto bytes = serialize_klpo_batch(value);
    const auto recovered = deserialize_klpo_batch(bytes);
    require(serialize_klpo_batch(recovered) == bytes, "El registro no conserva sus bytes");
    require(klpo_terminal_returns(recovered) == klpo_terminal_returns(value),
            "La recuperación altera el retorno");
    rejected([&] { static_cast<void>(deserialize_klpo_batch(bytes.substr(0, bytes.size() - 1))); },
             "Se aceptó un cuerpo incompleto");
    rejected([&] { static_cast<void>(deserialize_klpo_batch(bytes + "x")); },
             "Se aceptaron bytes ajenos");
    rejected([&] { static_cast<void>(deserialize_klpo_batch(bytes, 1)); },
             "Se leyó antes de aplicar el límite");
}

void maximum_population_and_history_are_checked_before_payload() {
    auto value = batch();
    value.observation_width = 1;
    auto episode = value.episodes.front();
    episode.steps.clear();
    episode.spec.close_times.clear();
    for (std::size_t time = 0; time <= maximum_klpo_steps; ++time) {
        episode.spec.close_times.push_back(static_cast<int64_t>(time + 1));
    }
    value.episodes.clear();
    for (std::size_t index = 0; index < maximum_klpo_episodes; ++index) {
        episode.spec.id = "episode-" + std::to_string(index);
        value.episodes.push_back(episode);
    }
    validate_klpo_batch(value, false);
    const auto bytes = serialize_klpo_batch(value);
    require(deserialize_klpo_batch(bytes).episodes.size() == maximum_klpo_episodes,
            "El límite admitido no se puede recuperar");
    auto enlarged = bytes;
    // El ancho sigue a la firma de ocho bytes y al presupuesto uint64.
    for (std::size_t byte = 0; byte < 8; ++byte) {
        enlarged[16 + byte] = static_cast<char>((uint64_t{32768} >> (byte * 8)) & 0xffU);
    }
    rejected([&] { static_cast<void>(deserialize_klpo_batch(enlarged)); },
             "Se materializó una geometría mayor que el presupuesto");
    value.episodes.push_back(episode);
    rejected([&] { validate_klpo_batch(value, false); }, "Se admitieron más de 128 episodios");
    value.episodes.pop_back();
    value.episodes[0].spec.close_times.push_back(9999);
    rejected([&] { validate_klpo_batch(value, false); },
             "Se cortó o admitió una historia demasiado larga");
}
// El colector valida en cada paso solo lo añadido. Debe decidir igual que la validación completa.
void appended_steps_match_the_complete_validation() {
    const auto complete = batch();
    for (std::size_t prefix = 0; prefix <= complete.episodes[0].steps.size(); ++prefix) {
        auto value = complete;
        value.episodes[0].steps.resize(prefix);
        for (std::size_t validated = 0; validated <= prefix; ++validated) {
            const std::vector<std::size_t> previous{validated};
            validate_klpo_batch(value, false, previous);
        }
    }
    // Un paso añadido tras un cierre ya validado se rechaza aunque el cierre no se repita.
    auto closed = complete;
    closed.episodes[0].steps[0].truncated = true;
    closed.episodes[0].spec.close_times = {10, 20, 30, 40};
    closed.episodes[0].steps.push_back(step(2, true, 1., true));
    rejected([&] { validate_klpo_batch(closed, false, std::vector<std::size_t>{1}); },
             "Se añadió un paso tras un cierre ya validado");
    rejected([&] { validate_klpo_batch(closed, false, std::vector<std::size_t>{2}); },
             "Se aceptó un paso posterior al cierre validado");
    // Un paso nuevo corrupto falla. Un recuento ajeno o mayor que el registro también.
    auto corrupt = complete;
    corrupt.episodes[0].steps.back().cursor = 5;
    rejected([&] { validate_klpo_batch(corrupt, false, std::vector<std::size_t>{1}); },
             "Se aceptó un paso añadido corrupto");
    rejected([&] { validate_klpo_batch(complete, false, std::vector<std::size_t>{3}); },
             "Se validaron más pasos de los registrados");
    rejected([&] { validate_klpo_batch(complete, false, std::vector<std::size_t>{0, 0}); },
             "Se aceptó un recuento de otra oleada");
    // Cabecera y presupuesto se comprueban aunque no se repita ningún paso.
    auto budget = complete;
    budget.max_bytes = 1;
    rejected([&] { validate_klpo_batch(budget, false, std::vector<std::size_t>{2}); },
             "Se ignoró el presupuesto en la validación incremental");
}
} // namespace

int main() {
    try {
        forced_steps_keep_real_reward_and_clock();
        incomplete_and_unvalued_batches_are_not_objectives();
        invalid_records_and_budgets_fail();
        appended_steps_match_the_complete_validation();
        codec_roundtrips_exactly_and_rejects_damage();
        maximum_population_and_history_are_checked_before_payload();
        std::cout << "Registro terminal contrastado sin aprendizaje\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
