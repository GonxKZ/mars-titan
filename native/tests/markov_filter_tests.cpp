#include "mars_titan/markov_filter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using mars_titan::simulation::MarkovFilter;
using mars_titan::simulation::MarkovObservation;
using mars_titan::simulation::MarkovParameters;
using mars_titan::simulation::MarkovSnapshot;

constexpr double tolerance = 2e-12;
constexpr double unknown = std::numeric_limits<double>::quiet_NaN();
constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double equal_probability = 0.5;
constexpr int64_t future_time = 99;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void near(double actual, double expected, std::string_view message) {
    require(std::isfinite(actual) &&
                std::abs(actual - expected) <= tolerance * std::max(1.0, std::abs(expected)),
            message);
}

template <class Function>
void rejected(Function&& function, std::string_view message) {
    try {
        std::forward<Function>(function)();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}

MarkovParameters parameters() {
    const MarkovParameters fixture{2, 2, {0.6, 0.4}, {0.85, 0.15, 0.25, 0.75},
                                   {0.5, -1.0, 1.5, 0.75}, {1.0, 2.0, 0.5, 1.25}};
    return fixture;
}

struct Frame {
    std::array<double, 2> values{};
    std::array<uint8_t, 2> present{1, 1};
    std::array<int64_t, 2> available_at{};
    int64_t decision_at = 0;

    [[nodiscard]] MarkovObservation observation() const {
        return {values, present, available_at, decision_at};
    }
};

std::array<Frame, 4> frames() {
    const std::array<Frame, 4> observations{{{{0.0, -0.5}, {1, 1}, {0, 0}, 1},
                                            {{1.0, 0.25}, {1, 1}, {1, 2}, 2},
                                            {{2.0, unknown}, {1, 0}, {3, future_time}, 3},
                                            {{-0.5, -2.0}, {1, 1}, {2, 4}, 4}}};
    return observations;
}

struct Enumerated {
    std::array<double, 2> probabilities{};
    double log_likelihood = 0;
    double change_probability = 0;
};

// La referencia suma todas las trayectorias, sin reutilizar la recurrencia del filtro.
Enumerated enumerate(const MarkovParameters& model, std::span<const Frame> observations) {
    std::array<long double, 2> terminal{};
    long double changed = 0;
    const std::size_t paths = std::size_t{1} << observations.size();
    for (std::size_t path = 0; path < paths; ++path) {
        std::size_t previous = path & 1U;
        long double weight = static_cast<long double>(model.prior[previous]);
        for (std::size_t step = 0; step < observations.size(); ++step) {
            const std::size_t state = (path >> step) & 1U;
            if (step != 0) {
                weight *= static_cast<long double>(model.transitions[previous * model.states + state]);
            }
            for (std::size_t feature = 0; feature < model.dimensions; ++feature) {
                if (observations[step].present.at(feature) == 0) {
                    continue;
                }
                const std::size_t offset = state * model.dimensions + feature;
                const long double variance = static_cast<long double>(model.variances[offset]);
                const long double error = static_cast<long double>(observations[step].values.at(feature)) -
                                          static_cast<long double>(model.means[offset]);
                weight *= std::exp(-error * error / (2 * variance)) /
                          std::sqrt(2 * std::numbers::pi_v<long double> * variance);
            }
            previous = state;
        }
        terminal.at(previous) += weight;
        if (observations.size() > 1 &&
            previous != ((path >> (observations.size() - 2)) & 1U)) {
            changed += weight;
        }
    }
    const long double total = terminal.at(0) + terminal.at(1);
    return {{static_cast<double>(terminal.at(0) / total), static_cast<double>(terminal.at(1) / total)},
            static_cast<double>(std::log(total)), static_cast<double>(changed / total)};
}

void matches_exact_path_enumeration() {
    const auto model = parameters();
    MarkovFilter filter(model);
    const auto data = frames();
    for (std::size_t index = 0; index < data.size(); ++index) {
        filter.step(data.at(index).observation());
        const auto expected = enumerate(model, std::span(data).first(index + 1));
        for (std::size_t state = 0; state < model.states; ++state) {
            near(filter.probabilities()[state], expected.probabilities.at(state),
                 "El filtro difiere de la enumeración exacta de trayectorias");
        }
        near(filter.log_likelihood(), expected.log_likelihood,
             "La log verosimilitud difiere de la enumeración exacta");
        near(filter.change_probability(), expected.change_probability,
             "La probabilidad de cambio no corresponde a las transiciones posteriores");
        require(filter.cursor() == index + 1 &&
                    filter.snapshot().last_decision_at == data.at(index).decision_at,
                "El cursor debe confirmar cada observación exactamente una vez");
    }
}

void future_suffix_does_not_change_the_prefix() {
    auto first_data = frames();
    auto second_data = first_data;
    constexpr std::array<double, 2> first_suffix{-20, 40};
    constexpr std::array<double, 2> last_suffix{30, -50};
    second_data.at(2).values = first_suffix;
    second_data.at(3).values = last_suffix;
    MarkovFilter first(parameters());
    MarkovFilter second(parameters());
    std::vector<MarkovSnapshot> prefix;
    for (std::size_t index = 0; index < first_data.size(); ++index) {
        first.step(first_data.at(index).observation());
        second.step(second_data.at(index).observation());
        if (index < 2) {
            require(first.snapshot() == second.snapshot(),
                    "Cambiar el sufijo futuro no puede alterar el prefijo filtrado");
            prefix.push_back(first.snapshot());
        }
    }
    require(first.snapshot() != second.snapshot(), "El sufijo observado debe afectar al filtro");
    MarkovFilter recovered(parameters());
    recovered.restore(prefix.back());
    recovered.step(first_data.at(2).observation());
    recovered.step(first_data.at(3).observation());
    require(first.snapshot() == recovered.snapshot(),
            "La recuperación del prefijo debe reproducir la continuación causal");
}

void missing_features_only_predict_and_ignore_their_payload() {
    constexpr double initial_probability = 0.6;
    constexpr double predicted_probability = 0.61;
    constexpr double predicted_change = 0.19;
    MarkovFilter filter(parameters());
    Frame absent{{unknown, infinity}, {0, 0}, {future_time, future_time}, 1};
    filter.step(absent.observation());
    near(filter.probabilities()[0], initial_probability, "Sin datos iniciales se conserva el prior");
    near(filter.log_likelihood(), 0, "Una observación ausente no añade verosimilitud");
    absent.decision_at = 2;
    filter.step(absent.observation());
    near(filter.probabilities()[0], predicted_probability, "Sin emisiones se aplica una transición");
    near(filter.change_probability(), predicted_change, "El cambio sin datos sigue la matriz de transición");
    near(filter.log_likelihood(), 0, "La predicción sola no cambia la verosimilitud");

    auto first_data = frames();
    auto second_data = first_data;
    second_data.at(2).values[1] = -infinity;
    second_data.at(2).available_at[1] = std::numeric_limits<int64_t>::max();
    MarkovFilter first(parameters());
    MarkovFilter second(parameters());
    for (std::size_t index = 0; index < first_data.size(); ++index) {
        first.step(first_data.at(index).observation());
        second.step(second_data.at(index).observation());
        require(first.snapshot() == second.snapshot(), "Una variable ausente no debe aportar datos");
    }
}

void invalid_observations_leave_the_state_unchanged() {
    MarkovFilter filter(parameters());
    filter.step(frames()[0].observation());
    const auto confirmed = filter.snapshot();
    auto bad = frames()[1];
    for (double value : {unknown, infinity, -infinity}) {
        bad.values[0] = value;
        rejected([&] { filter.step(bad.observation()); }, "Los datos presentes deben ser finitos");
        require(filter.snapshot() == confirmed, "Un dato inválido no puede confirmar el estado");
    }
    bad = frames()[1];
    bad.available_at[0] = bad.decision_at + 1;
    rejected([&] { filter.step(bad.observation()); }, "Una variable futura debe rechazarse");
    bad = frames()[1];
    bad.present[0] = 2;
    rejected([&] { filter.step(bad.observation()); }, "La máscara solo admite cero y uno");
    for (int64_t time : {int64_t{0}, int64_t{1}}) {
        bad = frames()[1];
        bad.decision_at = time;
        rejected([&] { filter.step(bad.observation()); }, "La decisión debe avanzar estrictamente");
    }
    bad = frames()[1];
    auto observation = bad.observation();
    observation.values = observation.values.first(1);
    rejected([&] { filter.step(observation); }, "Las variables deben tener la dimensión declarada");
    observation = bad.observation();
    observation.present = observation.present.first(1);
    rejected([&] { filter.step(observation); }, "La máscara debe tener la dimensión declarada");
    observation = bad.observation();
    observation.available_at = observation.available_at.first(1);
    rejected([&] { filter.step(observation); }, "Las fechas deben tener la dimensión declarada");
    require(filter.snapshot() == confirmed, "Un rechazo no puede alterar el estado confirmado");
}

void invalid_parameters_fail_before_filtering() {
    const auto reject = [](const MarkovParameters& model) {
        rejected([&] { const MarkovFilter invalid(model); }, "Los parámetros inválidos deben fallar");
    };
    for (std::size_t states : {std::size_t{0}, std::size_t{17},
                               std::numeric_limits<std::size_t>::max()}) {
        auto model = parameters();
        model.states = states;
        reject(model);
    }
    for (std::size_t dimensions : {std::size_t{0}, std::size_t{65}}) {
        auto model = parameters();
        model.dimensions = dimensions;
        reject(model);
    }
    for (std::size_t field = 0; field < 4; ++field) {
        auto model = parameters();
        const std::array<std::vector<double>*, 4> fields{
            &model.prior, &model.transitions, &model.means, &model.variances};
        fields.at(field)->pop_back();
        reject(model);
    }
    for (double value : {unknown, infinity, -infinity, -0.1, 1.1}) {
        auto model = parameters();
        model.prior[0] = value;
        reject(model);
        model = parameters();
        model.transitions[0] = value;
        reject(model);
    }
    auto model = parameters();
    model.prior = {0, 0};
    reject(model);
    model = parameters();
    constexpr double incomplete_row_probability = 0.75;
    model.transitions[0] = incomplete_row_probability;
    reject(model);
    for (double value : {unknown, infinity, -infinity, 0.0, -1.0}) {
        model = parameters();
        model.variances[0] = value;
        reject(model);
    }
    for (double value : {unknown, infinity, -infinity}) {
        model = parameters();
        model.means[0] = value;
        reject(model);
    }
}

void logarithmic_state_survives_extreme_underflow_and_recovery() {
    constexpr double separation = 1000;
    const MarkovParameters model{2, 1, {0.5, 0.5}, {1, 0, 0, 1}, {0, separation}, {1, 1}};
    MarkovFilter filter(model);
    std::array<double, 1> values{0};
    const std::array<uint8_t, 1> present{1};
    const std::array<int64_t, 1> available{0};
    filter.step({values, present, available, 0});
    require(filter.probabilities()[1] == 0 &&
                std::isfinite(filter.snapshot().log_probabilities[1]),
            "La cola que no cabe como probabilidad debe conservarse en logaritmos");
    MarkovFilter restored(model);
    restored.restore(filter.snapshot());
    values[0] = separation;
    filter.step({values, present, available, 1});
    restored.step({values, present, available, 1});
    near(filter.probabilities()[0], equal_probability, "La evidencia posterior debe recuperar la cola pequeña");
    constexpr double squared_error_penalty = 500000;
    near(filter.log_likelihood(), -squared_error_penalty - std::log(2 * std::numbers::pi),
         "La verosimilitud extrema debe conservar su escala");
    near(filter.change_probability(), 0, "Las transiciones imposibles deben seguir a cero");
    require(filter.snapshot() == restored.snapshot(), "La recuperación debe conservar colas pequeñas");

    values[0] = std::numeric_limits<double>::max();
    const auto confirmed = filter.snapshot();
    rejected([&] { filter.step({values, present, available, 2}); },
             "Una verosimilitud no representable debe fallar sin publicar infinito");
    require(filter.snapshot() == confirmed, "El desbordamiento debe conservar el estado");
}

void unreachable_emissions_do_not_set_the_numerical_scale() {
    constexpr double distant_value = 1e100;
    constexpr double first_prior = 0.3;
    constexpr double second_prior = 0.7;
    const MarkovParameters model{3, 1, {0.3, 0.7, 0}, {1, 0, 0, 0, 1, 0, 0, 0, 1},
                                 {0, 0, distant_value}, {1, 1, 1}};
    MarkovFilter filter(model);
    const std::array<double, 1> values{distant_value};
    const std::array<uint8_t, 1> present{1};
    const std::array<int64_t, 1> available{0};
    for (int64_t decision = 0; decision < 2; ++decision) {
        filter.step({values, present, available, decision});
        near(filter.probabilities()[0], first_prior,
             "Una emisión imposible no puede borrar el prior de los estados alcanzables");
        near(filter.probabilities()[1], second_prior, "La densidad común extrema debe conservar la normalización");
        near(filter.probabilities()[2], 0, "La evidencia no puede activar un estado inalcanzable");
        near(filter.change_probability(), 0, "La escala numérica no puede crear transiciones");
    }
}

void nearby_means_remain_distinguishable_far_from_the_observation() {
    constexpr double distant_value = 1e10;
    constexpr double separation = 1e-10;
    constexpr double first_posterior = 0.2689414213699951;
    constexpr double second_posterior = 0.7310585786300049;
    constexpr double absolute_log_density = -5e19;
    const MarkovParameters model{2, 1, {0.5, 0.5}, {1, 0, 0, 1}, {0, separation}, {1, 1}};
    const std::array<double, 1> values{distant_value};
    const std::array<uint8_t, 1> present{1};
    const std::array<int64_t, 1> available{0};
    MarkovFilter filter(model);
    filter.step({values, present, available, 0});
    near(filter.probabilities()[0], first_posterior,
         "Las medias cercanas deben conservar una razón de emisiones de exp(1)");
    near(filter.probabilities()[1], second_posterior,
         "La densidad absoluta grande no puede borrar la diferencia entre emisiones");
    near(filter.log_likelihood(), absolute_log_density,
         "El cálculo relativo debe conservar la escala absoluta de verosimilitud");
    MarkovFilter restored(model);
    restored.restore(filter.snapshot());
    restored.step({values, present, available, 1});
    constexpr double repeated_posterior = 0.8807970779778824;
    near(restored.probabilities()[1], repeated_posterior,
         "La continuación debe acumular la evidencia relativa después de recuperar");
}

void extreme_relative_emissions_choose_the_reachable_state() {
    constexpr double distant_value = 1e100;
    const MarkovParameters model{3, 1, {0, 0.5, 0.5}, {1, 0, 0, 0, 1, 0, 0, 0, 1},
                                 {distant_value, 0, 1}, {1, 1, 1}};
    const std::array<double, 1> values{distant_value};
    const std::array<uint8_t, 1> present{1};
    const std::array<int64_t, 1> available{0};
    MarkovFilter filter(model);
    filter.step({values, present, available, 0});
    near(filter.probabilities()[0], 0, "La referencia de emisiones debe ser alcanzable");
    near(filter.probabilities()[1], 0, "Las emisiones extremas deben conservar su orden");
    near(filter.probabilities()[2], 1, "La diferencia de medias debe resolver la emisión extrema");
    require(std::isfinite(filter.snapshot().log_probabilities[1]),
            "La comparación relativa debe conservar el peso logarítmico de la cola");
    constexpr double absolute_log_density = -5e199;
    near(filter.log_likelihood(), absolute_log_density,
         "La referencia alcanzable debe aportar la log verosimilitud absoluta");
}

void opposite_large_means_preserve_the_small_centered_observation() {
    const std::array<std::array<double, 4>, 2> cases{
        {{{1, -1e100, 1e100, 1}}, {{5e99, 1, 1e100, 0}}}};
    const std::array<uint8_t, 1> present{1};
    const std::array<int64_t, 1> available{0};
    for (const auto& item : cases) {
        const MarkovParameters model{2, 1, {0.5, 0.5}, {1, 0, 0, 1},
                                     {item[1], item[2]}, {1, 1}};
        MarkovFilter filter(model);
        const std::array<double, 1> values{item[0]};
        filter.step({values, present, available, 0});
        near(filter.probabilities()[1], item[3],
             "La suma centrada no puede perder el término pequeño entre medias grandes");
    }
}

void nearby_variances_remain_distinguishable_at_large_scale() {
    constexpr double distant_value = 1e8;
    // Referencia calculada con aritmética decimal de 110 cifras para los valores binary64 exactos.
    constexpr double expected_posterior = 0.752170687785598318922479466663;
    const MarkovParameters model{2, 1, {0.5, 0.5}, {1, 0, 0, 1}, {0, 0},
                                 {1, std::nextafter(1.0, 2.0)}};
    const std::array<double, 1> values{distant_value};
    const std::array<uint8_t, 1> present{1};
    const std::array<int64_t, 1> available{0};
    MarkovFilter filter(model);
    filter.step({values, present, available, 0});
    near(filter.probabilities()[1], expected_posterior,
         "La diferencia de varianzas no puede perderse entre densidades absolutas grandes");
}

void unresolved_relative_cancellation_leaves_the_state_unchanged() {
    const MarkovParameters model{2, 1, {0.5, 0.5}, {1, 0, 0, 1}, {2e10, 1e10}, {4, 1}};
    constexpr double first_observation = 1e10;
    std::array<double, 1> values{first_observation};
    const std::array<uint8_t, 1> present{1};
    const std::array<int64_t, 1> available{0};
    MarkovFilter filter(model);
    filter.step({values, present, available, 0});
    const auto confirmed = filter.snapshot();
    values[0] = 0;
    rejected([&] { filter.step({values, present, available, 1}); },
             "La cancelación sin precisión suficiente debe rechazarse de forma explícita");
    require(filter.snapshot() == confirmed, "Un rechazo numérico debe conservar el estado confirmado");
}

void snapshot_validation_is_atomic_and_binds_the_model() {
    MarkovFilter filter(parameters());
    const auto pristine = filter.snapshot();
    for (const auto& frame : frames()) {
        filter.step(frame.observation());
    }
    const auto confirmed = filter.snapshot();
    const auto reject = [&](const MarkovSnapshot& snapshot) {
        rejected([&] { filter.restore(snapshot); }, "La recuperación debe rechazar estados inválidos");
        require(filter.snapshot() == confirmed, "Una recuperación fallida debe ser atómica");
    };
    auto corrupted = confirmed;
    corrupted.parameters.means[0] += 1;
    reject(corrupted);
    corrupted = confirmed;
    corrupted.log_probabilities.pop_back();
    reject(corrupted);
    for (double value : {unknown, infinity, 0.0}) {
        corrupted = confirmed;
        corrupted.log_probabilities[0] = value;
        reject(corrupted);
    }
    corrupted = confirmed;
    corrupted.log_probabilities = {-infinity, -infinity};
    reject(corrupted);
    for (double value : {unknown, infinity, -infinity}) {
        corrupted = confirmed;
        corrupted.log_likelihood = value;
        reject(corrupted);
    }
    for (double value : {unknown, infinity, -0.1, 1.1}) {
        corrupted = confirmed;
        corrupted.change_probability = value;
        reject(corrupted);
    }
    corrupted = confirmed;
    corrupted.last_decision_at.reset();
    reject(corrupted);
    corrupted = confirmed;
    corrupted.cursor = 0;
    reject(corrupted);
    corrupted = confirmed;
    corrupted.cursor = 1;
    reject(corrupted);
    corrupted = pristine;
    corrupted.log_likelihood = 1;
    reject(corrupted);
    corrupted = pristine;
    corrupted.log_probabilities = {std::log(equal_probability), std::log(equal_probability)};
    reject(corrupted);
    filter.restore(pristine);
    require(filter.snapshot() == pristine, "Debe poder recuperarse el prior sin observaciones");
}

void state_permutation_preserves_the_likelihood_and_change() {
    const auto model = parameters();
    auto permuted = model;
    std::reverse(permuted.prior.begin(), permuted.prior.end());
    std::reverse(permuted.transitions.begin(), permuted.transitions.end());
    for (std::size_t feature = 0; feature < model.dimensions; ++feature) {
        std::swap(permuted.means[feature], permuted.means[model.dimensions + feature]);
        std::swap(permuted.variances[feature], permuted.variances[model.dimensions + feature]);
    }
    MarkovFilter first(model);
    MarkovFilter second(permuted);
    for (const auto& frame : frames()) {
        first.step(frame.observation());
        second.step(frame.observation());
        near(first.probabilities()[0], second.probabilities()[1],
             "Permutar los estados solo debe permutar sus probabilidades");
        near(first.log_likelihood(), second.log_likelihood(),
             "Permutar los estados no puede cambiar la verosimilitud");
        near(first.change_probability(), second.change_probability(),
             "Permutar los estados no puede cambiar su probabilidad de cambio");
    }
}

void dimensions_are_bounded_and_parameters_are_frozen() {
    auto model = parameters();
    MarkovFilter frozen(model);
    constexpr double changed_mean = 100;
    model.means[0] = changed_mean;
    require(frozen.snapshot().parameters == parameters(), "El filtro debe poseer parámetros congelados");

    model = {1, 1, {1}, {1}, {0}, {1}};
    MarkovFilter single(model);
    const std::array<double, 1> value{0};
    const std::array<uint8_t, 1> present{1};
    const std::array<int64_t, 1> earliest{std::numeric_limits<int64_t>::min()};
    single.step({value, present, earliest, earliest[0]});
    near(single.probabilities()[0], 1, "Un estado conserva probabilidad uno");
    near(single.change_probability(), 0, "Un solo estado no puede cambiar");
    near(single.log_likelihood(), -std::log(2 * std::numbers::pi) / 2,
         "El primer paso debe incluir la densidad de la observación");

    constexpr std::size_t states = 16;
    constexpr std::size_t dimensions = 64;
    model = {states, dimensions, std::vector<double>(states, 1.0 / states),
             std::vector<double>(states * states, 1.0 / states),
             std::vector<double>(states * dimensions, 0),
             std::vector<double>(states * dimensions, 1)};
    MarkovFilter largest(model);
    const std::vector<double> values(dimensions, 0);
    const std::vector<uint8_t> mask(dimensions, 1);
    const std::vector<int64_t> dates(dimensions, 0);
    largest.step({values, mask, dates, 0});
    largest.step({values, mask, dates, 1});
    for (double probability : largest.probabilities()) {
        near(probability, 1.0 / states, "Las dimensiones máximas deben filtrar todos los estados");
    }
    near(largest.change_probability(), 1.0 - 1.0 / states,
         "El cambio uniforme debe incluir todos los pares distintos");
    auto exhausted = largest.snapshot();
    exhausted.cursor = std::numeric_limits<std::size_t>::max();
    largest.restore(exhausted);
    rejected([&] { largest.step({values, mask, dates, 2}); }, "El cursor no puede desbordarse");
    require(largest.snapshot() == exhausted, "El cursor agotado no puede modificar el estado");
}
}

int main() {
    try {
        matches_exact_path_enumeration();
        future_suffix_does_not_change_the_prefix();
        missing_features_only_predict_and_ignore_their_payload();
        invalid_observations_leave_the_state_unchanged();
        invalid_parameters_fail_before_filtering();
        logarithmic_state_survives_extreme_underflow_and_recovery();
        unreachable_emissions_do_not_set_the_numerical_scale();
        nearby_means_remain_distinguishable_far_from_the_observation();
        extreme_relative_emissions_choose_the_reachable_state();
        opposite_large_means_preserve_the_small_centered_observation();
        nearby_variances_remain_distinguishable_at_large_scale();
        unresolved_relative_cancellation_leaves_the_state_unchanged();
        snapshot_validation_is_atomic_and_binds_the_model();
        state_permutation_preserves_the_likelihood_and_change();
        dimensions_are_bounded_and_parameters_are_frozen();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Filtro causal, disponibilidad, estabilidad y recuperación comprobados\n";
    return 0;
}
