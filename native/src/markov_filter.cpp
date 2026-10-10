#include "mars_titan/markov_filter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mars_titan::simulation {
namespace {
constexpr long double probability_tolerance = 1e-12L;
constexpr long double emission_tolerance = 1e-12L;
constexpr long double negative_infinity = -std::numeric_limits<long double>::infinity();
// Precisión declarada del control de cancelación: el épsilon del formato extendido de x86-64,
// con 64 bits de mantisa. En aarch64 long double es binary128 y su épsilon, mucho menor, aceptaría
// cancelaciones que x86-64 rechaza. Con la misma cota las dos arquitecturas aplican la misma regla
// y aarch64 calcula con más precisión de la exigida. Un long double menos preciso, como el double
// de MSVC, conserva su propio épsilon.
constexpr long double declared_epsilon =
    std::max(0x1p-63L, std::numeric_limits<long double>::epsilon());
using LogWeights = std::array<long double, maximum_markov_states>;

class EmissionSum {
public:
    void add(long double value) noexcept {
        const long double next = sum_ + value;
        correction_ += std::abs(sum_) >= std::abs(value) ? (sum_ - next) + value
                                                       : (value - next) + sum_;
        sum_ = next;
    }

    [[nodiscard]] long double value() const noexcept { return sum_ + correction_; }

private:
    long double sum_ = 0;
    long double correction_ = 0;
};

long double probability_sum(std::span<const double> probabilities) {
    long double sum = 0;
    for (double probability : probabilities) {
        if (!std::isfinite(probability) || probability < 0 || probability > 1) {
            throw std::invalid_argument("Las probabilidades deben ser finitas y estar entre cero y uno");
        }
        sum += static_cast<long double>(probability);
    }
    if (std::abs(sum - 1) > probability_tolerance) {
        throw std::invalid_argument("El prior y cada fila de transición deben sumar uno");
    }
    return sum;
}

MarkovParameters validated_parameters(const MarkovParameters& parameters) {
    if (parameters.states == 0 || parameters.states > maximum_markov_states ||
        parameters.dimensions == 0 || parameters.dimensions > maximum_markov_dimensions) {
        throw std::invalid_argument("El filtro admite entre 1 y 16 estados y entre 1 y 64 variables");
    }
    const std::size_t emission_size = parameters.states * parameters.dimensions;
    if (parameters.prior.size() != parameters.states ||
        parameters.transitions.size() != parameters.states * parameters.states ||
        parameters.means.size() != emission_size || parameters.variances.size() != emission_size) {
        throw std::invalid_argument("Las dimensiones de los parámetros del filtro no coinciden");
    }
    static_cast<void>(probability_sum(parameters.prior));
    for (std::size_t state = 0; state < parameters.states; ++state) {
        static_cast<void>(probability_sum(std::span(parameters.transitions)
                                              .subspan(state * parameters.states, parameters.states)));
    }
    for (std::size_t index = 0; index < emission_size; ++index) {
        if (!std::isfinite(parameters.means[index]) ||
            !std::isfinite(parameters.variances[index]) || parameters.variances[index] <= 0) {
            throw std::invalid_argument("Las medias deben ser finitas y las varianzas positivas finitas");
        }
    }
    return parameters;
}

long double log_sum(std::span<const long double> weights) {
    const long double maximum = *std::max_element(weights.begin(), weights.end());
    if (maximum == negative_infinity) {
        return maximum;
    }
    long double sum = 0;
    for (long double weight : weights) {
        sum += std::exp(weight - maximum);
    }
    return maximum + std::log(sum);
}

double finite_double(long double value, std::string_view message) {
    constexpr long double limit = static_cast<long double>(std::numeric_limits<double>::max());
    if (!std::isfinite(value) || value < -limit || value > limit) {
        throw std::overflow_error(std::string(message));
    }
    return static_cast<double>(value);
}

double normalized_log(double probability, long double total) {
    if (probability == 0) {
        return -std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(std::log(static_cast<long double>(probability) / total));
}

void validate_observation(const MarkovObservation& observation, std::size_t dimensions,
                          std::optional<int64_t> last_decision_at) {
    if (observation.values.size() != dimensions || observation.present.size() != dimensions ||
        observation.available_at.size() != dimensions) {
        throw std::invalid_argument("Las variables, máscaras y fechas deben tener la dimensión declarada");
    }
    if (last_decision_at.has_value() && observation.decision_at <= last_decision_at.value()) {
        throw std::invalid_argument("La fecha de decisión debe avanzar estrictamente");
    }
    for (std::size_t feature = 0; feature < dimensions; ++feature) {
        if (observation.present[feature] > 1) {
            throw std::invalid_argument("La máscara de presencia solo admite cero y uno");
        }
        if (observation.present[feature] != 0 &&
            (!std::isfinite(observation.values[feature]) ||
             observation.available_at[feature] > observation.decision_at)) {
            throw std::invalid_argument("Cada variable presente debe ser finita y estar disponible al decidir");
        }
    }
}

long double absolute_emission(const MarkovParameters& parameters,
                              const MarkovObservation& observation, std::size_t state) {
    EmissionSum density;
    const long double log_two_pi = std::log(2 * std::numbers::pi_v<long double>);
    for (std::size_t feature = 0; feature < parameters.dimensions; ++feature) {
        if (observation.present[feature] == 0) {
            continue;
        }
        const std::size_t index = state * parameters.dimensions + feature;
        const long double variance = static_cast<long double>(parameters.variances[index]);
        const long double difference = static_cast<long double>(observation.values[feature]) -
                                       static_cast<long double>(parameters.means[index]);
        const long double scaled = difference / std::sqrt(variance);
        density.add(-(log_two_pi + std::log(variance) + scaled * scaled) / 2);
    }
    if (!std::isfinite(density.value())) {
        throw std::overflow_error("La densidad de emisión excede el rango numérico disponible");
    }
    return density.value();
}

long double relative_emission(const MarkovParameters& parameters,
                              const MarkovObservation& observation,
                              std::size_t state, std::size_t reference) {
    if (state == reference) {
        return 0;
    }
    EmissionSum difference;
    long double magnitude = 0;
    for (std::size_t feature = 0; feature < parameters.dimensions; ++feature) {
        if (observation.present[feature] == 0) {
            continue;
        }
        const std::size_t index = state * parameters.dimensions + feature;
        const std::size_t reference_index = reference * parameters.dimensions + feature;
        const long double mean = static_cast<long double>(parameters.means[index]);
        const long double reference_mean = static_cast<long double>(parameters.means[reference_index]);
        const long double variance = static_cast<long double>(parameters.variances[index]);
        const long double reference_variance =
            static_cast<long double>(parameters.variances[reference_index]);
        const long double value = static_cast<long double>(observation.values[feature]);
        const long double variance_delta = variance - reference_variance;
        const long double reference_error = value - reference_mean;
        EmissionSum centered;
        centered.add(2 * value);
        centered.add(-mean);
        centered.add(-reference_mean);
        // Diferencia factorizada de cuadrados, sin restar dos densidades absolutas grandes.
        const long double mean_term = (mean - reference_mean) * centered.value() / variance;
        const long double variance_term =
            (variance_delta / variance) * (reference_error / reference_variance) * reference_error;
        const long double log_ratio = std::abs(variance_delta) <= variance / 2
                                          ? std::log1p(-variance_delta / variance)
                                          : std::log(reference_variance) - std::log(variance);
        for (long double term : {mean_term, variance_term, log_ratio}) {
            difference.add(term);
            magnitude += std::abs(term);
        }
    }
    const long double relative = difference.value() / 2;
    if (!std::isfinite(relative) || !std::isfinite(magnitude)) {
        throw std::overflow_error("La diferencia entre emisiones excede el rango numérico disponible");
    }
    constexpr long double roundoff_factor = 16;
    constexpr long double unit_log_scale = 1;
    const long double roundoff = roundoff_factor * declared_epsilon * magnitude / 2;
    // El control del condicionamiento evita confirmar diferencias dominadas por el redondeo.
    if (roundoff > emission_tolerance * std::max(unit_log_scale, std::abs(relative))) {
        throw std::runtime_error("La cancelación entre emisiones supera la precisión disponible");
    }
    return relative;
}
}

MarkovFilter::MarkovFilter(const MarkovParameters& parameters)
    : parameters_(validated_parameters(parameters)) {
    const long double prior_total = probability_sum(parameters_.prior);
    for (std::size_t state = 0; state < parameters_.states; ++state) {
        log_prior_.at(state) = normalized_log(parameters_.prior[state], prior_total);
        log_probabilities_.at(state) = log_prior_.at(state);
        probabilities_.at(state) = std::exp(log_prior_.at(state));
        const auto row = std::span(parameters_.transitions)
                             .subspan(state * parameters_.states, parameters_.states);
        const long double row_total = probability_sum(row);
        for (std::size_t next = 0; next < parameters_.states; ++next) {
            log_transitions_.at(state * parameters_.states + next) = normalized_log(row[next], row_total);
        }
    }
}

void MarkovFilter::step(const MarkovObservation& observation) {
    validate_observation(observation, parameters_.dimensions, last_decision_at_);
    if (cursor_ == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("El cursor del filtro ha agotado su rango");
    }
    const std::size_t states = parameters_.states;
    LogWeights predictions{};
    LogWeights change_given_destination{};
    LogWeights joint{};
    for (std::size_t next = 0; next < states; ++next) {
        predictions.at(next) = static_cast<long double>(log_prior_.at(next));
        if (cursor_ != 0) {
            LogWeights incoming{};
            for (std::size_t previous = 0; previous < states; ++previous) {
                incoming.at(previous) = static_cast<long double>(log_probabilities_.at(previous)) +
                                     static_cast<long double>(log_transitions_.at(previous * states + next));
            }
            const auto weights = std::span(incoming).first(states);
            const long double maximum = *std::max_element(weights.begin(), weights.end());
            predictions.at(next) = negative_infinity;
            if (maximum != negative_infinity) {
                long double total = 0;
                long double changed = 0;
                for (std::size_t previous = 0; previous < states; ++previous) {
                    const long double weight = std::exp(incoming.at(previous) - maximum);
                    total += weight;
                    if (previous != next) {
                        changed += weight;
                    }
                }
                predictions.at(next) = maximum + std::log(total);
                change_given_destination.at(next) = changed / total;
            }
        }
    }
    const auto prediction_span = std::span(predictions).first(states);
    auto reference = static_cast<std::size_t>(
        std::max_element(prediction_span.begin(), prediction_span.end()) - prediction_span.begin());
    for (std::size_t next = 0; next < states; ++next) {
        if (predictions.at(next) != negative_infinity &&
            relative_emission(parameters_, observation, next, reference) > 0) {
            reference = next;
        }
    }
    const long double largest_emission = absolute_emission(parameters_, observation, reference);
    for (std::size_t next = 0; next < states; ++next) {
        joint.at(next) = predictions.at(next) == negative_infinity
                             ? negative_infinity
                             : predictions.at(next) +
                                   relative_emission(parameters_, observation, next, reference);
    }
    const auto joint_span = std::span(joint).first(states);
    const long double largest_joint = *std::max_element(joint_span.begin(), joint_span.end());
    for (long double& weight : joint_span) {
        weight -= largest_joint;
    }
    const long double normalization = log_sum(std::span(joint).first(states));
    const double likelihood = finite_double(static_cast<long double>(log_likelihood_) +
                                                largest_emission + largest_joint + normalization,
                                            "La log verosimilitud acumulada no cabe en double");
    std::array<double, maximum_markov_states> next_logs{};
    std::array<double, maximum_markov_states> next_probabilities{};
    long double changed = 0;
    for (std::size_t state = 0; state < states; ++state) {
        next_logs.at(state) = joint.at(state) == negative_infinity
                               ? -std::numeric_limits<double>::infinity()
                               : finite_double(joint.at(state) - normalization,
                                               "El peso logarítmico del estado no cabe en double");
        next_probabilities.at(state) = std::exp(next_logs.at(state));
        changed += change_given_destination.at(state) *
                   static_cast<long double>(next_probabilities.at(state));
    }
    const double change = static_cast<double>(std::clamp(changed, 0.0L, 1.0L));
    log_probabilities_ = next_logs;
    probabilities_ = next_probabilities;
    log_likelihood_ = likelihood;
    change_probability_ = change;
    last_decision_at_ = observation.decision_at;
    ++cursor_;
}

std::span<const double> MarkovFilter::probabilities() const & {
    return std::span(probabilities_).first(parameters_.states);
}

std::size_t MarkovFilter::cursor() const noexcept { return cursor_; }
double MarkovFilter::log_likelihood() const noexcept { return log_likelihood_; }
double MarkovFilter::change_probability() const noexcept { return change_probability_; }

MarkovSnapshot MarkovFilter::snapshot() const {
    const auto logs = std::span(log_probabilities_).first(parameters_.states);
    return {parameters_, {logs.begin(), logs.end()}, cursor_, last_decision_at_,
            log_likelihood_, change_probability_};
}

void MarkovFilter::restore(const MarkovSnapshot& snapshot) {
    if (snapshot.parameters != parameters_ || snapshot.log_probabilities.size() != parameters_.states ||
        !std::isfinite(snapshot.log_likelihood) || !std::isfinite(snapshot.change_probability) ||
        snapshot.change_probability < 0 || snapshot.change_probability > 1 ||
        (snapshot.cursor == 0) != !snapshot.last_decision_at.has_value() ||
        (snapshot.cursor <= 1 && snapshot.change_probability != 0)) {
        throw std::invalid_argument("La identidad, el cursor o las métricas del estado no son válidos");
    }
    LogWeights logs{};
    std::array<double, maximum_markov_states> restored_logs{};
    std::array<double, maximum_markov_states> restored_probabilities{};
    for (std::size_t state = 0; state < parameters_.states; ++state) {
        const double value = snapshot.log_probabilities[state];
        if (std::isnan(value) || value > 0 ||
            (snapshot.cursor == 0 && value != log_prior_.at(state))) {
            throw std::invalid_argument("Los pesos logarítmicos recuperados no son válidos");
        }
        logs.at(state) = static_cast<long double>(value);
        restored_logs.at(state) = value;
        restored_probabilities.at(state) = std::exp(value);
    }
    const long double normalization = log_sum(std::span(logs).first(parameters_.states));
    if (!std::isfinite(normalization) || std::abs(normalization) > probability_tolerance ||
        (snapshot.cursor == 0 && snapshot.log_likelihood != 0)) {
        throw std::invalid_argument("El estado recuperado no está normalizado o su prior fue modificado");
    }
    log_probabilities_ = restored_logs;
    probabilities_ = restored_probabilities;
    cursor_ = snapshot.cursor;
    last_decision_at_ = snapshot.last_decision_at;
    log_likelihood_ = snapshot.log_likelihood;
    change_probability_ = snapshot.change_probability;
}
}
