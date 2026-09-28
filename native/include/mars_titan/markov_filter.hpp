#ifndef MARS_TITAN_MARKOV_FILTER_HPP
#define MARS_TITAN_MARKOV_FILTER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#if __has_cpp_attribute(clang::lifetimebound)
#define MARS_TITAN_MARKOV_LIFETIME_BOUND [[clang::lifetimebound]]
#elif __has_cpp_attribute(msvc::lifetimebound)
#define MARS_TITAN_MARKOV_LIFETIME_BOUND [[msvc::lifetimebound]]
#else
#define MARS_TITAN_MARKOV_LIFETIME_BOUND
#endif

namespace mars_titan::simulation {

inline constexpr std::size_t maximum_markov_states = 16;
inline constexpr std::size_t maximum_markov_dimensions = 64;

struct MarkovParameters {
    std::size_t states = 0;
    std::size_t dimensions = 0;
    std::vector<double> prior;
    // Matrices por filas: origen y destino para transiciones, estado y variable para emisiones.
    // Se admite error absoluto de 1e-12 en las sumas y se normaliza al construir el filtro.
    std::vector<double> transitions;
    std::vector<double> means;
    std::vector<double> variances;
    bool operator==(const MarkovParameters&) const = default;
};

struct MarkovObservation {
    std::span<const double> values;
    // Cero indica ausencia. Su valor y su fecha no se consumen.
    std::span<const uint8_t> present;
    std::span<const int64_t> available_at;
    int64_t decision_at = 0;
};

struct MarkovSnapshot {
    MarkovParameters parameters;
    // Conserva colas que no se pueden representar como probabilidades double.
    std::vector<double> log_probabilities;
    std::size_t cursor = 0;
    std::optional<int64_t> last_decision_at;
    double log_likelihood = 0;
    double change_probability = 0;
    bool operator==(const MarkovSnapshot&) const = default;
};

// Filtrado hacia delante con parámetros aportados por el consumidor, sin ajuste ni suavizado.
class MarkovFilter {
public:
    explicit MarkovFilter(const MarkovParameters& parameters);

    // El primer paso condiciona el prior. Los posteriores aplican una transición y una emisión.
    // Los errores conservan el estado confirmado. Los logaritmos deben caber en double.
    // Las diferencias de emisiones mal condicionadas se rechazan de forma explícita.
    void step(const MarkovObservation& observation);
    [[nodiscard]] std::span<const double>
    probabilities() const & MARS_TITAN_MARKOV_LIFETIME_BOUND;
    std::span<const double> probabilities() const && = delete;
    [[nodiscard]] std::size_t cursor() const noexcept;
    [[nodiscard]] double log_likelihood() const noexcept;
    // P(estado actual distinto al anterior | observaciones hasta ahora), cero en el primer paso.
    [[nodiscard]] double change_probability() const noexcept;
    [[nodiscard]] MarkovSnapshot snapshot() const;
    // Comprueba parámetros, dimensiones y coherencia del estado antes de confirmarlo.
    void restore(const MarkovSnapshot& snapshot);

private:
    MarkovParameters parameters_;
    std::array<double, maximum_markov_states * maximum_markov_states> log_transitions_{};
    std::array<double, maximum_markov_states> log_prior_{};
    std::array<double, maximum_markov_states> log_probabilities_{};
    std::array<double, maximum_markov_states> probabilities_{};
    std::size_t cursor_ = 0;
    std::optional<int64_t> last_decision_at_;
    double log_likelihood_ = 0;
    double change_probability_ = 0;
};

}

#undef MARS_TITAN_MARKOV_LIFETIME_BOUND

#endif
