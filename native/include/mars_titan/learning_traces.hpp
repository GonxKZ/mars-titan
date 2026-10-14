#ifndef MARS_TITAN_LEARNING_TRACES_HPP
#define MARS_TITAN_LEARNING_TRACES_HPP

#include <nlohmann/json.hpp>

#include <compare>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace mars_titan::learning {
// Mismo contrato que `mars_titan.learning_traces.recorder` en Python (#448): manifiesto con
// versión, identidad, configuración, unidad del paso y partes Parquet numeradas con las
// columnas step, phase, metric, group, stat y value.
inline constexpr std::size_t minimum_learning_trace_bytes = 1024;
inline constexpr std::size_t maximum_learning_trace_bytes = std::size_t{1024} * 1024 * 1024;
// Filas pendientes a partir de las que se vuelca sin esperar a la confirmación siguiente.
inline constexpr std::size_t maximum_pending_trace_rows = 65536;

struct LearningTraceOptions {
    // Cero: sin trazas. Si no, cada cuántas unidades del paso se registra una actualización.
    std::size_t every = 0;
    std::size_t max_bytes = 0;
    [[nodiscard]] bool enabled() const noexcept { return every != 0; }
};

// Marca de confirmación de una parte. Una parte es válida si su marca no supera la del estado
// recuperado, y las marcas de las partes crecen en orden. PPO usa (transiciones,
// evaluaciones) y KLPO (oleada, 0 para la actualización y 1 para su validación).
struct TraceCommit {
    uint64_t major = 0;
    uint64_t minor = 0;
    auto operator<=>(const TraceCommit&) const = default;
};

class LearningTraceWriter {
  public:
    // Crea la carpeta o la recupera. Al recuperar exige la misma identidad, configuración y
    // unidad, borra las partes cuya marca supera `confirmed` y las que no llegaron a entrar en
    // el manifiesto, y comprueba el tamaño de las que quedan.
    LearningTraceWriter(std::filesystem::path directory, LearningTraceOptions options,
                        nlohmann::json identity, std::string step_unit, TraceCommit confirmed);
    ~LearningTraceWriter();
    LearningTraceWriter(const LearningTraceWriter&) = delete;
    LearningTraceWriter& operator=(const LearningTraceWriter&) = delete;
    LearningTraceWriter(LearningTraceWriter&&) = delete;
    LearningTraceWriter& operator=(LearningTraceWriter&&) = delete;

    [[nodiscard]] bool active() const noexcept;
    // Si el avance de `previous` a `current` cruza un múltiplo de la cadencia.
    [[nodiscard]] bool due(uint64_t previous, uint64_t current) const noexcept;
    void scalar(uint64_t step, std::string_view phase, std::string_view metric, double value,
                std::string_view group = "all");
    [[nodiscard]] std::size_t pending() const noexcept;
    // Escribe lo pendiente como una parte nueva con la marca de la confirmación que la sigue.
    // Si la parte no cabe en el presupuesto, no se escribe y las trazas quedan agotadas.
    void flush(TraceCommit commit);
    [[nodiscard]] std::optional<uint64_t> exhausted_at() const noexcept;
    [[nodiscard]] std::size_t parts() const noexcept;
    [[nodiscard]] std::size_t bytes() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace mars_titan::learning
#endif
