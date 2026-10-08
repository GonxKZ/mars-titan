#ifndef MARS_TITAN_COHORT_EXECUTION_HPP
#define MARS_TITAN_COHORT_EXECUTION_HPP

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mars_titan::cohorts {
using Json = nlohmann::json;
inline constexpr std::size_t mebibyte = std::size_t{1024} * 1024;
inline constexpr std::size_t default_max_assets = 4096;
inline constexpr std::size_t default_max_pending = 32768;
inline constexpr std::size_t default_file_bytes = 16 * mebibyte;
inline constexpr std::size_t default_log_bytes = 1024 * mebibyte;
inline constexpr std::size_t default_max_cohorts = 100000;
inline constexpr std::size_t default_batch_rows = 64;
inline constexpr std::size_t default_history_reads = 4096;

struct Identity {
    std::string source_sha256;
    std::string view_sha256;
    std::string representation_sha256;
    std::string model_sha256;
};
struct Task {
    std::string name;
    std::uint32_t horizon = 1;
    bool operator==(const Task&) const = default;
};
struct Limits {
    std::size_t feature_width = 1;
    std::size_t max_assets = default_max_assets;
    std::size_t max_pending = default_max_pending;
    std::size_t max_state_bytes = mebibyte;
    std::size_t max_checkpoint_bytes = default_file_bytes;
    std::size_t max_record_bytes = default_file_bytes;
    std::size_t max_log_bytes = default_log_bytes;
    std::size_t max_cohorts = default_max_cohorts;
};
enum class PredictionMode : std::uint8_t { stateless, prepared, financial };
enum class EventKind : std::uint8_t { warmup, decision, settlement };
enum class PrefixReason : std::uint8_t { insufficient_pairs, zero_market_variance };
struct PhaseContract {
    std::string partition;
    std::int64_t warmup_start = 0;
    std::int64_t decision_start = 0;
    std::int64_t decision_end = 0;
    std::int64_t close_at = 0;
    std::string prefix_policy_sha256;
};
struct Definition {
    Identity identity;
    std::vector<Task> tasks;
    Json initial_state;
    Limits limits;
    PredictionMode prediction_mode = PredictionMode::stateless;
    PhaseContract phase = {};
};
struct Observation {
    std::string asset;
    std::int64_t available_at = 0;
    std::vector<double> features;
};
struct PrefixExclusion {
    std::string asset;
    Task task;
    std::int64_t decision_at = 0;
    PrefixReason reason = PrefixReason::insufficient_pairs;
    std::size_t history_pairs = 0;
    std::optional<double> market_variance;
    std::string evidence_sha256;
};
struct Cohort {
    std::size_t cursor = 0;
    std::int64_t cutoff = 0;
    std::vector<Observation> observations;
    EventKind kind = EventKind::decision;
    bool close_phase = false;
    std::vector<PrefixExclusion> prefix_exclusions = {};
};
struct Prediction {
    std::string id;
    std::string asset;
    Task task;
    std::size_t generation = 0;
    std::int64_t decision_at = 0;
    double value = 0;
};
struct Feedback {
    std::string prediction_id;
    std::uint32_t revision = 0;
    std::int64_t available_at = 0;
    double value = 0;
};
struct ResolvedFeedback {
    std::string id;
    Prediction prediction;
    Feedback label;
};
struct ResolvedPrefixExclusion {
    Prediction prediction;
    PrefixExclusion evidence;
};
struct AdministrativeFinalization {
    Prediction prediction;
    std::int64_t closed_at = 0;
};
struct PreparedCohort {
    // Orden canónico activo/tarea. Una propuesta compartida por toda la cohorte.
    std::vector<double> values;
    Json proposed_state;
};
struct Callbacks {
    // Funciones deterministas. Todo el estado del consumidor debe estar en el JSON explícito.
    std::function<std::vector<double>(std::span<const Observation>, const Task&, const Json&)>
        predict;
    std::function<Json(const Json&, std::span<const ResolvedFeedback>)> update;
    // Se elige prepare o predict. Los tensores externos necesitan referencias verificables.
    std::function<PreparedCohort(std::span<const Observation>, std::span<const Task>, std::int64_t,
                                 const Json&, std::size_t)>
        prepare = {};
    std::function<PreparedCohort(EventKind, std::span<const Observation>, std::span<const Task>,
                                 std::int64_t, const Json&, std::size_t)>
        prepare_event = {};
    std::function<Json(const Json&, std::span<const ResolvedFeedback>,
                       std::span<const ResolvedPrefixExclusion>,
                       std::span<const AdministrativeFinalization>)>
        resolve = {};
};
enum class Boundary : std::uint8_t {
    before_predictions,
    predictions_ready,
    record_written,
    feedback_applied,
    checkpoint_written,
    before_commit,
    committed
};
struct Commit {
    std::size_t generation = 0;
    std::vector<Prediction> predictions;
    std::vector<ResolvedFeedback> applied;
    std::string record_sha256;
    std::vector<ResolvedPrefixExclusion> excluded = {};
    std::vector<AdministrativeFinalization> finalized = {};
};

[[nodiscard]] std::string replay_id(const ResolvedFeedback& feedback, std::uint64_t visit);

class Executor {
  public:
    Executor(const std::filesystem::path& output, Definition definition, Callbacks callbacks,
             bool resume = false);
    ~Executor();
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    Executor(Executor&&) = delete;
    Executor& operator=(Executor&&) = delete;

    Commit step(const Cohort& cohort, std::span<const Feedback> mature_feedback,
                std::size_t batch_rows = default_batch_rows,
                const std::function<void(Boundary)>& fault = {});
    [[nodiscard]] std::size_t cursor() const;
    [[nodiscard]] Json snapshot() const;
    [[nodiscard]] std::vector<Prediction> pending() const;
    // La cadena se verifica desde el último registro confirmado, con un máximo explícito.
    [[nodiscard]] Json record(std::size_t generation,
                              std::size_t max_hops = default_history_reads) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace mars_titan::cohorts

#endif
