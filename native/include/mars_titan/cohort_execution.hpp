#ifndef MARS_TITAN_COHORT_EXECUTION_HPP
#define MARS_TITAN_COHORT_EXECUTION_HPP

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
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
struct Definition {
    Identity identity;
    std::vector<Task> tasks;
    Json initial_state;
    Limits limits;
};
struct Observation {
    std::string asset;
    std::int64_t available_at = 0;
    std::vector<double> features;
};
struct Cohort {
    std::size_t cursor = 0;
    std::int64_t cutoff = 0;
    std::vector<Observation> observations;
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
struct Callbacks {
    // Funciones deterministas. Todo el estado del consumidor debe estar en el JSON explícito.
    std::function<std::vector<double>(std::span<const Observation>, const Task&, const Json&)>
        predict;
    std::function<Json(const Json&, std::span<const ResolvedFeedback>)> update;
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
