#ifndef MARS_TITAN_ADAPTER_CONTROL_HPP
#define MARS_TITAN_ADAPTER_CONTROL_HPP

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mars_titan::controls {
struct SemanticVersions {
    uint64_t view = 1;
    uint64_t representation = 1;
    uint64_t keys = 1;
    uint64_t query = 1;
    uint64_t output = 1;
    bool operator==(const SemanticVersions&) const = default;
};
enum class ArtifactKind : uint8_t { representation, memory, read, prediction };
struct InvalidatedArtifacts {
    bool representations = false;
    bool memory = false;
    bool reads = false;
    bool predictions = false;
};
[[nodiscard]] InvalidatedArtifacts invalidated(const SemanticVersions& before,
                                               const SemanticVersions& after);
void require_compatible(ArtifactKind artifact, const SemanticVersions& stored,
                        const SemanticVersions& current);

enum class AdapterKind : uint8_t { full, residual, low_rank };
struct AdapterConfig {
    std::string problem_id = "matrix-control-v1";
    std::string validation_id = "matrix-validation-v1";
    AdapterKind kind = AdapterKind::full;
    std::size_t inputs = 4;
    std::size_t outputs = 3;
    std::size_t rank = 2;
    uint64_t seed = 71;
    double learning_rate = 0.05;
    double momentum = 0.5;
    double minimum_improvement = 0;
    std::size_t max_steps = 64;
    std::size_t max_rows = 4096;
    std::size_t max_bytes = 128U << 20;
    SemanticVersions versions{};
    bool operator==(const AdapterConfig&) const = default;
};
struct AdapterSnapshot {
    AdapterConfig config;
    at::Tensor base;
    at::Tensor validation_inputs;
    at::Tensor validation_targets;
    std::vector<at::Tensor> parameters;
    std::vector<at::Tensor> momentum;
    std::vector<at::Tensor> best_parameters;
    std::size_t steps = 0;
    std::size_t best_step = 0;
    double best_mse = 0;
};

// Control técnico de una proyección de salida. Los codificadores y las claves son externos.
class AdapterControl {
  public:
    AdapterControl(AdapterConfig config, const at::Tensor& base,
                   const at::Tensor& validation_inputs, const at::Tensor& validation_targets,
                   std::string_view device = "cpu");
    AdapterControl(const AdapterControl&) = delete;
    AdapterControl& operator=(const AdapterControl&) = delete;
    AdapterControl(AdapterControl&&) noexcept;
    AdapterControl& operator=(AdapterControl&&) noexcept;
    ~AdapterControl();

    [[nodiscard]] at::Tensor predict(const at::Tensor& inputs, bool selected = false) const;
    [[nodiscard]] double train(const at::Tensor& inputs, const at::Tensor& targets);
    [[nodiscard]] AdapterSnapshot snapshot() const;
    void restore(const AdapterSnapshot& state);
    [[nodiscard]] SemanticVersions versions(bool selected = false) const;
    [[nodiscard]] std::size_t trainable_parameters() const noexcept;
    [[nodiscard]] std::size_t state_tensor_bytes() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
[[nodiscard]] std::string serialize_adapter(const AdapterSnapshot& state);
[[nodiscard]] AdapterSnapshot deserialize_adapter(std::string_view archive);
[[nodiscard]] std::string_view adapter_kind_name(AdapterKind kind);
} // namespace mars_titan::controls
#endif
