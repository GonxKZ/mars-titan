#ifndef MARS_TITAN_CANDIDATE_HPP
#define MARS_TITAN_CANDIDATE_HPP

#include <ATen/core/Tensor.h>
#include <torch/nn/module.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace mars_titan::candidate {
inline constexpr int64_t modality_count = 5;
inline constexpr int64_t hidden_width = 128;
inline constexpr int64_t feature_width = 256;
inline constexpr int64_t quantile_count = 5;
inline constexpr int64_t price_window = 64;
inline constexpr double normalization_epsilon = 1e-12;
inline constexpr int64_t maximum_batch = 256;
inline constexpr int64_t maximum_episodes = 8192;
inline constexpr int64_t maximum_neighbors = 8;
inline constexpr std::size_t maximum_codec_bytes = std::size_t{64} << 20;

struct Config {
    static constexpr std::array<int64_t, modality_count> default_dimensions{5, 384, 512, 45, 420};
    static constexpr int64_t default_parameter_seed = 42;
    static constexpr int64_t default_feature_seed = 43;
    static constexpr int64_t default_key_seed = 44;
    std::array<int64_t, modality_count> dimensions = default_dimensions;
    int64_t max_batch = maximum_batch;
    int64_t max_episodes = maximum_episodes;
    int64_t neighbors = maximum_neighbors;
    int64_t parameter_seed = default_parameter_seed;
    int64_t feature_seed = default_feature_seed;
    int64_t key_seed = default_key_seed;
    double temperature = 1;
    std::string normalization_id;
    std::string input_policy = "strict_inputs_v1";
    bool operator==(const Config&) const = default;
};

struct Inputs {
    at::Tensor prices;
    at::Tensor news;
    at::Tensor charts;
    at::Tensor fundamentals;
    at::Tensor macro;
    at::Tensor presence;
};

class MemorySnapshot {
  public:
    [[nodiscard]] int64_t size() const;

  private:
    friend class Candidate;
    MemorySnapshot() = default;
    at::Tensor keys_;
    at::Tensor features_;
    at::Tensor returns_;
    at::Tensor ids_;
    std::string representation_id_;
};

struct Encoded {
    at::Tensor fused;
    at::Tensor episode_features;
    at::Tensor episode_keys;
};
struct Read {
    at::Tensor values;
    at::Tensor weights;
    at::Tensor ids;
    at::Tensor presence;
};
struct Prediction {
    at::Tensor quantiles;
    at::Tensor state;
    Encoded encoded;
    Read read;
};

struct EpisodeEncoding {
    at::Tensor keys;
    at::Tensor values;
};

class Candidate;

// Solo proyecciones CPU propias. No conserva el propietario ni parámetros aprendidos.
class CpuEpisodeCodec final {
  public:
    explicit CpuEpisodeCodec(const Candidate& source,
                             std::size_t max_working_bytes = maximum_codec_bytes);
    [[nodiscard]] EpisodeEncoding encode(const Inputs& inputs) const;
    [[nodiscard]] const std::string& projection_id() const noexcept;
    [[nodiscard]] std::size_t estimated_bytes(int64_t batch) const;
    [[nodiscard]] at::ScalarType dtype() const noexcept;

  private:
    Config config_;
    at::Tensor feature_projection_;
    at::Tensor key_projection_;
    std::string projection_id_;
    std::size_t max_working_bytes_;
};

// Cálculo puro. La admisión temporal y la publicación de memoria pertenecen al ejecutor.
class Candidate final : public torch::nn::Module {
  public:
    explicit Candidate(Config config, at::ScalarType dtype = at::kFloat,
                       const at::Device& device = at::Device(at::kCPU));
    [[nodiscard]] const Config& config() const noexcept;
    [[nodiscard]] std::string representation_id() const;
    [[nodiscard]] std::string parameter_fingerprint() const;
    // Traslado explícito a la política histórica, conservando sus proyecciones fijas.
    [[nodiscard]] std::vector<std::string> transfer_strict_parameters(const Candidate& source);
    [[nodiscard]] MemorySnapshot empty_memory() const;
    // Copia y separa del grafo una instantánea ya madura, con IDs crecientes únicos.
    [[nodiscard]] MemorySnapshot snapshot(const at::Tensor& keys, const at::Tensor& features,
                                          const at::Tensor& returns, const at::Tensor& ids,
                                          const std::string& representation_id) const;
    [[nodiscard]] Encoded encode(const Inputs& inputs) const;
    [[nodiscard]] at::Tensor encode_context(const Inputs& inputs) const;
    [[nodiscard]] std::shared_ptr<CpuEpisodeCodec>
    cpu_episode_codec(std::size_t max_working_bytes = maximum_codec_bytes) const;
    [[nodiscard]] at::Tensor initial_state(const at::Tensor& fused) const;
    [[nodiscard]] Read read(const at::Tensor& state, const MemorySnapshot& memory) const;
    [[nodiscard]] at::Tensor refine(const at::Tensor& state, const at::Tensor& fused,
                                    const Read& memory_read) const;
    [[nodiscard]] at::Tensor quantiles(const at::Tensor& state) const;
    [[nodiscard]] Prediction forward(const Inputs& inputs, const MemorySnapshot& memory,
                                     int64_t refinements = 1) const;
    void to(torch::Device device, torch::Dtype dtype, bool non_blocking = false) override;
    void to(torch::Dtype dtype, bool non_blocking = false) override;
    void to(torch::Device device, bool non_blocking = false) override;
    void save_state(std::ostream& destination) const;
    [[nodiscard]] static std::shared_ptr<Candidate>
    load_state(std::istream& source, const at::Device& device = at::Device(at::kCPU));
    [[nodiscard]] static std::shared_ptr<Candidate>
    load_state(std::istream& source, const at::Device& device, const std::string& expected_policy);

  private:
    friend class CpuEpisodeCodec;
    [[nodiscard]] static int64_t validate_inputs(const Inputs& inputs, const Config& config,
                                                 const at::Tensor& reference);
    [[nodiscard]] static at::Tensor flatten_inputs(const Inputs& inputs, const Config& config);
    [[nodiscard]] static at::Tensor normalize(const at::Tensor& value);
    [[nodiscard]] static EpisodeEncoding project_episodes(const at::Tensor& flattened,
                                                          const at::Tensor& feature_projection,
                                                          const at::Tensor& key_projection);
    void refresh_representation();
    struct Linear {
        at::Tensor weight;
        at::Tensor bias;
        [[nodiscard]] at::Tensor operator()(const at::Tensor& value) const;
    };
    [[nodiscard]] Linear linear(const std::string& name, int64_t in, int64_t out,
                                at::Generator& generator, at::ScalarType dtype);
    Config config_;
    std::vector<at::Tensor> gru_;
    std::array<Linear, modality_count - 1> modalities_;
    Linear fusion_;
    Linear initial_;
    Linear query_;
    Linear value_;
    Linear update_;
    Linear head_;
    at::Tensor step_logit_;
    at::Tensor feature_projection_;
    at::Tensor key_projection_;
    std::string representation_id_;
};
} // namespace mars_titan::candidate
#endif
