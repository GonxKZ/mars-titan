#ifndef MARS_TITAN_REPLAY_SCHEDULE_HPP
#define MARS_TITAN_REPLAY_SCHEDULE_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mars_titan::controls {
enum class ReplayOrder : uint8_t { uniform, recent, spaced };

struct MatureEpisode {
    uint64_t id = 0;
    uint64_t observed_at = 0;
    uint64_t available_at = 0;
    uint32_t exposures = 1;
    bool operator==(const MatureEpisode&) const = default;
};

struct ReplayScheduleConfig {
    std::string run_id;
    ReplayOrder order = ReplayOrder::uniform;
    uint64_t cutoff = 0;
    uint64_t order_seed = 0;
    std::size_t batch_size = 1;
    std::size_t minimum_distance = 1;
    std::size_t max_exposures = 1U << 20;
    std::size_t max_bytes = 128U << 20;
    bool operator==(const ReplayScheduleConfig&) const = default;
};

struct ReplayToken {
    std::string run_id;
    ReplayOrder order = ReplayOrder::uniform;
    uint64_t order_seed = 0;
    std::size_t cursor = 0;
    std::size_t count = 0;
    std::size_t update = 0;
    bool operator==(const ReplayToken&) const = default;
};

struct PreparedReplay {
    ReplayToken token;
    std::span<const uint32_t> indices;
};

struct ReplayScheduleSnapshot {
    ReplayScheduleConfig config;
    std::vector<MatureEpisode> episodes;
    std::size_t cursor = 0;
    std::size_t updates = 0;
};

// Un propietario confirma cada lote después de guardar el estado coherente del consumidor.
class ReplaySchedule {
  public:
    ReplaySchedule(ReplayScheduleConfig config, std::span<const MatureEpisode> episodes);
    [[nodiscard]] PreparedReplay prepare() const;
    void commit(const ReplayToken& token);
    [[nodiscard]] ReplayScheduleSnapshot snapshot() const;
    void restore(const ReplayScheduleSnapshot& state);
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t cursor() const noexcept;
    [[nodiscard]] std::size_t updates() const noexcept;
    [[nodiscard]] std::size_t index_bytes() const noexcept;

  private:
    ReplayScheduleConfig config_;
    std::vector<MatureEpisode> episodes_;
    std::vector<uint32_t> indices_;
    std::size_t cursor_ = 0;
    std::size_t updates_ = 0;
};

[[nodiscard]] std::string serialize_schedule(const ReplayScheduleSnapshot& state);
[[nodiscard]] ReplayScheduleSnapshot deserialize_schedule(std::string_view archive);
[[nodiscard]] std::string_view replay_order_name(ReplayOrder order);
} // namespace mars_titan::controls
#endif
