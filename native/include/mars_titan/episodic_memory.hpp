#ifndef MARS_TITAN_EPISODIC_MEMORY_HPP
#define MARS_TITAN_EPISODIC_MEMORY_HPP

#include <ATen/core/Tensor.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mars_titan::learning {
inline constexpr std::size_t episodic_memory_width = 64;
inline constexpr std::size_t episodic_memory_capacity = 1024;
inline constexpr std::size_t episodic_memory_neighbors = 4;
inline constexpr std::size_t maximum_retention_batch = 8192;
inline constexpr std::size_t maximum_episodic_archive_bytes = std::size_t{2} * 1024 * 1024;
inline constexpr std::size_t maximum_reservoir_capacity = 8192;
using MemoryVector = std::array<float, episodic_memory_width>;

struct MemoryScope {
    std::string world;
    std::string partition;
    std::string fold;
    std::string representation;
    uint64_t lane = 0;
    bool operator==(const MemoryScope&) const = default;
};

struct MemoryRecord {
    uint64_t id = 0;
    int64_t decision_at = 0;
    int64_t available_at = 0;
    int64_t maturity_at = 0;
    MemoryVector key{};
    MemoryVector value{};
    double reward = 0;
    double label = 0;
    bool reward_valid = false;
    bool label_valid = false;
    bool operator==(const MemoryRecord&) const = default;
};

struct MemoryNeighbor {
    MemoryRecord record;
    double similarity = 0;
};
struct MemoryQuery {
    std::array<MemoryNeighbor, episodic_memory_neighbors> neighbors{};
    std::size_t count = 0;
};

// Tensores CPU independientes: claves/valores FP32 [R,64], metadatos int64
// [R,6] y resultados FP64 [R,2]. El ámbito se guarda una vez por banco.
struct MemorySnapshot {
    MemoryScope scope;
    std::size_t capacity = episodic_memory_capacity;
    uint64_t seed = 0;
    uint64_t seen = 0;
    uint64_t last_id = 0;
    int64_t confirmed_at = 0;
    at::Tensor keys;
    at::Tensor values;
    at::Tensor metadata;
    at::Tensor outcomes;
    std::string reservoir_rng;
    std::uint32_t schema_version = 1;
};

class PreparedMemoryWrite {
  public:
    PreparedMemoryWrite(PreparedMemoryWrite&&) noexcept;
    PreparedMemoryWrite& operator=(PreparedMemoryWrite&&) noexcept;
    PreparedMemoryWrite(const PreparedMemoryWrite&) = delete;
    PreparedMemoryWrite& operator=(const PreparedMemoryWrite&) = delete;
    ~PreparedMemoryWrite();

  private:
    friend class EpisodicMemory;
    struct Impl;
    explicit PreparedMemoryWrite(std::unique_ptr<Impl> candidate);
    std::unique_ptr<Impl> impl_;
};

// Un banco por lane y episodio lógico. Un único controlador, sin acceso concurrente.
class EpisodicMemory {
  public:
    explicit EpisodicMemory(MemoryScope scope, uint64_t seed,
                            std::size_t capacity = episodic_memory_capacity);
    EpisodicMemory(MemoryScope scope, uint64_t seed, std::size_t capacity,
                   std::uint32_t schema_version);
    EpisodicMemory(const EpisodicMemory&) = delete;
    EpisodicMemory& operator=(const EpisodicMemory&) = delete;
    EpisodicMemory(EpisodicMemory&&) = delete;
    EpisodicMemory& operator=(EpisodicMemory&&) = delete;
    ~EpisodicMemory();

    // Exige IDs crecientes y resultados maduros. Normaliza una clave finita no nula.
    [[nodiscard]] PreparedMemoryWrite prepare_write(const MemoryRecord& record,
                                                    int64_t confirmed_at) const;
    // False identifica un token ajeno o caducado. Nunca cambia el banco al rechazarlo.
    [[nodiscard]] bool commit(PreparedMemoryWrite&& candidate) noexcept;
    [[nodiscard]] MemoryQuery query(const MemoryVector& key, int64_t cutoff,
                                    std::optional<uint64_t> exclude_id = std::nullopt) const;
    [[nodiscard]] MemoryQuery
    query_prepared(const MemoryVector& key, int64_t cutoff, const PreparedMemoryWrite& candidate,
                   std::optional<uint64_t> exclude_id = std::nullopt) const;
    [[nodiscard]] MemorySnapshot snapshot() const;
    void restore(const MemorySnapshot& snapshot);
    // Selecciona solo IDs de los registros actuales y del lote maduro. Sustitución atómica.
    void retain_batch(std::span<const MemoryRecord> incoming,
                      std::span<const uint64_t> retained_ids, int64_t confirmed_at);
    [[nodiscard]] std::vector<MemoryRecord> validate_batch(std::span<const MemoryRecord> incoming,
                                                           int64_t confirmed_at) const;
    [[nodiscard]] std::vector<MemoryRecord> retained_records() const;
    [[nodiscard]] const MemoryScope& scope() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] uint64_t seen() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::string serialize_memory(const MemorySnapshot& snapshot);
[[nodiscard]] MemorySnapshot deserialize_memory(std::string_view archive);
[[nodiscard]] MemoryVector normalize_memory_key(const MemoryVector& key);

// Sorteos del reservorio causal v2 para bancos de otra geometría. Una plaza -1 descarta el
// episodio. El estado es el texto canónico de mt19937_64 y la entrada nunca se modifica.
struct ReservoirDraws {
    std::vector<int64_t> slots;
    std::string state;
};
[[nodiscard]] std::string causal_reservoir_state(uint64_t seed);
[[nodiscard]] ReservoirDraws causal_reservoir_draws(std::string_view state, uint64_t seen,
                                                    std::size_t capacity, std::size_t count);
} // namespace mars_titan::learning
#endif
