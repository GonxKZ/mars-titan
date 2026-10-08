#include "mars_titan/episodic_memory.hpp"
#include "accurate_sum.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/util/ScopeExit.h>
#include <torch/serialize/archive.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <locale>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mars_titan::learning {
namespace {
constexpr std::size_t maximum_scope_bytes = 128;
constexpr std::size_t maximum_rng_bytes = 8192;
constexpr uint64_t maximum_seen = uint64_t{1} << 32;
constexpr double unit_norm_tolerance = 1e-6;
constexpr int64_t snapshot_version = 1;
constexpr int64_t metadata_width = 6;
constexpr int64_t outcome_width = 2;
constexpr int64_t id_column = 0;
constexpr int64_t decision_column = 1;
constexpr int64_t available_column = 2;
constexpr int64_t maturity_column = 3;
constexpr int64_t reward_valid_column = 4;
constexpr int64_t label_valid_column = 5;
using ReservoirEngine = std::mt19937_64;
using QueryRow = std::array<double, episodic_memory_width>;
static_assert(std::is_trivially_copyable_v<MemoryRecord>);
static_assert(std::is_nothrow_copy_assignable_v<ReservoirEngine>);
// La admisión del contexto reserva hasta 2 MiB por banco.
static_assert(sizeof(MemoryRecord) + sizeof(QueryRow) <=
              maximum_episodic_archive_bytes / episodic_memory_capacity);

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

void validate_scope(const MemoryScope& scope) {
    for (const auto* text : {&scope.world, &scope.partition, &scope.fold, &scope.representation}) {
        require(!text->empty() && text->size() <= maximum_scope_bytes &&
                    std::all_of(text->begin(), text->end(),
                                [](unsigned char character) {
                                    return character >= '!' && character <= '~';
                                }),
                "El ámbito episódico necesita identificadores ASCII no vacíos y acotados");
    }
    require(scope.partition == "train" || scope.partition == "validation",
            "La memoria episódica solo admite entrenamiento o validación");
}

MemoryVector normalize(const MemoryVector& input) {
    double scale = 0;
    for (const float value : input) {
        require(std::isfinite(value), "La clave episódica contiene NaN o infinito");
        scale = std::max(scale, std::abs(static_cast<double>(value)));
    }
    require(scale > 0, "La clave episódica no puede tener norma cero");
    simulation::AccurateSum squared;
    for (const float value : input) {
        const double scaled = static_cast<double>(value) / scale;
        require(squared.add(scaled * scaled), "La norma episódica no cabe en FP64");
    }
    const double norm = std::sqrt(squared.value());
    MemoryVector result{};
    for (std::size_t index = 0; index < input.size(); ++index) {
        result.at(index) =
            static_cast<float>((static_cast<double>(input.at(index)) / scale) / norm);
    }
    return result;
}

void validate_record(const MemoryRecord& record, int64_t confirmed_at, bool normalized) {
    require(record.id > 0 &&
                record.id <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
                record.decision_at >= 0 && record.available_at >= 0 &&
                record.available_at <= record.decision_at &&
                record.maturity_at > record.decision_at && record.maturity_at <= confirmed_at,
            "El episodio no conserva ID, disponibilidad y maduración confirmada");
    require((record.reward_valid || record.label_valid) && std::isfinite(record.reward) &&
                std::isfinite(record.label) && (record.reward_valid || record.reward == 0) &&
                (record.label_valid || record.label == 0),
            "El episodio necesita un resultado maduro y máscaras coherentes");
    require(std::all_of(record.value.begin(), record.value.end(),
                        [](float value) { return std::isfinite(value); }),
            "El valor episódico contiene NaN o infinito");
    simulation::AccurateSum squared;
    for (const float component : record.key) {
        require(std::isfinite(component), "La clave episódica contiene NaN o infinito");
        const double value = static_cast<double>(component);
        require(squared.add(value * value), "La norma episódica no cabe en FP64");
    }
    require(squared.value() > 0 &&
                (!normalized || std::abs(squared.value() - 1) <= unit_norm_tolerance),
            "La clave guardada no conserva una norma unitaria");
}

ReservoirEngine initial_rng(uint64_t seed, const MemoryScope& scope) {
    constexpr unsigned int word_bits = 32;
    std::vector<uint32_t> words{
        static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> word_bits),
        static_cast<uint32_t>(scope.lane), static_cast<uint32_t>(scope.lane >> word_bits)};
    for (const auto* text : {&scope.world, &scope.partition, &scope.fold, &scope.representation}) {
        words.push_back(static_cast<uint32_t>(text->size()));
        for (const char character : *text) {
            words.push_back(static_cast<unsigned char>(character));
        }
    }
    std::seed_seq sequence(words.begin(), words.end());
    return ReservoirEngine(sequence);
}

std::string encode_rng(const ReservoirEngine& engine) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << engine;
    auto result = stream.str();
    require(stream.good() && result.size() <= maximum_rng_bytes,
            "El estado del reservorio excede el presupuesto");
    return result;
}

ReservoirEngine decode_rng(const std::string& bytes) {
    require(!bytes.empty() && bytes.size() <= maximum_rng_bytes,
            "El estado del reservorio excede el presupuesto");
    ReservoirEngine engine;
    std::istringstream stream(bytes);
    stream.imbue(std::locale::classic());
    stream >> engine;
    require(!stream.fail() && encode_rng(engine) == bytes,
            "El estado aleatorio del reservorio no tiene un formato canónico válido");
    return engine;
}

uint64_t choose_slot(ReservoirEngine& engine, uint64_t bound) {
    // El rechazo elimina el sesgo del resto sin depender de una distribución de la STL.
    const uint64_t threshold = (uint64_t{0} - bound) % bound;
    uint64_t value = engine();
    while (value < threshold) {
        value = engine();
    }
    return value % bound;
}

at::Tensor vector_tensor(const MemoryVector& values) {
    return at::tensor(at::ArrayRef<float>(values.data(), values.size()),
                      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
}

void tensor_shape(const at::Tensor& tensor, at::ScalarType dtype, int64_t rows, int64_t width) {
    require(tensor.defined() && tensor.device().is_cpu() && tensor.layout() == at::kStrided &&
                tensor.scalar_type() == dtype && tensor.dim() == 2 && tensor.size(0) == rows &&
                tensor.size(1) == width && tensor.is_contiguous() && !tensor.requires_grad(),
            "El tensor episódico no conserva dispositivo CPU, tipo, forma o contigüidad");
}

std::vector<MemoryRecord> snapshot_records(const MemorySnapshot& snapshot) {
    validate_scope(snapshot.scope);
    require(snapshot.capacity > 0 && snapshot.capacity <= episodic_memory_capacity &&
                snapshot.seen <= maximum_seen && snapshot.last_id >= snapshot.seen &&
                snapshot.last_id <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
                snapshot.confirmed_at >= 0 &&
                (snapshot.seen != 0 || (snapshot.last_id == 0 && snapshot.confirmed_at == 0)),
            "El snapshot episódico no conserva capacidad, contadores o tiempo");
    const auto count =
        static_cast<std::size_t>(std::min(snapshot.seen, static_cast<uint64_t>(snapshot.capacity)));
    const auto rows = static_cast<int64_t>(count);
    tensor_shape(snapshot.keys, at::kFloat, rows, static_cast<int64_t>(episodic_memory_width));
    tensor_shape(snapshot.values, at::kFloat, rows, static_cast<int64_t>(episodic_memory_width));
    tensor_shape(snapshot.metadata, at::kLong, rows, metadata_width);
    tensor_shape(snapshot.outcomes, at::kDouble, rows, outcome_width);
    const auto keys = snapshot.keys.accessor<float, 2>();
    const auto values = snapshot.values.accessor<float, 2>();
    const auto metadata = snapshot.metadata.accessor<int64_t, 2>();
    const auto outcomes = snapshot.outcomes.accessor<double, 2>();
    std::vector<MemoryRecord> records;
    records.reserve(snapshot.capacity);
    std::unordered_set<uint64_t> ids;
    for (int64_t row = 0; row < rows; ++row) {
        require(
            metadata[row][id_column] > 0 &&
                (metadata[row][reward_valid_column] == 0 ||
                 metadata[row][reward_valid_column] == 1) &&
                (metadata[row][label_valid_column] == 0 || metadata[row][label_valid_column] == 1),
            "El snapshot contiene un identificador o una máscara inválidos");
        MemoryRecord record;
        record.id = static_cast<uint64_t>(metadata[row][id_column]);
        record.decision_at = metadata[row][decision_column];
        record.available_at = metadata[row][available_column];
        record.maturity_at = metadata[row][maturity_column];
        record.reward_valid = metadata[row][reward_valid_column] != 0;
        record.label_valid = metadata[row][label_valid_column] != 0;
        record.reward = outcomes[row][0];
        record.label = outcomes[row][1];
        for (std::size_t column = 0; column < episodic_memory_width; ++column) {
            record.key.at(column) = keys[row][static_cast<int64_t>(column)];
            record.value.at(column) = values[row][static_cast<int64_t>(column)];
        }
        validate_record(record, snapshot.confirmed_at, true);
        require(record.id <= snapshot.last_id && ids.insert(record.id).second,
                "El snapshot contiene identificadores repetidos o posteriores al cursor");
        records.push_back(record);
    }
    return records;
}
} // namespace

struct PreparedMemoryWrite::Impl {
    std::shared_ptr<const MemoryScope> owner;
    uint64_t generation;
    std::optional<std::size_t> slot;
    MemoryRecord record;
    int64_t confirmed_at;
    ReservoirEngine rng;
};
PreparedMemoryWrite::PreparedMemoryWrite(std::unique_ptr<Impl> candidate)
    : impl_(std::move(candidate)) {}
PreparedMemoryWrite::PreparedMemoryWrite(PreparedMemoryWrite&&) noexcept = default;
PreparedMemoryWrite& PreparedMemoryWrite::operator=(PreparedMemoryWrite&&) noexcept = default;
PreparedMemoryWrite::~PreparedMemoryWrite() = default;

struct EpisodicMemory::Impl {
    Impl(uint64_t initial_seed, MemoryScope initial_scope, std::size_t initial_capacity)
        : scope(std::move(initial_scope)), seed(initial_seed), capacity(initial_capacity),
          rng(initial_rng(seed, scope)),
          keys(at::zeros(
              {static_cast<int64_t>(capacity), static_cast<int64_t>(episodic_memory_width)},
              at::kDouble)) {
        records.reserve(capacity);
        key_storage = {keys.data_ptr<double>(), capacity * episodic_memory_width};
    }

    [[nodiscard]] bool valid(const PreparedMemoryWrite::Impl* candidate) const noexcept {
        return candidate != nullptr && candidate->owner == owner &&
               candidate->generation == generation;
    }

    [[nodiscard]] MemoryQuery query(const MemoryVector& key, int64_t cutoff,
                                    std::optional<uint64_t> excluded,
                                    const PreparedMemoryWrite::Impl* candidate = nullptr) const {
        require(cutoff >= 0, "El corte de la consulta episódica no puede ser negativo");
        const at::NoGradGuard no_grad;
        const auto normalized = vector_tensor(normalize(key)).to(at::kDouble);
        const auto rows =
            records.size() +
            ((candidate && candidate->slot && *candidate->slot == records.size()) ? 1 : 0);
        const auto matrix = keys.narrow(0, 0, static_cast<int64_t>(rows));
        QueryRow saved_row{};
        std::span<double> overwritten;
        const auto restore_row = c10::make_scope_exit([&]() noexcept {
            if (!overwritten.empty()) {
                std::copy_n(saved_row.begin(), overwritten.size(), overwritten.begin());
            }
        });
        if (candidate && candidate->slot) {
            overwritten = key_storage.subspan(*candidate->slot * episodic_memory_width,
                                              episodic_memory_width);
            std::copy(overwritten.begin(), overwritten.end(), saved_row.begin());
            std::copy(candidate->record.key.begin(), candidate->record.key.end(),
                      overwritten.begin());
        }
        // La GEMV conserva el redondeo. La fila provisional se restaura incluso si falla la
        // consulta.
        const auto scores = at::matmul(matrix, normalized);
        std::array<std::pair<double, const MemoryRecord*>, episodic_memory_capacity> eligible{};
        std::size_t count = 0;
        const auto admit = [&](const MemoryRecord& record, double score) {
            if (record.maturity_at <= cutoff && record.available_at <= cutoff &&
                record.decision_at < cutoff && (!excluded || record.id != *excluded)) {
                require(std::isfinite(score), "La similitud episódica no es finita");
                eligible.at(count++) = {score, &record};
            }
        };
        const auto similarity = scores.accessor<double, 1>();
        for (std::size_t index = 0; index < records.size(); ++index) {
            if (!candidate || !candidate->slot || index != *candidate->slot) {
                admit(records[index], similarity[static_cast<int64_t>(index)]);
            }
        }
        if (candidate && candidate->slot) {
            admit(candidate->record, similarity[static_cast<int64_t>(*candidate->slot)]);
        }
        const auto selected = std::min(count, episodic_memory_neighbors);
        const auto begin = eligible.begin();
        std::partial_sort(begin, std::next(begin, static_cast<std::ptrdiff_t>(selected)),
                          std::next(begin, static_cast<std::ptrdiff_t>(count)),
                          [](const auto& left, const auto& right) {
                              return left.first == right.first ? left.second->id < right.second->id
                                                               : left.first > right.first;
                          });
        MemoryQuery result;
        result.count = selected;
        for (std::size_t index = 0; index < selected; ++index) {
            result.neighbors.at(index) = {*eligible.at(index).second, eligible.at(index).first};
        }
        return result;
    }

    MemoryScope scope;
    std::shared_ptr<const MemoryScope> owner = std::make_shared<const MemoryScope>(scope);
    uint64_t seed;
    std::size_t capacity;
    uint64_t seen = 0;
    uint64_t last_id = 0;
    int64_t confirmed_at = 0;
    uint64_t generation = 0;
    ReservoirEngine rng;
    std::vector<MemoryRecord> records;
    at::Tensor keys;
    std::span<double> key_storage;
};

EpisodicMemory::EpisodicMemory(MemoryScope scope, uint64_t seed, std::size_t capacity) {
    validate_scope(scope);
    require(capacity > 0 && capacity <= episodic_memory_capacity,
            "La capacidad episódica debe estar entre 1 y 1024");
    impl_ = std::make_unique<Impl>(seed, std::move(scope), capacity);
}
EpisodicMemory::~EpisodicMemory() = default;

MemoryVector normalize_memory_key(const MemoryVector& key) { return normalize(key); }

std::vector<MemoryRecord> EpisodicMemory::retained_records() const { return impl_->records; }

std::vector<MemoryRecord> EpisodicMemory::validate_batch(std::span<const MemoryRecord> incoming,
                                                         int64_t confirmed_at) const {
    require(!incoming.empty() && incoming.size() <= maximum_retention_batch &&
                incoming.size() <= maximum_seen - impl_->seen &&
                confirmed_at >= impl_->confirmed_at,
            "El lote de retención está vacío, retrocede o supera el presupuesto");
    std::vector<MemoryRecord> normalized;
    normalized.reserve(incoming.size());
    auto last_id = impl_->last_id;
    for (const auto& input : incoming) {
        validate_record(input, confirmed_at, false);
        require(input.id > last_id, "El lote repite o desordena una admisión episódica");
        auto record = input;
        record.key = normalize(record.key);
        normalized.push_back(record);
        last_id = input.id;
    }
    return normalized;
}

void EpisodicMemory::retain_batch(std::span<const MemoryRecord> incoming,
                                  std::span<const uint64_t> retained_ids, int64_t confirmed_at) {
    const auto normalized = validate_batch(incoming, confirmed_at);
    const auto seen = impl_->seen + static_cast<uint64_t>(incoming.size());
    require(retained_ids.size() == std::min(static_cast<uint64_t>(impl_->capacity), seen) &&
                std::ranges::is_sorted(retained_ids) &&
                std::ranges::adjacent_find(retained_ids) == retained_ids.end(),
            "La selección necesita IDs crecientes únicos y la capacidad completa");
    auto eligible = impl_->records;
    eligible.insert(eligible.end(), normalized.begin(), normalized.end());
    std::ranges::sort(eligible, {}, &MemoryRecord::id);
    const at::NoGradGuard no_grad;
    auto next = std::make_unique<Impl>(impl_->seed, impl_->scope, impl_->capacity);
    for (const auto id : retained_ids) {
        const auto found = std::ranges::lower_bound(eligible, id, {}, &MemoryRecord::id);
        require(found != eligible.end() && found->id == id,
                "La retención incluye un episodio ajeno al conjunto admisible");
        const auto row = next->records.size();
        next->records.push_back(*found);
        std::copy(
            found->key.begin(), found->key.end(),
            next->key_storage.subspan(row * episodic_memory_width, episodic_memory_width).begin());
    }
    next->seen = seen;
    next->last_id = incoming.back().id;
    next->confirmed_at = confirmed_at;
    next->rng = impl_->rng;
    impl_ = std::move(next);
}

PreparedMemoryWrite EpisodicMemory::prepare_write(const MemoryRecord& input,
                                                  int64_t confirmed_at) const {
    validate_record(input, confirmed_at, false);
    require(input.id > impl_->last_id && confirmed_at >= impl_->confirmed_at &&
                impl_->seen < maximum_seen &&
                impl_->generation < std::numeric_limits<uint64_t>::max(),
            "La escritura episódica retrocede o supera su contador");
    auto record = input;
    record.key = normalize(record.key);
    auto rng = impl_->rng;
    const auto chosen =
        impl_->seen < impl_->capacity ? impl_->seen : choose_slot(rng, impl_->seen + 1);
    const auto slot = chosen < impl_->capacity
                          ? std::optional<std::size_t>(static_cast<std::size_t>(chosen))
                          : std::nullopt;
    return PreparedMemoryWrite(
        std::make_unique<PreparedMemoryWrite::Impl>(PreparedMemoryWrite::Impl{
            impl_->owner, impl_->generation, slot, record, confirmed_at, rng}));
}

bool EpisodicMemory::commit(PreparedMemoryWrite&& candidate) noexcept {
    if (!impl_->valid(candidate.impl_.get())) {
        return false;
    }
    const auto owned = std::move(candidate);
    const auto& write = *owned.impl_;
    if (write.slot) {
        if (*write.slot == impl_->records.size()) {
            impl_->records.push_back(write.record);
        } else {
            impl_->records[*write.slot] = write.record;
        }
        const auto target =
            impl_->key_storage.subspan(*write.slot * episodic_memory_width, episodic_memory_width);
        std::copy(write.record.key.begin(), write.record.key.end(), target.begin());
    }
    impl_->rng = write.rng;
    impl_->confirmed_at = write.confirmed_at;
    impl_->last_id = write.record.id;
    ++impl_->seen;
    ++impl_->generation;
    return true;
}

MemoryQuery EpisodicMemory::query(const MemoryVector& key, int64_t cutoff,
                                  std::optional<uint64_t> exclude_id) const {
    return impl_->query(key, cutoff, exclude_id);
}
MemoryQuery EpisodicMemory::query_prepared(const MemoryVector& key, int64_t cutoff,
                                           const PreparedMemoryWrite& candidate,
                                           std::optional<uint64_t> exclude_id) const {
    require(impl_->valid(candidate.impl_.get()),
            "La consulta provisional usa un token ajeno o caducado");
    return impl_->query(key, cutoff, exclude_id, candidate.impl_.get());
}
MemorySnapshot EpisodicMemory::snapshot() const {
    MemorySnapshot result;
    result.scope = impl_->scope;
    result.capacity = impl_->capacity;
    result.seed = impl_->seed;
    result.seen = impl_->seen;
    result.last_id = impl_->last_id;
    result.confirmed_at = impl_->confirmed_at;
    result.reservoir_rng = encode_rng(impl_->rng);
    const auto rows = static_cast<int64_t>(impl_->records.size());
    result.keys = impl_->keys.narrow(0, 0, rows).to(at::kFloat);
    result.values = at::empty({rows, static_cast<int64_t>(episodic_memory_width)}, at::kFloat);
    result.metadata = at::empty({rows, metadata_width}, at::kLong);
    result.outcomes = at::empty({rows, outcome_width}, at::kDouble);
    auto values = result.values.accessor<float, 2>();
    auto metadata = result.metadata.accessor<int64_t, 2>();
    auto outcomes = result.outcomes.accessor<double, 2>();
    for (int64_t row = 0; row < rows; ++row) {
        const auto& record = impl_->records[static_cast<std::size_t>(row)];
        for (std::size_t column = 0; column < episodic_memory_width; ++column) {
            values[row][static_cast<int64_t>(column)] = record.value.at(column);
        }
        metadata[row][id_column] = static_cast<int64_t>(record.id);
        metadata[row][decision_column] = record.decision_at;
        metadata[row][available_column] = record.available_at;
        metadata[row][maturity_column] = record.maturity_at;
        metadata[row][reward_valid_column] = record.reward_valid ? 1 : 0;
        metadata[row][label_valid_column] = record.label_valid ? 1 : 0;
        outcomes[row][0] = record.reward;
        outcomes[row][1] = record.label;
    }
    return result;
}
void EpisodicMemory::restore(const MemorySnapshot& snapshot) {
    require(snapshot.scope == impl_->scope && snapshot.seed == impl_->seed &&
                snapshot.capacity == impl_->capacity &&
                impl_->generation < std::numeric_limits<uint64_t>::max(),
            "El snapshot pertenece a otro ámbito, semilla o capacidad");
    auto records = snapshot_records(snapshot);
    auto rng = decode_rng(snapshot.reservoir_rng);
    if (snapshot.seen <= snapshot.capacity) {
        require(rng == initial_rng(snapshot.seed, snapshot.scope),
                "El reservorio sin reemplazos no conserva su estado aleatorio inicial");
    }
    const auto next_generation = impl_->generation + 1;
    auto candidate = std::make_unique<Impl>(snapshot.seed, snapshot.scope, snapshot.capacity);
    candidate->records = std::move(records);
    for (std::size_t index = 0; index < candidate->records.size(); ++index) {
        const auto target =
            candidate->key_storage.subspan(index * episodic_memory_width, episodic_memory_width);
        std::copy(candidate->records[index].key.begin(), candidate->records[index].key.end(),
                  target.begin());
    }
    candidate->rng = rng;
    candidate->seen = snapshot.seen;
    candidate->last_id = snapshot.last_id;
    candidate->confirmed_at = snapshot.confirmed_at;
    candidate->generation = next_generation;
    impl_.swap(candidate);
}
const MemoryScope& EpisodicMemory::scope() const noexcept { return impl_->scope; }
std::size_t EpisodicMemory::size() const noexcept { return impl_->records.size(); }
uint64_t EpisodicMemory::seen() const noexcept { return impl_->seen; }

std::string serialize_memory(const MemorySnapshot& snapshot) {
    EpisodicMemory validated(snapshot.scope, snapshot.seed, snapshot.capacity);
    validated.restore(snapshot);
    const auto state = validated.snapshot();
    torch::serialize::OutputArchive archive;
    archive.write("schema_version", c10::IValue(snapshot_version));
    archive.write("capacity", c10::IValue(static_cast<int64_t>(state.capacity)));
    archive.write("seed", c10::IValue(std::bit_cast<int64_t>(state.seed)));
    archive.write("lane", c10::IValue(std::bit_cast<int64_t>(state.scope.lane)));
    archive.write("seen", c10::IValue(static_cast<int64_t>(state.seen)));
    archive.write("last_id", c10::IValue(static_cast<int64_t>(state.last_id)));
    archive.write("confirmed_at", c10::IValue(state.confirmed_at));
    archive.write("world", c10::IValue(state.scope.world));
    archive.write("partition", c10::IValue(state.scope.partition));
    archive.write("fold", c10::IValue(state.scope.fold));
    archive.write("representation", c10::IValue(state.scope.representation));
    archive.write("reservoir_rng", c10::IValue(state.reservoir_rng));
    archive.write("keys", state.keys, true);
    archive.write("values", state.values, true);
    archive.write("metadata", state.metadata, true);
    archive.write("outcomes", state.outcomes, true);
    std::ostringstream destination;
    archive.save_to(destination);
    auto bytes = destination.str();
    require(bytes.size() <= maximum_episodic_archive_bytes, "El archivo episódico supera 2 MiB");
    return bytes;
}
MemorySnapshot deserialize_memory(std::string_view bytes) {
    require(!bytes.empty() && bytes.size() <= maximum_episodic_archive_bytes,
            "El archivo episódico está vacío o supera 2 MiB");
    constexpr std::size_t zip_footer_bytes = 22;
    constexpr std::string_view zip_start = "PK\003\004";
    constexpr std::string_view zip_end = "PK\005\006";
    // El escritor LibTorch no añade comentarios. Rechaza cortes antes de abrir su lector ZIP.
    require(bytes.size() >= zip_footer_bytes && bytes.starts_with(zip_start) &&
                bytes.substr(bytes.size() - zip_footer_bytes).starts_with(zip_end) &&
                bytes[bytes.size() - 1] == '\0' && bytes[bytes.size() - 2] == '\0',
            "El archivo episódico no conserva el principio y final de su contenedor");
    std::istringstream source{std::string(bytes)};
    torch::serialize::InputArchive archive;
    archive.load_from(source, at::Device(at::kCPU));
    const auto integer = [&archive](const char* name) {
        c10::IValue value;
        archive.read(name, value);
        require(value.isInt(), "El archivo episódico necesita un contador entero");
        return value.toInt();
    };
    const auto text = [&archive](const char* name) {
        c10::IValue value;
        archive.read(name, value);
        require(value.isString(), "El archivo episódico necesita una identidad de texto");
        return value.toStringRef();
    };
    require(integer("schema_version") == snapshot_version,
            "La versión de memoria episódica no está admitida");
    const auto capacity = integer("capacity");
    const auto seen = integer("seen");
    const auto last_id = integer("last_id");
    require(capacity > 0 && capacity <= static_cast<int64_t>(episodic_memory_capacity) &&
                seen >= 0 && last_id >= 0,
            "El archivo episódico contiene contadores fuera de rango");
    MemorySnapshot result;
    result.scope = {text("world"), text("partition"), text("fold"), text("representation"),
                    std::bit_cast<uint64_t>(integer("lane"))};
    result.capacity = static_cast<std::size_t>(capacity);
    result.seed = std::bit_cast<uint64_t>(integer("seed"));
    result.seen = static_cast<uint64_t>(seen);
    result.last_id = static_cast<uint64_t>(last_id);
    result.confirmed_at = integer("confirmed_at");
    result.reservoir_rng = text("reservoir_rng");
    archive.read("keys", result.keys, true);
    archive.read("values", result.values, true);
    archive.read("metadata", result.metadata, true);
    archive.read("outcomes", result.outcomes, true);
    EpisodicMemory validated(result.scope, result.seed, result.capacity);
    validated.restore(result);
    return validated.snapshot();
}
} // namespace mars_titan::learning
