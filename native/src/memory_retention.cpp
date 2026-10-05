#include "mars_titan/memory_stress.hpp"
#include "memory_stress_internal.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>

#include <array>
#include <iterator>
#include <span>
#include <unordered_set>
#include <vector>

namespace mars_titan::stress {
using namespace detail;
using namespace learning;
namespace {
constexpr uint32_t retention_stream = 3;
constexpr uint64_t seed_maximum = std::numeric_limits<uint64_t>::max();
constexpr double unit_norm_tolerance = 1e-6;

std::string_view policy_name(Retention policy) {
    switch (policy) {
    case Retention::uniform:
        return "uniform";
    case Retention::recent:
        return "recent";
    case Retention::selective:
        return "selective";
    }
    throw std::invalid_argument("Política de retención desconocida");
}
} // namespace
struct RetentionBank::Impl {
    Impl(std::size_t slots, Retention selected, uint64_t initial_seed)
        : policy(selected), capacity(slots), seed(initial_seed),
          rng(engine(seed, retention_stream)) {
        static_cast<void>(policy_name(policy));
        require(capacity > 0 && capacity <= episodic_memory_capacity,
                "La capacidad debe estar entre 1 y 1024");
        records.reserve(capacity);
        priorities.reserve(capacity);
        keys =
            at::zeros({static_cast<int64_t>(capacity), static_cast<int64_t>(episodic_memory_width)},
                      at::TensorOptions().dtype(at::kDouble).device(at::kCPU));
    }
    void cache(std::size_t slot) {
        auto storage = std::span<double>(keys.data_ptr<double>(), capacity * episodic_memory_width);
        auto row = storage.subspan(slot * episodic_memory_width, episodic_memory_width);
        std::copy(records.at(slot).key.begin(), records.at(slot).key.end(), row.begin());
    }
    Retention policy;
    std::size_t capacity;
    uint64_t seed;
    Engine rng;
    uint64_t count = 0;
    uint64_t last_id = 0;
    int64_t confirmed_at = 0;
    std::vector<MemoryRecord> records;
    std::vector<double> priorities;
    at::Tensor keys;
};

RetentionBank::RetentionBank(Retention policy, std::size_t capacity, uint64_t seed)
    : impl_(std::make_unique<Impl>(capacity, policy, seed)) {}
RetentionBank::~RetentionBank() = default;

void RetentionBank::write(const MemoryRecord& input, int64_t confirmed_at, double priority) {
    valid_record(input);
    require(input.id > impl_->last_id && confirmed_at >= impl_->confirmed_at &&
                input.maturity_at <= confirmed_at && std::isfinite(priority) && priority >= 0 &&
                impl_->count < maximum_steps,
            "La escritura repite identidad, usa futuro o excede el presupuesto");
    auto record = input;
    record.key = normalized(record.key);
    std::size_t slot = impl_->records.size();
    if (slot == impl_->capacity) {
        if (impl_->policy == Retention::uniform) {
            slot = std::uniform_int_distribution<std::size_t>(0, impl_->capacity - 1)(impl_->rng);
        } else if (impl_->policy == Retention::recent) {
            slot = static_cast<std::size_t>(impl_->count % impl_->capacity);
        } else {
            slot = static_cast<std::size_t>(std::distance(
                impl_->priorities.begin(),
                std::min_element(impl_->priorities.begin(), impl_->priorities.end())));
        }
        impl_->records.at(slot) = record;
        impl_->priorities.at(slot) = priority;
    } else {
        impl_->records.push_back(record);
        impl_->priorities.push_back(priority);
    }
    impl_->cache(slot);
    ++impl_->count;
    impl_->last_id = record.id;
    impl_->confirmed_at = confirmed_at;
}

MemoryQuery RetentionBank::query(const MemoryVector& key, int64_t cutoff) const {
    require(cutoff >= 0, "El corte no puede ser negativo");
    const auto unit = normalized(key);
    const at::NoGradGuard no_grad;
    const auto tensor = at::tensor(at::ArrayRef<float>(unit.data(), unit.size()),
                                   at::TensorOptions().dtype(at::kFloat).device(at::kCPU))
                            .to(at::kDouble);
    const auto scores = at::mv(impl_->keys.narrow(0, 0, static_cast<int64_t>(size())), tensor);
    const auto values = scores.accessor<double, 1>();
    std::array<std::pair<double, const MemoryRecord*>, episodic_memory_capacity> eligible{};
    std::size_t count = 0;
    for (std::size_t row = 0; row < size(); ++row) {
        const auto& record = impl_->records.at(row);
        if (record.maturity_at <= cutoff && record.decision_at < cutoff &&
            record.available_at <= cutoff) {
            eligible.at(count++) = {values[static_cast<int64_t>(row)], &record};
        }
    }
    const auto selected = std::min(count, episodic_memory_neighbors);
    std::partial_sort(eligible.begin(),
                      std::next(eligible.begin(), static_cast<std::ptrdiff_t>(selected)),
                      std::next(eligible.begin(), static_cast<std::ptrdiff_t>(count)),
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

std::string RetentionBank::checkpoint() const {
    Json records = Json::array();
    for (const auto& record : impl_->records) {
        records.push_back(record_json(record));
    }
    return Json{{"version", 1},
                {"policy", policy_name(impl_->policy)},
                {"capacity", impl_->capacity},
                {"seed", impl_->seed},
                {"writes", impl_->count},
                {"last_id", impl_->last_id},
                {"confirmed_at", impl_->confirmed_at},
                {"rng", encode_rng(impl_->rng)},
                {"records", std::move(records)},
                {"priorities", impl_->priorities}}
        .dump();
}
void RetentionBank::restore(std::string_view bytes) {
    const auto value = parse(bytes);
    require(integer(value.at("version")) == 1 && value.at("policy") == policy_name(impl_->policy) &&
                integer(value.at("capacity")) == impl_->capacity &&
                integer(value.at("seed"), seed_maximum) == impl_->seed,
            "El banco pertenece a otra versión, capacidad, política o semilla");
    auto next = std::make_unique<Impl>(impl_->capacity, impl_->policy, impl_->seed);
    next->count = integer(value.at("writes"));
    next->last_id = integer(value.at("last_id"));
    next->confirmed_at =
        static_cast<int64_t>(integer(value.at("confirmed_at"), 2 * maximum_steps + maximum_delay));
    next->rng = decode_rng(value.at("rng"));
    const auto& records = value.at("records");
    const auto& priorities = value.at("priorities");
    require(records.is_array() && records.size() == std::min(next->count, next->capacity) &&
                priorities.is_array() && priorities.size() == records.size() &&
                next->last_id >= next->count &&
                (next->count > 0 || (next->last_id == 0 && next->confirmed_at == 0)),
            "El banco no conserva contadores, prioridades y filas coherentes");
    std::unordered_set<uint64_t> ids;
    for (std::size_t index = 0; index < records.size(); ++index) {
        auto record = read_record(records.at(index));
        double norm = 0;
        for (float component : record.key) {
            norm += static_cast<double>(component) * static_cast<double>(component);
        }
        const auto priority = number(priorities.at(index));
        require(record.id <= next->last_id && record.maturity_at <= next->confirmed_at &&
                    ids.insert(record.id).second && std::abs(norm - 1) <= unit_norm_tolerance &&
                    priority >= 0,
                "El banco contiene un registro duplicado, futuro o no normalizado");
        next->records.push_back(record);
        next->priorities.push_back(priority);
        next->cache(index);
    }
    impl_ = std::move(next);
}
std::size_t RetentionBank::size() const noexcept { return impl_->records.size(); }
uint64_t RetentionBank::writes() const noexcept { return impl_->count; }
} // namespace mars_titan::stress
