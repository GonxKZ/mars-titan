#include "../src/accurate_sum.hpp"
#include "mars_titan/episodic_memory.hpp"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/ThreadLocalDebugInfo.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace mars_titan::learning;
constexpr uint64_t seed = 71;
constexpr int64_t final_time = 10'000;
constexpr double tolerance = 1e-12;
constexpr std::size_t small_capacity = 8;
constexpr uint64_t dense_records = 80;
constexpr uint64_t pause_record = 19;
constexpr uint64_t last_record = 100;
constexpr int64_t second_confirmation = 5;
constexpr double reward_scale = 10;
constexpr std::size_t minimum_inclusion = 20;
constexpr std::size_t maximum_inclusion = 85;

struct QueryAllocations final : c10::MemoryReportingInfoBase {
    std::size_t bytes = 0;
    std::size_t largest = 0;
    void reportMemoryUsage(void*, int64_t allocation, std::size_t, std::size_t,
                           c10::Device device) override {
        if (device.is_cpu() && allocation > 0) {
            const auto size = static_cast<std::size_t>(allocation);
            bytes += size;
            largest = std::max(largest, size);
        }
    }
    [[nodiscard]] bool memoryProfilingEnabled() const override { return true; }
};

template <typename Operation> QueryAllocations allocations(Operation&& operation) {
    const auto report = std::make_shared<QueryAllocations>();
    const c10::DebugInfoGuard guard(c10::DebugInfoKind::PROFILER_STATE, report);
    std::forward<Operation>(operation)();
    return *report;
}

struct RejectScoreAllocation final : c10::Allocator {
    explicit RejectScoreAllocation(c10::Allocator& allocator) : upstream(allocator) {}
    c10::DataPtr allocate(std::size_t bytes) override {
        if (bytes == sizeof(double)) {
            throw std::bad_alloc();
        }
        return upstream.get().allocate(bytes);
    }
    void copy_data(void* destination, const void* source, std::size_t count) const override {
        upstream.get().copy_data(destination, source, count);
    }
    std::reference_wrapper<c10::Allocator> upstream;
};

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Function> void rejected(Function&& action) {
    try {
        std::forward<Function>(action)();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("La entrada episódica inválida no fue rechazada");
}

MemoryScope scope(uint64_t lane = 0) { return {"world-1", "train", "fold-0", "fixed-v1", lane}; }
MemoryVector key(float first = 1, float second = 0) {
    MemoryVector result{};
    result[0] = first;
    result[1] = second;
    return result;
}
MemoryRecord record(uint64_t id, MemoryVector vector = key()) {
    MemoryRecord result;
    result.id = id;
    result.decision_at = static_cast<int64_t>(id) * 2;
    result.available_at = result.decision_at;
    result.maturity_at = result.decision_at + 1;
    result.key = vector;
    result.value.fill(static_cast<float>(id));
    result.reward = static_cast<double>(id) / reward_scale;
    result.reward_valid = true;
    return result;
}
void write(EpisodicMemory& memory, const MemoryRecord& value) {
    auto candidate = memory.prepare_write(value, value.maturity_at);
    require(memory.commit(std::move(candidate)), "La escritura preparada no se confirmó");
}

void same_query(const MemoryQuery& first, const MemoryQuery& second) {
    require(first.count == second.count, "Las consultas no conservan el número de vecinos");
    for (std::size_t index = 0; index < first.count; ++index) {
        require(first.neighbors.at(index).record == second.neighbors.at(index).record &&
                    first.neighbors.at(index).similarity == second.neighbors.at(index).similarity,
                "Las consultas no conservan sus vecinos y puntuaciones");
    }
}

void writes_are_staged_and_provisional_queries_match_commit() {
    EpisodicMemory memory(scope(), seed, 4);
    auto first = memory.prepare_write(record(1), 3);
    require(memory.size() == 0 && memory.seen() == 0 && memory.query(key(), 3).count == 0,
            "La preparación modificó el banco confirmado");
    const auto provisional = memory.query_prepared(key(), 3, first);
    require(provisional.count == 1 && provisional.neighbors[0].record.id == 1,
            "La consulta provisional no ve el resultado maduro");
    require(memory.commit(std::move(first)), "No se confirmó el primer candidato");
    same_query(provisional, memory.query(key(), 3));
    for (uint64_t id = 2; id <= dense_records; ++id) {
        MemoryVector dense{};
        for (std::size_t coordinate = 0; coordinate < dense.size(); ++coordinate) {
            dense.at(coordinate) =
                static_cast<float>(std::sin(static_cast<double>(id * (coordinate + 1))));
        }
        auto candidate = memory.prepare_write(record(id, dense), static_cast<int64_t>(id) * 2 + 1);
        const auto expected = memory.query_prepared(dense, final_time, candidate);
        const auto seen = memory.seen();
        require(memory.seen() == seen, "La consulta provisional consume el RNG");
        require(memory.commit(std::move(candidate)), "El candidato del reservorio no se confirmó");
        same_query(expected, memory.query(dense, final_time));
        require(memory.size() <= 4 && memory.seen() == id,
                "El reservorio perdió su límite o contador");
    }
}

void time_cutoff_self_exclusion_and_ties_are_explicit() {
    EpisodicMemory memory(scope(), seed, small_capacity);
    write(memory, record(1, key(3, 4)));
    write(memory, record(2));
    write(memory, record(3, key(-1)));
    write(memory, record(4));
    write(memory, record(static_cast<uint64_t>(second_confirmation), key(0, 1)));
    const auto result = memory.query(key(), final_time);
    const std::array<uint64_t, 4> expected{2, 4, 1, 5};
    require(result.count == expected.size(), "La consulta no limita a cuatro vecinos");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require(result.neighbors.at(index).record.id == expected.at(index),
                "La consulta no ordena por coseno y por ID en los empates");
    }
    require(memory.query(key(), 2).count == 0, "Se consultó un resultado todavía no maduro");
    require(memory.query(key(), 3).count == 1, "No se admitió un resultado maduro en el corte");
    const auto excluded = memory.query(key(), final_time, 2);
    require(excluded.count == 4 && excluded.neighbors[0].record.id == 4,
            "El registro propio no fue excluido");
    same_query(result, memory.query(key(static_cast<float>(small_capacity)), final_time));
    EpisodicMemory delayed(scope(), seed, small_capacity);
    auto label = record(1);
    label.reward_valid = false;
    label.reward = 0;
    label.label_valid = true;
    label.label = 3;
    constexpr int64_t delayed_maturity = 10;
    label.maturity_at = delayed_maturity;
    write(delayed, label);
    require(delayed.query(key(), second_confirmation).count == 0 &&
                delayed.query(key(), delayed_maturity).count == 1,
            "La consulta no distingue publicación de entrada y maduración de etiqueta");
}

void matrix_query_matches_an_independent_accurate_reference() {
    constexpr std::size_t samples = 37;
    EpisodicMemory memory(scope(), seed, samples);
    for (uint64_t id = 1; id <= samples; ++id) {
        MemoryVector vector{};
        for (std::size_t coordinate = 0; coordinate < vector.size(); ++coordinate) {
            vector.at(coordinate) =
                static_cast<float>(std::sin(static_cast<double>(id * (coordinate + 1))));
        }
        write(memory, record(id, vector));
    }
    const auto snapshot = memory.snapshot();
    const auto keys = snapshot.keys.accessor<float, 2>();
    const auto metadata = snapshot.metadata.accessor<int64_t, 2>();
    std::vector<std::pair<double, uint64_t>> reference;
    reference.reserve(samples);
    const auto query = key(3, 4);
    const auto normalized = key(0.6F, 0.8F);
    for (std::size_t row = 0; row < samples; ++row) {
        mars_titan::simulation::AccurateSum dot;
        for (std::size_t column = 0; column < normalized.size(); ++column) {
            require(dot.add(static_cast<double>(
                                keys[static_cast<int64_t>(row)][static_cast<int64_t>(column)]) *
                            static_cast<double>(normalized.at(column))),
                    "Desbordamiento de la referencia de producto escalar");
        }
        reference.emplace_back(dot.value(),
                               static_cast<uint64_t>(metadata[static_cast<int64_t>(row)][0]));
    }
    std::sort(reference.begin(), reference.end(), [](const auto& left, const auto& right) {
        return left.first == right.first ? left.second < right.second : left.first > right.first;
    });
    const auto actual = memory.query(query, final_time);
    for (std::size_t index = 0; index < actual.count; ++index) {
        require(actual.neighbors.at(index).record.id == reference[index].second &&
                    std::abs(actual.neighbors.at(index).similarity - reference[index].first) <=
                        tolerance,
                "La consulta matricial difiere de la referencia escalar precisa");
    }
}

void reservoir_recovers_its_rng_and_snapshot_owns_data() {
    EpisodicMemory uninterrupted(scope(), seed, small_capacity);
    for (uint64_t id = 1; id <= pause_record; ++id) {
        write(uninterrupted, record(id));
    }
    auto snapshot = uninterrupted.snapshot();
    const auto bytes = serialize_memory(snapshot);
    EpisodicMemory resumed(scope(), seed, small_capacity);
    resumed.restore(deserialize_memory(bytes));
    snapshot.values.fill_(0);
    require(uninterrupted.query(key(), final_time).neighbors[0].record.value[0] != 0,
            "El snapshot conserva una vista mutable del banco");
    for (uint64_t id = pause_record + 1; id <= last_record; ++id) {
        write(uninterrupted, record(id));
        write(resumed, record(id));
    }
    same_query(uninterrupted.query(key(), final_time), resumed.query(key(), final_time));
    const auto expected = uninterrupted.snapshot();
    const auto observed = resumed.snapshot();
    require(expected.reservoir_rng == observed.reservoir_rng &&
                at::equal(expected.metadata, observed.metadata) &&
                at::equal(expected.values, observed.values),
            "El reservorio recuperado no conserva RNG y contenido exactos");
    rejected([&] {
        static_cast<void>(deserialize_memory(std::string_view(bytes).substr(0, bytes.size() / 2)));
    });
    rejected([&] { static_cast<void>(deserialize_memory("archivo corrupto")); });
}

void stale_and_foreign_tokens_are_rejected_without_changes() {
    EpisodicMemory first(scope(), seed, small_capacity);
    EpisodicMemory other(scope(1), seed, small_capacity);
    auto candidate = first.prepare_write(record(1), 3);
    rejected([&] { static_cast<void>(other.query_prepared(key(), 3, candidate)); });
    require(!other.commit(std::move(candidate)) && other.seen() == 0,
            "Se confirmó una escritura en otra lane");
    auto current = first.prepare_write(record(1), 3);
    auto stale = first.prepare_write(record(1), 3);
    require(first.commit(std::move(current)), "Falla la escritura propia");
    require(!first.commit(std::move(stale)) && first.seen() == 1,
            "Se confirmó un token de una generación anterior");
    auto old = first.prepare_write(record(2), second_confirmation);
    first.restore(first.snapshot());
    require(!first.commit(std::move(old)), "La recuperación no invalida tokens anteriores");
}

void malformed_inputs_and_incompatible_scopes_fail_atomically() {
    EpisodicMemory memory(scope(), seed, small_capacity);
    write(memory, record(1));
    const auto before = memory.query(key(), final_time);
    auto wrong = record(2);
    wrong.key.fill(0);
    rejected([&] { static_cast<void>(memory.prepare_write(wrong, second_confirmation)); });
    wrong = record(2);
    wrong.value[0] = std::numeric_limits<float>::quiet_NaN();
    rejected([&] { static_cast<void>(memory.prepare_write(wrong, second_confirmation)); });
    wrong = record(2);
    wrong.key[0] = std::numeric_limits<float>::infinity();
    rejected([&] { static_cast<void>(memory.prepare_write(wrong, second_confirmation)); });
    wrong = record(2);
    wrong.available_at = second_confirmation;
    rejected([&] { static_cast<void>(memory.prepare_write(wrong, second_confirmation)); });
    wrong = record(2);
    rejected([&] { static_cast<void>(memory.prepare_write(wrong, 4)); });
    wrong.maturity_at = wrong.decision_at;
    rejected([&] { static_cast<void>(memory.prepare_write(wrong, second_confirmation)); });
    wrong = record(2);
    wrong.reward_valid = false;
    rejected([&] { static_cast<void>(memory.prepare_write(wrong, second_confirmation)); });
    wrong = record(2);
    wrong.reward = std::numeric_limits<double>::quiet_NaN();
    rejected([&] { static_cast<void>(memory.prepare_write(wrong, second_confirmation)); });
    rejected([&] { static_cast<void>(memory.prepare_write(record(1), 3)); });
    rejected([&] { static_cast<void>(memory.query(MemoryVector{}, final_time)); });
    same_query(before, memory.query(key(), final_time));
    auto snapshot = memory.snapshot();
    snapshot.scope.partition = "validation";
    rejected([&] { memory.restore(snapshot); });
    snapshot = memory.snapshot();
    snapshot.scope.world = "other-world";
    rejected([&] { memory.restore(snapshot); });
    snapshot = memory.snapshot();
    snapshot.scope.fold = "other-fold";
    rejected([&] { memory.restore(snapshot); });
    snapshot = memory.snapshot();
    snapshot.scope.representation = "other-representation";
    rejected([&] { memory.restore(snapshot); });
    snapshot = memory.snapshot();
    snapshot.metadata[0][0] = -1;
    rejected([&] { memory.restore(snapshot); });
    snapshot = memory.snapshot();
    snapshot.keys[0][0] = 2;
    rejected([&] { memory.restore(snapshot); });
    snapshot = memory.snapshot();
    snapshot.reservoir_rng = "sin estado";
    rejected([&] { memory.restore(snapshot); });
    same_query(before, memory.query(key(), final_time));
    rejected([&] { EpisodicMemory invalid(scope(), seed, 0); });
    rejected([&] { EpisodicMemory invalid(scope(), seed, episodic_memory_capacity + 1); });
    auto invalid_scope = scope();
    invalid_scope.partition = "test";
    rejected([&] { EpisodicMemory invalid(invalid_scope, seed); });
    auto extreme = record(2, key(std::numeric_limits<float>::max()));
    write(memory, extreme);
    require(memory.query(key(std::numeric_limits<float>::denorm_min()), final_time).count == 2,
            "La normalización no admite claves finitas extremas");
}

void reservoir_inclusion_is_not_biased_to_recent_or_early_records() {
    constexpr std::size_t population = 20;
    constexpr uint64_t repetitions = 256;
    std::array<std::size_t, population> occurrences{};
    for (uint64_t trial = 0; trial < repetitions; ++trial) {
        EpisodicMemory memory(scope(), trial, 4);
        for (uint64_t id = 1; id <= population; ++id) {
            write(memory, record(id));
        }
        const auto actual = memory.query(key(), final_time);
        for (std::size_t index = 0; index < actual.count; ++index) {
            ++occurrences.at(actual.neighbors.at(index).record.id - 1);
        }
    }
    require(std::all_of(occurrences.begin(), occurrences.end(),
                        [](auto count) {
                            return count >= minimum_inclusion && count <= maximum_inclusion;
                        }),
            "El muestreo no conserva una inclusión compatible con el reservorio uniforme");
}

void queries_bound_temporary_storage_to_scores() {
    EpisodicMemory memory(scope(), seed);
    for (uint64_t id = 1; id <= episodic_memory_capacity; ++id) {
        write(memory, record(id));
    }
    auto candidate =
        memory.prepare_write(record(episodic_memory_capacity + 1, key(0, 1)), final_time);
    const auto committed = allocations([&] { static_cast<void>(memory.query(key(), final_time)); });
    const auto prepared = allocations(
        [&] { static_cast<void>(memory.query_prepared(key(), final_time, candidate)); });
    std::cout << "Asignaciones por consulta (confirmada/provisional): " << committed.bytes << '/'
              << prepared.bytes << " bytes\n";
    constexpr std::size_t score_bytes = episodic_memory_capacity * sizeof(double);
    for (const auto& measured : {committed, prepared}) {
        require(measured.bytes > 0 && measured.largest <= score_bytes &&
                    measured.bytes <= 2 * score_bytes,
                "La consulta vuelve a materializar la matriz de claves completa");
    }
}

void abandoned_queries_leave_committed_keys_and_rng_unchanged() {
    for (const std::size_t capacity : {std::size_t{1}, small_capacity, episodic_memory_capacity}) {
        EpisodicMemory memory(scope(), seed, capacity);
        for (uint64_t id = 1; id <= capacity; ++id) {
            write(memory, record(id));
        }
        const auto before = memory.snapshot();
        const auto expected = memory.query(key(), final_time);
        for (uint64_t id = capacity + 1; id <= capacity + dense_records; ++id) {
            auto candidate = memory.prepare_write(record(id, key(0, 1)), final_time);
            static_cast<void>(memory.query_prepared(key(), final_time, candidate));
            static_cast<void>(memory.query_prepared(key(0, 1), final_time, candidate, id));
            rejected([&] {
                static_cast<void>(memory.query_prepared(MemoryVector{}, final_time, candidate));
            });
            same_query(expected, memory.query(key(), final_time));
        }
        const auto after = memory.snapshot();
        require(at::equal(before.keys, after.keys) && before.reservoir_rng == after.reservoir_rng &&
                    before.seen == after.seen && before.last_id == after.last_id,
                "Descartar consultas provisionales cambió claves, contadores o RNG");
    }
}

void failed_provisional_query_restores_the_committed_row() {
    EpisodicMemory memory(scope(), seed, 1);
    write(memory, record(1));
    const auto before = memory.query(key(), final_time);
    for (uint64_t id = 2; id <= dense_records; ++id) {
        auto candidate = memory.prepare_write(record(id, key(0, 1)), final_time);
        const auto provisional = memory.query_prepared(key(), final_time, candidate);
        if (provisional.neighbors[0].record.id != id) {
            require(memory.commit(std::move(candidate)),
                    "No avanzó el RNG tras descartar el registro");
            continue;
        }
        bool failed = false;
        {
            auto* upstream = c10::GetCPUAllocator();
            RejectScoreAllocation allocator(*upstream);
            const auto restore = c10::make_scope_exit([&] { c10::SetCPUAllocator(upstream); });
            c10::SetCPUAllocator(&allocator);
            try {
                static_cast<void>(memory.query_prepared(key(), final_time, candidate));
            } catch (const std::bad_alloc&) {
                failed = true;
            }
        }
        require(failed, "No se ejercitó el fallo al asignar las puntuaciones");
        same_query(before, memory.query(key(), final_time));
        same_query(provisional, memory.query_prepared(key(), final_time, candidate));
        require(memory.commit(std::move(candidate)),
                "El fallo invalidó una escritura todavía coherente");
        same_query(provisional, memory.query(key(), final_time));
        return;
    }
    throw std::runtime_error("La prueba no encontró una sustitución del reservorio");
}
void external_retention_changes_only_the_selected_records() {
    EpisodicMemory memory(scope(), seed, 2);
    write(memory, record(1, key(3, 4)));
    write(memory, record(2));
    const auto before = memory.retained_records();
    auto stale = memory.prepare_write(record(3), final_time);
    const std::array incoming{record(3, key(4, 3)), record(4, key(0, 2))};
    const std::array<uint64_t, 2> retained{1, 4};
    memory.retain_batch(incoming, retained, final_time);
    const auto after = memory.retained_records();
    require(after.size() == 2 && memory.seen() == 4 && after.front() == before.front(),
            "La retención alteró un centro fijo o perdió admisiones");
    auto expected = incoming.back();
    expected.key = normalize_memory_key(expected.key);
    require(after.back() == expected, "La retención modificó el episodio seleccionado");
    require(!memory.commit(std::move(stale)), "Una escritura anterior sobrevive a la sustitución");
    auto detached = memory.retained_records();
    detached.front().value.fill(0);
    require(memory.retained_records().front() == before.front(), "La copia comparte registros");
    EpisodicMemory resumed(scope(), seed, 2);
    resumed.restore(deserialize_memory(serialize_memory(memory.snapshot())));
    require(resumed.retained_records() == after && resumed.seen() == 4,
            "El archivo no recupera la retención y sus contadores");
    same_query(memory.query(key(), final_time), resumed.query(key(), final_time));
}

void external_retention_rejects_invalid_batches_without_changing_the_bank() {
    EpisodicMemory memory(scope(), seed, 2);
    write(memory, record(1));
    const auto before = memory.retained_records();
    const auto rng = memory.snapshot().reservoir_rng;
    const auto reject = [&](std::vector<MemoryRecord> records, std::vector<uint64_t> retained,
                            int64_t confirmed_at = final_time) {
        rejected([&] { memory.retain_batch(records, retained, confirmed_at); });
        require(memory.retained_records() == before && memory.seen() == 1 &&
                    memory.snapshot().reservoir_rng == rng,
                "El rechazo dejó una sustitución o avanzó el contador");
    };
    reject({}, {1});
    reject({record(2)}, {2});
    reject({record(2)}, {1, 3});
    reject({record(2)}, {1, 1});
    reject({record(2)}, {2, 1});
    reject({record(2), record(2)}, {1, 2});
    reject({record(3), record(2)}, {1, 2});
    reject({record(1)}, {1, 2});
    reject({record(2)}, {1, 2}, 2);
    auto future = record(2);
    future.maturity_at = final_time + 1;
    reject({future}, {1, 2});
    auto invalid = record(2);
    invalid.value.front() = std::numeric_limits<float>::quiet_NaN();
    reject({invalid}, {1, 2});
    reject(std::vector<MemoryRecord>(maximum_retention_batch + 1, record(2)), {1, 2});
    const std::array incoming{record(2)};
    const std::array<uint64_t, 2> retained{1, 2};
    memory.retain_batch(incoming, retained, final_time);
    require(memory.size() == 2 && memory.seen() == 2 && memory.snapshot().reservoir_rng == rng,
            "La retención externa consumió el RNG del reservorio");
}

void causal_rng_pairs_scopes_but_keeps_recovery_isolated() {
    constexpr uint64_t other_lane = 42;
    constexpr unsigned int seed_word_bits = 32;
    auto other = scope(other_lane);
    other.world = "another-arm";
    other.partition = "evaluation";
    other.fold = "fold-9";
    other.representation = "different-future-source";
    EpisodicMemory first(scope(), seed, small_capacity, 2);
    EpisodicMemory second(other, seed, small_capacity, 2);
    EpisodicMemory resumed(scope(), seed, small_capacity, 2);
    for (uint64_t id = 1; id <= last_record; ++id) {
        write(first, record(id));
        write(second, record(id));
        if (id == pause_record) {
            resumed.restore(deserialize_memory(serialize_memory(first.snapshot())));
        } else if (id > pause_record) {
            write(resumed, record(id));
        }
        require(first.retained_records() == second.retained_records() &&
                    first.snapshot().reservoir_rng == second.snapshot().reservoir_rng,
                "Cambiar el ámbito alteró los sorteos de v2");
    }
    require(first.retained_records() == resumed.retained_records() &&
                first.snapshot().reservoir_rng == resumed.snapshot().reservoir_rng,
            "La recuperación de v2 no conserva los siguientes sorteos");
    const auto before = serialize_memory(first.snapshot());
    rejected([&] { first.restore(second.snapshot()); });
    require(before == serialize_memory(first.snapshot()),
            "El rechazo de un ámbito ajeno modificó el banco de v2");
    EpisodicMemory high_seed(scope(), seed + (uint64_t{1} << seed_word_bits), small_capacity, 2);
    require(high_seed.snapshot().reservoir_rng !=
                EpisodicMemory(scope(), seed, small_capacity, 2).snapshot().reservoir_rng,
            "La semilla de v2 perdió sus 32 bits superiores");
}

void memory_contract_versions_are_explicit_and_incompatible() {
    EpisodicMemory legacy(scope(), seed, small_capacity);
    EpisodicMemory explicit_legacy(scope(), seed, small_capacity, 1);
    EpisodicMemory causal(scope(), seed, small_capacity, 2);
    for (uint64_t id = 1; id <= dense_records; ++id) {
        write(legacy, record(id));
        write(explicit_legacy, record(id));
    }
    require(serialize_memory(legacy.snapshot()) == serialize_memory(explicit_legacy.snapshot()),
            "La selección explícita de v1 cambió su archivo o sus sorteos");
    require(deserialize_memory(serialize_memory(causal.snapshot())).schema_version == 2,
            "El archivo de memoria perdió la versión de v2");
    rejected([&] { causal.restore(legacy.snapshot()); });
    rejected([&] { legacy.restore(causal.snapshot()); });
    rejected([&] { EpisodicMemory invalid(scope(), seed, small_capacity, 0); });
    rejected([&] { EpisodicMemory invalid(scope(), seed, small_capacity, 3); });
    for (const std::string partition : {"train", "validation", "calibration", "evaluation"}) {
        auto current = scope();
        current.partition = partition;
        EpisodicMemory phase(current, seed, small_capacity, 2);
        write(phase, record(1));
        phase.restore(deserialize_memory(serialize_memory(phase.snapshot())));
        require(phase.seen() == 1, "La partición de v2 no se recupera");
        if (partition == "calibration" || partition == "evaluation") {
            rejected([&] { EpisodicMemory invalid(current, seed, small_capacity); });
        }
    }
    auto invalid = causal.snapshot();
    invalid.schema_version = 3;
    rejected([&] { static_cast<void>(serialize_memory(invalid)); });
}
} // namespace

int main() {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        writes_are_staged_and_provisional_queries_match_commit();
        time_cutoff_self_exclusion_and_ties_are_explicit();
        matrix_query_matches_an_independent_accurate_reference();
        reservoir_recovers_its_rng_and_snapshot_owns_data();
        stale_and_foreign_tokens_are_rejected_without_changes();
        malformed_inputs_and_incompatible_scopes_fail_atomically();
        reservoir_inclusion_is_not_biased_to_recent_or_early_records();
        abandoned_queries_leave_committed_keys_and_rng_unchanged();
        failed_provisional_query_restores_the_committed_row();
        queries_bound_temporary_storage_to_scores();
        external_retention_changes_only_the_selected_records();
        external_retention_rejects_invalid_batches_without_changing_the_bank();
        causal_rng_pairs_scopes_but_keeps_recovery_isolated();
        memory_contract_versions_are_explicit_and_incompatible();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Memoria episódica, reservorio, temporalidad y recuperación comprobados\n";
    return 0;
}
