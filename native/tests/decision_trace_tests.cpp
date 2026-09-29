#include "mars_titan/decision_trace.hpp"
#include "mars_titan/simulation_files.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace mars_titan::learning;
constexpr std::size_t digest_width = 64;
constexpr std::size_t manifest_limit = std::size_t{4} * 1024 * 1024;
constexpr int64_t fixture_time = 100;
constexpr double fixture_critic = 0.1;
constexpr double fixture_reward = 0.01;
constexpr int defect_count = 6;
constexpr int future_outcome_defect = 5;
constexpr std::size_t tiny_budget = 1024;
constexpr std::size_t reserved_shard_budget = std::size_t{1} * 1024 * 1024 + 2048;
constexpr std::size_t staged_blocks = 8;

void require(bool value, std::string_view message) {
    if (!value) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Function> void rejected(Function&& action) {
    try {
        std::forward<Function>(action)();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("Se aceptó una traza que debía rechazarse");
}

class TemporaryDirectory {
  public:
    TemporaryDirectory() {
        auto pattern = (std::filesystem::temp_directory_path() / "mars-trace-test-XXXXXX").string();
        const auto* created = ::mkdtemp(pattern.data());
        require(created != nullptr, "No se pudo preparar la carpeta temporal");
        path = created;
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    TemporaryDirectory(TemporaryDirectory&&) = delete;
    TemporaryDirectory& operator=(TemporaryDirectory&&) = delete;
    std::filesystem::path path;
};

DecisionRecord record(uint64_t id) {
    DecisionRecord result;
    result.decision_id = id;
    result.world_sha256.assign(digest_width, 'a');
    result.context_sha256.assign(digest_width, 'b');
    result.decision_at = fixture_time;
    result.outcome_at = fixture_time + 1;
    result.probabilities.fill(1.0F / static_cast<float>(trace_action_count));
    result.critic = fixture_critic;
    result.reward = fixture_reward;
    result.reward_valid = true;
    result.costs_delta = fixture_reward;
    return result;
}

nlohmann::json trace_index(const std::filesystem::path& output) {
    return mars_titan::simulation::parse_bounded_json(
        mars_titan::simulation::read_bounded_file(output / "trace-index.json", manifest_limit));
}

void append_blocks(TraceWriter& writer, std::size_t blocks) {
    std::vector<DecisionRecord> records;
    records.reserve(trace_shard_records);
    for (std::size_t block = 0; block < blocks; ++block) {
        records.clear();
        const auto first = writer.cursor() + 1;
        for (std::size_t offset = 0; offset < trace_shard_records; ++offset) {
            records.push_back(record(first + offset));
        }
        writer.append(records);
    }
}

std::size_t parquet_files(const std::filesystem::path& output) {
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(output)) {
        count += entry.path().extension() == ".parquet" ? 1 : 0;
    }
    return count;
}

std::size_t stored_bytes(const std::filesystem::path& output) {
    std::size_t bytes = 0;
    for (const auto& entry : std::filesystem::directory_iterator(output)) {
        if (entry.is_regular_file()) {
            bytes += static_cast<std::size_t>(entry.file_size());
        }
    }
    return bytes;
}

void partial_shard_rewind_preserves_confirmed_prefix_and_foreign_files() {
    TemporaryDirectory directory;
    const auto output = directory.path / "trace";
    const std::string identity(digest_width, 'c');
    {
        TraceWriter writer(output, identity);
        const std::array first{record(1), record(2), record(3)};
        writer.append(first);
        require(writer.cursor() == 3 && writer.bytes() > 0, "Falta la reserva del lote admitido");
        writer.flush();
        const auto bytes = writer.bytes();
        writer.flush();
        require(writer.bytes() == bytes, "El flush repetido no debe añadir archivos");
        rejected([&] { TraceWriter other(output, identity, true, 3); });
    }
    {
        std::ofstream unrelated(output / "foreign.txt");
        unrelated << "original";
    }
    {
        TraceWriter writer(output, identity, true, 2);
        require(writer.cursor() == 2, "La recuperación debe descartar la cola no confirmada");
        writer.append(std::array{record(3)});
        writer.flush();
    }
    TraceWriter restored(output, identity, true, 3);
    require(restored.cursor() == 3 && std::filesystem::is_regular_file(output / "foreign.txt"),
            "La recuperación alteró un archivo ajeno o perdió el prefijo");
}

void invalid_rows_and_capacity_do_not_advance_the_cursor() {
    TemporaryDirectory directory;
    TraceWriter writer(directory.path / "trace", std::string(digest_width, 'c'));
    writer.append(std::array{record(1)});
    const auto before = writer.bytes();
    for (int defect = 0; defect < defect_count; ++defect) {
        auto invalid = record(2);
        if (defect == 0) {
            invalid.decision_id = 1;
        }
        if (defect == 1) {
            invalid.action = static_cast<uint8_t>(trace_action_count);
        }
        if (defect == 2) {
            invalid.reward = std::numeric_limits<double>::quiet_NaN();
        }
        if (defect == 3) {
            invalid.probabilities.fill(1);
        }
        if (defect == 4) {
            invalid.retrieved_count = 1;
            invalid.matured_at[0] = fixture_time + 1;
        }
        if (defect == future_outcome_defect) {
            invalid.outcome_at = invalid.decision_at;
        }
        rejected([&] { writer.append(std::array{invalid}); });
        require(writer.cursor() == 1 && writer.bytes() == before,
                "Una fila rechazada cambia el estado de la traza");
    }
    writer.flush();
    TraceWriter small(directory.path / "small", std::string(digest_width, 'c'), false, 0,
                      tiny_budget);
    bool capacity = false;
    try {
        small.append(std::array{record(1)});
    } catch (const TraceCapacityError&) {
        capacity = true;
    }
    require(capacity && small.cursor() == 0, "El límite debe pedir pausa antes de admitir filas");
}

void batches_are_atomic_and_reserved_rows_always_fit_at_flush() {
    TemporaryDirectory directory;
    const auto output = directory.path / "trace";
    TraceWriter writer(output, std::string(digest_width, 'c'), false, 0, reserved_shard_budget);
    std::vector<DecisionRecord> first;
    for (uint64_t id = 1; id < trace_shard_records; ++id) {
        first.push_back(record(id));
    }
    writer.append(first);
    const auto initial_index =
        mars_titan::simulation::read_bounded_file(output / "trace-index.json", manifest_limit);
    const auto before = writer.bytes();
    const std::array next{record(trace_shard_records), record(trace_shard_records + 1)};
    rejected([&] { writer.append(next); });
    require(writer.cursor() == trace_shard_records - 1 && writer.bytes() == before &&
                mars_titan::simulation::read_bounded_file(output / "trace-index.json",
                                                          manifest_limit) == initial_index,
            "La falta de reserva confirmó solo una parte del lote");
    writer.flush();
    require(writer.cursor() == trace_shard_records - 1 && writer.bytes() < before,
            "Las filas reservadas no se pudieron confirmar dentro de su presupuesto");
}

void full_batch_write_failure_preserves_the_pending_prefix() {
    TemporaryDirectory directory;
    const auto output = directory.path / "trace";
    TraceWriter writer(output, std::string(digest_width, 'c'));
    writer.append(std::array{record(1)});
    const auto original =
        mars_titan::simulation::read_bounded_file(output / "trace-index.json", manifest_limit);
    const auto before = writer.bytes();
    const auto moved = directory.path / "moved";
    std::filesystem::rename(output, moved);
    {
        std::ofstream obstruction(output);
        obstruction << "El destino no es una carpeta";
    }
    std::vector<DecisionRecord> following;
    for (uint64_t id = 2; id <= trace_shard_records; ++id) {
        following.push_back(record(id));
    }
    rejected([&] { writer.append(following); });
    require(writer.cursor() == 1 && writer.bytes() == before,
            "El fallo de publicación cambió el cursor o el buffer admitido");
    std::filesystem::remove(output);
    std::filesystem::rename(moved, output);
    require(mars_titan::simulation::read_bounded_file(output / "trace-index.json",
                                                      manifest_limit) == original,
            "El fallo al escribir un bloque cambió el índice confirmado");
    writer.append(following);
    require(writer.cursor() == trace_shard_records,
            "El lote válido no conserva el prefijo después de un fallo");
    writer.flush();
}

void complete_blocks_are_published_only_by_flush_and_recovery_prunes_the_tail() {
    TemporaryDirectory directory;
    const auto output = directory.path / "trace";
    const std::string identity(digest_width, 'c');
    const auto confirmed = staged_blocks * trace_shard_records;
    {
        TraceWriter writer(output, identity);
        append_blocks(writer, staged_blocks);
        require(writer.cursor() == confirmed && parquet_files(output) == staged_blocks,
                "El bloque completo no se escribió de forma incremental");
        require(trace_index(output).at("payload").at("cursor") == 0,
                "append publicó decisiones antes del flush de checkpoint");
    }
    rejected([&] { TraceWriter invalid(output, identity, true, 1); });
    require(parquet_files(output) == staged_blocks,
            "Un cursor imposible eliminó bloques antes de validar el checkpoint");
    {
        TraceWriter writer(output, identity, true, 0);
        require(writer.cursor() == 0 && parquet_files(output) == 0,
                "La recuperación no retiró los ocho bloques propios sin confirmar");
        append_blocks(writer, staged_blocks);
        writer.flush();
        require(trace_index(output).at("payload").at("cursor") == confirmed,
                "flush no publicó todos los bloques admitidos");
        require(writer.bytes() == stored_bytes(output),
                "Los contadores incrementales no coinciden con los archivos confirmados");
        append_blocks(writer, staged_blocks);
        require(trace_index(output).at("payload").at("cursor") == confirmed,
                "Una cola posterior al checkpoint sustituyó su índice");
    }
    {
        TraceWriter writer(output, identity, true, confirmed);
        require(writer.cursor() == confirmed && parquet_files(output) == staged_blocks,
                "La recuperación perdió el prefijo o conservó una cola sin índice");
        writer.append(std::array{record(confirmed + 1)});
        writer.flush();
    }
    TraceWriter previous_checkpoint(output, identity, true, confirmed);
    require(previous_checkpoint.cursor() == confirmed && parquet_files(output) == staged_blocks,
            "Un corte tras flush y antes del checkpoint no conserva el estado anterior");
    require(previous_checkpoint.bytes() == stored_bytes(output),
            "La recuperación no reconstruye el recuento de bytes confirmado");
}

void failed_flush_preserves_all_accepted_records_for_a_retry() {
    TemporaryDirectory directory;
    const auto output = directory.path / "trace";
    TraceWriter writer(output, std::string(digest_width, 'c'));
    append_blocks(writer, staged_blocks);
    const auto original =
        mars_titan::simulation::read_bounded_file(output / "trace-index.json", manifest_limit);
    const auto before = writer.bytes();
    mars_titan::simulation::atomic_binary_file(output / "trace-index.json", "alterado",
                                               manifest_limit);
    rejected([&] { writer.flush(); });
    require(writer.cursor() == staged_blocks * trace_shard_records && writer.bytes() == before,
            "Un fallo de flush cambia los registros ya admitidos");
    mars_titan::simulation::atomic_binary_file(output / "trace-index.json", original,
                                               manifest_limit);
    writer.flush();
    require(trace_index(output).at("payload").at("cursor") == writer.cursor(),
            "El reintento no publicó el prefijo admitido íntegro");
}

void incremental_capacity_rejects_a_full_block_before_creating_its_file() {
    TemporaryDirectory directory;
    const std::string identity(digest_width, 'c');
    std::size_t budget = 0;
    {
        TraceWriter probe(directory.path / "probe", identity);
        append_blocks(probe, 1);
        probe.flush();
        budget = probe.bytes();
    }
    const auto output = directory.path / "limited";
    TraceWriter writer(output, identity, false, 0, budget);
    append_blocks(writer, 1);
    const auto before = writer.bytes();
    bool capacity = false;
    try {
        append_blocks(writer, 1);
    } catch (const TraceCapacityError&) {
        capacity = true;
    }
    require(capacity && writer.cursor() == trace_shard_records && writer.bytes() == before &&
                parquet_files(output) == 1 && trace_index(output).at("payload").at("cursor") == 0,
            "El límite incremental admitió o publicó parte del siguiente bloque");
    writer.flush();
    require(writer.bytes() == stored_bytes(output) && writer.bytes() <= budget,
            "El bloque admitido no pudo confirmarse dentro del presupuesto reservado");
}

void recovery_removes_only_own_unindexed_shards() {
    TemporaryDirectory directory;
    const auto output = directory.path / "trace";
    const std::string identity(digest_width, 'c');
    std::string original;
    std::filesystem::path own;
    {
        TraceWriter writer(output, identity);
        original =
            mars_titan::simulation::read_bounded_file(output / "trace-index.json", manifest_limit);
        writer.append(std::array{record(1)});
        writer.flush();
        const auto index = mars_titan::simulation::parse_bounded_json(
            mars_titan::simulation::read_bounded_file(output / "trace-index.json", manifest_limit));
        own = output / index.at("payload").at("shards").at(0).at("path").get<std::string>();
    }
    const auto foreign_output = directory.path / "other";
    std::filesystem::path foreign;
    {
        TraceWriter writer(foreign_output, std::string(digest_width, 'd'));
        writer.append(std::array{record(1)});
        writer.flush();
        const auto index =
            mars_titan::simulation::parse_bounded_json(mars_titan::simulation::read_bounded_file(
                foreign_output / "trace-index.json", manifest_limit));
        const auto name = index.at("payload").at("shards").at(0).at("path").get<std::string>();
        foreign = output / name;
        std::filesystem::copy_file(foreign_output / name, foreign);
    }
    mars_titan::simulation::atomic_binary_file(output / "trace-index.json", original,
                                               manifest_limit);
    TraceWriter restored(output, identity, true, 0);
    require(!std::filesystem::exists(own) && std::filesystem::is_regular_file(foreign),
            "La recuperación conserva un bloque huérfano propio o elimina uno ajeno");
}

void corrupt_shards_and_foreign_identities_are_rejected() {
    TemporaryDirectory directory;
    const auto output = directory.path / "trace";
    const std::string identity(digest_width, 'c');
    {
        TraceWriter writer(output, identity);
        writer.append(std::array{record(1)});
        writer.flush();
    }
    rejected([&] { TraceWriter other(output, std::string(digest_width, 'd'), true, 1); });
    const auto index = mars_titan::simulation::parse_bounded_json(
        mars_titan::simulation::read_bounded_file(output / "trace-index.json", manifest_limit));
    const auto shard =
        output / index.at("payload").at("shards").at(0).at("path").get<std::string>();
    {
        std::ofstream corrupted(shard);
        corrupted << "alterado";
    }
    rejected([&] { TraceWriter broken(output, identity, true, 1); });
}

void memory_sensitivity_requires_a_consistent_second_distribution() {
    TemporaryDirectory directory;
    TraceWriter writer(directory.path / "trace", std::string(digest_width, 'c'));
    auto first = record(1);
    first.memory_sensitivity = MemorySensitivity{first.probabilities, first.action, 1., false};
    rejected([&] { writer.append(std::array{first}); });
    require(writer.cursor() == 0, "Una distancia falsa de sensibilidad fue admitida");
    first.memory_sensitivity->probability_l1 = 0;
    writer.append(std::array{first});
    writer.flush();
}
} // namespace

int main() {
    try {
        partial_shard_rewind_preserves_confirmed_prefix_and_foreign_files();
        invalid_rows_and_capacity_do_not_advance_the_cursor();
        batches_are_atomic_and_reserved_rows_always_fit_at_flush();
        full_batch_write_failure_preserves_the_pending_prefix();
        complete_blocks_are_published_only_by_flush_and_recovery_prunes_the_tail();
        failed_flush_preserves_all_accepted_records_for_a_retry();
        incremental_capacity_rejects_a_full_block_before_creating_its_file();
        recovery_removes_only_own_unindexed_shards();
        corrupt_shards_and_foreign_identities_are_rejected();
        memory_sensitivity_requires_a_consistent_second_distribution();
        std::cout << "Trazas acotadas, recuperación y validación comprobadas\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
