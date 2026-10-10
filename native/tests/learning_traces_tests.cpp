// Trazas de aprendizaje nativas con el contrato de Python (#448): manifiesto, partes Parquet,
// recuperación por marcas de confirmación y agotamiento del presupuesto. No hay modelos.
#include "mars_titan/learning_traces.hpp"
#include "mars_titan/simulation_files.hpp"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {
using namespace mars_titan::learning;
using Json = nlohmann::json;
constexpr std::size_t budget = std::size_t{1} << 20;

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
    throw std::runtime_error("Se aceptó una operación que debía rechazarse");
}

class TemporaryDirectory {
  public:
    TemporaryDirectory() {
        auto pattern =
            (std::filesystem::temp_directory_path() / "mars-learning-traces-XXXXXX").string();
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

const Json identity{{"family", "ppo"}, {"run_identity_sha256", std::string(64, 'a')}};

Json manifest(const std::filesystem::path& folder) {
    return mars_titan::simulation::parse_bounded_json(
        mars_titan::simulation::read_bounded_file(folder / "manifest.json", budget));
}

std::shared_ptr<arrow::Table> part(const std::filesystem::path& path) {
    auto input = arrow::io::ReadableFile::Open(path.string()).ValueOrDie();
    auto reader = parquet::arrow::OpenFile(input, arrow::default_memory_pool()).ValueOrDie();
    return reader->ReadTable().ValueOrDie();
}

void parts_follow_the_python_contract() {
    TemporaryDirectory directory;
    const auto folder = directory.path / "traces";
    LearningTraceWriter writer(folder, {4, budget}, identity, "transition", {});
    writer.scalar(8, "train", "policy.loss", 0.25);
    writer.scalar(8, "train", "rl.grad_l2_total", 1.5, "all");
    writer.scalar(8, "validation", "rl.mean_log_growth", -0.125);
    require(writer.pending() == 3, "Faltan filas pendientes");
    writer.flush({8, 1});
    require(writer.pending() == 0 && writer.parts() == 1, "La parte no se escribió");
    const auto saved = manifest(folder);
    require(saved.at("kind") == "learning_traces" && saved.at("schema_version") == 1 &&
                saved.at("identity") == identity &&
                saved.at("config") == Json{{"every", 4}, {"max_bytes", budget}, {"enabled", true}} &&
                saved.at("step_unit") == "transition" &&
                saved.at("columns") ==
                    Json::array({"step", "phase", "metric", "group", "stat", "value"}) &&
                saved.at("stats").size() == 7 && saved.at("exhausted_at_step").is_null(),
            "El manifiesto no sigue el contrato de Python");
    const auto& listed = saved.at("native").at("parts");
    require(listed.size() == 1 && listed[0].at("sequence") == 1 &&
                listed[0].at("commit") == Json::array({8, 1}) &&
                listed[0].at("bytes") == writer.bytes() &&
                std::filesystem::file_size(folder / "parts/part-00000001.parquet") ==
                    writer.bytes(),
            "La parte no conserva su número, marca o tamaño");
    const auto table = part(folder / "parts/part-00000001.parquet");
    const auto expected = arrow::schema(
        {arrow::field("step", arrow::int64()), arrow::field("phase", arrow::utf8()),
         arrow::field("metric", arrow::utf8()), arrow::field("group", arrow::utf8()),
         arrow::field("stat", arrow::utf8()), arrow::field("value", arrow::float64())});
    require(table->schema()->Equals(*expected, false) && table->num_rows() == 3,
            "La parte no tiene el esquema de pyarrow");
    const auto values = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("value")->chunk(0));
    const auto stats = std::static_pointer_cast<arrow::StringArray>(
        table->GetColumnByName("stat")->chunk(0));
    require(values->Value(0) == 0.25 && values->Value(2) == -0.125 &&
                stats->GetString(1) == "value",
            "La parte no conserva los valores exactos");
    writer.flush({9, 0});
    require(writer.parts() == 1, "Un volcado sin filas no crea partes");
}

void cadence_counts_crossed_multiples() {
    TemporaryDirectory directory;
    const LearningTraceWriter writer(directory.path / "traces", {4, budget}, identity, "wave", {});
    require(!writer.due(0, 3) && writer.due(3, 4) && writer.due(4, 9) && !writer.due(5, 7) &&
                writer.due(0, 100),
            "La cadencia no cuenta los múltiplos cruzados");
}

void recovery_keeps_only_confirmed_parts() {
    TemporaryDirectory directory;
    const auto folder = directory.path / "traces";
    {
        LearningTraceWriter writer(folder, {1, budget}, identity, "wave", {});
        for (const TraceCommit commit : {TraceCommit{0, 1}, TraceCommit{1, 0}, TraceCommit{1, 1},
                                         TraceCommit{2, 0}}) {
            writer.scalar(commit.major, commit.minor == 0 ? "train" : "validation", "rl.metric",
                          static_cast<double>(commit.major));
            writer.flush(commit);
        }
        require(writer.parts() == 4, "Faltan partes antes del corte");
    }
    // Restos de una escritura cortada que nunca llegó al manifiesto.
    std::ofstream(folder / "parts/part-00000005.parquet") << "sin confirmar";
    std::ofstream(folder / "parts/.part-00000006.parquet.pending-abcdef") << "temporal";
    LearningTraceWriter resumed(folder, {1, budget}, identity, "wave", {1, 1});
    require(resumed.parts() == 3 && !std::filesystem::exists(folder / "parts/part-00000004.parquet") &&
                !std::filesystem::exists(folder / "parts/part-00000005.parquet") &&
                !std::filesystem::exists(folder / "parts/.part-00000006.parquet.pending-abcdef") &&
                manifest(folder).at("native").at("parts").size() == 3,
            "La recuperación no borra las partes sin confirmar");
    resumed.scalar(2, "train", "rl.metric", 2.);
    resumed.flush({2, 0});
    require(resumed.parts() == 4 && std::filesystem::exists(folder / "parts/part-00000004.parquet"),
            "La parte repetida no recupera su número");
    resumed.scalar(3, "train", "rl.metric", 3.);
    rejected([&] { resumed.flush({1, 0}); });
}

void recovery_rejects_other_runs_and_changed_parts() {
    TemporaryDirectory directory;
    const auto folder = directory.path / "traces";
    {
        LearningTraceWriter writer(folder, {2, budget}, identity, "transition", {});
        writer.scalar(2, "train", "policy.loss", 1.);
        writer.flush({2, 0});
    }
    auto other = identity;
    other["run_identity_sha256"] = std::string(64, 'b');
    rejected([&] { LearningTraceWriter changed(folder, {2, budget}, other, "transition", {2, 0}); });
    rejected([&] { LearningTraceWriter changed(folder, {3, budget}, identity, "transition", {2, 0}); });
    rejected([&] { LearningTraceWriter changed(folder, {2, budget}, identity, "wave", {2, 0}); });
    std::ofstream(folder / "parts/part-00000001.parquet", std::ios::app) << "x";
    rejected([&] { LearningTraceWriter changed(folder, {2, budget}, identity, "transition", {2, 0}); });
    std::ofstream(folder / "parts/notes.txt") << "ajeno";
    rejected([&] { LearningTraceWriter changed(folder, {2, budget}, identity, "transition", {0, 0}); });
}

void invalid_names_steps_and_options_are_rejected() {
    TemporaryDirectory directory;
    LearningTraceWriter writer(directory.path / "traces", {1, budget}, identity, "transition", {});
    rejected([&] { writer.scalar(1, "Train", "policy.loss", 1.); });
    rejected([&] { writer.scalar(1, "train", "1loss", 1.); });
    rejected([&] { writer.scalar(1, "train", "policy loss", 1.); });
    rejected([&] { writer.scalar(1, "train", "policy.loss", 1., std::string(97, 'a')); });
    rejected([&] { writer.scalar(uint64_t{1} << 63U, "train", "policy.loss", 1.); });
    require(writer.pending() == 0, "Se admitió una fila inválida");
    rejected([&] { LearningTraceWriter off(directory.path / "off", {0, budget}, identity, "wave", {}); });
    rejected([&] { LearningTraceWriter small(directory.path / "small", {1, 512}, identity, "wave", {}); });
    rejected([&] { LearningTraceWriter unit(directory.path / "unit", {1, budget}, identity, "epoch", {}); });
}

void exhaustion_is_declared_and_rewound_with_its_commit() {
    TemporaryDirectory directory;
    std::size_t single = 0;
    {
        LearningTraceWriter probe(directory.path / "probe", {1, budget}, identity, "wave", {});
        probe.scalar(1, "train", "rl.metric", 1.);
        probe.flush({1, 0});
        single = probe.bytes();
    }
    // Cabe una parte de una fila y no una de muchas.
    const auto limit = single + single / 2;
    const auto folder = directory.path / "traces";
    {
        LearningTraceWriter writer(folder, {1, limit}, identity, "wave", {});
        writer.scalar(1, "train", "rl.metric", 1.);
        writer.flush({1, 0});
        require(writer.parts() == 1 && writer.active(), "La primera parte debía caber");
        for (int step = 2; step < 64; ++step) {
            writer.scalar(static_cast<uint64_t>(step), "train", "rl.metric", 1.);
        }
        writer.flush({2, 0});
        require(!writer.active() && writer.exhausted_at() == 2 && writer.parts() == 1 &&
                    manifest(folder).at("exhausted_at_step") == 2,
                "El agotamiento no declara el primer paso perdido");
        writer.scalar(70, "train", "rl.metric", 1.);
        require(writer.pending() == 0, "Unas trazas agotadas no admiten filas");
    }
    {
        const LearningTraceWriter kept(folder, {1, limit}, identity, "wave", {2, 0});
        require(!kept.active() && kept.exhausted_at() == 2, "El agotamiento confirmado se perdió");
    }
    const LearningTraceWriter rewound(folder, {1, limit}, identity, "wave", {1, 0});
    require(rewound.active() && !rewound.exhausted_at() && rewound.parts() == 1 &&
                manifest(folder).at("exhausted_at_step").is_null(),
            "Un agotamiento sin confirmar sigue declarado");
}
} // namespace

int main() {
    try {
        parts_follow_the_python_contract();
        cadence_counts_crossed_multiples();
        recovery_keeps_only_confirmed_parts();
        recovery_rejects_other_runs_and_changed_parts();
        invalid_names_steps_and_options_are_rejected();
        exhaustion_is_declared_and_rewound_with_its_commit();
        std::cout << "Trazas de aprendizaje nativas comprobadas\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
