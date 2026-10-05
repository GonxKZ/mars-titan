#include "mars_titan/ppo_checkpoints.hpp"
#include "mars_titan/simulation_files.hpp"

#include <algorithm>
#include <array>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {
using Json = nlohmann::json;
using mars_titan::learning::PpoCheckpointStore;
using mars_titan::simulation::atomic_json_file;
using mars_titan::simulation::content_sha256;
using mars_titan::simulation::read_bounded_file;
constexpr std::size_t small_file_limit = std::size_t{64} * 1024;
constexpr int test_seed = 71;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Function> void rejected(Function&& action, std::string_view message) {
    try {
        std::forward<Function>(action)();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}

class TemporaryDirectory {
  public:
    TemporaryDirectory() {
        auto pattern = (std::filesystem::temp_directory_path() / "mars-ppo-test-XXXXXX").string();
        const auto* created = ::mkdtemp(pattern.data());
        if (created == nullptr) {
            throw std::runtime_error("No se puede preparar el directorio de prueba");
        }
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

Json identity() { return Json{{"experiment", "ppo-local"}, {"seed", test_seed}, {"schema", 1}}; }

std::vector<std::filesystem::path> bundles(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> result;
    for (const auto& item : std::filesystem::directory_iterator(directory)) {
        if (item.is_directory() && item.path().filename().string().starts_with("ppo-")) {
            result.push_back(item.path());
        }
    }
    return result;
}

void write_bytes(const std::filesystem::path& path, std::string_view bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        throw std::runtime_error("No se puede preparar el archivo de prueba");
    }
}

void archives_survive_reopening_and_exclusive_access() {
    TemporaryDirectory temporary;
    const auto output = temporary.path / "output";
    const std::string policy("pesos\0\xff", 7);
    const std::string rollout("pasos\0\x01", 7);
    const Json metadata{{"cursor", 4}, {"environment", Json{{"cash", 1000.0}}}};
    {
        PpoCheckpointStore store(output, identity());
        rejected([&] { PpoCheckpointStore other(output, identity(), true); },
                 "El bloqueo debe excluir una segunda ejecución");
        const auto saved = store.save(metadata, policy, rollout, true);
        require(saved.at("selected_best") == true, "Falta la selección explícita del mejor");
    }
    PpoCheckpointStore resumed(output, identity(), true);
    const auto latest = resumed.load_latest();
    require(latest.metadata == metadata && latest.policy_archive == policy &&
                latest.rollout_archive == rollout,
            "La reapertura no conserva exactamente los tres archivos");
    require(latest.receipt.at("recovered_from_previous") == false,
            "La lectura íntegra no debe aparentar una recuperación anterior");
    require(resumed.load_best().metadata == metadata, "Se ha perdido el mejor checkpoint");
}

void retention_keeps_two_recent_and_the_selected_best() {
    TemporaryDirectory temporary;
    const auto output = temporary.path / "output";
    PpoCheckpointStore store(output, identity());
    constexpr int checkpoints_before_improvement = 5;
    constexpr int improved_cursor = checkpoints_before_improvement + 1;
    for (int cursor = 1; cursor <= checkpoints_before_improvement; ++cursor) {
        static_cast<void>(store.save(Json{{"cursor", cursor}}, "pesos", "rollout", cursor == 1));
    }
    require(bundles(output).size() == 3, "La retención debe limitarse a dos recientes y el mejor");
    require(store.load_best().metadata.at("cursor") == 1, "El mejor no cambia sin selección");
    require(store.load_latest().metadata.at("cursor") == checkpoints_before_improvement,
            "El reciente no conserva el cursor");
    static_cast<void>(store.save(Json{{"cursor", improved_cursor}}, "pesos", "rollout", true));
    require(bundles(output).size() == 2 &&
                store.load_best().metadata.at("cursor") == improved_cursor,
            "Una mejora debe liberar el mejor anterior tras confirmar el índice");
}

void repeated_checkpoint_is_idempotent_and_foreign_files_survive() {
    TemporaryDirectory temporary;
    const auto output = temporary.path / "output";
    PpoCheckpointStore store(output, identity());
    const auto first = store.save(Json{{"cursor", 1}}, "pesos", "rollout");
    const auto repeated = store.save(Json{{"cursor", 1}}, "pesos", "rollout", true);
    require(first.at("bundle") == repeated.at("bundle") && bundles(output).size() == 1,
            "Repetir el mismo estado no debe duplicarlo");
    write_bytes(output / "weights-external.pt", "contenido ajeno");
    const auto foreign = output / "ppo-externo";
    std::filesystem::create_directory(foreign);
    write_bytes(foreign / "weights.pt", "pesos ajenos");
    constexpr int replacement_checkpoints = 5;
    for (int cursor = 2; cursor < 2 + replacement_checkpoints; ++cursor) {
        static_cast<void>(store.save(Json{{"cursor", cursor}}, "pesos", "rollout", true));
    }
    require(read_bounded_file(output / "weights-external.pt", small_file_limit) ==
                    "contenido ajeno" &&
                read_bounded_file(foreign / "weights.pt", small_file_limit) == "pesos ajenos",
            "La limpieza no puede alterar archivos ajenos");
}

void corrupted_latest_falls_back_with_an_explicit_receipt() {
    TemporaryDirectory temporary;
    const auto output = temporary.path / "output";
    Json latest;
    {
        PpoCheckpointStore store(output, identity());
        static_cast<void>(store.save(Json{{"cursor", 1}}, "pesos buenos", "rollout"));
        latest = store.save(Json{{"cursor", 2}}, "pesos nuevos", "rollout");
    }
    write_bytes(output / latest.at("bundle").get<std::string>() / "policy.pt", "truncado");
    PpoCheckpointStore resumed(output, identity(), true);
    const auto restored = resumed.load_latest();
    require(restored.metadata.at("cursor") == 1 && restored.policy_archive == "pesos buenos" &&
                restored.receipt.at("recovered_from_previous") == true &&
                !restored.receipt.at("discarded").empty(),
            "El fallback debe devolver el estado anterior y declarar la corrupción");
    static_cast<void>(resumed.save(Json{{"cursor", 3}}, "pesos recuperados", "rollout"));
    static_cast<void>(resumed.save(Json{{"cursor", 4}}, "pesos posteriores", "rollout"));
    require(resumed.load_latest().metadata.at("cursor") == 4 && bundles(output).size() == 2,
            "La continuación debe retirar también un checkpoint propio que quedó corrupto");
}

void identity_index_and_all_blobs_are_validated() {
    TemporaryDirectory temporary;
    const auto output = temporary.path / "output";
    Json receipt;
    {
        PpoCheckpointStore store(output, identity());
        receipt = store.save(Json{{"cursor", 1}}, "pesos", "rollout", true);
    }
    auto different = identity();
    different["seed"] = test_seed + 1;
    rejected([&] { PpoCheckpointStore wrong(output, different, true); },
             "Una identidad distinta debe rechazarse");
    const auto folder = output / receipt.at("bundle").get<std::string>();
    write_bytes(folder / "rollout.pt", "corrupt");
    {
        PpoCheckpointStore resumed(output, identity(), true);
        rejected([&] { static_cast<void>(resumed.load_latest()); },
                 "No se deben devolver pesos cuando el rollout está alterado");
        rejected([&] { static_cast<void>(resumed.load_best()); },
                 "El mejor también debe verificar todos sus archivos");
    }
    write_bytes(folder / "rollout.pt", "rollout");
    atomic_json_file(folder / "metadata.json", Json{{"cursor", 2}});
    {
        PpoCheckpointStore resumed(output, identity(), true);
        rejected([&] { static_cast<void>(resumed.load_latest()); },
                 "Los metadatos también deben conservar su huella");
    }
    const auto original_index = read_bounded_file(output / "ppo-index.json", small_file_limit);
    auto index = Json::parse(original_index);
    index["payload"]["best"] = nullptr;
    atomic_json_file(output / "ppo-index.json", index);
    rejected([&] { PpoCheckpointStore wrong(output, identity(), true); },
             "Un índice sintácticamente válido también debe conservar su checksum");
    index = Json::parse(original_index);
    index["payload"]["recent"][0]["bundle"] = "../source";
    index["sha256"] = content_sha256(index.at("payload").dump());
    atomic_json_file(output / "ppo-index.json", index);
    rejected([&] { PpoCheckpointStore wrong(output, identity(), true); },
             "El índice no puede introducir una ruta fuera de la salida");
    write_bytes(output / "ppo-index.json", "{\"payload\":{}}");
    rejected([&] { PpoCheckpointStore wrong(output, identity(), true); },
             "Un índice corrupto no debe reconstruirse ni ignorarse");
}

void symlinks_sources_and_unknown_bundle_contents_are_not_replaced() {
    TemporaryDirectory temporary;
    const auto source = temporary.path / "source";
    std::filesystem::create_directory(source);
    write_bytes(source / "policy.pt", "origen");
    rejected([&] { PpoCheckpointStore wrong(source, identity()); },
             "Una salida nueva no puede sobrescribir un directorio existente");
    const auto link = temporary.path / "link";
    std::filesystem::create_directory_symlink(source, link);
    rejected([&] { PpoCheckpointStore wrong(link / "new", identity()); },
             "Una ruta con enlaces simbólicos debe rechazarse");
    const auto output = temporary.path / "output";
    PpoCheckpointStore store(output, identity());
    const auto saved = store.save(Json{{"cursor", 1}}, "pesos", "rollout");
    const auto policy = output / saved.at("bundle").get<std::string>() / "policy.pt";
    std::filesystem::remove(policy);
    std::filesystem::create_symlink(source / "policy.pt", policy);
    rejected([&] { static_cast<void>(store.load_latest()); },
             "Un blob enlazado al exterior debe rechazarse");
    rejected([&] { static_cast<void>(store.save(Json{{"cursor", 1}}, "pesos", "rollout")); },
             "La repetición no debe sustituir un archivo desconocido");
    require(read_bounded_file(source / "policy.pt", small_file_limit) == "origen",
            "El archivo de origen se ha modificado");
}

void confirmed_identity_recovers_an_interrupted_initialization() {
    TemporaryDirectory temporary;
    const auto output = temporary.path / "output";
    {
        PpoCheckpointStore initial(output, identity());
    }
    std::filesystem::remove(output / "ppo-index.json");
    PpoCheckpointStore resumed(output, identity(), true);
    static_cast<void>(resumed.save(Json{{"cursor", 0}}, "inicial", "vacío serializado", true));
    require(resumed.load_latest().metadata.at("cursor") == 0,
            "La identidad confirmada debe permitir completar el arranque interrumpido");
}

void interrupted_blob_write_can_complete_without_replacing_existing_files() {
    TemporaryDirectory temporary;
    const auto output = temporary.path / "output";
    std::string previous_index;
    Json interrupted;
    {
        PpoCheckpointStore store(output, identity());
        static_cast<void>(store.save(Json{{"cursor", 1}}, "anterior", "rollout"));
        previous_index = read_bounded_file(output / "ppo-index.json", small_file_limit);
        interrupted = store.save(Json{{"cursor", 2}}, "posterior", "rollout");
    }
    write_bytes(output / "ppo-index.json", previous_index);
    const auto directory = output / interrupted.at("bundle").get<std::string>();
    std::filesystem::remove(directory / "policy.pt");
    const auto pending = directory / ".policy.pt.pending-ABC123";
    write_bytes(pending, "post");
    PpoCheckpointStore resumed(output, identity(), true);
    require(resumed.load_latest().metadata.at("cursor") == 1,
            "El índice anterior debe seguir siendo recuperable durante una escritura parcial");
    static_cast<void>(resumed.save(Json{{"cursor", 2}}, "posterior", "rollout"));
    require(
        resumed.load_latest().policy_archive == "posterior" && !std::filesystem::exists(pending),
        "El reintento debe completar el archivo y retirar su temporal tras confirmar el índice");
}

void failed_index_publication_preserves_the_previous_checkpoint() {
    TemporaryDirectory temporary;
    const auto output = temporary.path / "output";
    PpoCheckpointStore store(output, identity());
    for (int cursor = 1; cursor <= 3; ++cursor) {
        static_cast<void>(store.save(Json{{"cursor", cursor}}, "pesos", "rollout", cursor == 1));
    }
    std::uintmax_t largest_bundle_file = 0;
    for (const auto& folder : bundles(output)) {
        for (const auto& file : std::filesystem::directory_iterator(folder)) {
            largest_bundle_file = std::max(largest_bundle_file, file.file_size());
        }
    }
    const auto child = ::fork();
    require(child >= 0, "No se pudo crear el proceso que simula falta de espacio");
    if (child == 0) {
        static_cast<void>(std::signal(SIGXFSZ, SIG_IGN));
        const auto limit = static_cast<rlim_t>(largest_bundle_file + 16);
        const rlimit restricted{limit, limit};
        if (::setrlimit(RLIMIT_FSIZE, &restricted) != 0) {
            ::_exit(2);
        }
        try {
            static_cast<void>(store.save(Json{{"cursor", 4}}, "pesos", "rollout"));
        } catch (const std::exception&) {
            ::_exit(0);
        }
        ::_exit(3);
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "La publicación debe fallar al superar el presupuesto de archivo del proceso");
    require(store.load_latest().metadata.at("cursor") == 3 &&
                store.load_best().metadata.at("cursor") == 1,
            "Una publicación fallida no debe borrar el estado anterior ni el mejor");
    require(bundles(output).size() == 3,
            "Una publicación fallida no debe acumular otro bundle sin confirmar");
}

void empty_and_oversized_archives_fail_before_publication() {
    TemporaryDirectory temporary;
    PpoCheckpointStore store(temporary.path / "output", identity());
    rejected([&] { static_cast<void>(store.save(Json::object(), "", "rollout")); },
             "Un archivo vacío no debe publicarse");
    const std::string oversized_metadata(mars_titan::learning::maximum_ppo_metadata_bytes, 'x');
    rejected([&] { static_cast<void>(store.save(Json{{"text", oversized_metadata}}, "p", "r")); },
             "Los metadatos que exceden 32 MiB deben rechazarse");
    require(bundles(temporary.path / "output").empty(),
            "Los estados rechazados no deben dejar bundles publicados");
    std::string oversized_archive(mars_titan::learning::maximum_ppo_archive_bytes + 1, 'x');
    rejected([&] { static_cast<void>(store.save(Json::object(), oversized_archive, "r")); },
             "Los pesos que exceden 128 MiB deben rechazarse");
    rejected([&] { static_cast<void>(store.save(Json::object(), "p", oversized_archive)); },
             "El rollout que excede 128 MiB debe rechazarse");
}
} // namespace

int main() {
    try {
        archives_survive_reopening_and_exclusive_access();
        retention_keeps_two_recent_and_the_selected_best();
        repeated_checkpoint_is_idempotent_and_foreign_files_survive();
        corrupted_latest_falls_back_with_an_explicit_receipt();
        identity_index_and_all_blobs_are_validated();
        symlinks_sources_and_unknown_bundle_contents_are_not_replaced();
        confirmed_identity_recovers_an_interrupted_initialization();
        interrupted_blob_write_can_complete_without_replacing_existing_files();
        failed_index_publication_preserves_the_previous_checkpoint();
        empty_and_oversized_archives_fail_before_publication();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Persistencia PPO, integridad, retención y recuperación comprobadas\n";
    return 0;
}
