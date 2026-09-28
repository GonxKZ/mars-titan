#include "mars_titan/ppo_checkpoints.hpp"
#include "mars_titan/simulation_files.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mars_titan::learning {
namespace {
using Json = nlohmann::json;
using simulation::atomic_binary_file;
using simulation::atomic_json_file;
using simulation::content_sha256;
using simulation::parse_bounded_json;
using simulation::read_bounded_file;
using simulation::read_json_int64;
using simulation::require_safe_path;
constexpr std::size_t index_limit = 64 * 1024;
constexpr std::size_t identity_limit = 1024 * 1024;
constexpr std::size_t digest_length = 64;
constexpr std::size_t retained_recent = 2;
constexpr std::size_t maximum_retired = 3;
constexpr std::array<std::string_view, 3> payload_names{"metadata.json", "policy.pt", "rollout.pt"};

bool valid_digest(std::string_view digest) {
    return digest.size() == digest_length && std::ranges::all_of(digest, [](char ch) {
               return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
           });
}

std::string object_bytes(const Json& value, std::size_t maximum) {
    if (!value.is_object()) {
        throw std::invalid_argument("La identidad y los metadatos deben ser objetos JSON");
    }
    const auto bytes = value.dump();
    if (bytes.size() > maximum || parse_bounded_json(bytes) != value) {
        throw std::invalid_argument(
            "El JSON excede su presupuesto o contiene valores no serializables");
    }
    return bytes;
}

Json sealed(const Json& payload) {
    return Json{{"payload", payload}, {"sha256", content_sha256(payload.dump())}};
}

Json unseal(const Json& envelope) {
    if (!envelope.is_object() || envelope.size() != 2 || !envelope.contains("payload") ||
        !envelope.contains("sha256") || !envelope.at("sha256").is_string() ||
        content_sha256(envelope.at("payload").dump()) != envelope.at("sha256").get<std::string>()) {
        throw std::invalid_argument("El índice no conserva su estructura y huella SHA256");
    }
    return envelope.at("payload");
}

std::filesystem::path checked_output(const std::filesystem::path& path) {
    if (path.empty()) {
        throw std::invalid_argument("La salida del checkpoint está vacía");
    }
    require_safe_path(path);
    return std::filesystem::absolute(path).lexically_normal();
}

void check_record(const Json& record) {
    if (!record.is_object() || record.size() != 2 || !record.contains("bundle") ||
        !record.contains("sha256") || !record.at("sha256").is_string() ||
        !record.at("bundle").is_string()) {
        throw std::invalid_argument("El índice contiene una referencia de checkpoint inválida");
    }
    const auto digest = record.at("sha256").get<std::string>();
    if (!valid_digest(digest) || record.at("bundle") != "ppo-" + digest) {
        throw std::invalid_argument(
            "La referencia de checkpoint no es una ruta local por contenido");
    }
}

std::vector<Json> active_records(const Json& index) {
    auto records = index.at("recent").get<std::vector<Json>>();
    if (!index.at("best").is_null() &&
        std::ranges::find(records, index.at("best")) == records.end()) {
        records.push_back(index.at("best"));
    }
    return records;
}

void validate_index(const Json& index, const std::string& identity_digest) {
    if (!index.is_object() || index.size() != 5 ||
        read_json_int64(index.at("schema_version")) != 1 ||
        index.at("identity_sha256") != identity_digest || !index.at("recent").is_array() ||
        index.at("recent").size() > retained_recent || !index.at("retired").is_array() ||
        index.at("retired").size() > maximum_retired) {
        throw std::invalid_argument("El índice no conserva identidad, versión o retención");
    }
    const auto active = active_records(index);
    for (const auto& record : active) {
        check_record(record);
    }
    if (index.at("recent").size() == retained_recent &&
        index.at("recent").front() == index.at("recent").back()) {
        throw std::invalid_argument("El índice duplica un checkpoint reciente");
    }
    std::vector<Json> retired;
    for (const auto& record : index.at("retired")) {
        check_record(record);
        if (std::ranges::find(active, record) != active.end() ||
            std::ranges::find(retired, record) != retired.end()) {
            throw std::invalid_argument("La limpieza incluye un estado activo o duplicado");
        }
        retired.push_back(record);
    }
}

std::size_t recorded_size(const Json& file, std::size_t maximum) {
    if (!file.is_object() || file.size() != 2 || !file.contains("bytes") ||
        !file.contains("sha256") || !file.at("sha256").is_string() ||
        !valid_digest(file.at("sha256").get<std::string>())) {
        throw std::invalid_argument(
            "El manifiesto no conserva el tamaño y la huella de un archivo");
    }
    const auto bytes = read_json_int64(file.at("bytes"));
    if (bytes <= 0 || static_cast<std::uint64_t>(bytes) > maximum) {
        throw std::invalid_argument("Un archivo del checkpoint supera su presupuesto o está vacío");
    }
    return static_cast<std::size_t>(bytes);
}

Json file_record(std::string_view bytes) {
    return Json{{"bytes", bytes.size()}, {"sha256", content_sha256(bytes)}};
}

std::size_t payload_limit(std::string_view name) {
    return name == "metadata.json" ? maximum_ppo_metadata_bytes : maximum_ppo_archive_bytes;
}

std::optional<std::size_t> pending_limit(std::string_view name) {
    constexpr std::size_t temporary_suffix_length = 6;
    constexpr std::array<std::string_view, 4> names{"metadata.json", "policy.pt", "rollout.pt",
                                                    "manifest.json"};
    for (const auto file : names) {
        const auto prefix = "." + std::string(file) + ".pending-";
        if (name.starts_with(prefix)) {
            const auto suffix = name.substr(prefix.size());
            if (suffix.size() == temporary_suffix_length &&
                std::ranges::all_of(suffix, [](char ch) {
                    return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'z') ||
                           (ch >= 'A' && ch <= 'Z');
                })) {
                return file == "manifest.json" ? index_limit : payload_limit(file);
            }
        }
    }
    return std::nullopt;
}

void check_directory_contents(const std::filesystem::path& directory) {
    require_safe_path(directory);
    std::size_t count = 0;
    std::size_t pending_count = 0;
    for (const auto& item : std::filesystem::directory_iterator(directory)) {
        const auto name = item.path().filename().string();
        const auto temporary_limit = pending_limit(name);
        if (++count > payload_names.size() + 2 ||
            (!temporary_limit && name != "manifest.json" &&
             std::ranges::find(payload_names, name) == payload_names.end())) {
            throw std::invalid_argument(
                "El directorio de checkpoint contiene archivos desconocidos");
        }
        require_safe_path(item.path());
        if (!item.is_regular_file()) {
            throw std::invalid_argument("El checkpoint contiene un archivo que no es regular");
        }
        if (temporary_limit && (++pending_count > 1 || item.file_size() > *temporary_limit)) {
            throw std::invalid_argument("Los temporales del checkpoint exceden su presupuesto");
        }
    }
}

void remove_pending_files(const std::filesystem::path& directory) {
    require_safe_path(directory);
    if (!std::filesystem::exists(directory)) {
        return;
    }
    check_directory_contents(directory);
    for (const auto& item : std::filesystem::directory_iterator(directory)) {
        if (pending_limit(item.path().filename().string())) {
            std::filesystem::remove(item.path());
        }
    }
}

void remove_owned_bundle(const std::filesystem::path& directory) {
    require_safe_path(directory);
    if (!std::filesystem::exists(directory)) {
        return;
    }
    check_directory_contents(directory);
    remove_pending_files(directory);
    for (const auto name : payload_names) {
        std::filesystem::remove(directory / name);
    }
    std::filesystem::remove(directory / "manifest.json");
    std::filesystem::remove(directory);
}
} // namespace

struct PpoCheckpointStore::Impl {
    Impl(const std::filesystem::path& destination, const Json& identity, bool resume)
        : output(checked_output(destination)),
          identity_bytes(object_bytes(identity, identity_limit)),
          identity_digest(content_sha256(identity_bytes)), lock(output, resume) {
        const Json identity_record{{"schema_version", 1},
                                   {"kind", "ppo_checkpoints"},
                                   {"identity", identity},
                                   {"sha256", identity_digest}};
        const auto expected = identity_record.dump();
        if (expected.size() > identity_limit) {
            throw std::invalid_argument("La identidad completa excede 1 MiB");
        }
        if (resume) {
            if (read_bounded_file(output / "identity.json", identity_limit) != expected) {
                throw std::invalid_argument(
                    "La salida pertenece a otra identidad o su identidad está alterada");
            }
        } else {
            atomic_json_file(output / "identity.json", identity_record, identity_limit);
        }
        identity_bytes = expected;
        const auto index_path = output / "ppo-index.json";
        require_safe_path(index_path);
        if (std::filesystem::exists(index_path)) {
            index_bytes = read_bounded_file(index_path, index_limit);
            index = unseal(parse_bounded_json(index_bytes));
            validate_index(index, identity_digest);
        } else {
            // Una identidad confirmada sin estados permite terminar el primer arranque.
            std::size_t entries = 0;
            for (const auto& item : std::filesystem::directory_iterator(output)) {
                const auto name = item.path().filename().string();
                if (++entries > 16 || (name != "identity.json" && name != ".run.lock" &&
                                       !name.starts_with(".ppo-index.json.pending-"))) {
                    throw std::invalid_argument(
                        "Falta el índice y la salida ya contiene otros archivos");
                }
            }
            index = Json{{"schema_version", 1},
                         {"identity_sha256", identity_digest},
                         {"recent", Json::array()},
                         {"best", nullptr},
                         {"retired", Json::array()}};
            publish(index);
        }
        clean_retired();
        for (const auto& record : active_records(index)) {
            remove_pending_files(output / record.at("bundle").get<std::string>());
        }
    }

    void require_unchanged() const {
        if (read_bounded_file(output / "identity.json", identity_limit) != identity_bytes ||
            read_bounded_file(output / "ppo-index.json", index_limit) != index_bytes) {
            throw std::invalid_argument("La identidad o el índice cambiaron durante la ejecución");
        }
    }

    Json manifest(const Json& record) const {
        check_record(record);
        const auto directory = output / record.at("bundle").get<std::string>();
        check_directory_contents(directory);
        const auto bytes = read_bounded_file(directory / "manifest.json", index_limit);
        if (content_sha256(bytes) != record.at("sha256").get<std::string>()) {
            throw std::invalid_argument("El manifiesto del checkpoint no conserva su SHA256");
        }
        auto value = parse_bounded_json(bytes);
        if (!value.is_object() || value.size() != 3 ||
            read_json_int64(value.at("schema_version")) != 1 ||
            value.at("identity_sha256") != identity_digest || !value.at("files").is_object() ||
            value.at("files").size() != payload_names.size()) {
            throw std::invalid_argument("El manifiesto del checkpoint no conserva su contrato");
        }
        for (const auto name : payload_names) {
            static_cast<void>(recorded_size(value.at("files").at(name), payload_limit(name)));
        }
        return value;
    }

    std::string payload(const std::filesystem::path& directory, std::string_view name,
                        const Json& description) const {
        const auto size = recorded_size(description, payload_limit(name));
        auto bytes = read_bounded_file(directory / name, size);
        if (bytes.size() != size ||
            content_sha256(bytes) != description.at("sha256").get<std::string>()) {
            throw std::invalid_argument("El archivo " + std::string(name) +
                                        " no conserva tamaño y SHA256");
        }
        return bytes;
    }

    PpoCheckpointBundle load(const Json& record) const {
        const auto description = manifest(record);
        const auto directory = output / record.at("bundle").get<std::string>();
        auto metadata_bytes =
            payload(directory, "metadata.json", description.at("files").at("metadata.json"));
        auto policy = payload(directory, "policy.pt", description.at("files").at("policy.pt"));
        auto rollout = payload(directory, "rollout.pt", description.at("files").at("rollout.pt"));
        auto metadata = parse_bounded_json(metadata_bytes);
        if (!metadata.is_object()) {
            throw std::invalid_argument("Los metadatos del checkpoint no son un objeto JSON");
        }
        return PpoCheckpointBundle{std::move(metadata), std::move(policy), std::move(rollout),
                                   record};
    }

    void clean_retired() const {
        for (const auto& record : index.at("retired")) {
            // El índice confirmado acredita la propiedad, también si el contenido se corrompió.
            remove_owned_bundle(output / record.at("bundle").get<std::string>());
        }
    }

    void publish(const Json& next) {
        const auto envelope = sealed(next);
        atomic_json_file(output / "ppo-index.json", envelope, index_limit);
        index = next;
        index_bytes = envelope.dump();
    }

    void ensure_bundle(const Json& record, const Json& description, std::string_view metadata,
                       std::string_view policy, std::string_view rollout) const {
        const auto directory = output / record.at("bundle").get<std::string>();
        require_safe_path(directory);
        if (!std::filesystem::create_directory(directory)) {
            check_directory_contents(directory);
        }
        const std::array<std::string_view, 3> payloads{metadata, policy, rollout};
        // Completa únicamente archivos ausentes. Un archivo previo se verifica y nunca se
        // sustituye.
        for (std::size_t position = 0; position < payload_names.size(); ++position) {
            const auto name = payload_names[position];
            const auto path = directory / name;
            require_safe_path(path);
            if (std::filesystem::exists(path)) {
                static_cast<void>(payload(directory, name, description.at("files").at(name)));
            }
        }
        const auto manifest_path = directory / "manifest.json";
        if (std::filesystem::exists(manifest_path) &&
            read_bounded_file(manifest_path, index_limit) != description.dump()) {
            throw std::invalid_argument("No se sustituye un manifiesto previo distinto");
        }
        for (std::size_t position = 0; position < payload_names.size(); ++position) {
            const auto name = payload_names[position];
            if (!std::filesystem::exists(directory / name)) {
                atomic_binary_file(directory / name, payloads[position], payload_limit(name),
                                   false);
            }
        }
        if (!std::filesystem::exists(manifest_path)) {
            atomic_binary_file(manifest_path, description.dump(), index_limit, false);
        }
    }

    Json save(const Json& metadata, std::string_view policy, std::string_view rollout,
              bool select_best) {
        if (policy.empty() || rollout.empty() || policy.size() > maximum_ppo_archive_bytes ||
            rollout.size() > maximum_ppo_archive_bytes) {
            throw std::invalid_argument("Los archivos PPO deben contener entre 1 byte y 128 MiB");
        }
        const auto metadata_text = object_bytes(metadata, maximum_ppo_metadata_bytes);
        require_unchanged();
        clean_retired();
        const Json description{{"schema_version", 1},
                               {"identity_sha256", identity_digest},
                               {"files", Json{{"metadata.json", file_record(metadata_text)},
                                              {"policy.pt", file_record(policy)},
                                              {"rollout.pt", file_record(rollout)}}}};
        const auto digest = content_sha256(description.dump());
        const Json record{{"bundle", "ppo-" + digest}, {"sha256", digest}};
        auto next = index;
        next["recent"] = Json::array({record});
        for (const auto& previous : index.at("recent")) {
            if (previous != record && next.at("recent").size() < retained_recent) {
                next["recent"].push_back(previous);
            }
        }
        if (select_best) {
            next["best"] = record;
        }
        next["retired"] = Json::array();
        const auto active = active_records(next);
        for (const auto& previous : active_records(index)) {
            if (std::ranges::find(active, previous) == active.end()) {
                next["retired"].push_back(previous);
            }
        }
        const auto directory = output / record.at("bundle").get<std::string>();
        require_safe_path(directory);
        const auto existed = std::filesystem::exists(directory);
        const auto previous_index_bytes = index_bytes;
        try {
            ensure_bundle(record, description, metadata_text, policy, rollout);
            require_unchanged();
            publish(next);
        } catch (const std::exception& publication_error) {
            try {
                if (!existed && read_bounded_file(output / "ppo-index.json", index_limit) ==
                                    previous_index_bytes) {
                    remove_owned_bundle(directory);
                }
            } catch (const std::exception& cleanup_error) {
                throw std::runtime_error(
                    std::string(publication_error.what()) +
                    ". No se pudo retirar el intento sin confirmar: " + cleanup_error.what());
            }
            throw;
        }
        clean_retired();
        remove_pending_files(directory);
        auto receipt = record;
        receipt["selected_best"] = select_best;
        receipt["retained_bundles"] = active.size();
        receipt["payload_bytes"] = metadata_text.size() + policy.size() + rollout.size();
        receipt["maximum_bundle_bytes"] = maximum_ppo_bundle_bytes;
        receipt["maximum_retained_payload_bytes"] = maximum_ppo_retained_payload_bytes;
        return receipt;
    }

    std::filesystem::path output;
    std::string identity_bytes;
    std::string identity_digest;
    simulation::OutputLock lock;
    Json index;
    std::string index_bytes;
};

PpoCheckpointStore::PpoCheckpointStore(const std::filesystem::path& output, const Json& identity,
                                       bool resume)
    : impl_(std::make_unique<Impl>(output, identity, resume)) {}
PpoCheckpointStore::~PpoCheckpointStore() = default;

Json PpoCheckpointStore::save(const Json& metadata, std::string_view policy_archive,
                              std::string_view rollout_archive, bool select_best) {
    return impl_->save(metadata, policy_archive, rollout_archive, select_best);
}

PpoCheckpointBundle PpoCheckpointStore::load_latest() {
    impl_->require_unchanged();
    Json discarded = Json::array();
    for (const auto& record : impl_->index.at("recent")) {
        try {
            auto bundle = impl_->load(record);
            bundle.receipt["requested_bundle"] = impl_->index.at("recent").front().at("bundle");
            bundle.receipt["loaded_bundle"] = record.at("bundle");
            bundle.receipt["recovered_from_previous"] = !discarded.empty();
            bundle.receipt["discarded"] = discarded;
            return bundle;
        } catch (const std::bad_alloc&) {
            throw;
        } catch (const std::exception& error) {
            discarded.push_back(Json{{"bundle", record.at("bundle")}, {"reason", error.what()}});
        }
    }
    throw std::runtime_error("No hay un checkpoint reciente íntegro: " + discarded.dump());
}

PpoCheckpointBundle PpoCheckpointStore::load_best() {
    impl_->require_unchanged();
    if (impl_->index.at("best").is_null()) {
        throw std::invalid_argument("Todavía no se ha seleccionado un mejor checkpoint");
    }
    auto bundle = impl_->load(impl_->index.at("best"));
    bundle.receipt["requested_bundle"] = impl_->index.at("best").at("bundle");
    bundle.receipt["loaded_bundle"] = impl_->index.at("best").at("bundle");
    bundle.receipt["recovered_from_previous"] = false;
    bundle.receipt["discarded"] = Json::array();
    return bundle;
}
} // namespace mars_titan::learning
