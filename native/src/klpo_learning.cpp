#include "mars_titan/klpo_learning.hpp"
#include "mars_titan/simulation_files.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace mars_titan::learning {
namespace {
using Json = nlohmann::json;
constexpr std::size_t maximum_reference_updates = 64;
constexpr uint64_t maximum_confirmed_updates = uint64_t{1} << 26;
constexpr uint64_t maximum_consumed_waves = uint64_t{1} << 32;
constexpr std::size_t learning_metadata_fields = 7;
constexpr std::size_t counter_fields = 4;

void require(bool value, const char* message) {
    if (!value) {
        throw std::invalid_argument(message);
    }
}
PpoHyperparameters hyperparameters(const KlpoLearningConfig& config) {
    PpoHyperparameters result;
    result.learning_rate = config.adam.learning_rate;
    return result;
}
std::size_t decisions(const KlpoEpisodeBatch& records) {
    std::size_t result = 0;
    for (const auto& episode : records.episodes) {
        result += static_cast<std::size_t>(
            std::ranges::count_if(episode.steps, [](const auto& step) { return step.sampled; }));
    }
    return result;
}
Json counters_json(const KlpoLearningCounters& value) {
    return {{"consumed_waves", value.consumed_waves},
            {"confirmed_updates", value.confirmed_updates},
            {"reference_version", value.reference_version},
            {"updates_since_reference", value.updates_since_reference}};
}
KlpoLearningCounters read_counters(const Json& value, std::size_t cadence) {
    require(value.is_object() && value.size() == counter_fields, "Faltan contadores terminales");
    const auto number = [&](const char* name) {
        const auto item = simulation::read_json_int64(value.at(name));
        require(item >= 0, "El contador terminal no puede ser negativo");
        return static_cast<uint64_t>(item);
    };
    KlpoLearningCounters result{number("consumed_waves"), number("confirmed_updates"),
                                number("reference_version"), number("updates_since_reference")};
    result.validate(cadence);
    return result;
}
std::string phase_name(KlpoLearningPhase phase) {
    switch (phase) {
    case KlpoLearningPhase::collecting:
        return "collecting";
    case KlpoLearningPhase::ready:
        return "ready";
    case KlpoLearningPhase::blocked:
        return "blocked";
    case KlpoLearningPhase::consumed:
        return "consumed";
    }
    throw std::invalid_argument("La fase terminal no está admitida");
}
KlpoLearningPhase phase_of(const KlpoTerminalCollector& collector, bool consumed) {
    if (consumed) {
        return KlpoLearningPhase::consumed;
    }
    switch (collector.phase()) {
    case KlpoCollectionPhase::collecting:
        return KlpoLearningPhase::collecting;
    case KlpoCollectionPhase::ready:
        return KlpoLearningPhase::ready;
    case KlpoCollectionPhase::blocked:
        return KlpoLearningPhase::blocked;
    }
    throw std::invalid_argument("La recogida no conserva su fase");
}
} // namespace

void KlpoLearningConfig::validate() const {
    adam.validate();
    require(confirmed_updates_per_reference > 0 &&
                confirmed_updates_per_reference <= maximum_reference_updates &&
                gradient_block_episodes > 0 && gradient_block_episodes <= maximum_klpo_episodes,
            "La cadencia o el bloque terminal exceden sus límites");
}
void KlpoLearningCounters::validate(std::size_t cadence) const {
    require(cadence > 0 && cadence <= maximum_reference_updates &&
                consumed_waves <= maximum_consumed_waves &&
                confirmed_updates <= maximum_confirmed_updates &&
                confirmed_updates <= consumed_waves && reference_version <= confirmed_updates &&
                updates_since_reference <= cadence &&
                confirmed_updates == reference_version * cadence + updates_since_reference,
            "Los contadores no conservan oleadas y actualizaciones por referencia");
}
KlpoLearningCounters klpo_after_consumption(KlpoLearningCounters value, std::size_t cadence,
                                            bool updated) {
    value.validate(cadence);
    require(value.consumed_waves < maximum_consumed_waves &&
                value.updates_since_reference < cadence &&
                (!updated || value.confirmed_updates < maximum_confirmed_updates),
            "La oleada necesita refrescar la referencia o agotó el presupuesto");
    ++value.consumed_waves;
    if (updated) {
        ++value.confirmed_updates;
        ++value.updates_since_reference;
    }
    value.validate(cadence);
    return value;
}
KlpoLearningCounters klpo_before_collection(KlpoLearningCounters value, std::size_t cadence) {
    value.validate(cadence);
    if (value.updates_since_reference == cadence) {
        ++value.reference_version;
        value.updates_since_reference = 0;
    }
    value.validate(cadence);
    return value;
}

struct KlpoLearningController::Impl {
    Impl(std::vector<simulation::BatchInput> sources, KlpoLearningConfig settings)
        : inputs(std::move(sources)), config(std::move(settings)) {
        config.validate();
        collector = std::make_unique<KlpoTerminalCollector>(inputs, config.collection);
        actor = std::make_unique<PpoPolicy>(collector->policy().observation_width(),
                                            hyperparameters(config), config.collection.seed,
                                            config.collection.device, default_ppo_memory_bytes,
                                            config.collection.architecture);
        actor->enable_terminal_adam(config.adam);
        require(actor->parameter_fingerprint() == collector->records().reference_sha256,
                "El actor inicial no coincide con la referencia de recogida");
        identity_value = {
            {"schema_version", 1},
            {"kind", "klpo_terminal_actor_updates"},
            {"algorithm", "klpo_full_fresh_waves_v1"},
            {"initial_collection", collector->identity()},
            {"initial_actor_sha256", actor->parameter_fingerprint()},
            {"critic_sha256", actor->critic_fingerprint()},
            {"adam",
             {{"contract", "klpo_actor_adam_zero_critic_v1"},
              {"learning_rate", config.adam.learning_rate},
              {"gradient_norm", config.adam.gradient_norm}}},
            {"confirmed_updates_per_reference", config.confirmed_updates_per_reference},
            {"gradient_block_episodes", config.gradient_block_episodes},
            {"policy_budget_per_instance", default_ppo_memory_bytes},
            {"episode_identity", "run_identity_wave_index_source_record_id"},
            {"reference_refresh", "confirmed_cadence_before_next_collection"}};
    }

    void healthy() const {
        require(!failed && !busy,
                "El controlador terminal necesita recuperar un estado confirmado");
    }
    void check_store(const PpoCheckpointStore& store) const {
        require(store.identity_sha256() == simulation::content_sha256(identity_value.dump()),
                "El escritor pertenece a otra ejecución terminal");
    }
    void validate(const PpoPolicy& policy, const KlpoTerminalCollector& selected,
                  const KlpoLearningCounters& counts, bool is_consumed) const {
        counts.validate(config.confirmed_updates_per_reference);
        require(policy.terminal_adam_enabled() && policy.terminal_adam_options() == config.adam &&
                    policy.hyperparameters() == hyperparameters(config) &&
                    policy.architecture() == config.collection.architecture &&
                    policy.device() == config.collection.device &&
                    policy.memory_budget() == default_ppo_memory_bytes &&
                    selected.policy().memory_budget() == default_ppo_memory_bytes &&
                    policy.critic_fingerprint() ==
                        identity_value.at("critic_sha256").get_ref<const std::string&>() &&
                    selected.policy().critic_fingerprint() ==
                        identity_value.at("critic_sha256").get_ref<const std::string&>() &&
                    policy.observation_width() == selected.policy().observation_width() &&
                    policy.optimizer_steps() == counts.confirmed_updates,
                "El actor no corresponde a los contadores o a la configuración terminal");
        const auto count = decisions(selected.records());
        if (is_consumed) {
            require(
                counts.consumed_waves > 0 && selected.phase() == KlpoCollectionPhase::ready &&
                    (count == 0
                         ? counts.updates_since_reference < config.confirmed_updates_per_reference
                         : counts.confirmed_updates > 0 && counts.updates_since_reference > 0),
                "La oleada consumida no conserva su cierre y actualización");
        } else {
            require(counts.updates_since_reference < config.confirmed_updates_per_reference,
                    "Falta refrescar q antes de recoger otra oleada");
        }
        if (counts.updates_since_reference == 0) {
            require(policy.parameter_fingerprint() == selected.records().reference_sha256,
                    "La referencia recién publicada no coincide con el actor");
        }
    }
    PpoCheckpointBundle snapshot_for(const KlpoTerminalCollector& selected,
                                     const KlpoLearningCounters& counts, bool is_consumed) const {
        validate(*actor, selected, counts, is_consumed);
        auto collection = selected.snapshot();
        std::ostringstream stream;
        actor->save(stream);
        auto actor_bytes = stream.str();
        require(actor_bytes.size() <= maximum_ppo_archive_bytes - collection.policy_archive.size(),
                "Actor y referencia exceden el archivo de recuperación");
        Json metadata = {{"schema_version", 1},
                         {"kind", "klpo_learning_state"},
                         {"identity", identity_value},
                         {"counters", counters_json(counts)},
                         {"phase", phase_name(phase_of(selected, is_consumed))},
                         {"actor_archive_bytes", actor_bytes.size()},
                         {"collection", std::move(collection.metadata)}};
        require(metadata.dump().size() <= maximum_ppo_metadata_bytes,
                "Los metadatos terminales exceden el presupuesto");
        actor_bytes += collection.policy_archive;
        return {std::move(metadata), std::move(actor_bytes), std::move(collection.rollout_archive),
                Json::object()};
    }

    void publish(PpoCheckpointStore& store, const KlpoLearningCounters& next, bool is_consumed,
                 std::unique_ptr<KlpoTerminalCollector> following,
                 const std::function<void(KlpoLearningBoundary)>& failure) {
        check_store(store);
        busy = true;
        try {
            const auto& selected = following ? *following : *collector;
            const auto payload = snapshot_for(selected, next, is_consumed);
            if (failure) {
                failure(KlpoLearningBoundary::before_publish);
            }
            static_cast<void>(
                store.save(payload.metadata, payload.policy_archive, payload.rollout_archive));
            if (failure) {
                failure(KlpoLearningBoundary::confirmed);
            }
            if (following) {
                collector = std::move(following);
            }
            counters = next;
            consumed = is_consumed;
            busy = false;
        } catch (...) {
            failed = true;
            busy = false;
            throw;
        }
    }

    std::vector<simulation::BatchInput> inputs;
    KlpoLearningConfig config;
    std::unique_ptr<KlpoTerminalCollector> collector;
    std::unique_ptr<PpoPolicy> actor;
    KlpoLearningCounters counters;
    Json identity_value;
    bool consumed = false;
    bool busy = false;
    bool failed = false;
};

KlpoLearningController::KlpoLearningController(std::vector<simulation::BatchInput> inputs,
                                               KlpoLearningConfig config)
    : impl_(std::make_unique<Impl>(std::move(inputs), std::move(config))) {}
KlpoLearningController::~KlpoLearningController() = default;
bool KlpoLearningController::collect_tick() {
    impl_->healthy();
    require(!impl_->consumed, "La oleada ya está consumida y necesita una publicación nueva");
    return impl_->collector->collect_tick();
}
KlpoLearningPhase KlpoLearningController::phase() const {
    impl_->healthy();
    return phase_of(*impl_->collector, impl_->consumed);
}
KlpoLearningCounters KlpoLearningController::counters() const {
    impl_->healthy();
    return impl_->counters;
}
Json KlpoLearningController::identity() const { return impl_->identity_value; }
std::string KlpoLearningController::actor_fingerprint() const {
    impl_->healthy();
    return impl_->actor->parameter_fingerprint();
}
std::string KlpoLearningController::reference_fingerprint() const {
    impl_->healthy();
    return impl_->collector->records().reference_sha256;
}
PpoCheckpointBundle KlpoLearningController::snapshot() const {
    impl_->healthy();
    return impl_->snapshot_for(*impl_->collector, impl_->counters, impl_->consumed);
}
Json KlpoLearningController::save(PpoCheckpointStore& store) const {
    impl_->check_store(store);
    const auto state = snapshot();
    return store.save(state.metadata, state.policy_archive, state.rollout_archive);
}

KlpoGradientSummary KlpoLearningController::backward_ready() {
    require(phase() == KlpoLearningPhase::ready && at::GradMode::is_enabled(),
            "Los gradientes requieren una oleada completa y autograd habilitado");
    const auto& records = impl_->collector->records();
    KlpoGradientSummary summary;
    summary.episodes = records.episodes.size();
    summary.sampled_decisions = decisions(records);
    summary.no_policy_decisions = summary.sampled_decisions == 0;
    impl_->actor->terminal_zero_grad();
    if (summary.no_policy_decisions) {
        return summary;
    }
    for (std::size_t begin = 0; begin < summary.episodes;
         begin += impl_->config.gradient_block_episodes) {
        const auto end = std::min(summary.episodes, begin + impl_->config.gradient_block_episodes);
        KlpoEpisodeBatch block{records.fold,
                               records.reference_sha256,
                               records.beta,
                               records.gamma,
                               records.observation_width,
                               records.max_bytes,
                               {}};
        block.episodes.assign(records.episodes.begin() + static_cast<std::ptrdiff_t>(begin),
                              records.episodes.begin() + static_cast<std::ptrdiff_t>(end));
        const auto result = klpo_episode_objective(*impl_->actor, block);
        if (!result.no_policy_decisions) {
            const auto loss = result.per_episode.sum() / static_cast<double>(summary.episodes);
            summary.surrogate += loss.item<double>();
            loss.backward();
        }
        ++summary.blocks;
    }
    require(std::isfinite(summary.surrogate), "El sustituto acumulado no es finito");
    summary.gradient_norm = impl_->actor->validate_terminal_gradients();
    return summary;
}
std::vector<at::Tensor> KlpoLearningController::gradient_snapshot() const {
    impl_->healthy();
    return impl_->actor->terminal_gradients();
}

void KlpoLearningController::consume_without_update(
    PpoCheckpointStore& store, const std::function<void(KlpoLearningBoundary)>& failure) {
    require(phase() == KlpoLearningPhase::ready && decisions(impl_->collector->records()) == 0,
            "Solo se consume sin update una oleada completa sin decisiones");
    const auto next = klpo_after_consumption(impl_->counters,
                                             impl_->config.confirmed_updates_per_reference, false);
    impl_->publish(store, next, true, {}, failure);
}
void KlpoLearningController::start_next_wave(
    PpoCheckpointStore& store, const std::function<void(KlpoLearningBoundary)>& failure) {
    require(phase() == KlpoLearningPhase::consumed, "Falta consumir la oleada anterior");
    impl_->check_store(store);
    const auto next =
        klpo_before_collection(impl_->counters, impl_->config.confirmed_updates_per_reference);
    const auto rng = impl_->collector->policy().random_state();
    const auto& source = next.reference_version != impl_->counters.reference_version
                             ? *impl_->actor
                             : impl_->collector->policy();
    auto reference = source.frozen_reference(rng);
    auto following = std::make_unique<KlpoTerminalCollector>(
        impl_->inputs, impl_->config.collection, std::move(reference));
    impl_->publish(store, next, false, std::move(following), failure);
}

KlpoGradientSummary
KlpoLearningController::update_ready(PpoCheckpointStore& store,
                                     const std::function<void(KlpoLearningBoundary)>& failure) {
    require(phase() == KlpoLearningPhase::ready && decisions(impl_->collector->records()) > 0,
            "La actualización necesita una oleada completa con decisiones");
    impl_->check_store(store);
    static_cast<void>(save(store));
    auto summary = backward_ready();
    impl_->busy = true;
    try {
        if (failure) {
            failure(KlpoLearningBoundary::before_update);
        }
        summary.gradient_norm = impl_->actor->terminal_step();
        if (failure) {
            failure(KlpoLearningBoundary::after_update);
        }
        const auto next = klpo_after_consumption(
            impl_->counters, impl_->config.confirmed_updates_per_reference, true);
        impl_->publish(store, next, true, {}, failure);
    } catch (...) {
        impl_->busy = false;
        impl_->failed = true;
        throw;
    }
    return summary;
}

void KlpoLearningController::restore(const PpoCheckpointBundle& state) {
    require(!impl_->busy && state.metadata.is_object() &&
                state.metadata.size() == learning_metadata_fields &&
                state.metadata.dump().size() <= maximum_ppo_metadata_bytes &&
                state.metadata.at("identity").dump() == impl_->identity_value.dump() &&
                state.policy_archive.size() <= maximum_ppo_archive_bytes &&
                state.rollout_archive.size() <= maximum_ppo_archive_bytes,
            "El checkpoint no corresponde a esta ejecución terminal");
    require(simulation::read_json_int64(state.metadata.at("schema_version")) == 1 &&
                state.metadata.at("kind") == "klpo_learning_state",
            "Contrato de recuperación desconocido");
    const auto split = simulation::read_json_int64(state.metadata.at("actor_archive_bytes"));
    require(split > 0 && static_cast<uint64_t>(split) < state.policy_archive.size(),
            "Los archivos del actor y referencia no conservan sus longitudes");
    const auto counters =
        read_counters(state.metadata.at("counters"), impl_->config.confirmed_updates_per_reference);
    std::istringstream archive(state.policy_archive.substr(0, static_cast<std::size_t>(split)));
    auto actor = std::make_unique<PpoPolicy>(
        PpoPolicy::load_terminal(archive, impl_->config.collection.device));
    PpoCheckpointBundle collection{state.metadata.at("collection"),
                                   state.policy_archive.substr(static_cast<std::size_t>(split)),
                                   state.rollout_archive, Json::object()};
    auto collector =
        KlpoTerminalCollector::from_snapshot(impl_->inputs, impl_->config.collection, collection);
    const bool consumed = state.metadata.at("phase") == "consumed";
    require(state.metadata.at("phase") == phase_name(phase_of(*collector, consumed)),
            "La fase declarada no concuerda con la recogida");
    impl_->validate(*actor, *collector, counters, consumed);
    impl_->actor = std::move(actor);
    impl_->collector = std::move(collector);
    impl_->counters = counters;
    impl_->consumed = consumed;
    impl_->failed = false;
}
} // namespace mars_titan::learning
