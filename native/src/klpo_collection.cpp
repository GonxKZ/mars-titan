#include "mars_titan/klpo_collection.hpp"
#include "mars_titan/ppo_objectives.hpp"
#include "mars_titan/simulation_files.hpp"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <torch/version.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mars_titan::learning {
static_assert(klpo_record_action_count == static_cast<std::size_t>(ppo_action_count));
static_assert(maximum_klpo_steps == static_cast<std::size_t>(maximum_klpo_terminal_length));
namespace {
void require(bool value, const char* message) {
    if (!value) {
        throw std::invalid_argument(message);
    }
}

at::Tensor long_tensor(const std::vector<int64_t>& values, const at::Device& device) {
    auto result = at::empty({static_cast<int64_t>(values.size())}, at::kLong);
    std::copy(values.begin(), values.end(), result.data_ptr<int64_t>());
    return result.to(device);
}
} // namespace

KlpoEpisodeObjective klpo_episode_objective(const PpoPolicy& policy, const KlpoEpisodeBatch& batch,
                                            KlpoTerminalBudget budget) {
    const auto returns = klpo_terminal_returns(batch);
    require(policy.observation_width() == batch.observation_width,
            "El actor no corresponde a la representación registrada");
    const at::Device device(policy.device());
    const auto options = at::TensorOptions().dtype(at::kDouble).device(device);
    const auto total = static_cast<int64_t>(batch.episodes.size());
    KlpoEpisodeObjective result{at::zeros({total}, options), at::zeros({}, options), 0, true};
    std::vector<std::size_t> active;
    std::size_t horizon = 0;
    std::size_t maximum_decisions = 0;
    for (std::size_t lane = 0; lane < batch.episodes.size(); ++lane) {
        const auto& steps = batch.episodes[lane].steps;
        const auto count = static_cast<std::size_t>(
            std::ranges::count_if(steps, [](const auto& step) { return step.sampled; }));
        if (count == 0) {
            continue;
        }
        active.push_back(lane);
        horizon = std::max(horizon, steps.size());
        maximum_decisions = std::max(maximum_decisions, count);
        result.sampled_decisions += count;
    }
    if (active.empty()) {
        return result;
    }
    // Presupuesta las copias densas y su padding antes de reservar tensores.
    constexpr std::size_t history_copies = 3;
    constexpr std::size_t scalar_working_bytes = 256;
    const auto row_bytes =
        history_copies * batch.observation_width * sizeof(float) + scalar_working_bytes;
    require(horizon <= batch.max_bytes / active.size() / row_bytes,
            "El padding de la historia supera el presupuesto del consumidor");
    const auto lanes = static_cast<int64_t>(active.size());
    const auto time = static_cast<int64_t>(horizon);
    const auto decisions = static_cast<int64_t>(maximum_decisions);
    auto history =
        at::zeros({time, lanes, static_cast<int64_t>(batch.observation_width)}, at::kFloat);
    std::vector<int64_t> lengths;
    lengths.reserve(active.size());
    for (std::size_t column = 0; column < active.size(); ++column) {
        const auto& steps = batch.episodes[active[column]].steps;
        lengths.push_back(static_cast<int64_t>(steps.size()));
        for (std::size_t index = 0; index < steps.size(); ++index) {
            auto row = history[static_cast<int64_t>(index)][static_cast<int64_t>(column)];
            std::copy(steps[index].observation.begin(), steps[index].observation.end(),
                      row.data_ptr<float>());
        }
    }
    const auto current = policy.terminal_forward(history.to(device), long_tensor(lengths, device));
    std::vector<at::Tensor> logp_rows;
    std::vector<at::Tensor> logq_rows;
    auto actions = at::zeros({lanes, decisions}, at::kLong);
    auto mask = at::zeros({lanes, decisions}, at::kBool);
    auto terminal = at::zeros({lanes}, at::kDouble);
    for (std::size_t column = 0; column < active.size(); ++column) {
        const auto row_index = static_cast<int64_t>(column);
        const auto& episode = batch.episodes[active[column]];
        std::vector<int64_t> positions;
        for (const auto& step : episode.steps) {
            if (step.sampled) {
                positions.push_back(static_cast<int64_t>(step.cursor));
            }
        }
        const auto count = static_cast<int64_t>(positions.size());
        auto historical = at::zeros({count, ppo_action_count}, at::kFloat);
        for (int64_t index = 0; index < count; ++index) {
            const auto& step =
                episode.steps[static_cast<std::size_t>(positions[static_cast<std::size_t>(index)])];
            std::copy(step.behavior.begin(), step.behavior.end(),
                      historical[index].data_ptr<float>());
            actions[row_index][index] = static_cast<int64_t>(step.action);
        }
        const auto logits =
            current.logits.select(1, row_index).index_select(0, long_tensor(positions, device));
        const auto logp = ppo_behavior_log_probabilities(logits.log_softmax(1).exp());
        const auto logq = ppo_behavior_log_probabilities(historical.to(device));
        const auto padding = at::zeros({decisions - count, ppo_action_count}, options);
        logp_rows.push_back(at::cat({logp, padding}, 0));
        logq_rows.push_back(at::cat({logq, padding}, 0));
        mask[row_index].narrow(0, 0, count).fill_(true);
        terminal[row_index] = returns[active[column]];
    }
    const auto active_loss =
        klpo_terminal_full_loss(at::stack(logp_rows), at::stack(logq_rows), actions.to(device),
                                terminal.to(device), mask.to(device), batch.beta, budget);
    std::vector<at::Tensor> losses;
    losses.reserve(batch.episodes.size());
    std::size_t column = 0;
    for (std::size_t index = 0; index < batch.episodes.size(); ++index) {
        if (column < active.size() && active[column] == index) {
            losses.push_back(active_loss[static_cast<int64_t>(column++)]);
        } else {
            losses.push_back(at::zeros({}, options));
        }
    }
    result.per_episode = at::stack(losses);
    result.mean = result.per_episode.sum() / static_cast<double>(batch.episodes.size());
    result.no_policy_decisions = false;
    return result;
}

namespace {
using Json = nlohmann::json;
constexpr std::size_t metadata_base_bytes = std::size_t{1024} * 1024;
constexpr std::size_t metadata_position_bytes = 1024;
constexpr std::size_t metadata_action_bytes = 256;
constexpr std::size_t context_growth_per_lane = 2048;
constexpr std::size_t collection_metadata_fields = 8;

Json precision_identity() {
    return {{"matmul_tf32", at::globalContext().allowTF32CuBLAS()},
            {"cudnn_conv_tf32", at::globalContext().allowTF32CuDNN(at::Float32Op::CONV)},
            {"cudnn_rnn_tf32", at::globalContext().allowTF32CuDNN(at::Float32Op::RNN)},
            {"cudnn_benchmark", at::globalContext().benchmarkCuDNN()},
            {"cudnn_deterministic", at::globalContext().deterministicCuDNN()},
            {"deterministic_algorithms", at::globalContext().deterministicAlgorithms()}};
}

at::Tensor observations(std::span<const float> values, std::size_t count, std::size_t width) {
    auto result = at::empty({static_cast<int64_t>(count), static_cast<int64_t>(width)}, at::kFloat);
    std::copy(values.begin(), values.end(), result.data_ptr<float>());
    return result;
}

KlpoCollectionPhase collection_phase(const KlpoEpisodeBatch& records) {
    bool complete = true;
    for (const auto& episode : records.episodes) {
        const auto status = klpo_episode_status(episode);
        if (status == KlpoEpisodeStatus::missing_valuation) {
            return KlpoCollectionPhase::blocked;
        }
        complete = complete && status != KlpoEpisodeStatus::open;
    }
    return complete ? KlpoCollectionPhase::ready : KlpoCollectionPhase::collecting;
}

std::string phase_name(KlpoCollectionPhase phase) {
    switch (phase) {
    case KlpoCollectionPhase::collecting:
        return "collecting";
    case KlpoCollectionPhase::ready:
        return "ready";
    case KlpoCollectionPhase::blocked:
        return "blocked";
    }
    throw std::invalid_argument("La fase terminal no está admitida");
}

Json environment_json(const simulation::BatchSnapshot& state) {
    Json sessions = Json::array();
    for (const auto& session : state.sessions) {
        sessions.push_back(simulation::snapshot_json(session));
    }
    return {{"sessions", std::move(sessions)}, {"context_sources", state.context_sources}};
}

Json hidden_json(const at::Tensor& hidden) {
    const auto values = hidden.detach().to(at::kCPU).contiguous();
    std::vector<float> storage(static_cast<std::size_t>(values.numel()));
    if (!storage.empty()) {
        std::copy_n(values.const_data_ptr<float>(), storage.size(), storage.begin());
    }
    return {{"rows", values.size(0)}, {"width", values.size(1)}, {"values", storage}};
}

void check_source_budget(const std::vector<simulation::BatchInput>& inputs,
                         const KlpoCollectionOptions& options) {
    require(!inputs.empty() && inputs.size() <= maximum_klpo_episodes && options.workers > 0 &&
                options.workers <= simulation::maximum_batch_workers && options.record_bytes > 0 &&
                options.record_bytes <= maximum_klpo_record_bytes / 2 &&
                std::isfinite(options.beta) && options.beta > 0 && std::isfinite(options.gamma) &&
                options.gamma > 0 && options.gamma <= 1 &&
                options.seed <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            "La oleada terminal supera sus límites o declara parámetros inválidos");
    options.architecture.validate();
    require(!options.architecture.auxiliary && !options.architecture.double_dqn,
            "El registro terminal no admite auxiliares ni Double DQN");
    require(!options.context.markov && options.context.markov_fields.empty(),
            "El registro terminal no admite HMM");
    if (options.context.enabled) {
        const auto expected = options.architecture.kind == PpoNetworkKind::gru ? "ppo_gru" : "ppo";
        require(options.context.variant == expected && options.context.fold == options.fold,
                "El contexto no conserva el perfil y fold del actor");
    } else {
        require(!options.context.trading_field, "La máscara requiere su contexto explícito");
    }
    std::size_t estimated = metadata_base_bytes;
    const simulation::MarketTape* first = nullptr;
    for (const auto& input : inputs) {
        require(input.tape && input.tape->partition == "train" &&
                    input.tape->close_times.size() >= 2 &&
                    input.tape->close_times.size() <= maximum_klpo_steps + 1,
                "La cinta no define un episodio completo de entrenamiento admitido");
        if (first == nullptr) {
            first = input.tape.get();
        }
        require(input.tape->domain == first->domain && input.tape->parent_id == first->parent_id,
                "Las fuentes mezclan dominios o predictores padre");
        const auto account = [&](std::size_t count, std::size_t bytes) {
            require(count <= (maximum_ppo_metadata_bytes - estimated) / bytes,
                    "El estado de las carteras excedería el límite de recuperación");
            estimated += count * bytes;
        };
        account(input.tape->assets.size(), metadata_position_bytes);
        account(input.tape->actions.size(), metadata_action_bytes);
    }
}
} // namespace

struct KlpoTerminalCollector::Impl {
    Impl(std::vector<simulation::BatchInput> sources, KlpoCollectionOptions configuration)
        : inputs(std::move(sources)), options(std::move(configuration)) {
        check_source_budget(inputs, options);
        environment = std::make_unique<simulation::FinancialBatch>(inputs, options.workers);
        auto width = environment->observation_width();
        if (options.context.enabled) {
            context = std::make_unique<PolicyContext>(inputs, options.context, options.seed,
                                                      environment->observations());
            width = context->observation_width();
        }
        actor =
            std::make_unique<PpoPolicy>(width, PpoHyperparameters{}, options.seed, options.device,
                                        default_ppo_memory_bytes, options.architecture);
        hidden = actor->initial_state(inputs.size());
        recorded.fold = options.fold;
        recorded.reference_sha256 = actor->parameter_fingerprint();
        recorded.beta = options.beta;
        recorded.gamma = options.gamma;
        recorded.observation_width = width;
        recorded.max_bytes = options.record_bytes;
        Json source_identity = Json::array();
        const auto initial = environment->snapshot();
        for (std::size_t lane = 0; lane < inputs.size(); ++lane) {
            const auto& source = inputs[lane];
            KlpoEpisodeRecord episode;
            auto& spec = episode.spec;
            spec.id = std::to_string(lane) + ":" + source.tape->source_sha256;
            spec.source_sha256 = source.tape->source_sha256;
            spec.context_sha256 = source.context ? source.context->source_sha256 : std::string{};
            spec.close_times = source.tape->close_times;
            if (options.context.trading_field) {
                if (!source.context.has_value()) {
                    throw std::invalid_argument("Falta la fuente de elegibilidad");
                }
                const auto field = options.context.trading_field.value();
                const auto& source_context = source.context.value();
                require(field < source_context.fields.size(), "Falta el campo de elegibilidad");
                bool enabled = false;
                for (std::size_t time = 0; time + 1 < spec.close_times.size(); ++time) {
                    const auto& value =
                        source_context.values.at(time * source_context.fields.size() + field);
                    require(value.present && (value.value == 0 || value.value == 1) &&
                                (!enabled || value.value == 1),
                            "El calentamiento no conserva un prefijo conocido");
                    enabled = enabled || value.value == 1;
                    if (!enabled) {
                        ++spec.forced_prefix;
                    }
                }
            }
            const auto session = simulation::snapshot_json(initial.sessions[lane]);
            source_identity.push_back(
                {{"id", spec.id},
                 {"source", spec.source_sha256},
                 {"context", spec.context_sha256},
                 {"calendar", simulation::content_sha256(Json(spec.close_times).dump())},
                 {"parameters", session.at("parameters")},
                 {"domain", source.tape->domain},
                 {"parent", source.tape->parent_id},
                 {"currency", source.tape->currency},
                 {"forced_prefix", spec.forced_prefix}});
            recorded.episodes.push_back(std::move(episode));
        }
        validate_klpo_batch(recorded, false);
        for (auto& episode : recorded.episodes) {
            episode.steps.reserve(episode.spec.close_times.size() - 1);
        }
        const auto context_bytes = context ? context->snapshot().archive.size() : 0;
        require(context_bytes <= maximum_ppo_archive_bytes - options.record_bytes &&
                    inputs.size() * context_growth_per_lane <=
                        maximum_ppo_archive_bytes - options.record_bytes - context_bytes,
                "El registro y el contexto no cabrían juntos en el checkpoint");
        identity_value = {
            {"schema_version", 1},
            {"kind", "klpo_terminal_collection"},
            {"objective", klpo_terminal_contract},
            {"records", klpo_episode_contract},
            {"sampler", ppo_sampler_contract},
            {"support", "positive_fp32_six_actions"},
            {"draw_order", "active_sampled_lanes_ascending_v1"},
            {"fold", options.fold},
            {"beta", options.beta},
            {"gamma", options.gamma},
            {"seed", options.seed},
            {"device", options.device},
            {"workers", options.workers},
            {"record_bytes", options.record_bytes},
            {"reference_sha256", recorded.reference_sha256},
            {"sources", std::move(source_identity)},
            {"architecture",
             {{"kind", static_cast<unsigned>(options.architecture.kind)},
              {"hidden_width", options.architecture.hidden_width}}},
            {"context",
             {{"enabled", options.context.enabled},
              {"variant", options.context.variant},
              {"fold", options.context.fold},
              {"representation", options.context.representation},
              {"environments", options.context.environments},
              {"trading_field", options.context.trading_field ? Json(*options.context.trading_field)
                                                              : Json(nullptr)}}},
            {"precision", precision_identity()},
            {"torch", TORCH_VERSION},
            {"source_code", MARS_TITAN_NATIVE_SOURCE_SHA256},
            {"build", MARS_TITAN_NATIVE_BUILD_SHA256}};
    }

    void healthy() const {
        require(!failed && !busy, "El colector necesita recuperar un estado confirmado");
        require(precision_identity().dump() == identity_value.at("precision").dump(),
                "La precisión cambió durante la oleada");
    }

    bool tick(const std::function<void(KlpoCollectionBoundary)>& failure) {
        healthy();
        if (collection_phase(recorded) != KlpoCollectionPhase::collecting) {
            return false;
        }
        const auto previous_rng = actor->random_state();
        bool committed = false;
        busy = true;
        try {
            const auto current = context ? context->observations()
                                         : observations(environment->observations(), inputs.size(),
                                                        environment->observation_width());
            auto proposed_hidden = hidden.clone();
            std::vector<KlpoEpisodeStep> prepared(inputs.size());
            std::vector<uint8_t> active(inputs.size(), 0), actions(inputs.size(), 1);
            std::vector<int64_t> forced, selected;
            const auto eligible =
                context ? context->training_mask() : std::vector<uint8_t>(inputs.size(), 1);
            for (std::size_t lane = 0; lane < inputs.size(); ++lane) {
                if (klpo_episode_status(recorded.episodes[lane]) != KlpoEpisodeStatus::open) {
                    continue;
                }
                active[lane] = 1;
                auto& step = prepared[lane];
                const auto& episode = recorded.episodes[lane];
                step.cursor = episode.steps.size();
                require(step.cursor + 1 < episode.spec.close_times.size(),
                        "Falta el cierre de una cinta");
                step.decision_at = episode.spec.close_times[step.cursor];
                step.outcome_at = episode.spec.close_times[step.cursor + 1];
                step.sampled = step.cursor >= episode.spec.forced_prefix;
                require(step.sampled == (eligible[lane] != 0),
                        "La elegibilidad cambió respecto al prefijo declarado");
                const auto row = current[static_cast<int64_t>(lane)].contiguous();
                const std::span values(row.const_data_ptr<float>(),
                                       static_cast<std::size_t>(row.numel()));
                step.observation.assign(values.begin(), values.end());
                (step.sampled ? selected : forced).push_back(static_cast<int64_t>(lane));
            }
            const at::Device device(actor->device());
            const auto infer = [&](const std::vector<int64_t>& lanes, bool sampled) {
                if (lanes.empty()) {
                    return;
                }
                const auto host_indices = long_tensor(lanes, at::Device(at::kCPU));
                const auto indices = host_indices.to(device);
                const auto rows = current.index_select(0, host_indices).to(device);
                const auto state = hidden.index_select(0, indices);
                at::Tensor next;
                if (sampled) {
                    const auto chosen = actor->act_recurrent(rows, state);
                    next = chosen.next_state;
                    const auto packed = chosen.packed.to(at::kCPU).contiguous();
                    const auto weights = chosen.probabilities.to(at::kCPU).contiguous();
                    const auto valid = at::ones({static_cast<int64_t>(lanes.size())}, at::kBool);
                    validate_ppo_behavior(weights, packed.select(1, 0).to(at::kLong),
                                          packed.select(1, 1), valid);
                    require((weights > 0).all().item<bool>(),
                            "KLPO requiere soporte positivo en las seis acciones");
                    for (std::size_t index = 0; index < lanes.size(); ++index) {
                        const auto lane = static_cast<std::size_t>(lanes[index]);
                        prepared[lane].action = static_cast<uint8_t>(
                            packed[static_cast<int64_t>(index)][0].item<int64_t>());
                        actions[lane] = prepared[lane].action;
                        const auto row = weights[static_cast<int64_t>(index)];
                        std::copy_n(row.const_data_ptr<float>(), klpo_record_action_count,
                                    prepared[lane].behavior.begin());
                    }
                } else {
                    next = actor->infer(rows, state).next_state;
                }
                proposed_hidden.index_copy_(0, indices, next);
            };
            infer(forced, false);
            infer(selected, true);
            static_cast<void>(environment->step_checked(
                actions, active, [&](auto following, const auto& transition) {
                    if (context) {
                        static_cast<void>(context->prepare(following, actions, transition, active));
                    }
                    for (std::size_t lane = 0; lane < inputs.size(); ++lane) {
                        if (active[lane] == 0) {
                            continue;
                        }
                        auto& step = prepared[lane];
                        step.reward = transition.rewards[lane];
                        step.valuation_valid = transition.reward_valid[lane] != 0;
                        step.terminated = transition.terminated[lane] != 0;
                        step.truncated = transition.truncated[lane] != 0;
                    }
                    if (failure) {
                        failure(KlpoCollectionBoundary::before_commit);
                    }
                }));
            committed = true;
            if (context) {
                context->commit();
            }
            hidden = std::move(proposed_hidden);
            for (std::size_t lane = 0; lane < inputs.size(); ++lane) {
                if (active[lane] != 0) {
                    recorded.episodes[lane].steps.push_back(std::move(prepared[lane]));
                }
            }
            validate_klpo_batch(recorded, false);
            if (failure) {
                failure(KlpoCollectionBoundary::committed);
            }
            busy = false;
            return true;
        } catch (...) {
            if (context) {
                context->cancel();
            }
            if (committed) {
                failed = true;
            } else {
                actor->restore_random_state(previous_rng);
            }
            busy = false;
            throw;
        }
    }

    PpoCheckpointBundle snapshot() const {
        healthy();
        require(actor->parameter_fingerprint() == recorded.reference_sha256 &&
                    actor->optimizer_steps() == 0,
                "El actor cambió dentro de la oleada");
        const auto records = serialize_klpo_batch(recorded);
        const auto context_bytes = context ? context->snapshot().archive : std::string{};
        require(records.size() <= maximum_ppo_archive_bytes &&
                    context_bytes.size() <= maximum_ppo_archive_bytes - records.size(),
                "La oleada excede el payload de recuperación");
        std::ostringstream archive;
        actor->save(archive);
        Json metadata = {{"schema_version", 1},
                         {"kind", "klpo_collection_state"},
                         {"identity", identity_value},
                         {"phase", phase_name(collection_phase(recorded))},
                         {"record_bytes", records.size()},
                         {"context_bytes", context_bytes.size()},
                         {"environment", environment_json(environment->snapshot())},
                         {"hidden", hidden_json(hidden)}};
        require(metadata.dump().size() <= maximum_ppo_metadata_bytes,
                "Los metadatos exceden el presupuesto");
        return {std::move(metadata), archive.str(), records + context_bytes, Json::object()};
    }

    std::vector<simulation::BatchInput> inputs;
    KlpoCollectionOptions options;
    std::unique_ptr<simulation::FinancialBatch> environment;
    std::unique_ptr<PolicyContext> context;
    std::unique_ptr<PpoPolicy> actor;
    at::Tensor hidden;
    KlpoEpisodeBatch recorded;
    Json identity_value;
    bool busy = false;
    bool failed = false;
};

KlpoTerminalCollector::KlpoTerminalCollector(std::vector<simulation::BatchInput> inputs,
                                             KlpoCollectionOptions options)
    : impl_(std::make_unique<Impl>(std::move(inputs), std::move(options))) {}
KlpoTerminalCollector::~KlpoTerminalCollector() = default;
bool KlpoTerminalCollector::collect_tick(
    const std::function<void(KlpoCollectionBoundary)>& failure) {
    return impl_->tick(failure);
}
KlpoCollectionPhase KlpoTerminalCollector::phase() const {
    impl_->healthy();
    return collection_phase(impl_->recorded);
}
const KlpoEpisodeBatch& KlpoTerminalCollector::records() const& {
    impl_->healthy();
    return impl_->recorded;
}
const PpoPolicy& KlpoTerminalCollector::policy() const& {
    impl_->healthy();
    return *impl_->actor;
}
KlpoEpisodeObjective KlpoTerminalCollector::objective(KlpoTerminalBudget budget) const {
    impl_->healthy();
    return klpo_episode_objective(*impl_->actor, impl_->recorded, budget);
}
Json KlpoTerminalCollector::identity() const { return impl_->identity_value; }
PpoCheckpointBundle KlpoTerminalCollector::snapshot() const { return impl_->snapshot(); }
Json KlpoTerminalCollector::save(PpoCheckpointStore& store) const {
    require(store.identity_sha256() == simulation::content_sha256(impl_->identity_value.dump()),
            "El escritor pertenece a otra identidad");
    const auto state = snapshot();
    return store.save(state.metadata, state.policy_archive, state.rollout_archive);
}

void KlpoTerminalCollector::restore(const PpoCheckpointBundle& state) {
    require(!impl_->busy, "No se recupera dentro de una transición activa");
    require(state.metadata.is_object() && state.metadata.size() == collection_metadata_fields &&
                state.metadata.dump().size() <= maximum_ppo_metadata_bytes &&
                !state.policy_archive.empty() &&
                state.policy_archive.size() <= maximum_ppo_archive_bytes &&
                state.rollout_archive.size() <= maximum_ppo_archive_bytes &&
                state.metadata.at("identity").dump() == impl_->identity_value.dump(),
            "El checkpoint no corresponde a esta oleada");
    const auto count = simulation::read_json_int64(state.metadata.at("record_bytes"));
    const auto context_size = simulation::read_json_int64(state.metadata.at("context_bytes"));
    require(count > 0 && context_size >= 0 &&
                static_cast<uint64_t>(count) <= state.rollout_archive.size() &&
                static_cast<uint64_t>(context_size) ==
                    state.rollout_archive.size() - static_cast<std::size_t>(count),
            "El payload no conserva sus longitudes");
    const auto recorded = deserialize_klpo_batch(
        std::string_view(state.rollout_archive).substr(0, static_cast<std::size_t>(count)),
        impl_->options.record_bytes);
    auto candidate = std::make_unique<Impl>(impl_->inputs, impl_->options);
    std::size_t ticks = 0;
    for (const auto& episode : recorded.episodes) {
        ticks = std::max(ticks, episode.steps.size());
    }
    // Reproduce el prefijo con el sampler fijo, sin actualizar ni escribir otro estado.
    for (std::size_t tick = 0; tick < ticks; ++tick) {
        require(candidate->tick({}), "El registro continúa después de finalizar o bloquearse");
    }
    const auto expected = candidate->snapshot();
    require(expected.metadata.dump() == state.metadata.dump() &&
                expected.rollout_archive == state.rollout_archive,
            "El registro no concuerda con el sampler, las fuentes o las carteras");
    std::istringstream archive(state.policy_archive);
    const auto restored = PpoPolicy::load(archive, impl_->options.device);
    const auto actual_rng = restored.random_state();
    const auto expected_rng = candidate->actor->random_state();
    require(restored.parameter_fingerprint() == candidate->recorded.reference_sha256 &&
                restored.hyperparameters() == candidate->actor->hyperparameters() &&
                restored.architecture() == candidate->actor->architecture() &&
                !restored.objective().enabled() && restored.optimizer_steps() == 0 &&
                at::equal(actual_rng.sampling, expected_rng.sampling) &&
                at::equal(actual_rng.shuffle, expected_rng.shuffle),
            "El archivo del actor no conserva referencia, RNG o ausencia de actualizaciones");
    impl_ = std::move(candidate);
}
} // namespace mars_titan::learning
