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

TerminalDecisions terminal_decisions(const PpoPolicy& policy, const KlpoEpisodeBatch& batch) {
    require(policy.observation_width() == batch.observation_width,
            "El actor no corresponde a la representación registrada");
    const at::Device device(policy.device());
    const auto options = at::TensorOptions().dtype(at::kDouble).device(device);
    TerminalDecisions result;
    std::size_t horizon = 0;
    std::size_t maximum_decisions = 0;
    for (std::size_t lane = 0; lane < batch.episodes.size(); ++lane) {
        const auto& steps = batch.episodes[lane].steps;
        const auto count = static_cast<std::size_t>(
            std::ranges::count_if(steps, [](const auto& step) { return step.sampled; }));
        if (count == 0) {
            continue;
        }
        result.active.push_back(lane);
        result.counts.push_back(static_cast<int64_t>(count));
        horizon = std::max(horizon, steps.size());
        maximum_decisions = std::max(maximum_decisions, count);
        result.sampled_decisions += count;
    }
    if (result.active.empty()) {
        return result;
    }
    // Presupuesta las copias densas y su padding antes de reservar tensores.
    constexpr std::size_t history_copies = 3;
    constexpr std::size_t scalar_working_bytes = 256;
    const auto row_bytes =
        history_copies * batch.observation_width * sizeof(float) + scalar_working_bytes;
    require(horizon <= batch.max_bytes / result.active.size() / row_bytes,
            "El padding de la historia supera el presupuesto del consumidor");
    const auto lanes = static_cast<int64_t>(result.active.size());
    const auto time = static_cast<int64_t>(horizon);
    const auto decisions = static_cast<int64_t>(maximum_decisions);
    auto history =
        at::zeros({time, lanes, static_cast<int64_t>(batch.observation_width)}, at::kFloat);
    std::vector<int64_t> lengths;
    lengths.reserve(result.active.size());
    for (std::size_t column = 0; column < result.active.size(); ++column) {
        const auto& steps = batch.episodes[result.active[column]].steps;
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
    for (std::size_t column = 0; column < result.active.size(); ++column) {
        const auto row_index = static_cast<int64_t>(column);
        const auto& episode = batch.episodes[result.active[column]];
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
    }
    result.logp = at::stack(logp_rows);
    result.logq = at::stack(logq_rows);
    result.actions = actions.to(device);
    result.mask = mask.to(device);
    return result;
}

at::Tensor scatter_episodes(const at::Tensor& active_values, const std::vector<std::size_t>& active,
                            std::size_t episodes) {
    std::vector<at::Tensor> losses;
    losses.reserve(episodes);
    std::size_t column = 0;
    for (std::size_t index = 0; index < episodes; ++index) {
        if (column < active.size() && active[column] == index) {
            losses.push_back(active_values[static_cast<int64_t>(column++)]);
        } else {
            losses.push_back(at::zeros({}, active_values.options()));
        }
    }
    return at::stack(losses);
}

KlpoEpisodeObjective klpo_episode_objective(const PpoPolicy& policy, const KlpoEpisodeBatch& batch,
                                            KlpoTerminalBudget budget) {
    const auto returns = klpo_terminal_returns(batch);
    const at::Device device(policy.device());
    const auto options = at::TensorOptions().dtype(at::kDouble).device(device);
    const auto total = static_cast<int64_t>(batch.episodes.size());
    KlpoEpisodeObjective result{at::zeros({total}, options), at::zeros({}, options), 0, true};
    const auto decisions = terminal_decisions(policy, batch);
    result.sampled_decisions = decisions.sampled_decisions;
    if (decisions.active.empty()) {
        return result;
    }
    auto terminal = at::zeros({static_cast<int64_t>(decisions.active.size())}, at::kDouble);
    for (std::size_t column = 0; column < decisions.active.size(); ++column) {
        terminal[static_cast<int64_t>(column)] = returns[decisions.active[column]];
    }
    const auto active_loss =
        klpo_terminal_full_loss(decisions.logp, decisions.logq, decisions.actions,
                                terminal.to(device), decisions.mask, batch.beta, budget);
    result.per_episode = scatter_episodes(active_loss, decisions.active, batch.episodes.size());
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
constexpr std::size_t maximum_initial_rng_bytes = 16384;

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

Json random_json(const PpoRandomState& state) {
    const auto bytes = [](const at::Tensor& value) {
        require(value.defined() && value.device().is_cpu() && value.scalar_type() == at::kByte &&
                    value.dim() == 1 && value.is_contiguous() && value.numel() > 0 &&
                    static_cast<std::size_t>(value.numel()) <= maximum_initial_rng_bytes,
                "El RNG inicial no conserva forma, tipo o presupuesto");
        const std::span elements(value.const_data_ptr<uint8_t>(),
                                 static_cast<std::size_t>(value.numel()));
        return std::vector<uint8_t>(elements.begin(), elements.end());
    };
    return {{"sampling", bytes(state.sampling)}, {"shuffle", bytes(state.shuffle)}};
}

PpoRandomState read_random(const Json& state) {
    require(state.is_object() && state.size() == 2, "El RNG inicial tiene campos incompatibles");
    const auto bytes = [](const Json& values) {
        require(values.is_array() && !values.empty() && values.size() <= maximum_initial_rng_bytes,
                "El RNG inicial excede su presupuesto");
        auto tensor = at::empty({static_cast<int64_t>(values.size())}, at::kByte);
        const std::span output(tensor.data_ptr<uint8_t>(), values.size());
        for (std::size_t index = 0; index < values.size(); ++index) {
            const auto value = simulation::read_json_int64(values[index]);
            require(value >= 0 && value <= std::numeric_limits<uint8_t>::max(),
                    "El RNG inicial contiene un byte inválido");
            output[index] = static_cast<uint8_t>(value);
        }
        return tensor;
    };
    return {bytes(state.at("sampling")), bytes(state.at("shuffle"))};
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
        require(simulation::same_policy_origin(*input.tape, *first),
                "Las fuentes mezclan dominios, predictores padre o bases históricas");
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
    Impl(std::vector<simulation::BatchInput> sources, KlpoCollectionOptions configuration,
         std::unique_ptr<PpoPolicy> reference = {})
        : inputs(std::move(sources)), options(std::move(configuration)) {
        check_source_budget(inputs, options);
        environment = std::make_unique<simulation::FinancialBatch>(inputs, options.workers);
        auto width = environment->observation_width();
        if (options.context.enabled) {
            context = std::make_unique<PolicyContext>(inputs, options.context, options.seed,
                                                      environment->observations());
            width = context->observation_width();
        }
        supplied_reference = static_cast<bool>(reference);
        if (reference) {
            require(reference->observation_width() == width &&
                        reference->device() == options.device &&
                        reference->architecture() == options.architecture &&
                        reference->optimizer_steps() == 0 && !reference->objective().enabled() &&
                        !reference->terminal_adam_enabled(),
                    "La referencia recibida no corresponde a la recogida congelada");
            actor = std::move(reference);
            initial_rng = random_json(actor->random_state());
        } else {
            actor = std::make_unique<PpoPolicy>(width, PpoHyperparameters{}, options.seed,
                                                options.device, default_ppo_memory_bytes,
                                                options.architecture);
        }
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
        if (supplied_reference) {
            identity_value["reference_origin"] = "provided_parameters_and_rng_v1";
            identity_value["initial_rng_sha256"] = simulation::content_sha256(initial_rng.dump());
        }
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
            const auto current = (context ? context->observations()
                                          : observations(environment->observations(), inputs.size(),
                                                         environment->observation_width()))
                                     .contiguous();
            const auto width = static_cast<std::size_t>(current.size(1));
            const std::span current_values(current.const_data_ptr<float>(), inputs.size() * width);
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
                const auto values = current_values.subspan(lane * width, width);
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
                    // Filas contiguas en CPU: acción, logaritmo y valor, y los seis pesos.
                    constexpr std::size_t packed_width = 3;
                    const std::span packed_values(packed.const_data_ptr<double>(),
                                                  lanes.size() * packed_width);
                    const std::span weight_values(weights.const_data_ptr<float>(),
                                                  lanes.size() * klpo_record_action_count);
                    for (std::size_t index = 0; index < lanes.size(); ++index) {
                        const auto lane = static_cast<std::size_t>(lanes[index]);
                        prepared[lane].action = static_cast<uint8_t>(
                            static_cast<int64_t>(packed_values[index * packed_width]));
                        actions[lane] = prepared[lane].action;
                        const auto row = weight_values.subspan(index * klpo_record_action_count,
                                                               klpo_record_action_count);
                        std::copy(row.begin(), row.end(), prepared[lane].behavior.begin());
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
            std::vector<std::size_t> validated(inputs.size());
            for (std::size_t lane = 0; lane < inputs.size(); ++lane) {
                validated[lane] = recorded.episodes[lane].steps.size();
                if (active[lane] != 0) {
                    recorded.episodes[lane].steps.push_back(std::move(prepared[lane]));
                }
            }
            // Los pasos anteriores ya se validaron y solo cambian al restaurar otro estado.
            // La oleada completa se valida entera al serializarla y al calcular su objetivo.
            validate_klpo_batch(recorded, false, validated);
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
        Json metadata = {{"schema_version", supplied_reference ? 2 : 1},
                         {"kind", "klpo_collection_state"},
                         {"identity", identity_value},
                         {"phase", phase_name(collection_phase(recorded))},
                         {"record_bytes", records.size()},
                         {"context_bytes", context_bytes.size()},
                         {"environment", environment_json(environment->snapshot())},
                         {"hidden", hidden_json(hidden)}};
        if (supplied_reference) {
            metadata["initial_rng"] = initial_rng;
        }
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
    Json initial_rng;
    bool supplied_reference = false;
    bool busy = false;
    bool failed = false;
};

KlpoTerminalCollector::KlpoTerminalCollector(std::vector<simulation::BatchInput> inputs,
                                             KlpoCollectionOptions options)
    : impl_(std::make_unique<Impl>(std::move(inputs), std::move(options))) {}
KlpoTerminalCollector::KlpoTerminalCollector(std::vector<simulation::BatchInput> inputs,
                                             KlpoCollectionOptions options, PpoPolicy reference)
    : impl_(std::make_unique<Impl>(std::move(inputs), std::move(options),
                                   std::make_unique<PpoPolicy>(std::move(reference)))) {}
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
    require(state.metadata.is_object() &&
                state.metadata.at("identity").dump() == impl_->identity_value.dump(),
            "El checkpoint pertenece a otra referencia o recogida");
    auto candidate = from_snapshot(impl_->inputs, impl_->options, state);
    impl_ = std::move(candidate->impl_);
}

std::unique_ptr<KlpoTerminalCollector>
KlpoTerminalCollector::from_snapshot(std::vector<simulation::BatchInput> inputs,
                                     KlpoCollectionOptions options,
                                     const PpoCheckpointBundle& state) {
    require(state.metadata.is_object(), "Faltan los metadatos de la recogida");
    const auto version = simulation::read_json_int64(state.metadata.at("schema_version"));
    require((version == 1 || version == 2) &&
                state.metadata.size() == collection_metadata_fields + (version == 2 ? 1 : 0) &&
                state.metadata.dump().size() <= maximum_ppo_metadata_bytes &&
                !state.policy_archive.empty() &&
                state.policy_archive.size() <= maximum_ppo_archive_bytes &&
                state.rollout_archive.size() <= maximum_ppo_archive_bytes,
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
        options.record_bytes);
    std::istringstream archive(state.policy_archive);
    const auto restored = PpoPolicy::load(archive, options.device);
    std::unique_ptr<KlpoTerminalCollector> result;
    if (version == 2) {
        const auto initial = read_random(state.metadata.at("initial_rng"));
        result = std::make_unique<KlpoTerminalCollector>(std::move(inputs), std::move(options),
                                                         restored.frozen_reference(initial));
    } else {
        result = std::make_unique<KlpoTerminalCollector>(std::move(inputs), std::move(options));
    }
    auto& candidate = result->impl_;
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
    const auto actual_rng = restored.random_state();
    const auto expected_rng = candidate->actor->random_state();
    require(restored.parameter_fingerprint() == candidate->recorded.reference_sha256 &&
                restored.hyperparameters() == candidate->actor->hyperparameters() &&
                restored.architecture() == candidate->actor->architecture() &&
                !restored.objective().enabled() && restored.optimizer_steps() == 0 &&
                at::equal(actual_rng.sampling, expected_rng.sampling) &&
                at::equal(actual_rng.shuffle, expected_rng.shuffle),
            "El archivo del actor no conserva referencia, RNG o ausencia de actualizaciones");
    return result;
}
} // namespace mars_titan::learning
