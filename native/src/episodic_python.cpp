#include "mars_titan/cohort_execution.hpp"
#include "mars_titan/episodic_memory.hpp"
#include "mars_titan/episodic_projection.hpp"
#include "mars_titan/medoid_kernels.hpp"
#include "mars_titan/simulation_files.hpp"

#include <ATen/Parallel.h>
#include <c10/util/ScopeExit.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace cohorts = mars_titan::cohorts;
namespace memory = mars_titan::learning;
namespace {
cohorts::Json state_json(const std::string& bytes) {
    constexpr std::size_t maximum = 16 * cohorts::mebibyte;
    if (bytes.size() > maximum) {
        throw std::invalid_argument("El estado del enlace supera 16 MiB");
    }
    return mars_titan::simulation::parse_bounded_json(bytes);
}

struct BoundExecutor {
    BoundExecutor(const std::string& output, cohorts::Definition definition, py::function prepare,
                  py::function update, bool resume) {
        cohorts::Callbacks callbacks;
        if (definition.prediction_mode == cohorts::PredictionMode::financial) {
            callbacks.prepare_event = [prepare = std::move(prepare)](
                                          cohorts::EventKind kind,
                                          std::span<const cohorts::Observation> rows,
                                          std::span<const cohorts::Task> tasks, std::int64_t cutoff,
                                          const cohorts::Json& before, std::size_t batch_rows) {
                const auto returned = prepare(kind, std::vector(rows.begin(), rows.end()),
                                              std::vector(tasks.begin(), tasks.end()), cutoff,
                                              before.dump(), batch_rows)
                                          .cast<std::pair<std::vector<double>, std::string>>();
                return cohorts::PreparedCohort{returned.first, state_json(returned.second)};
            };
            callbacks.resolve =
                [update = std::move(update)](
                    const cohorts::Json& proposed,
                    std::span<const cohorts::ResolvedFeedback> feedback,
                    std::span<const cohorts::ResolvedPrefixExclusion> excluded,
                    std::span<const cohorts::AdministrativeFinalization> finalized) {
                    return state_json(update(proposed.dump(),
                                             std::vector(feedback.begin(), feedback.end()),
                                             std::vector(excluded.begin(), excluded.end()),
                                             std::vector(finalized.begin(), finalized.end()))
                                          .cast<std::string>());
                };
        } else {
            callbacks.prepare = [prepare = std::move(prepare)](
                                    std::span<const cohorts::Observation> rows,
                                    std::span<const cohorts::Task> tasks, std::int64_t cutoff,
                                    const cohorts::Json& before, std::size_t batch_rows) {
                const auto returned = prepare(std::vector(rows.begin(), rows.end()),
                                              std::vector(tasks.begin(), tasks.end()), cutoff,
                                              before.dump(), batch_rows)
                                          .cast<std::pair<std::vector<double>, std::string>>();
                return cohorts::PreparedCohort{returned.first, state_json(returned.second)};
            };
            callbacks.update =
                [update = std::move(update)](const cohorts::Json& proposed,
                                             std::span<const cohorts::ResolvedFeedback> feedback) {
                    return state_json(
                        update(proposed.dump(), std::vector(feedback.begin(), feedback.end()))
                            .cast<std::string>());
                };
        }
        executor = std::make_unique<cohorts::Executor>(output, std::move(definition),
                                                       std::move(callbacks), resume);
    }
    cohorts::Executor& get() {
        if (!executor) {
            throw std::runtime_error("El ejecutor nativo está cerrado");
        }
        return *executor;
    }
    void close() {
        if (step_active) {
            throw std::runtime_error("No se puede cerrar el ejecutor mientras step está activo");
        }
        executor.reset();
    }
    std::unique_ptr<cohorts::Executor> executor;
    bool step_active = false;
};

void bind_memory(py::module_& module) {
    py::class_<memory::MemoryScope>(module, "MemoryScope")
        .def(py::init<>())
        .def_readwrite("world", &memory::MemoryScope::world)
        .def_readwrite("partition", &memory::MemoryScope::partition)
        .def_readwrite("fold", &memory::MemoryScope::fold)
        .def_readwrite("representation", &memory::MemoryScope::representation)
        .def_readwrite("lane", &memory::MemoryScope::lane);
    py::class_<memory::MemoryRecord>(module, "MemoryRecord")
        .def(py::init<>())
        .def_readwrite("id", &memory::MemoryRecord::id)
        .def_readwrite("decision_at", &memory::MemoryRecord::decision_at)
        .def_readwrite("available_at", &memory::MemoryRecord::available_at)
        .def_readwrite("maturity_at", &memory::MemoryRecord::maturity_at)
        .def_readwrite("key", &memory::MemoryRecord::key)
        .def_readwrite("value", &memory::MemoryRecord::value)
        .def_readwrite("label", &memory::MemoryRecord::label)
        .def_readwrite("label_valid", &memory::MemoryRecord::label_valid)
        .def_readwrite("reward", &memory::MemoryRecord::reward)
        .def_readwrite("reward_valid", &memory::MemoryRecord::reward_valid);
    py::class_<memory::MemoryNeighbor>(module, "MemoryNeighbor")
        .def_readonly("record", &memory::MemoryNeighbor::record)
        .def_readonly("similarity", &memory::MemoryNeighbor::similarity);
    module.def("normalize_key", &memory::normalize_memory_key);
    // Claves FP32 de varios registros en una matriz [R, 64], con los mismos bits que convertir
    // cada `record.key` con NumPy y sin una lista de Python por registro.
    module.def(
        "record_keys",
        [](const std::vector<memory::MemoryRecord>& records) {
            constexpr auto width = static_cast<py::ssize_t>(memory::episodic_memory_width);
            py::array_t<float> keys({static_cast<py::ssize_t>(records.size()), width});
            float* target = keys.mutable_data();
            for (const auto& record : records) {
                target = std::copy(record.key.begin(), record.key.end(), target);
            }
            return keys;
        },
        py::arg("records"));
    module.def("causal_reservoir_state", &memory::causal_reservoir_state, py::arg("seed"));
    module.def(
        "causal_reservoir_draws",
        [](const std::string& state, uint64_t seen, std::size_t capacity, std::size_t count) {
            auto draws = memory::causal_reservoir_draws(state, seen, capacity, count);
            return py::make_tuple(std::move(draws.slots), py::str(draws.state));
        },
        py::arg("state"), py::arg("seen"), py::arg("capacity"), py::arg("count"));
    py::class_<memory::EpisodicMemory>(module, "EpisodicMemory")
        .def(py::init<memory::MemoryScope, uint64_t, std::size_t>())
        .def(py::init<memory::MemoryScope, uint64_t, std::size_t, std::uint32_t>())
        .def("write",
             [](memory::EpisodicMemory& bank, const memory::MemoryRecord& record,
                int64_t confirmed_at) {
                 auto write = bank.prepare_write(record, confirmed_at);
                 if (!bank.commit(std::move(write))) {
                     throw std::runtime_error("La escritura episódica perdió su propietario");
                 }
             })
        .def("retain_batch",
             [](memory::EpisodicMemory& bank, const std::vector<memory::MemoryRecord>& incoming,
                const std::vector<uint64_t>& retained,
                int64_t confirmed_at) { bank.retain_batch(incoming, retained, confirmed_at); })
        .def("retained_records", &memory::EpisodicMemory::retained_records)
        // Columnas de los registros retenidos ordenadas por ID, como las listas que arma
        // `bank_snapshot` registro a registro y con los mismos bits.
        .def("retained_arrays",
             [](const memory::EpisodicMemory& bank) {
                 auto records = bank.retained_records();
                 std::ranges::sort(records, {}, &memory::MemoryRecord::id);
                 const auto rows = static_cast<py::ssize_t>(records.size());
                 constexpr auto width = static_cast<py::ssize_t>(memory::episodic_memory_width);
                 py::array_t<float> keys({rows, width});
                 py::array_t<float> values({rows, width});
                 py::array_t<double> labels(rows);
                 py::array_t<std::int64_t> ids(rows);
                 py::array_t<std::int64_t> decision(rows);
                 py::array_t<std::int64_t> available(rows);
                 py::array_t<std::int64_t> maturity(rows);
                 float* key = keys.mutable_data();
                 float* value = values.mutable_data();
                 for (py::ssize_t row = 0; row < rows; ++row) {
                     const auto& record = records[static_cast<std::size_t>(row)];
                     if (record.id > static_cast<std::uint64_t>(
                                         std::numeric_limits<std::int64_t>::max())) {
                         throw std::overflow_error("Un ID episódico no cabe en int64");
                     }
                     key = std::ranges::copy(record.key, key).out;
                     value = std::ranges::copy(record.value, value).out;
                     labels.mutable_at(row) = record.label;
                     ids.mutable_at(row) = static_cast<std::int64_t>(record.id);
                     decision.mutable_at(row) = record.decision_at;
                     available.mutable_at(row) = record.available_at;
                     maturity.mutable_at(row) = record.maturity_at;
                 }
                 return py::dict(py::arg("keys") = keys, py::arg("values") = values,
                                 py::arg("labels") = labels, py::arg("ids") = ids,
                                 py::arg("decision_at") = decision,
                                 py::arg("available_at") = available,
                                 py::arg("maturity_at") = maturity);
             })
        .def("validate_batch",
             [](const memory::EpisodicMemory& bank,
                const std::vector<memory::MemoryRecord>& incoming,
                int64_t confirmed_at) { return bank.validate_batch(incoming, confirmed_at); })
        .def(
            "query",
            [](const memory::EpisodicMemory& bank, const memory::MemoryVector& key, int64_t cutoff,
               std::optional<uint64_t> exclude_id) {
                const auto result = bank.query(key, cutoff, exclude_id);
                const auto selected = std::span(result.neighbors).first(result.count);
                return std::vector(selected.begin(), selected.end());
            },
            py::arg("key"), py::arg("cutoff"), py::arg("exclude_id") = std::nullopt)
        .def("snapshot_bytes",
             [](const memory::EpisodicMemory& bank) {
                 return py::bytes(memory::serialize_memory(bank.snapshot()));
             })
        .def("snapshot_metadata",
             [](const memory::EpisodicMemory& bank) {
                 const auto state = bank.snapshot();
                 return py::dict(py::arg("seen") = state.seen, py::arg("last_id") = state.last_id,
                                 py::arg("confirmed_at") = state.confirmed_at);
             })
        .def("restore_bytes",
             [](memory::EpisodicMemory& bank, const py::bytes& bytes) {
                 if (py::len(bytes) > memory::maximum_episodic_archive_bytes) {
                     throw std::invalid_argument(
                         "El archivo del enlace excede el presupuesto episódico");
                 }
                 const auto archive = bytes.cast<std::string>();
                 bank.restore(memory::deserialize_memory(archive));
             })
        // El mismo estado que restore_bytes(source.snapshot_bytes()), sin pasar por el archivo:
        // el instantáneo de un banco ya es canónico y el archivo lo conserva bit a bit.
        .def(
            "restore_from",
            [](memory::EpisodicMemory& bank, const memory::EpisodicMemory& source) {
                bank.restore(source.snapshot());
            },
            py::arg("source"))
        .def_property_readonly("size", &memory::EpisodicMemory::size)
        .def_property_readonly("seen", &memory::EpisodicMemory::seen);
}

using Matrix = py::array_t<double, py::array::c_style>;

// El pool inter-op de ATen solo se usa si se registró el aviso de fork y este proceso no es un
// hijo de fork. Un hijo hereda el pool sin sus hilos y una tarea lanzada allí no terminaría.
std::atomic<bool> inter_op_pool_usable{false};

// Hilos de una proyección: los de ATen en este proceso, que siguen OMP_NUM_THREADS o
// torch.set_num_threads, si la llamada es grande y el pool se puede usar, y uno si no. El
// resultado no depende de los hilos.
std::size_t projection_threads(std::size_t rows, std::size_t width, std::size_t outputs) {
    if (!inter_op_pool_usable.load(std::memory_order_relaxed)) {
        return 1;
    }
    return mars_titan::episodic::projection_threads(
        rows, width, outputs, static_cast<std::size_t>(std::max(at::get_num_threads(), 1)));
}

// Tareas de una proyección repartida que siguen en el pool. Vive en el montón y cada tarea guarda
// una referencia, así que la última puede avisar aunque quien llama ya haya vuelto.
struct PendingRanges {
    std::mutex mutex;
    std::condition_variable done;
    std::size_t left = 0;
    std::exception_ptr failure;
};

// Con varios hilos, quien llama calcula el primer tramo de grupos de ocho salidas y una tarea
// del pool inter-op de ATen (at::launch) cada uno de los demás. Cada tramo escribe solo sus
// columnas. at::parallel_for no sirve aquí porque se expande en esta unidad, y sin OpenMP al
// compilarla calcula todo en un hilo. Clang no genera OpenMP para el libgomp que trae torch. El
// pool persiste entre llamadas, así que no se crean hilos en cada una.
void project(std::span<const double> source, std::size_t rows, std::size_t width,
             std::span<const double> weights, std::size_t outputs, std::span<double> target,
             std::size_t threads) {
    namespace episodic = mars_titan::episodic;
    const auto isa = episodic::default_projection_isa();
    const std::size_t groups = episodic::projection_groups(outputs);
    const std::size_t parts = std::min(threads, groups);
    if (parts <= 1) {
        episodic::project_block(source, rows, width, weights, outputs, target, isa);
        return;
    }
    const std::size_t chunk = (groups + parts - 1) / parts;
    const auto range = [&](std::size_t first) noexcept -> std::exception_ptr {
        try {
            episodic::project_block_groups(source, rows, width, weights, outputs, target, isa,
                                           first, std::min(groups, first + chunk));
            return nullptr;
        } catch (...) {
            return std::current_exception();
        }
    };
    auto pending = std::make_shared<PendingRanges>();
    pending->left = (groups + chunk - 1) / chunk - 1;
    for (std::size_t first = chunk; first < groups; first += chunk) {
        // Cada tramo se calcula una sola vez, en el pool o aquí si el pool no lo admite.
        const auto task = [pending, &range, first] {
            auto failure = range(first);
            const std::lock_guard lock(pending->mutex);
            if (failure && !pending->failure) {
                pending->failure = std::move(failure);
            }
            if (--pending->left == 0) {
                pending->done.notify_one();
            }
        };
        try {
            at::launch(task);
        } catch (...) {
            task();
        }
    }
    auto failure = range(0);
    // Las tareas leen las entradas y escriben la salida, así que se espera a todas antes de
    // volver o de lanzar un error.
    std::unique_lock lock(pending->mutex);
    pending->done.wait(lock, [&] { return pending->left == 0; });
    if (!failure) {
        failure = pending->failure;
    }
    lock.unlock();
    if (failure) {
        std::rethrow_exception(failure);
    }
}

// Proyección del codec episódico. Solo acepta float64 contiguo, sin conversiones implícitas,
// y calcula sin el GIL porque la salida es nueva y las entradas son inmutables en el codec.
void bind_projection(py::module_& module) {
    module.def(
        "project_codec_block",
        [](const Matrix& block, const Matrix& transposed) {
            if (block.ndim() != 2 || transposed.ndim() != 2 ||
                block.shape(1) != transposed.shape(0)) {
                throw std::invalid_argument("Las formas de la proyección del codec no coinciden");
            }
            const auto rows = static_cast<std::size_t>(block.shape(0));
            const auto width = static_cast<std::size_t>(block.shape(1));
            const auto outputs = static_cast<std::size_t>(transposed.shape(1));
            Matrix out({block.shape(0), transposed.shape(1)});
            const std::span<const double> source(block.data(), rows * width);
            const std::span<const double> weights(transposed.data(), width * outputs);
            const std::span<double> target(out.mutable_data(), rows * outputs);
            const std::size_t threads = projection_threads(rows, width, outputs);
            {
                const py::gil_scoped_release released;
                project(source, rows, width, weights, outputs, target, threads);
            }
            return out;
        },
        py::arg("block").noconvert(), py::arg("transposed").noconvert());
    // Hilos que usaría project_codec_block con esas formas, para registrarlos en las medidas.
    module.def("codec_projection_threads", &projection_threads, py::arg("rows"),
               py::arg("width"), py::arg("outputs"));
    // Versión del núcleo que usa project_codec_block en este procesador, para registrarla en las
    // medidas. Todas dan los mismos bits.
    module.attr("codec_projection_isa") = std::string(
        mars_titan::episodic::projection_isa_name(mars_titan::episodic::default_projection_isa()));
}

// Fondo de los medoids anclados tras el producto de NumPy. Solo acepta float64 contiguo, sin
// conversiones implícitas, y escribe los cuadrados en `products` y el mínimo en `nearest`, que
// son vistas de buffers de `_native_bounded_minimum`. Calcula sin el GIL.
void bind_medoid_kernels(py::module_& module) {
    module.def(
        "nearest_reference_block",
        [](Matrix& products, const Matrix& client_norms, const Matrix& center_norms,
           const Matrix& clients, const Matrix& fixed, Matrix& nearest) {
            if (products.ndim() != 2 || client_norms.ndim() != 1 || center_norms.ndim() != 1 ||
                clients.ndim() != 2 || fixed.ndim() != 2 || nearest.ndim() != 1 ||
                clients.shape(1) != fixed.shape(1)) {
                throw std::invalid_argument("Las formas del fondo de medoids no coinciden");
            }
            const auto size = [](const Matrix& array) {
                return static_cast<std::size_t>(array.size());
            };
            const std::span<double> squares(products.mutable_data(), size(products));
            const std::span<double> target(nearest.mutable_data(), size(nearest));
            const std::span<const double> rows(client_norms.data(), size(client_norms));
            const std::span<const double> columns(center_norms.data(), size(center_norms));
            const std::span<const double> left(clients.data(), size(clients));
            const std::span<const double> right(fixed.data(), size(fixed));
            const auto dimensions = static_cast<std::size_t>(clients.shape(1));
            const py::gil_scoped_release released;
            mars_titan::medoids::nearest_reference_block(squares, rows, columns, left, right,
                                                         dimensions, target);
        },
        py::arg("products").noconvert(), py::arg("client_norms").noconvert(),
        py::arg("center_norms").noconvert(), py::arg("clients").noconvert(),
        py::arg("fixed").noconvert(), py::arg("nearest").noconvert());
    // Sumas exactas de las propuestas de `_Costs`. `distance` puede ser una vista con pasos del
    // búfer de distancias. Devuelve None si la referencia debe repetir la suma.
    module.def(
        "min_column_sums",
        [](const py::array_t<double>& distance, const Matrix& base) -> py::object {
            if (distance.ndim() != 2 || base.ndim() != 1) {
                throw std::invalid_argument("Las formas de las sumas de los medoids no coinciden");
            }
            const auto step = [](py::ssize_t stride) {
                if (stride < 0 || stride % static_cast<py::ssize_t>(sizeof(double)) != 0) {
                    throw std::invalid_argument("Las distancias necesitan pasos positivos");
                }
                return static_cast<std::size_t>(stride) / sizeof(double);
            };
            const mars_titan::medoids::StridedMatrix view{
                distance.data(), static_cast<std::size_t>(distance.shape(0)),
                static_cast<std::size_t>(distance.shape(1)), step(distance.strides(0)),
                step(distance.strides(1))};
            const std::span<const double> limits(base.data(),
                                                 static_cast<std::size_t>(base.size()));
            std::vector<double> totals(view.columns);
            bool exact = false;
            {
                const py::gil_scoped_release released;
                exact = mars_titan::medoids::min_column_sums(view, limits, totals);
            }
            return exact ? py::cast(totals) : py::none();
        },
        py::arg("distance").noconvert(), py::arg("base").noconvert());
}

void bind_cohorts(py::module_& module) {
    py::enum_<cohorts::PredictionMode>(module, "PredictionMode")
        .value("stateless", cohorts::PredictionMode::stateless)
        .value("prepared", cohorts::PredictionMode::prepared)
        .value("financial", cohorts::PredictionMode::financial);
    py::enum_<cohorts::EventKind>(module, "EventKind")
        .value("warmup", cohorts::EventKind::warmup)
        .value("decision", cohorts::EventKind::decision)
        .value("settlement", cohorts::EventKind::settlement);
    py::enum_<cohorts::PrefixReason>(module, "PrefixReason")
        .value("insufficient_pairs", cohorts::PrefixReason::insufficient_pairs)
        .value("zero_market_variance", cohorts::PrefixReason::zero_market_variance);
    py::class_<cohorts::PhaseContract>(module, "PhaseContract")
        .def(py::init<std::string, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
                      std::string>())
        .def_readonly("partition", &cohorts::PhaseContract::partition)
        .def_readonly("warmup_start", &cohorts::PhaseContract::warmup_start)
        .def_readonly("decision_start", &cohorts::PhaseContract::decision_start)
        .def_readonly("decision_end", &cohorts::PhaseContract::decision_end)
        .def_readonly("close_at", &cohorts::PhaseContract::close_at)
        .def_readonly("prefix_policy_sha256", &cohorts::PhaseContract::prefix_policy_sha256);
    py::enum_<cohorts::Boundary>(module, "Boundary")
        .value("before_predictions", cohorts::Boundary::before_predictions)
        .value("predictions_ready", cohorts::Boundary::predictions_ready)
        .value("record_written", cohorts::Boundary::record_written)
        .value("feedback_applied", cohorts::Boundary::feedback_applied)
        .value("checkpoint_written", cohorts::Boundary::checkpoint_written)
        .value("before_commit", cohorts::Boundary::before_commit)
        .value("committed", cohorts::Boundary::committed);
    py::class_<cohorts::Identity>(module, "Identity")
        .def(py::init<>())
        .def_readwrite("source_sha256", &cohorts::Identity::source_sha256)
        .def_readwrite("view_sha256", &cohorts::Identity::view_sha256)
        .def_readwrite("representation_sha256", &cohorts::Identity::representation_sha256)
        .def_readwrite("model_sha256", &cohorts::Identity::model_sha256);
    py::class_<cohorts::Task>(module, "Task")
        .def(py::init<std::string, std::uint32_t>())
        .def_readonly("name", &cohorts::Task::name)
        .def_readonly("horizon", &cohorts::Task::horizon);
    py::class_<cohorts::Limits>(module, "Limits")
        .def(py::init<>())
        .def_readwrite("feature_width", &cohorts::Limits::feature_width)
        .def_readwrite("max_assets", &cohorts::Limits::max_assets)
        .def_readwrite("max_pending", &cohorts::Limits::max_pending)
        .def_readwrite("max_state_bytes", &cohorts::Limits::max_state_bytes)
        .def_readwrite("max_checkpoint_bytes", &cohorts::Limits::max_checkpoint_bytes)
        .def_readwrite("max_record_bytes", &cohorts::Limits::max_record_bytes)
        .def_readwrite("max_log_bytes", &cohorts::Limits::max_log_bytes)
        .def_readwrite("max_cohorts", &cohorts::Limits::max_cohorts);
    py::class_<cohorts::Definition>(module, "Definition")
        .def(py::init<>())
        .def_readwrite("identity", &cohorts::Definition::identity)
        .def_readwrite("tasks", &cohorts::Definition::tasks)
        .def_readwrite("limits", &cohorts::Definition::limits)
        .def_readwrite("prediction_mode", &cohorts::Definition::prediction_mode)
        .def_readwrite("phase", &cohorts::Definition::phase)
        .def_property(
            "initial_state_json",
            [](const cohorts::Definition& definition) { return definition.initial_state.dump(); },
            [](cohorts::Definition& definition, const std::string& bytes) {
                definition.initial_state = state_json(bytes);
            });
    py::class_<cohorts::Observation>(module, "Observation")
        .def(py::init<std::string, std::int64_t, std::vector<double>>())
        .def_readonly("asset", &cohorts::Observation::asset)
        .def_readonly("available_at", &cohorts::Observation::available_at)
        .def_readonly("features", &cohorts::Observation::features);
    py::class_<cohorts::Cohort>(module, "Cohort")
        .def(py::init<>())
        .def_readwrite("cursor", &cohorts::Cohort::cursor)
        .def_readwrite("cutoff", &cohorts::Cohort::cutoff)
        .def_readwrite("observations", &cohorts::Cohort::observations)
        .def_readwrite("kind", &cohorts::Cohort::kind)
        .def_readwrite("close_phase", &cohorts::Cohort::close_phase)
        .def_readwrite("prefix_exclusions", &cohorts::Cohort::prefix_exclusions);
    py::class_<cohorts::PrefixExclusion>(module, "PrefixExclusion")
        .def(py::init<std::string, cohorts::Task, std::int64_t, cohorts::PrefixReason, std::size_t,
                      std::optional<double>, std::string>())
        .def_readonly("asset", &cohorts::PrefixExclusion::asset)
        .def_readonly("task", &cohorts::PrefixExclusion::task)
        .def_readonly("decision_at", &cohorts::PrefixExclusion::decision_at)
        .def_readonly("reason", &cohorts::PrefixExclusion::reason)
        .def_readonly("history_pairs", &cohorts::PrefixExclusion::history_pairs)
        .def_readonly("market_variance", &cohorts::PrefixExclusion::market_variance)
        .def_readonly("evidence_sha256", &cohorts::PrefixExclusion::evidence_sha256);
    py::class_<cohorts::Prediction>(module, "Prediction")
        .def_readonly("id", &cohorts::Prediction::id)
        .def_readonly("asset", &cohorts::Prediction::asset)
        .def_readonly("task", &cohorts::Prediction::task)
        .def_readonly("generation", &cohorts::Prediction::generation)
        .def_readonly("decision_at", &cohorts::Prediction::decision_at)
        .def_readonly("value", &cohorts::Prediction::value);
    py::class_<cohorts::Feedback>(module, "Feedback")
        .def(py::init<std::string, std::uint32_t, std::int64_t, double>())
        .def_readonly("prediction_id", &cohorts::Feedback::prediction_id)
        .def_readonly("revision", &cohorts::Feedback::revision)
        .def_readonly("available_at", &cohorts::Feedback::available_at)
        .def_readonly("value", &cohorts::Feedback::value);
    py::class_<cohorts::ResolvedFeedback>(module, "ResolvedFeedback")
        .def_readonly("id", &cohorts::ResolvedFeedback::id)
        .def_readonly("prediction", &cohorts::ResolvedFeedback::prediction)
        .def_readonly("label", &cohorts::ResolvedFeedback::label);
    py::class_<cohorts::ResolvedPrefixExclusion>(module, "ResolvedPrefixExclusion")
        .def_readonly("prediction", &cohorts::ResolvedPrefixExclusion::prediction)
        .def_readonly("evidence", &cohorts::ResolvedPrefixExclusion::evidence);
    py::class_<cohorts::AdministrativeFinalization>(module, "AdministrativeFinalization")
        .def_readonly("prediction", &cohorts::AdministrativeFinalization::prediction)
        .def_readonly("closed_at", &cohorts::AdministrativeFinalization::closed_at);
    py::class_<cohorts::Commit>(module, "Commit")
        .def_readonly("generation", &cohorts::Commit::generation)
        .def_readonly("predictions", &cohorts::Commit::predictions)
        .def_readonly("applied", &cohorts::Commit::applied)
        .def_readonly("record_sha256", &cohorts::Commit::record_sha256)
        .def_readonly("excluded", &cohorts::Commit::excluded)
        .def_readonly("finalized", &cohorts::Commit::finalized);
    py::class_<BoundExecutor>(module, "Executor")
        .def(py::init<const std::string&, cohorts::Definition, py::function, py::function, bool>())
        .def(
            "step",
            [](BoundExecutor& owner, const cohorts::Cohort& cohort,
               const std::vector<cohorts::Feedback>& feedback, std::size_t batch_rows,
               const py::object& fault) {
                if (std::exchange(owner.step_active, true)) {
                    throw std::runtime_error("El ejecutor ya está procesando un step");
                }
                const auto reset = c10::make_scope_exit([&owner] { owner.step_active = false; });
                std::function<void(cohorts::Boundary)> hook;
                if (!fault.is_none()) {
                    hook = [callback = fault.cast<py::function>()](cohorts::Boundary point) {
                        callback(point);
                    };
                }
                return owner.get().step(cohort, feedback, batch_rows, hook);
            },
            py::arg("cohort"), py::arg("feedback"),
            py::arg("batch_rows") = cohorts::default_batch_rows, py::arg("fault") = py::none())
        .def("close", &BoundExecutor::close)
        .def("snapshot_json", [](BoundExecutor& owner) { return owner.get().snapshot().dump(); })
        .def("record_json",
             [](BoundExecutor& owner, std::size_t generation) {
                 return owner.get().record(generation).dump();
             })
        .def("pending", [](BoundExecutor& owner) { return owner.get().pending(); })
        .def_property_readonly("cursor", [](BoundExecutor& owner) { return owner.get().cursor(); });
}
} // namespace

#ifdef MARS_TITAN_CANDIDATE_PYTHON
namespace mars_titan::candidate {
void bind_candidate(py::module_& module);
}
#endif

PYBIND11_MODULE(_episodic_native, module) {
    module.doc() = "Banco CPU y ejecución de cohortes con una sola publicación confirmada";
    // Si no se puede registrar el aviso, la proyección no usa el pool inter-op.
    inter_op_pool_usable.store(
        pthread_atfork(nullptr, nullptr, [] { inter_op_pool_usable.store(false); }) == 0);
    module.attr("abi_version") = 1;
    module.attr("torch_version") = MARS_TITAN_EPISODIC_TORCH_VERSION;
#ifdef MARS_TITAN_CANDIDATE_PYTHON
    mars_titan::candidate::bind_candidate(module);
#endif
    module.def("require_safe_path",
               [](const std::string& path) { mars_titan::simulation::require_safe_path(path); });
    module.def("seal_blob", [](const std::string& directory, std::size_t maximum,
                               const py::bytes& value) {
        constexpr std::size_t limit = 64 * cohorts::mebibyte;
        if (maximum == 0 || maximum > limit || py::len(value) == 0 || py::len(value) > maximum) {
            throw std::invalid_argument("El artefacto excede su presupuesto de archivo");
        }
        const auto bytes = value.cast<std::string>();
        const auto digest = mars_titan::simulation::content_sha256(bytes);
        const auto folder = std::filesystem::path(directory);
        mars_titan::simulation::require_safe_path(folder);
        std::filesystem::create_directories(folder);
        const auto path = folder / (digest + ".bin");
        mars_titan::simulation::require_safe_path(path);
        if (std::filesystem::exists(path)) {
            if (mars_titan::simulation::read_bounded_file(path, maximum) != bytes) {
                throw std::invalid_argument("El artefacto sellado contiene otros bytes");
            }
        } else {
            mars_titan::simulation::atomic_binary_file(path, bytes, maximum, false);
        }
        return py::dict(py::arg("name") = path.filename().string(), py::arg("sha256") = digest,
                        py::arg("bytes") = bytes.size());
    });
    module.def("read_blob", [](const std::string& path, std::size_t maximum) {
        constexpr std::size_t limit = 64 * cohorts::mebibyte;
        if (maximum == 0 || maximum > limit) {
            throw std::invalid_argument("El límite del artefacto no es válido");
        }
        return py::bytes(mars_titan::simulation::read_bounded_file(path, maximum));
    });
    bind_memory(module);
    bind_projection(module);
    bind_medoid_kernels(module);
    bind_cohorts(module);
}
