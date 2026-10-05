#include "mars_titan/adapter_control.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/DeviceAccelerator.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using namespace mars_titan::controls;
using Clock = std::chrono::steady_clock;
struct Options {
    std::string device = "cpu";
    std::size_t rows = 128;
    std::size_t inputs = 64;
    std::size_t outputs = 32;
    std::size_t rank = 4;
    std::size_t steps = 16;
    std::size_t repetitions = 5;
    uint64_t seed = 71;
};
std::size_t number(std::string_view text, std::size_t limit) {
    std::size_t result = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size() || result == 0 || result > limit)
        throw std::invalid_argument("Un argumento numérico no es positivo o supera el presupuesto");
    return result;
}
Options parse(std::span<char*> arguments) {
    if (arguments.size() > 9)
        throw std::invalid_argument("El control admite como máximo ocho argumentos");
    Options options;
    if (arguments.size() > 1)
        options.device = arguments[1];
    if (options.device != "cpu" && options.device != "cuda:0")
        throw std::invalid_argument("El control necesita cpu o cuda:0 de forma explícita");
    if (arguments.size() > 2)
        options.rows = number(arguments[2], 4096);
    if (arguments.size() > 3)
        options.inputs = number(arguments[3], 512);
    if (arguments.size() > 4)
        options.outputs = number(arguments[4], 512);
    if (arguments.size() > 5)
        options.rank = number(arguments[5], 512);
    if (arguments.size() > 6)
        options.steps = number(arguments[6], 1024);
    if (arguments.size() > 7)
        options.repetitions = number(arguments[7], 31);
    if (arguments.size() > 8)
        options.seed = number(arguments[8], 1U << 30);
    const auto fixture_elements = 2 * options.rows * (options.inputs + options.outputs) +
                                  5 * options.inputs * options.outputs;
    if (options.rank > std::min(options.inputs, options.outputs) ||
        fixture_elements > (64U << 20) / sizeof(double))
        throw std::invalid_argument(
            "Las formas del control exceden el rango o 64 MiB de datos preparados");
    if (options.device == "cuda:0" && (!at::hasCUDA() || at::getNumGPUs() == 0))
        throw std::invalid_argument("Se solicitó cuda:0 y no hay una GPU CUDA disponible");
    return options;
}
void synchronize(const std::string& device) {
    if (device == "cuda:0")
        at::accelerator::synchronizeDevice(0);
}
double elapsed(Clock::time_point before, Clock::time_point after) {
    return std::chrono::duration<double, std::milli>(after - before).count();
}
std::string encoded_identity(const SemanticVersions& versions) {
    if (!versions.output_state)
        throw std::runtime_error("El control no ha acreditado la identidad de su salida");
    constexpr std::string_view digits = "0123456789abcdef";
    std::string result;
    result.reserve(2 * versions.output_state->size());
    for (const auto byte : *versions.output_state) {
        result.push_back(digits[byte >> 4U]);
        result.push_back(digits[byte & 0x0fU]);
    }
    return result;
}
struct Data {
    at::Tensor base;
    at::Tensor train_inputs;
    at::Tensor train_targets;
    at::Tensor validation_inputs;
    at::Tensor validation_targets;
};
Data fixture(const Options& options) {
    auto rng = at::detail::createCPUGenerator(options.seed ^ 0xd1b54a32d192ed03ULL);
    const auto rows = static_cast<int64_t>(options.rows);
    const auto in = static_cast<int64_t>(options.inputs);
    const auto out = static_cast<int64_t>(options.outputs);
    const auto rank = static_cast<int64_t>(std::min<std::size_t>(2, options.rank));
    auto parent = at::randn({out, in}, rng, at::kDouble) / std::sqrt(static_cast<double>(in));
    const auto correction =
        at::mm(at::randn({out, rank}, rng, at::kDouble), at::randn({rank, in}, rng, at::kDouble)) *
        0.05;
    auto train = at::randn({rows, in}, rng, at::kDouble);
    auto validation = at::randn({rows, in}, rng, at::kDouble);
    const auto teacher = parent + correction;
    return {parent, train, at::mm(train, teacher.t()), validation, at::mm(validation, teacher.t())};
}
struct Trial {
    double setup_ms = 0;
    double training_ms = 0;
    double recovery_ms = 0;
    double total_ms = 0;
    double selected_mse = 0;
    double reference_error = 0;
    std::size_t best_step = 0;
    std::size_t trainable = 0;
    std::size_t tensor_bytes = 0;
    std::size_t archive_bytes = 0;
    int64_t peak_cuda_allocated_bytes = 0;
    int64_t peak_cuda_reserved_bytes = 0;
    std::string current_identity{};
    std::string selected_identity{};
};
Trial run(const Options& options, const AdapterConfig& config, const Data& data,
          const at::Tensor& expected, const at::Tensor& expected_selected) {
    if (options.device == "cuda:0") {
        at::globalContext().lazyInitDevice(at::kCUDA);
        at::accelerator::resetPeakStats(0);
    }
    synchronize(options.device);
    const auto start = Clock::now();
    const auto device = at::Device(options.device);
    const auto x = data.train_inputs.to(device);
    const auto y = data.train_targets.to(device);
    AdapterControl control(config, data.base, data.validation_inputs, data.validation_targets,
                           options.device);
    synchronize(options.device);
    const auto initialized = Clock::now();
    double recovery = 0;
    std::size_t checkpoint_bytes = 0;
    for (std::size_t step = 0; step < options.steps; ++step) {
        static_cast<void>(control.train(x, y));
        static_cast<void>(control.versions());
        if (step + 1 == (options.steps + 1) / 2) {
            synchronize(options.device);
            const auto checkpoint_start = Clock::now();
            const auto current_identity = control.versions();
            const auto selected_identity = control.versions(true);
            const auto archive = serialize_adapter(control.snapshot());
            checkpoint_bytes = archive.size();
            const auto state = deserialize_adapter(archive);
            AdapterControl restored(config, data.base, data.validation_inputs,
                                    data.validation_targets, options.device);
            restored.restore(state);
            if (restored.versions() != current_identity ||
                restored.versions(true) != selected_identity)
                throw std::runtime_error(
                    "La recuperación cambia la identidad del estado de salida");
            control = std::move(restored);
            synchronize(options.device);
            recovery += elapsed(checkpoint_start, Clock::now());
        }
    }
    synchronize(options.device);
    const auto trained = Clock::now();
    const auto prediction = control.predict(x).cpu();
    const auto selected = control.predict(x, true).cpu();
    const auto error = (prediction - expected).abs().max().item<double>();
    if (!at::allclose(prediction, expected, 1e-10, 1e-11) ||
        !at::allclose(selected, expected_selected, 1e-10, 1e-11))
        throw std::runtime_error(
            "La ejecución recuperada no coincide con la referencia CPU ininterrumpida");
    const auto state = control.snapshot();
    const auto current_identity = encoded_identity(control.versions());
    const auto selected_identity = encoded_identity(control.versions(true));
    Trial result{elapsed(start, initialized),
                 elapsed(initialized, trained) - recovery,
                 recovery,
                 elapsed(start, Clock::now()),
                 state.best_mse,
                 error,
                 state.best_step,
                 control.trainable_parameters(),
                 control.state_tensor_bytes(),
                 checkpoint_bytes,
                 0,
                 0,
                 current_identity,
                 selected_identity};
    if (options.device == "cuda:0") {
        const auto stats = at::accelerator::getDeviceStats(0);
        result.peak_cuda_allocated_bytes = stats.allocated_bytes[0].peak;
        result.peak_cuda_reserved_bytes = stats.reserved_bytes[0].peak;
    }
    return result;
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "Uso: mars-titan-adapter-control [cpu|cuda:0 filas entradas salidas rango "
                         "pasos repeticiones semilla]\n"
                         "Regresión matricial FP64 con padre fijo y validación independiente.\n";
            return 0;
        }
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        const auto options = parse(std::span(argv, static_cast<std::size_t>(argc)));
        const auto data = fixture(options);
        const auto parent_mse =
            (at::mm(data.validation_inputs, data.base.t()) - data.validation_targets)
                .square()
                .mean()
                .item<double>();
        std::cout << std::setprecision(10) << "{\"schema\":1,\"device\":\"" << options.device
                  << "\",\"dtype\":\"float64\",\"rows\":" << options.rows
                  << ",\"inputs\":" << options.inputs << ",\"outputs\":" << options.outputs
                  << ",\"rank\":" << options.rank << ",\"steps\":" << options.steps
                  << ",\"seed\":" << options.seed
                  << ",\"warmups\":1,\"repetitions\":" << options.repetitions
                  << ",\"parent_validation_mse\":" << parent_mse
                  << ",\"reference_steps_per_kind\":" << options.steps
                  << ",\"transfer_bytes_measured\":null,\"results\":[";
        bool first = true;
        for (const auto kind : {AdapterKind::full, AdapterKind::residual, AdapterKind::low_rank}) {
            const AdapterConfig config{.kind = kind,
                                       .inputs = options.inputs,
                                       .outputs = options.outputs,
                                       .rank = options.rank,
                                       .seed = options.seed,
                                       .max_steps = options.steps};
            AdapterControl reference(config, data.base, data.validation_inputs,
                                     data.validation_targets);
            for (std::size_t step = 0; step < options.steps; ++step)
                static_cast<void>(reference.train(data.train_inputs, data.train_targets));
            const auto expected = reference.predict(data.train_inputs);
            const auto selected = reference.predict(data.train_inputs, true);
            static_cast<void>(run(options, config, data, expected, selected));
            std::vector<double> times;
            Trial result;
            double setup = 0;
            double training = 0;
            double recovery = 0;
            for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition) {
                result = run(options, config, data, expected, selected);
                setup += result.setup_ms;
                training += result.training_ms;
                recovery += result.recovery_ms;
                times.push_back(result.total_ms);
            }
            std::sort(times.begin(), times.end());
            if (!first)
                std::cout << ',';
            first = false;
            const auto repetitions = static_cast<double>(options.repetitions);
            std::cout << "{\"kind\":\"" << adapter_kind_name(kind)
                      << "\",\"trainable_parameters\":" << result.trainable
                      << ",\"state_tensor_bytes\":" << result.tensor_bytes
                      << ",\"checkpoint_bytes\":" << result.archive_bytes
                      << ",\"best_step\":" << result.best_step
                      << ",\"selected_validation_mse\":" << result.selected_mse
                      << ",\"max_reference_error\":" << result.reference_error
                      << ",\"current_output_sha256\":\"" << result.current_identity
                      << "\",\"selected_output_sha256\":\"" << result.selected_identity << '"'
                      << ",\"setup_mean_ms\":" << setup / repetitions
                      << ",\"training_mean_ms\":" << training / repetitions
                      << ",\"recovery_mean_ms\":" << recovery / repetitions
                      << ",\"total_min_ms\":" << times.front()
                      << ",\"total_median_ms\":" << times[times.size() / 2]
                      << ",\"total_max_ms\":" << times.back() << ",\"training_rows_per_second\":"
                      << static_cast<double>(options.rows * options.steps) * 1000.0 * repetitions /
                             training
                      << ",\"peak_cuda_allocated_bytes\":"
                      << (options.device == "cuda:0"
                              ? std::to_string(result.peak_cuda_allocated_bytes)
                              : "null")
                      << ",\"peak_cuda_reserved_bytes\":"
                      << (options.device == "cuda:0"
                              ? std::to_string(result.peak_cuda_reserved_bytes)
                              : "null")
                      << '}';
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Falló el control de adaptadores: " << error.what() << '\n';
        return 1;
    }
}
