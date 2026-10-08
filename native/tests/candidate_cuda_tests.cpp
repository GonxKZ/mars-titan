#include "mars_titan/candidate.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/DeviceAccelerator.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

// Los literales definen fixtures y tolerancias, sin datos ni objetivos financieros.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using namespace mars_titan::candidate;
void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}
Inputs inputs(const Config& config, at::ScalarType dtype) {
    const auto options = at::TensorOptions().dtype(dtype);
    return {at::linspace(-1., 1., 2 * price_window * config.dimensions[0], options)
                .reshape({2, price_window, config.dimensions[0]}),
            at::ones({2, config.dimensions[1]}, options),
            at::ones({2, config.dimensions[2]}, options) * 2,
            at::ones({2, config.dimensions[3]}, options) * 3,
            at::ones({2, config.dimensions[4]}, options) * 4,
            at::ones({2, modality_count}, at::kBool)};
}
Inputs on_device(const Inputs& value, const at::Device& device) {
    return {value.prices.to(device),       value.news.to(device),  value.charts.to(device),
            value.fundamentals.to(device), value.macro.to(device), value.presence.to(device)};
}
struct CpuReference {
    at::Tensor values;
    double rtol;
    double atol;
};
struct Case {
    int64_t refinements;
    bool memory;
};
double compare(const at::Tensor& actual, const CpuReference& expected) {
    const auto copied = actual.cpu();
    require(at::isfinite(copied).all().item<bool>(), "La salida CUDA no es finita");
    require(at::allclose(copied, expected.values, expected.rtol, expected.atol),
            "No se conserva la tolerancia CPU/CUDA");
    return (copied - expected.values).abs().max().item<double>();
}
void check(at::ScalarType dtype) {
    Config config;
    config.normalization_id = "candidate-cuda-fixture-v1";
    Candidate cpu(config, dtype);
    std::stringstream initial;
    cpu.save_state(initial);
    const at::Device device("cuda:0");
    const auto gpu = Candidate::load_state(initial, device);
    const auto data = inputs(config, dtype);
    const auto cuda_data = on_device(data, device);
    const auto options = at::TensorOptions().dtype(dtype);
    auto keys = at::zeros({8, hidden_width}, options);
    for (int64_t index = 0; index < 8; ++index) {
        keys.index_put_({index, index}, 1.);
    }
    const auto features =
        at::linspace(-.5, .5, 8 * feature_width, options).reshape({8, feature_width});
    const auto returns = at::linspace(-.02, .02, 8, options);
    const auto ids = at::arange(1, 9, at::kLong);
    const auto memory = cpu.snapshot(keys, features, returns, ids, cpu.representation_id());
    const auto cuda_memory = gpu->snapshot(keys.to(device), features.to(device), returns.to(device),
                                           ids, gpu->representation_id());
    const double rtol = dtype == at::kDouble ? 1e-8 : 2e-4;
    const double atol = dtype == at::kDouble ? 1e-10 : 2e-6;
    double largest = 0;
    constexpr std::array<Case, 4> cases{{{1, false}, {1, true}, {2, true}, {4, true}}};
    for (const auto& scenario : cases) {
        const auto selected_memory = scenario.memory ? memory : cpu.empty_memory();
        const auto selected_cuda_memory = scenario.memory ? cuda_memory : gpu->empty_memory();
        cpu.zero_grad();
        gpu->zero_grad();
        const auto expected = cpu.forward(data, selected_memory, scenario.refinements);
        const auto actual = gpu->forward(cuda_data, selected_cuda_memory, scenario.refinements);
        largest = std::max(largest, compare(actual.quantiles, {expected.quantiles, rtol, atol}));
        static_cast<void>(compare(actual.state, {expected.state, rtol, atol}));
        if (scenario.memory) {
            static_cast<void>(compare(actual.read.weights, {expected.read.weights, rtol, atol}));
        }
        require(at::equal(actual.read.presence.cpu(), expected.read.presence),
                "La lectura cambia la presencia de memoria");
        static_cast<void>(compare(actual.encoded.fused, {expected.encoded.fused, rtol, atol}));
        static_cast<void>(compare(actual.encoded.episode_features,
                                  {expected.encoded.episode_features, rtol, atol}));
        static_cast<void>(
            compare(actual.encoded.episode_keys, {expected.encoded.episode_keys, rtol, atol}));
        require(at::equal(actual.read.ids.cpu(), expected.read.ids), "La lectura altera los IDs");
        expected.quantiles.square().sum().backward();
        actual.quantiles.square().sum().backward();
        const auto other = gpu->named_parameters();
        for (const auto& item : cpu.named_parameters()) {
            const auto& left = item.value().grad();
            const auto& right = other[item.key()].grad();
            require(left.defined() == right.defined(),
                    "Falta un gradiente en uno de los dispositivos");
            if (left.defined()) {
                static_cast<void>(compare(right, {left, rtol, atol}));
            }
        }
        std::stringstream saved;
        gpu->save_state(saved);
        const auto recovered = Candidate::load_state(saved, device);
        const auto restored_memory =
            scenario.memory
                ? recovered->snapshot(keys.to(device), features.to(device), returns.to(device), ids,
                                      recovered->representation_id())
                : recovered->empty_memory();
        const auto restored = recovered->forward(cuda_data, restored_memory, scenario.refinements);
        require(at::equal(actual.quantiles, restored.quantiles),
                "La recuperación cambia los cuantiles");
        require(at::equal(actual.read.ids, restored.read.ids),
                "La recuperación cambia los vecinos");
    }
    std::cout << "{\"dtype\":\"" << (dtype == at::kDouble ? "float64" : "float32")
              << "\",\"cases\":4,\"maximum_output_error\":" << largest
              << ",\"gradient_cases\":4,\"recovery_cases\":4,\"exact_recovery\":true}\n";
}
} // namespace

int main() {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        std::cout << std::setprecision(17);
        require(at::globalContext().hasCUDA(), "El ejecutable necesita CUDA, sin fallback");
        at::globalContext().setAllowTF32CuBLAS(false);
        at::globalContext().setAllowTF32CuDNN(false);
        const auto cpu_rng = at::detail::getDefaultCPUGenerator().get_state().clone();
        const auto cuda_rng =
            at::globalContext().defaultGenerator(at::Device("cuda:0")).get_state().clone();
        at::accelerator::resetPeakStats(0);
        check(at::kDouble);
        check(at::kFloat);
        require(at::equal(cpu_rng, at::detail::getDefaultCPUGenerator().get_state()),
                "Cambió el RNG CPU");
        require(at::equal(cuda_rng,
                          at::globalContext().defaultGenerator(at::Device("cuda:0")).get_state()),
                "Cambió el RNG CUDA");
        at::accelerator::synchronizeDevice(0);
        const auto stats = at::accelerator::getDeviceStats(0);
        std::cout << "{\"device\":\"cuda:0\",\"trained\":false,\"peak_allocated_bytes\":"
                  << stats.allocated_bytes[0].peak
                  << ",\"peak_reserved_bytes\":" << stats.reserved_bytes[0].peak
                  << ",\"rng_unchanged\":true}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
