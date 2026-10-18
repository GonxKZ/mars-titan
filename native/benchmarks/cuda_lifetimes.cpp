#include "cuda_lifetime_options.hpp"
#include "../tests/candidate_cuda_fixture.hpp"

#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/core/CachingHostAllocator.h>
#include <ATen/ops/_cudnn_rnn_flatten_weight.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <charconv>
#include <shared_mutex>
#include <span>
#include <tuple>

// Formas fijas de los controles técnicos. No hay datos financieros ni optimizadores.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
void workspaces(std::string_view name, at::cuda::WorkspaceMapWithMutex& pool) {
    std::shared_lock lock(pool.mutex);
    std::cout << '"' << name << "\":[";
    bool first = true;
    for (const auto& [key, allocation] : pool.map) {
        std::cout << (first ? "" : ",") << "{\"handle\":\"" << std::get<0>(key)
                  << "\",\"stream\":\"" << std::get<1>(key)
                  << "\",\"address\":\"" << allocation.first.get()
                  << "\",\"bytes\":" << allocation.second << '}';
        first = false;
    }
    std::cout << ']';
}

void snapshot(std::string_view phase, int iteration) {
    at::accelerator::synchronizeDevice(0);
    const auto stats = at::accelerator::getDeviceStats(0);
    const auto memory = c10::cuda::CUDACachingAllocator::snapshot({0, 0}, false);
    std::cout << "{\"record\":\"allocation_snapshot\",\"phase\":\"" << phase
              << "\",\"iteration\":" << iteration
              << ",\"allocated_bytes\":" << stats.allocated_bytes[0].current
              << ",\"active_bytes\":" << stats.active_bytes[0].current
              << ",\"reserved_bytes\":" << stats.reserved_bytes[0].current
              << ",\"segments\":[";
    bool first_segment = true;
    std::size_t allocated = 0, active = 0, reserved = 0;
    for (const auto& segment : memory.segments) {
        if (segment.device != 0) {
            continue;
        }
        std::cout << (first_segment ? "" : ",") << "{\"address\":\"0x" << std::hex
                  << segment.address << std::dec << "\",\"bytes\":" << segment.total_size
                  << ",\"allocated_bytes\":" << segment.allocated_size
                  << ",\"active_bytes\":" << segment.active_size
                  << ",\"stream\":\"" << segment.stream << "\",\"blocks\":[";
        std::size_t offset = 0;
        bool first_block = true;
        for (const auto& block : segment.blocks) {
            std::cout << (first_block ? "" : ",") << "{\"address\":\"0x" << std::hex
                      << segment.address + offset << std::dec << "\",\"bytes\":" << block.size
                      << ",\"requested_bytes\":" << block.requested_size
                      << ",\"allocated\":" << (block.allocated ? "true" : "false")
                      << ",\"active\":" << (block.active ? "true" : "false") << '}';
            offset += block.size;
            first_block = false;
        }
        require(offset == segment.total_size, "Los bloques no suman el tamaño del segmento");
        allocated += segment.allocated_size;
        active += segment.active_size;
        reserved += segment.total_size;
        std::cout << "]}";
        first_segment = false;
    }
    require(allocated == static_cast<std::size_t>(stats.allocated_bytes[0].current) &&
                active == static_cast<std::size_t>(stats.active_bytes[0].current) &&
                reserved == static_cast<std::size_t>(stats.reserved_bytes[0].current),
            "El snapshot no concilia con las estadísticas del asignador");
    std::cout << "],\"host_segments\":[";
    auto* host_allocator = at::getHostAllocator(at::kCUDA);
    require(host_allocator != nullptr, "No hay asignador de memoria host para CUDA");
    bool first_host = true;
    for (const auto& segment : host_allocator->get_segments()) {
        std::cout << (first_host ? "" : ",") << "{\"address\":\"0x" << std::hex
                  << segment.address << std::dec << "\",\"bytes\":" << segment.size
                  << ",\"allocated\":" << (segment.allocated ? "true" : "false")
                  << ",\"active\":" << (segment.active ? "true" : "false") << '}';
        first_host = false;
    }
    std::cout << "],";
    workspaces("cublas_workspaces", at::cuda::cublas_handle_stream_to_workspace());
    std::cout << ',';
    workspaces("cublaslt_workspaces", at::cuda::cublaslt_handle_stream_to_workspace());
    std::cout << "}\n" << std::flush;
}

void gemm(at::ScalarType dtype) {
    const auto options = at::TensorOptions().dtype(dtype).device(at::Device("cuda:0"));
    auto left = at::ones({32, 64}, options).set_requires_grad(true);
    auto right = at::ones({64, 32}, options).set_requires_grad(true);
    const auto product = at::mm(left, right);
    product.square().sum().backward();
    require(at::equal(product, at::full_like(product, 64)) &&
                at::equal(left.grad(), at::full_like(left, 4096)) &&
                at::equal(right.grad(), at::full_like(right, 4096)),
            "El control GEMM no coincide con su solución analítica");
}

void gru(at::ScalarType dtype) {
    const auto options = at::TensorOptions().dtype(dtype).device(at::Device("cuda:0"));
    auto input = at::ones({2, 64, 5}, options).set_requires_grad(true);
    const auto initial = at::zeros({1, 2, 128}, options);
    std::vector<at::Tensor> weights{
        at::full({384, 5}, 0.01, options).set_requires_grad(true),
        at::full({384, 128}, 0.01, options).set_requires_grad(true),
        at::zeros({384}, options).set_requires_grad(true),
        at::zeros({384}, options).set_requires_grad(true)};
    {
        const at::NoGradGuard no_grad;
        static_cast<void>(at::_cudnn_rnn_flatten_weight(weights, 4, 5, 3, 128, 0, 1, true, false));
    }
    const auto output = std::get<0>(at::gru(input, initial, weights, true, 1, 0., true, false, true));
    output.square().sum().backward();
    require(at::isfinite(output).all().item<bool>() && at::isfinite(input.grad()).all().item<bool>(),
            "El control GRU produjo valores no finitos");
    for (const auto& weight : weights) {
        require(weight.grad().defined() && at::isfinite(weight.grad()).all().item<bool>(),
                "El control GRU perdió un gradiente o produjo valores no finitos");
    }
}

void workload(std::string_view selected) {
    for (const auto dtype : {at::kDouble, at::kFloat}) {
        if (selected == "gemm") {
            gemm(dtype);
        } else if (selected == "gru") {
            gru(dtype);
        } else if (selected == "candidate") {
            check(dtype);
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::span arguments(argv, static_cast<std::size_t>(argc));
        if (arguments.size() != 4) {
            throw std::invalid_argument("Uso: mars-titan-cuda-lifetimes carga repeticiones modo");
        }
        const auto options = mars_titan::diagnostics::lifetime_options(
            std::array<std::string_view, 3>{arguments[1], arguments[2], arguments[3]});
        std::cout.exceptions(std::ios::badbit | std::ios::failbit);
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        require(at::globalContext().hasCUDA(), "El diagnóstico necesita cuda:0");
        at::globalContext().setAllowTF32CuBLAS(false);
        at::globalContext().setAllowTF32CuDNN(false);
        const auto cpu_rng = at::detail::getDefaultCPUGenerator().get_state().clone();
        const auto cuda_rng =
            at::globalContext().defaultGenerator(at::Device("cuda:0")).get_state().clone();
        std::cout << std::setprecision(17)
                  << "{\"record\":\"configuration\",\"schema_version\":1,\"workload\":\"" << options.workload
                  << "\",\"repetitions\":" << options.repetitions
                  << ",\"release_workspaces\":" << (options.release_workspaces ? "true" : "false")
                  << ",\"tf32\":false,\"optimizer_steps\":0}\n";
        snapshot("initial", 0);
        for (int iteration = 1; iteration <= options.repetitions; ++iteration) {
            workload(options.workload);
            snapshot("after_scope", iteration);
        }
        if (options.release_workspaces) {
            // Todas las operaciones han terminado. No quedan modelos, tensores ni trabajos en curso.
            at::cuda::clearCublasWorkspaces();
            snapshot("after_workspace_release", options.repetitions);
            workload(options.workload);
            snapshot("after_recreation", options.repetitions + 1);
            at::cuda::clearCublasWorkspaces();
            snapshot("final_workspace_release", options.repetitions + 1);
        }
        require(at::equal(cpu_rng, at::detail::getDefaultCPUGenerator().get_state()) &&
                    at::equal(cuda_rng, at::globalContext()
                                           .defaultGenerator(at::Device("cuda:0"))
                                           .get_state()),
                "El diagnóstico cambió los generadores globales");
        std::cout << "{\"record\":\"completed\",\"rng_unchanged\":true}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
