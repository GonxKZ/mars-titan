#include "candidate_cuda_fixture.hpp"

#include <limits>

int main() {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
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
