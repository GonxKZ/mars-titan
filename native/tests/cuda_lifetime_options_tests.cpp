#include "../benchmarks/cuda_lifetime_options.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
// Los límites literales son el contrato que se comprueba, independiente del del analizador.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
using mars_titan::diagnostics::lifetime_options;
using namespace std::string_view_literals;

void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("Las opciones del diagnóstico no cumplen su contrato");
    }
}

void rejected(const std::array<std::string_view, 3>& arguments) {
    bool failed = false;
    try {
        static_cast<void>(lifetime_options(arguments));
    } catch (const std::invalid_argument&) {
        failed = true;
    }
    require(failed);
}
} // namespace

int main() {
    try {
        for (const auto workload : {"empty"sv, "gemm"sv, "gru"sv, "candidate"sv}) {
            const auto first = lifetime_options(std::array{workload, "1"sv, "keep"sv});
            require(first.workload == workload && first.repetitions == 1 && !first.release_workspaces);
            const auto last = lifetime_options(std::array{workload, "300"sv, "release-workspaces"sv});
            require(last.repetitions == 300 && last.release_workspaces);
        }
        for (const auto value : {"0"sv, "301"sv, "-1"sv, "+1"sv, "1x"sv, " 1"sv,
                                 "99999999999999999999"sv, ""sv}) {
            rejected({"gemm", value, "keep"});
        }
        rejected({"unknown", "1", "keep"});
        rejected({"gemm", std::string_view{}, "keep"});
        rejected({"gemm", "1", "reset-device"});
        std::string caller_storage = "gemm";
        const auto independent = lifetime_options(
            std::array<std::string_view, 3>{caller_storage, "1", "keep"});
        caller_storage[0] = 'x';
        require(independent.workload == "gemm");
        std::cout << "Opciones y límites del diagnóstico CUDA comprobados\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
