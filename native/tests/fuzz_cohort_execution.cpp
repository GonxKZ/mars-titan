#include "mars_titan/cohort_execution.hpp"
#include "mars_titan/simulation_files.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>

namespace {
using namespace mars_titan::cohorts;
constexpr std::size_t maximum_input = 4096;
constexpr std::size_t digest_width = 64;
constexpr std::int64_t maximum_cutoff = 1000000;

class Directory {
  public:
    Directory() {
        auto pattern =
            (std::filesystem::temp_directory_path() / "mars-cohort-fuzz-XXXXXX").string();
        const auto* created = ::mkdtemp(pattern.data());
        if (created == nullptr) {
            throw std::runtime_error("No se puede crear el control de fuzzing");
        }
        path = created;
    }
    ~Directory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
    Directory(const Directory&) = delete;
    Directory& operator=(const Directory&) = delete;
    Directory(Directory&&) = delete;
    Directory& operator=(Directory&&) = delete;
    std::filesystem::path path;
};

void exercise(const Json& input) {
    const auto cutoff = mars_titan::simulation::read_json_int64(input.at("cutoff"));
    const auto available = mars_titan::simulation::read_json_int64(input.at("available_at"));
    const auto revision = mars_titan::simulation::read_json_int64(input.at("revision"));
    if (cutoff < 1 || cutoff > maximum_cutoff || revision < 0 || revision > 1) {
        return;
    }
    const auto value = input.at("value").get<double>();
    const auto target = input.at("target").get<double>();
    Definition definition{{std::string(digest_width, 'a'), std::string(digest_width, 'b'),
                           std::string(digest_width, 'c'), std::string(digest_width, 'd')},
                          {{"signal", 1}},
                          input.at("state"),
                          {}};
    definition.limits.max_assets = 1;
    definition.limits.max_pending = 2;
    definition.limits.max_state_bytes = maximum_input;
    Callbacks callbacks{
        [](std::span<const Observation> observations, const Task&, const Json& state) {
            return std::vector<double>{observations.front().features.front() +
                                       state.at("bias").get<double>()};
        },
        [](const Json& state, std::span<const ResolvedFeedback> feedback) {
            auto next = state;
            for (const auto& outcome : feedback) {
                next["bias"] = outcome.label.value - outcome.prediction.value;
            }
            return next;
        }};
    Directory directory;
    Json expected;
    {
        Executor executor(directory.path / "run", definition, callbacks);
        const auto first = executor.step(Cohort{0, cutoff, {{"A", available, {value}}}}, {}, 1);
        const Feedback label{first.predictions.front().id, static_cast<std::uint32_t>(revision),
                             cutoff + 1, target};
        executor.step(Cohort{1, cutoff + 1, {{"A", available, {value}}}}, std::span(&label, 1), 1);
        expected = executor.snapshot();
    }
    Executor restored(directory.path / "run", definition, callbacks, true);
    if (restored.snapshot() != expected) {
        std::abort();
    }
}
} // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size > maximum_input) {
        return 0;
    }
    std::string bytes;
    bytes.reserve(size);
    for (const auto value : std::span(data, size)) {
        bytes.push_back(static_cast<char>(value));
    }
    try {
        exercise(mars_titan::simulation::parse_bounded_json(bytes));
    } catch (const nlohmann::json::exception&) {
        return 0;
    } catch (const std::invalid_argument&) {
        return 0;
    }
    return 0;
}
