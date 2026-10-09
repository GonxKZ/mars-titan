#include "mars_titan/candidate.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace py = pybind11;
namespace mars_titan::candidate {
namespace {
constexpr std::size_t maximum_archive_bytes = 128U << 20;
at::ScalarType scalar_type(const std::string& dtype) {
    if (dtype == "float32") {
        return at::kFloat;
    }
    if (dtype == "float64") {
        return at::kDouble;
    }
    throw std::invalid_argument("El enlace candidato admite float32 o float64");
}
py::bytes save(const Candidate& model) {
    std::ostringstream output;
    model.save_state(output);
    return py::bytes(std::move(output).str());
}
// La frontera pybind exige bytes con noconvert. Intercambiarlos con texto falla antes de entrar.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::shared_ptr<Candidate> load(const py::bytes& payload, const std::string& device,
                                const std::string& policy) {
    if (py::len(payload) == 0 || py::len(payload) > maximum_archive_bytes) {
        throw std::invalid_argument("El archivo candidato debe ocupar entre uno y 128 MiB");
    }
    std::istringstream input(payload.cast<std::string>());
    return Candidate::load_state(input, at::Device(device), policy);
}
} // namespace

void bind_candidate(py::module_& module) {
    module.attr("candidate_abi_version") = 1;
    py::class_<Config>(module, "CandidateConfig")
        .def(py::init<>())
        .def_readwrite("dimensions", &Config::dimensions)
        .def_readwrite("max_batch", &Config::max_batch)
        .def_readwrite("max_episodes", &Config::max_episodes)
        .def_readwrite("neighbors", &Config::neighbors)
        .def_readwrite("parameter_seed", &Config::parameter_seed)
        .def_readwrite("feature_seed", &Config::feature_seed)
        .def_readwrite("key_seed", &Config::key_seed)
        .def_readwrite("temperature", &Config::temperature)
        .def_readwrite("normalization_id", &Config::normalization_id)
        .def_readwrite("input_policy", &Config::input_policy);
    py::class_<Inputs>(module, "CandidateInputs")
        .def(py::init<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(),
             py::arg("prices"), py::arg("news"), py::arg("charts"), py::arg("fundamentals"),
             py::arg("macro"), py::arg("presence"));
    py::class_<Encoded>(module, "CandidateEncoded")
        .def_readonly("fused", &Encoded::fused)
        .def_readonly("episode_features", &Encoded::episode_features)
        .def_readonly("episode_keys", &Encoded::episode_keys);
    py::class_<EpisodeEncoding>(module, "CandidateEpisodeEncoding")
        .def_readonly("keys", &EpisodeEncoding::keys)
        .def_readonly("values", &EpisodeEncoding::values);
    py::class_<CpuEpisodeCodec, std::shared_ptr<CpuEpisodeCodec>>(module, "CandidateCPUCodec")
        .def("encode", &CpuEpisodeCodec::encode)
        .def("estimated_bytes", &CpuEpisodeCodec::estimated_bytes)
        .def_property_readonly("projection_id", &CpuEpisodeCodec::projection_id)
        .def_property_readonly("dtype", [](const CpuEpisodeCodec& codec) {
            return codec.dtype() == at::kDouble ? "float64" : "float32";
        });
    py::class_<Read>(module, "CandidateRead")
        .def_readonly("values", &Read::values)
        .def_readonly("weights", &Read::weights)
        .def_readonly("ids", &Read::ids)
        .def_readonly("presence", &Read::presence);
    py::class_<Prediction>(module, "CandidatePrediction")
        .def_readonly("quantiles", &Prediction::quantiles)
        .def_readonly("state", &Prediction::state)
        .def_readonly("encoded", &Prediction::encoded)
        .def_readonly("read", &Prediction::read);
    py::class_<MemorySnapshot>(module, "CandidateMemory")
        .def_property_readonly("size", &MemorySnapshot::size);
    py::class_<Candidate, std::shared_ptr<Candidate>>(module, "Candidate")
        .def(
            py::init([](const Config& config, const std::string& dtype, const std::string& device) {
                return std::make_shared<Candidate>(config, scalar_type(dtype), at::Device(device));
            }),
            py::arg("config"), py::arg("dtype") = "float32", py::arg("device") = "cpu")
        .def_property_readonly("config", [](const Candidate& model) { return model.config(); })
        .def_property_readonly("training", &Candidate::is_training)
        .def("train", &Candidate::train, py::arg("mode") = true)
        .def("eval", &Candidate::eval)
        .def("encode", &Candidate::encode)
        .def("encode_context", &Candidate::encode_context)
        .def("cpu_episode_codec", &Candidate::cpu_episode_codec,
             py::arg("max_working_bytes") = maximum_codec_bytes)
        .def("empty_memory", &Candidate::empty_memory)
        .def("snapshot", &Candidate::snapshot)
        .def("forward", &Candidate::forward, py::arg("inputs"), py::arg("memory"),
             py::arg("refinements") = 1)
        .def("initial_state", &Candidate::initial_state)
        .def("read", &Candidate::read)
        .def("refine", &Candidate::refine)
        .def("quantiles", &Candidate::quantiles)
        .def("representation_id", &Candidate::representation_id)
        .def("parameter_fingerprint", &Candidate::parameter_fingerprint)
        .def("transfer_strict_parameters", &Candidate::transfer_strict_parameters)
        .def("named_parameters",
             [](const Candidate& model) {
                 py::dict parameters;
                 for (const auto& item : model.named_parameters()) {
                     parameters[py::str(item.key())] = py::cast(item.value());
                 }
                 return parameters;
             })
        .def("zero_grad", &Candidate::zero_grad, py::arg("set_to_none") = true)
        .def("save_state", &save)
        .def_static("load_state", &load, py::arg("payload").noconvert(), py::arg("device") = "cpu",
                    py::arg("expected_policy") = "strict_inputs_v1");
}
} // namespace mars_titan::candidate
