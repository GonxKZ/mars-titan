#include "mars_titan/adapter_control.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/core/grad_mode.h>
#include <torch/optim/sgd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace mars_titan::controls {
namespace {
constexpr std::size_t maximum_width = 2048;
constexpr std::size_t maximum_rows = 65536;
constexpr std::size_t maximum_steps = 1U << 20;
constexpr std::size_t maximum_bytes = 512U << 20;
constexpr std::size_t archive_limit = 128U << 20;
constexpr uint64_t archive_magic = 0x315250544144544dULL;
constexpr std::size_t archive_version = 1;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(message);
}
void check_versions(const SemanticVersions& versions) {
    require(versions.view > 0 && versions.representation > 0 && versions.keys > 0 &&
                versions.query > 0 && versions.output > 0,
            "Las versiones semánticas deben identificar todas las dependencias");
}
std::size_t parameter_count(const AdapterConfig& config) {
    return config.kind == AdapterKind::low_rank ? config.rank * (config.inputs + config.outputs)
                                                : config.inputs * config.outputs;
}
void check_config(const AdapterConfig& config) {
    static_cast<void>(adapter_kind_name(config.kind));
    check_versions(config.versions);
    require(!config.problem_id.empty() && config.problem_id.size() <= 128 &&
                !config.validation_id.empty() && config.validation_id.size() <= 128,
            "El control necesita identidades acotadas del problema y de la validación");
    require(config.inputs > 0 && config.inputs <= maximum_width && config.outputs > 0 &&
                config.outputs <= maximum_width && config.rank > 0 &&
                config.rank <= maximum_width &&
                (config.kind != AdapterKind::low_rank ||
                 config.rank <= std::min(config.inputs, config.outputs)),
            "Las dimensiones o el rango del adaptador están fuera del contrato");
    require(std::isfinite(config.learning_rate) && config.learning_rate > 0 &&
                config.learning_rate <= 1 && std::isfinite(config.momentum) &&
                config.momentum >= 0 && config.momentum < 1 &&
                std::isfinite(config.minimum_improvement) && config.minimum_improvement >= 0,
            "Las opciones de SGD o de selección no son finitas o están fuera de rango");
    require(config.max_steps > 0 && config.max_steps <= maximum_steps && config.max_rows > 0 &&
                config.max_rows <= maximum_rows && config.max_bytes > (64U << 10) &&
                config.max_bytes <= maximum_bytes &&
                config.versions.output <= std::numeric_limits<uint64_t>::max() - config.max_steps,
            "El control supera su presupuesto de pasos, filas, memoria o versiones");
}
void check_budget(const AdapterConfig& config, std::size_t validation_rows,
                  std::size_t batch_rows) {
    require(validation_rows > 0 && validation_rows <= config.max_rows &&
                batch_rows <= config.max_rows,
            "El lote o la validación exceden el número de filas admitido");
    const auto elements =
        uint64_t{6} * config.inputs * config.outputs + uint64_t{16} * parameter_count(config) +
        uint64_t{7} * validation_rows * (config.inputs + config.outputs) +
        uint64_t{3} * batch_rows * (config.inputs + 2 * config.outputs + 2 * config.rank);
    require(elements <= (config.max_bytes - (64U << 10)) / sizeof(double),
            "El presupuesto no cubre parámetros, momentum, selección y tensores temporales");
}
void check_tensor(const at::Tensor& tensor, at::IntArrayRef shape, const at::Device& device) {
    require(tensor.defined() && tensor.layout() == at::kStrided &&
                tensor.scalar_type() == at::kDouble && tensor.device() == device &&
                tensor.sizes() == shape && !tensor.requires_grad(),
            "El control necesita tensores FP64 sin gradiente con la forma y dispositivo acordados");
    require(at::isfinite(tensor).all().item<bool>(),
            "Un tensor del control contiene NaN o infinito");
}
std::size_t rows(const at::Tensor& tensor) {
    require(tensor.defined() && tensor.dim() == 2 && tensor.size(0) > 0,
            "El control necesita una matriz con al menos una fila");
    return static_cast<std::size_t>(tensor.size(0));
}
at::Device requested_device(std::string_view device) {
    require(device == "cpu" || device == "cuda:0", "El dispositivo debe ser cpu o cuda:0");
    if (device == "cuda:0")
        require(at::hasCUDA() && at::getNumGPUs() > 0,
                "Se solicitó cuda:0 y LibTorch no tiene una GPU CUDA disponible");
    return at::Device(std::string(device));
}
std::vector<std::array<int64_t, 2>> parameter_shapes(const AdapterConfig& config) {
    const auto input = static_cast<int64_t>(config.inputs);
    const auto output = static_cast<int64_t>(config.outputs);
    const auto rank = static_cast<int64_t>(config.rank);
    if (config.kind == AdapterKind::low_rank)
        return {{output, rank}, {rank, input}};
    return {{output, input}};
}
at::Tensor copy_to(const at::Tensor& tensor, const at::Device& device) {
    return tensor.detach().to(device).contiguous().clone();
}
std::vector<at::Tensor> copies(const std::vector<at::Tensor>& tensors, const at::Device& device) {
    std::vector<at::Tensor> result;
    result.reserve(tensors.size());
    for (const auto& tensor : tensors)
        result.push_back(copy_to(tensor, device));
    return result;
}
at::Tensor forward(const AdapterConfig& config, const at::Tensor& base,
                   const std::vector<at::Tensor>& parameters, const at::Tensor& inputs) {
    if (config.kind == AdapterKind::full)
        return at::mm(inputs, parameters[0].t());
    const auto parent = at::mm(inputs, base.t());
    if (config.kind == AdapterKind::residual)
        return parent + at::mm(inputs, parameters[0].t());
    return parent + at::mm(at::mm(inputs, parameters[1].t()), parameters[0].t());
}
double mse(const at::Tensor& prediction, const at::Tensor& target) {
    const auto value = (prediction - target).square().mean().item<double>();
    require(std::isfinite(value), "La pérdida de validación no es finita");
    return value;
}
void check_parameters(const std::vector<at::Tensor>& tensors, const AdapterConfig& config) {
    const auto shapes = parameter_shapes(config);
    require(tensors.size() == shapes.size(), "Falta un parámetro o un estado de momentum");
    for (std::size_t index = 0; index < shapes.size(); ++index)
        check_tensor(tensors[index], shapes[index], at::kCPU);
}
} // namespace

InvalidatedArtifacts invalidated(const SemanticVersions& before, const SemanticVersions& after) {
    check_versions(before);
    check_versions(after);
    const auto representations =
        before.view != after.view || before.representation != after.representation;
    const auto memory = representations || before.keys != after.keys;
    const auto reads = memory || before.query != after.query;
    return {representations, memory, reads, reads || before.output != after.output};
}
void require_compatible(ArtifactKind artifact, const SemanticVersions& stored,
                        const SemanticVersions& current) {
    const auto changes = invalidated(stored, current);
    bool incompatible = false;
    switch (artifact) {
    case ArtifactKind::representation:
        incompatible = changes.representations;
        break;
    case ArtifactKind::memory:
        incompatible = changes.memory;
        break;
    case ArtifactKind::read:
        incompatible = changes.reads;
        break;
    case ArtifactKind::prediction:
        incompatible = changes.predictions;
        break;
    default:
        throw std::invalid_argument("El tipo de artefacto no está definido");
    }
    require(!incompatible, "El artefacto pertenece a versiones semánticas incompatibles");
}
std::string_view adapter_kind_name(AdapterKind kind) {
    switch (kind) {
    case AdapterKind::full:
        return "full";
    case AdapterKind::residual:
        return "residual";
    case AdapterKind::low_rank:
        return "low_rank";
    }
    throw std::invalid_argument("El tipo de ajuste no está definido");
}

struct AdapterControl::Impl {
    AdapterConfig config;
    at::Device device;
    at::Tensor base;
    at::Tensor validation_inputs;
    at::Tensor validation_targets;
    std::vector<at::Tensor> parameters;
    std::vector<at::Tensor> best_parameters;
    std::unique_ptr<torch::optim::SGD> optimizer;
    std::size_t steps = 0;
    std::size_t best_step = 0;
    double best_mse = 0;

    Impl(AdapterConfig settings, const at::Tensor& parent, const at::Tensor& input,
         const at::Tensor& target, const at::Device& selected_device)
        : config(std::move(settings)), device(selected_device) {
        check_config(config);
        const auto validation_rows = rows(input);
        check_budget(config, validation_rows, 0);
        const auto in = static_cast<int64_t>(config.inputs);
        const auto out = static_cast<int64_t>(config.outputs);
        check_tensor(parent, {out, in}, at::kCPU);
        check_tensor(input, {static_cast<int64_t>(validation_rows), in}, at::kCPU);
        check_tensor(target, {static_cast<int64_t>(validation_rows), out}, at::kCPU);
        const at::NoGradGuard no_grad;
        base = copy_to(parent, device);
        validation_inputs = copy_to(input, device);
        validation_targets = copy_to(target, device);
        if (config.kind == AdapterKind::full)
            parameters.push_back(base.clone());
        else if (config.kind == AdapterKind::residual)
            parameters.push_back(at::zeros_like(base));
        else {
            auto rng = at::detail::createCPUGenerator(config.seed);
            parameters.push_back(
                at::zeros({out, static_cast<int64_t>(config.rank)}, base.options()));
            parameters.push_back(
                (at::randn({static_cast<int64_t>(config.rank), in}, rng, at::kDouble) /
                 std::sqrt(static_cast<double>(config.inputs)))
                    .to(device));
        }
        best_parameters = copies(parameters, device);
        for (auto& parameter : parameters)
            parameter.set_requires_grad(true);
        optimizer = std::make_unique<torch::optim::SGD>(
            parameters, torch::optim::SGDOptions(config.learning_rate).momentum(config.momentum));
        for (const auto& parameter : parameters) {
            auto state = std::make_unique<torch::optim::SGDParamState>();
            state->momentum_buffer(at::zeros_like(parameter));
            optimizer->state()[parameter.unsafeGetTensorImpl()] = std::move(state);
        }
        best_mse = mse(at::mm(validation_inputs, base.t()), validation_targets);
    }
    std::vector<at::Tensor> momentum() const {
        std::vector<at::Tensor> result;
        for (const auto& parameter : parameters) {
            const auto* state = dynamic_cast<const torch::optim::SGDParamState*>(
                optimizer->state().at(parameter.unsafeGetTensorImpl()).get());
            require(state != nullptr, "El optimizador no conserva un estado SGD compatible");
            result.push_back(state->momentum_buffer());
        }
        return result;
    }
};
AdapterControl::AdapterControl(AdapterConfig config, const at::Tensor& base,
                               const at::Tensor& validation_inputs,
                               const at::Tensor& validation_targets, std::string_view device)
    : impl_(std::make_unique<Impl>(std::move(config), base, validation_inputs, validation_targets,
                                   requested_device(device))) {}
AdapterControl::AdapterControl(AdapterControl&&) noexcept = default;
AdapterControl& AdapterControl::operator=(AdapterControl&&) noexcept = default;
AdapterControl::~AdapterControl() = default;

at::Tensor AdapterControl::predict(const at::Tensor& inputs, bool selected) const {
    const auto count = rows(inputs);
    check_budget(impl_->config, rows(impl_->validation_inputs), count);
    check_tensor(inputs, {static_cast<int64_t>(count), static_cast<int64_t>(impl_->config.inputs)},
                 impl_->device);
    const at::NoGradGuard no_grad;
    const auto result =
        selected && impl_->best_step == 0
            ? at::mm(inputs, impl_->base.t())
            : forward(impl_->config, impl_->base,
                      selected ? impl_->best_parameters : impl_->parameters, inputs);
    require(at::isfinite(result).all().item<bool>(), "La predicción del control no es finita");
    return result;
}
double AdapterControl::train(const at::Tensor& inputs, const at::Tensor& targets) {
    require(!impl_->base.requires_grad() && !impl_->base.grad().defined(),
            "La base del control debe permanecer congelada y sin gradientes");
    const auto count = rows(inputs);
    check_budget(impl_->config, rows(impl_->validation_inputs), count);
    check_tensor(inputs, {static_cast<int64_t>(count), static_cast<int64_t>(impl_->config.inputs)},
                 impl_->device);
    check_tensor(targets,
                 {static_cast<int64_t>(count), static_cast<int64_t>(impl_->config.outputs)},
                 impl_->device);
    require(impl_->steps < impl_->config.max_steps,
            "Se ha agotado el presupuesto de actualizaciones");
    impl_->optimizer->zero_grad();
    const auto loss =
        (forward(impl_->config, impl_->base, impl_->parameters, inputs) - targets).square().mean();
    const auto value = loss.item<double>();
    require(std::isfinite(value), "La pérdida de entrenamiento no es finita");
    loss.backward();
    for (const auto& parameter : impl_->parameters)
        require(parameter.grad().defined() && at::isfinite(parameter.grad()).all().item<bool>(),
                "El gradiente del control no es finito");
    const at::NoGradGuard no_grad;
    const auto previous = copies(impl_->parameters, impl_->device);
    const auto previous_momentum = copies(impl_->momentum(), impl_->device);
    try {
        impl_->optimizer->step();
        for (const auto& tensor : impl_->parameters)
            require(at::isfinite(tensor).all().item<bool>(),
                    "SGD ha producido un parámetro no finito");
        for (const auto& tensor : impl_->momentum())
            require(at::isfinite(tensor).all().item<bool>(),
                    "SGD ha producido un momentum no finito");
        const auto score =
            mse(forward(impl_->config, impl_->base, impl_->parameters, impl_->validation_inputs),
                impl_->validation_targets);
        if (impl_->best_mse - score > impl_->config.minimum_improvement) {
            auto best = copies(impl_->parameters, impl_->device);
            impl_->best_parameters = std::move(best);
            impl_->best_mse = score;
            impl_->best_step = impl_->steps + 1;
        }
        ++impl_->steps;
    } catch (...) {
        const auto momentum = impl_->momentum();
        for (std::size_t index = 0; index < previous.size(); ++index) {
            impl_->parameters[index].copy_(previous[index]);
            momentum[index].copy_(previous_momentum[index]);
        }
        throw;
    }
    return value;
}
AdapterSnapshot AdapterControl::snapshot() const {
    return {impl_->config,
            copy_to(impl_->base, at::kCPU),
            copy_to(impl_->validation_inputs, at::kCPU),
            copy_to(impl_->validation_targets, at::kCPU),
            copies(impl_->parameters, at::kCPU),
            copies(impl_->momentum(), at::kCPU),
            copies(impl_->best_parameters, at::kCPU),
            impl_->steps,
            impl_->best_step,
            impl_->best_mse};
}
void AdapterControl::restore(const AdapterSnapshot& state) {
    require(state.config == impl_->config,
            "El checkpoint cambia el problema, la validación o sus versiones");
    const auto current = snapshot();
    check_tensor(state.base, current.base.sizes(), at::kCPU);
    check_tensor(state.validation_inputs, current.validation_inputs.sizes(), at::kCPU);
    check_tensor(state.validation_targets, current.validation_targets.sizes(), at::kCPU);
    require(at::equal(state.base, current.base) &&
                at::equal(state.validation_inputs, current.validation_inputs) &&
                at::equal(state.validation_targets, current.validation_targets),
            "El checkpoint cambia los valores del padre o de la validación");
    require(state.steps <= impl_->config.max_steps && state.best_step <= state.steps &&
                std::isfinite(state.best_mse) && state.best_mse >= 0,
            "El checkpoint contiene cursores o selección inválidos");
    check_parameters(state.parameters, state.config);
    check_parameters(state.momentum, state.config);
    check_parameters(state.best_parameters, state.config);
    auto candidate = std::make_unique<Impl>(state.config, state.base, state.validation_inputs,
                                            state.validation_targets, impl_->device);
    const at::NoGradGuard no_grad;
    if (state.best_step == 0)
        for (std::size_t index = 0; index < state.best_parameters.size(); ++index)
            require(
                at::equal(state.best_parameters[index], candidate->best_parameters[index].cpu()),
                "La selección inicial no conserva los parámetros del padre");
    else
        require(candidate->best_mse - state.best_mse > state.config.minimum_improvement,
                "La selección no mejora el criterio del padre inicial");
    const auto momentum = candidate->momentum();
    for (std::size_t index = 0; index < state.parameters.size(); ++index) {
        if (state.steps == 0)
            require(
                at::equal(state.parameters[index], candidate->parameters[index].detach().cpu()) &&
                    at::count_nonzero(state.momentum[index]).item<int64_t>() == 0,
                "El checkpoint inicial contiene una actualización no confirmada");
        candidate->parameters[index].copy_(state.parameters[index].to(impl_->device));
        momentum[index].copy_(state.momentum[index].to(impl_->device));
    }
    candidate->best_parameters = copies(state.best_parameters, impl_->device);
    const auto score = state.best_step == 0
                           ? candidate->best_mse
                           : mse(forward(state.config, candidate->base, candidate->best_parameters,
                                         candidate->validation_inputs),
                                 candidate->validation_targets);
    require(std::abs(score - state.best_mse) <= 1e-12 * std::max(1.0, std::abs(score)),
            "La métrica guardada no corresponde al estado seleccionado");
    candidate->steps = state.steps;
    candidate->best_step = state.best_step;
    candidate->best_mse = state.best_mse;
    impl_ = std::move(candidate);
}
SemanticVersions AdapterControl::versions(bool selected) const {
    auto result = impl_->config.versions;
    result.output += selected ? impl_->best_step : impl_->steps;
    return result;
}
std::size_t AdapterControl::trainable_parameters() const noexcept {
    std::size_t count = 0;
    for (const auto& parameter : impl_->parameters)
        if (parameter.requires_grad())
            count += static_cast<std::size_t>(parameter.numel());
    return count;
}
std::size_t AdapterControl::state_tensor_bytes() const noexcept {
    const auto base = impl_->config.inputs * impl_->config.outputs;
    const auto validation = static_cast<std::size_t>(impl_->validation_inputs.numel() +
                                                     impl_->validation_targets.numel());
    return sizeof(double) * (base + 3 * trainable_parameters() + validation);
}

namespace {
void write_integer(std::string& output, uint64_t value) {
    for (unsigned int byte = 0; byte < 8; ++byte)
        output.push_back(static_cast<char>((value >> (8 * byte)) & 0xffU));
}
void write_string(std::string& output, const std::string& value) {
    write_integer(output, value.size());
    output += value;
}
void write_tensor(std::string& output, const at::Tensor& tensor) {
    const auto contiguous = tensor.contiguous();
    for (const auto value : std::span(contiguous.const_data_ptr<double>(),
                                      static_cast<std::size_t>(contiguous.numel())))
        write_integer(output, std::bit_cast<uint64_t>(value));
}
class Reader {
  public:
    explicit Reader(std::string_view data) : data_(data) {}
    uint64_t integer() {
        const auto bytes = take(8);
        uint64_t result = 0;
        for (unsigned int byte = 0; byte < 8; ++byte)
            result |= static_cast<uint64_t>(static_cast<unsigned char>(bytes[byte])) << (8 * byte);
        return result;
    }
    std::size_t size() {
        const auto value = integer();
        require(value <= std::numeric_limits<std::size_t>::max(),
                "Un tamaño del archivo no cabe en esta plataforma");
        return static_cast<std::size_t>(value);
    }
    std::string string() {
        const auto count = size();
        require(count > 0 && count <= 128, "La identidad del checkpoint no está acotada");
        return std::string(take(count));
    }
    double number() { return std::bit_cast<double>(integer()); }
    at::Tensor tensor(std::size_t row_count, std::size_t columns) {
        require(row_count * columns <= (data_.size() - cursor_) / sizeof(double),
                "Faltan valores de un tensor del checkpoint");
        auto result = at::empty({static_cast<int64_t>(row_count), static_cast<int64_t>(columns)},
                                at::kDouble);
        for (auto& value : std::span(result.data_ptr<double>(), row_count * columns))
            value = number();
        return result;
    }
    bool exhausted() const noexcept { return cursor_ == data_.size(); }

  private:
    std::string_view take(std::size_t count) {
        require(count <= data_.size() - cursor_, "El checkpoint de adaptadores está truncado");
        const auto result = data_.substr(cursor_, count);
        cursor_ += count;
        return result;
    }
    std::string_view data_;
    std::size_t cursor_ = 0;
};
} // namespace
std::string serialize_adapter(const AdapterSnapshot& state) {
    AdapterControl checked(state.config, state.base, state.validation_inputs,
                           state.validation_targets);
    checked.restore(state);
    require(checked.state_tensor_bytes() <= archive_limit - 512,
            "El estado no cabe en el presupuesto del archivo de recuperación");
    std::string output;
    const auto& config = state.config;
    write_integer(output, archive_magic);
    write_integer(output, archive_version);
    write_string(output, config.problem_id);
    write_string(output, config.validation_id);
    for (const auto value :
         {static_cast<uint64_t>(config.kind), static_cast<uint64_t>(config.inputs),
          static_cast<uint64_t>(config.outputs), static_cast<uint64_t>(config.rank), config.seed})
        write_integer(output, value);
    for (const auto value : {config.learning_rate, config.momentum, config.minimum_improvement})
        write_integer(output, std::bit_cast<uint64_t>(value));
    for (const auto value : {config.max_steps, config.max_rows, config.max_bytes})
        write_integer(output, value);
    const auto& versions = config.versions;
    for (const auto value :
         {versions.view, versions.representation, versions.keys, versions.query, versions.output})
        write_integer(output, value);
    write_integer(output, rows(state.validation_inputs));
    write_integer(output, state.steps);
    write_integer(output, state.best_step);
    write_integer(output, std::bit_cast<uint64_t>(state.best_mse));
    write_tensor(output, state.base);
    write_tensor(output, state.validation_inputs);
    write_tensor(output, state.validation_targets);
    for (const auto* list : {&state.parameters, &state.momentum, &state.best_parameters})
        for (const auto& tensor : *list)
            write_tensor(output, tensor);
    require(output.size() <= archive_limit, "El checkpoint supera el límite de archivo admitido");
    return output;
}
AdapterSnapshot deserialize_adapter(std::string_view archive) {
    require(!archive.empty() && archive.size() <= archive_limit,
            "El checkpoint está vacío o excede el límite de lectura");
    Reader input(archive);
    require(input.integer() == archive_magic && input.integer() == archive_version,
            "El formato del checkpoint no es compatible");
    AdapterSnapshot result;
    auto& config = result.config;
    config.problem_id = input.string();
    config.validation_id = input.string();
    const auto kind = input.integer();
    require(kind <= 2, "El archivo contiene un tipo de adaptador desconocido");
    config.kind = static_cast<AdapterKind>(kind);
    config.inputs = input.size();
    config.outputs = input.size();
    config.rank = input.size();
    config.seed = input.integer();
    config.learning_rate = input.number();
    config.momentum = input.number();
    config.minimum_improvement = input.number();
    config.max_steps = input.size();
    config.max_rows = input.size();
    config.max_bytes = input.size();
    config.versions = {input.integer(), input.integer(), input.integer(), input.integer(),
                       input.integer()};
    check_config(config);
    const auto validation_rows = input.size();
    check_budget(config, validation_rows, 0);
    result.steps = input.size();
    result.best_step = input.size();
    result.best_mse = input.number();
    result.base = input.tensor(config.outputs, config.inputs);
    result.validation_inputs = input.tensor(validation_rows, config.inputs);
    result.validation_targets = input.tensor(validation_rows, config.outputs);
    const auto shapes = parameter_shapes(config);
    for (auto* list : {&result.parameters, &result.momentum, &result.best_parameters})
        for (const auto& shape : shapes)
            list->push_back(input.tensor(static_cast<std::size_t>(shape[0]),
                                         static_cast<std::size_t>(shape[1])));
    require(input.exhausted(), "El checkpoint contiene datos sobrantes");
    AdapterControl checked(config, result.base, result.validation_inputs,
                           result.validation_targets);
    checked.restore(result);
    return result;
}
} // namespace mars_titan::controls
