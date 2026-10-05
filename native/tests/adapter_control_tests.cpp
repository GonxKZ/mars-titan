#include "mars_titan/adapter_control.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {
using namespace mars_titan::controls;
// Los literales son entradas, límites adversariales y resultados analíticos de las pruebas.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class Function> void rejected(Function&& action) {
    try {
        std::forward<Function>(action)();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Se aceptó un estado de adaptación inválido");
}
void close(const at::Tensor& left, const at::Tensor& right) {
    require(at::allclose(left.cpu(), right.cpu(), 1e-11, 1e-12),
            "El control difiere de la referencia FP64");
}
at::Tensor base() { return at::tensor({1.0, 2.0, -1.0, 0.5}, at::kDouble).reshape({2, 2}); }
at::Tensor inputs() { return at::eye(2, at::kDouble); }
AdapterConfig config(AdapterKind kind) {
    return {.kind = kind,
            .inputs = 2,
            .outputs = 2,
            .rank = 1,
            .learning_rate = 0.1,
            .momentum = 0.5,
            .max_steps = 8};
}
void dependencies_follow_the_changed_stage() {
    AdapterControl control(config(AdapterKind::full), base(), inputs(), base().t());
    const auto original = control.versions();
    auto changed = original;
    changed.output++;
    auto result = invalidated(original, changed);
    require(!result.representations && !result.memory && !result.reads && result.predictions,
            "Cambiar la salida invalida artefactos ajenos o conserva predicciones antiguas");
    require_compatible(ArtifactKind::memory, original, changed);
    require_compatible(ArtifactKind::read, original, changed);
    rejected([&] { require_compatible(ArtifactKind::prediction, original, changed); });
    changed = original;
    changed.query++;
    result = invalidated(original, changed);
    require(!result.representations && !result.memory && result.reads && result.predictions,
            "Cambiar la consulta no invalida todas las lecturas dependientes");
    changed = original;
    changed.keys++;
    result = invalidated(original, changed);
    require(!result.representations && result.memory && result.reads && result.predictions,
            "Cambiar las claves no invalida la memoria dependiente");
    for (const bool change_view : {true, false}) {
        changed = original;
        if (change_view)
            changed.view++;
        else
            changed.representation++;
        result = invalidated(original, changed);
        require(result.representations && result.memory && result.reads && result.predictions,
                "Cambiar representación o vista conserva un artefacto incompatible");
        rejected([&] { require_compatible(ArtifactKind::memory, original, changed); });
    }
    require_compatible(ArtifactKind::prediction, original, original);
    const SemanticVersions unbound;
    rejected([&] { require_compatible(ArtifactKind::prediction, unbound, unbound); });
    changed.view = 0;
    rejected([&] { static_cast<void>(invalidated(original, changed)); });
    // El enum tiene base uint8_t. Se comprueba el rechazo de un byte sin enumerador asociado.
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    rejected([&] { require_compatible(static_cast<ArtifactKind>(255), original, original); });
}
void divergent_restores_reject_stale_predictions(std::string_view device) {
    const auto x_cpu = at::ones({1, 1}, at::kDouble);
    const auto zero = at::zeros_like(x_cpu);
    const auto x = x_cpu.to(at::Device(std::string(device)));
    for (const auto kind : {AdapterKind::full, AdapterKind::residual, AdapterKind::low_rank}) {
        auto settings = config(kind);
        settings.inputs = settings.outputs = settings.rank = 1;
        AdapterControl control(settings, zero, x_cpu, zero, device);
        const auto initial = control.snapshot();
        const auto parent_identity = control.versions(true);
        static_cast<void>(control.train(x, x));
        const auto first_prediction = control.predict(x).item<double>();
        const auto first_identity = control.versions();
        require_compatible(ArtifactKind::prediction, parent_identity, control.versions(true));
        control.restore(initial);
        static_cast<void>(control.train(x, -x));
        const auto second_identity = control.versions();
        require(first_prediction != control.predict(x).item<double>() &&
                    first_identity.output == second_identity.output,
                "La regresión no reproduce trayectorias distintas con los mismos pasos");
        rejected(
            [&] { require_compatible(ArtifactKind::prediction, first_identity, second_identity); });
        require_compatible(ArtifactKind::memory, first_identity, second_identity);
        require_compatible(ArtifactKind::read, first_identity, second_identity);
        require_compatible(ArtifactKind::prediction, parent_identity, control.versions(true));
        const auto state = deserialize_adapter(serialize_adapter(control.snapshot()));
        AdapterControl resumed(settings, zero, x_cpu, zero, device);
        resumed.restore(state);
        require(resumed.versions() == second_identity && resumed.versions(true) == parent_identity,
                "La recuperación no reproduce las identidades del estado actual y seleccionado");
        static_cast<void>(resumed.predict(x));
        static_cast<void>(resumed.predict(x * 2));
        require(resumed.versions() == second_identity,
                "Una lectura cambia la identidad de los parámetros sin una actualización");
    }
}
void different_modes_and_states_have_distinct_identities(std::string_view device) {
    const auto x_cpu = at::ones({1, 1}, at::kDouble);
    const auto zero = at::zeros_like(x_cpu);
    const auto x = x_cpu.to(at::Device(std::string(device)));
    auto settings = config(AdapterKind::full);
    settings.inputs = settings.outputs = settings.rank = 1;
    AdapterControl full(settings, zero, x_cpu, zero, device);
    settings.kind = AdapterKind::low_rank;
    AdapterControl low_rank(settings, zero, x_cpu, zero, device);
    static_cast<void>(full.train(x, x));
    static_cast<void>(low_rank.train(x, x));
    require(full.predict(x).item<double>() != low_rank.predict(x).item<double>() &&
                full.versions().output == low_rank.versions().output,
            "La regresión no reproduce dos clases con salidas distintas y el mismo contador");
    rejected([&] {
        require_compatible(ArtifactKind::prediction, full.versions(), low_rank.versions());
    });
    AdapterControl same(settings, zero, x_cpu, zero, device);
    static_cast<void>(same.train(x, x));
    require(same.versions() == low_rank.versions(),
            "El mismo contenido produce identidades distintas en dos controles");
    const auto prior = low_rank.versions();
    auto changed = low_rank.snapshot();
    changed.parameters[0] += 0.25;
    low_rank.restore(changed);
    rejected([&] { require_compatible(ArtifactKind::prediction, prior, low_rank.versions()); });
}
void different_selected_states_at_the_same_step_are_incompatible(std::string_view device) {
    const auto x_cpu = at::ones({1, 1}, at::kDouble);
    const auto zero = at::zeros_like(x_cpu);
    const auto x = x_cpu.to(at::Device(std::string(device)));
    for (const auto kind : {AdapterKind::full, AdapterKind::residual, AdapterKind::low_rank}) {
        auto settings = config(kind);
        settings.inputs = settings.outputs = settings.rank = 1;
        settings.learning_rate = 0.01;
        AdapterControl control(settings, zero, x_cpu, x_cpu, device);
        const auto initial = control.snapshot();
        const auto initial_identity = control.versions(true);
        static_cast<void>(control.train(x, x));
        const auto first = control.versions(true);
        require(first.output_state != initial_identity.output_state,
                "La selección conserva una huella cacheada del padre después de mejorar");
        const auto first_value = control.predict(x, true).item<double>();
        require(control.snapshot().best_step == 1, "La primera actualización no mejora al padre");
        control.restore(initial);
        static_cast<void>(control.train(x, x * 2));
        require(control.snapshot().best_step == 1 &&
                    control.versions(true).output == first.output &&
                    first_value != control.predict(x, true).item<double>(),
                "La regresión no produce dos selecciones distintas en el mismo paso");
        rejected(
            [&] { require_compatible(ArtifactKind::prediction, first, control.versions(true)); });
        const auto selected = control.versions(true);
        static_cast<void>(control.train(x, x * -10));
        require(control.snapshot().best_step == 1 && control.versions(true) == selected &&
                    control.versions().output_state != selected.output_state,
                "Una actualización peor mezcla la identidad actual con la seleccionada");
        control.restore(deserialize_adapter(serialize_adapter(control.snapshot())));
        require(control.versions(true) == selected,
                "El checkpoint no conserva la identidad del mejor estado distinto del padre");
    }
}
void serialized_dependency_identities_are_preserved() {
    auto settings = config(AdapterKind::full);
    AdapterControl origin(settings, base(), inputs(), base().t());
    settings.versions = origin.versions();
    AdapterControl bound(settings, base(), inputs(), base().t());
    const auto state = bound.snapshot();
    const auto archive = serialize_adapter(state);
    const auto restored = deserialize_adapter(archive);
    require(restored.config.versions == settings.versions,
            "El archivo pierde una identidad de salida declarada como dependencia");
    const auto identity = bound.versions();
    bound.restore(restored);
    require(bound.versions() == identity, "La identidad cambia al recuperar las dependencias");
    auto previous_format = archive;
    previous_format[8] = 1;
    rejected([&] { static_cast<void>(deserialize_adapter(previous_format)); });
}
void full_and_residual_match_two_analytical_updates(std::string_view device) {
    for (const auto kind : {AdapterKind::full, AdapterKind::residual}) {
        const auto x = inputs().to(at::Device(std::string(device)));
        const auto y = at::zeros_like(x);
        AdapterControl control(config(kind), base(), inputs(), at::zeros_like(inputs()), device);
        const auto initial_identity = control.versions();
        close(control.predict(x), base().t());
        require(control.trainable_parameters() == 4,
                "El recuento de parámetros completos es incorrecto");
        const auto first_loss = control.train(x, y);
        require(std::abs(first_loss - 1.5625) < 1e-12, "La pérdida no usa el denominador global");
        close(control.predict(x), base().t() * 0.95);
        const auto first_identity = control.versions();
        require(first_identity.output_state != initial_identity.output_state,
                "La primera actualización conserva una huella cacheada de otros pesos");
        static_cast<void>(control.train(x, y));
        close(control.predict(x), base().t() * 0.8775);
        require(control.versions().output_state != first_identity.output_state,
                "La siguiente actualización no invalida la huella anterior");
        close(control.snapshot().base, base());
        require(control.versions().output == 3 && control.versions().representation == 1,
                "La actualización cambia versiones de etapas congeladas");
    }
}
void low_rank_matches_an_independent_scalar_gradient(std::string_view device) {
    const auto x = inputs().to(at::Device(std::string(device)));
    const auto y = at::zeros_like(x);
    AdapterControl control(config(AdapterKind::low_rank), base(), inputs(),
                           at::zeros_like(inputs()), device);
    const auto initial = control.snapshot();
    require(at::equal(control.predict(x).cpu(), base().t()),
            "El adaptador inicial no reproduce al padre");
    require(control.trainable_parameters() == 4, "El recuento de bajo rango omite un factor");
    auto u = initial.parameters[0].clone();
    auto v = initial.parameters[1].clone();
    auto mu = at::zeros_like(u);
    auto mv = at::zeros_like(v);
    const auto parent = base();
    for (int step = 0; step < 4; ++step) {
        auto du = at::zeros_like(u);
        auto dv = at::zeros_like(v);
        for (int64_t row = 0; row < 2; ++row)
            for (int64_t output = 0; output < 2; ++output) {
                const double error = parent[output][row].item<double>() +
                                     u[output][0].item<double>() * v[0][row].item<double>();
                du[output][0] += 0.5 * error * v[0][row].item<double>();
                dv[0][row] += 0.5 * error * u[output][0].item<double>();
            }
        mu = 0.5 * mu + du;
        mv = 0.5 * mv + dv;
        u -= 0.1 * mu;
        v -= 0.1 * mv;
        static_cast<void>(control.train(x, y));
        const auto actual = control.snapshot();
        close(actual.parameters[0], u);
        close(actual.parameters[1], v);
        close(actual.momentum[0], mu);
        close(actual.momentum[1], mv);
        require(at::equal(actual.base, parent),
                "El ajuste de bajo rango modifica la base congelada");
    }
}
void parent_selection_and_recovery(std::string_view device) {
    for (const auto kind : {AdapterKind::full, AdapterKind::residual, AdapterKind::low_rank}) {
        const auto x = inputs().to(at::Device(std::string(device)));
        const auto y = at::zeros_like(x);
        AdapterControl control(config(kind), base(), inputs(), base().t(), device);
        for (int step = 0; step < 3; ++step)
            static_cast<void>(control.train(x, y));
        const auto state = deserialize_adapter(serialize_adapter(control.snapshot()));
        require(state.best_step == 0 && state.best_mse == 0,
                "La selección pierde el padre inicial");
        require(at::equal(control.predict(x, true).cpu(), base().t()),
                "La selección no devuelve el padre exacto");
        require(!at::equal(control.predict(x).cpu(), base().t()),
                "La actualización no modifica la predicción");
        AdapterControl resumed(config(kind), base(), inputs(), base().t(), device);
        resumed.restore(state);
        for (int step = 3; step < 8; ++step) {
            static_cast<void>(control.train(x, y));
            static_cast<void>(resumed.train(x, y));
        }
        close(control.predict(x), resumed.predict(x));
        const auto complete = resumed.snapshot();
        require(complete.steps == 8 && complete.best_step == 0 &&
                    resumed.versions(true).output == 1,
                "La recuperación mezcla selección, cursor o versiones");
        const auto archive = serialize_adapter(complete);
        rejected([&] { static_cast<void>(resumed.train(x, y)); });
        require(serialize_adapter(resumed.snapshot()) == archive,
                "Un fallo altera el estado confirmado");
        rejected(
            [&] { static_cast<void>(deserialize_adapter(archive.substr(0, archive.size() / 2))); });
        rejected([&] { static_cast<void>(deserialize_adapter(archive + "basura")); });
    }
}
void rejects_corruption_and_separates_rng() {
    const auto before = at::detail::getDefaultCPUGenerator().get_state().clone();
    AdapterControl control(config(AdapterKind::low_rank), base(), inputs(), base().t());
    const auto state = control.snapshot();
    require(at::equal(before, at::detail::getDefaultCPUGenerator().get_state()),
            "La inicialización consume el RNG global");
    AdapterControl same(config(AdapterKind::low_rank), base(), inputs(), base().t());
    require(at::equal(state.parameters[1], same.snapshot().parameters[1]),
            "La semilla no reproduce los factores iniciales");
    auto other_seed = config(AdapterKind::low_rank);
    ++other_seed.seed;
    AdapterControl different(other_seed, base(), inputs(), base().t());
    require(!at::equal(state.parameters[1], different.snapshot().parameters[1]),
            "La semilla de inicialización no cambia el factor aleatorio");
    auto changed = state;
    changed.config.versions.view++;
    rejected([&] { control.restore(changed); });
    changed = control.snapshot();
    changed.parameters[0][0][0] = std::numeric_limits<double>::quiet_NaN();
    rejected([&] { control.restore(changed); });
    changed = control.snapshot();
    changed.best_step = 1;
    rejected([&] { control.restore(changed); });
    changed = control.snapshot();
    changed.best_mse = 0.25;
    rejected([&] { control.restore(changed); });
    changed = control.snapshot();
    changed.base[0][0] += 1;
    rejected([&] { control.restore(changed); });
    changed = control.snapshot();
    changed.validation_inputs[0][0] += 1;
    rejected([&] { control.restore(changed); });
    auto invalid = inputs();
    invalid[0][0] = std::numeric_limits<double>::infinity();
    rejected([&] { static_cast<void>(control.train(invalid, inputs())); });
    rejected([&] { static_cast<void>(control.train(inputs(), at::zeros({2, 1}, at::kDouble))); });
    rejected([&] { static_cast<void>(control.predict(at::empty({0, 2}, at::kDouble))); });
    require(serialize_adapter(control.snapshot()) == serialize_adapter(state),
            "Un rechazo modifica pesos o selección");
    auto settings = config(AdapterKind::low_rank);
    settings.rank = 3;
    rejected([&] { AdapterControl invalid_control(settings, base(), inputs(), base().t()); });
    settings = config(AdapterKind::full);
    settings.learning_rate = std::numeric_limits<double>::quiet_NaN();
    rejected([&] { AdapterControl invalid_control(settings, base(), inputs(), base().t()); });
    settings = config(AdapterKind::full);
    settings.max_bytes = 1;
    rejected([&] { AdapterControl invalid_control(settings, base(), inputs(), base().t()); });
    rejected([&] {
        AdapterControl invalid_control(config(AdapterKind::full), base(), inputs(), base().t(),
                                       "cuda:1");
    });
}
void failed_update_rolls_back_parameters_and_momentum() {
    auto settings = config(AdapterKind::full);
    settings.inputs = 1;
    settings.outputs = 1;
    settings.rank = 1;
    settings.learning_rate = 1;
    const auto zero = at::zeros({1, 1}, at::kDouble);
    AdapterControl control(settings, zero, at::full({1, 1}, 1e154, at::kDouble), zero);
    const auto before = serialize_adapter(control.snapshot());
    rejected([&] {
        static_cast<void>(
            control.train(at::ones({1, 1}, at::kDouble), at::full({1, 1}, 10.0, at::kDouble)));
    });
    require(serialize_adapter(control.snapshot()) == before,
            "El desbordamiento de validación confirma pesos o momentum parciales");
    settings.minimum_improvement = 100;
    AdapterControl threshold(settings, zero, at::ones({1, 1}, at::kDouble),
                             at::ones({1, 1}, at::kDouble));
    static_cast<void>(
        threshold.train(at::ones({1, 1}, at::kDouble), at::full({1, 1}, 0.5, at::kDouble)));
    require(threshold.snapshot().best_step == 0,
            "La selección acepta una mejora menor que el margen declarado");
}
void rejects_parameter_ranges_before_allocating() {
    auto check = [](const AdapterConfig& settings) {
        rejected([&] { AdapterControl invalid(settings, base(), inputs(), base().t()); });
    };
    for (const auto invalid : {0U, 2049U}) {
        auto settings = config(AdapterKind::full);
        settings.inputs = invalid;
        check(settings);
        settings = config(AdapterKind::full);
        settings.outputs = invalid;
        check(settings);
    }
    for (const auto invalid : {0.0, -1.0, 1.01}) {
        auto settings = config(AdapterKind::full);
        settings.learning_rate = invalid;
        check(settings);
    }
    for (const auto invalid : {-0.1, 1.0, std::numeric_limits<double>::infinity()}) {
        auto settings = config(AdapterKind::full);
        settings.momentum = invalid;
        check(settings);
    }
    auto settings = config(AdapterKind::full);
    settings.problem_id.clear();
    check(settings);
    settings = config(AdapterKind::full);
    settings.validation_id.clear();
    check(settings);
    settings = config(AdapterKind::full);
    settings.max_steps = 0;
    check(settings);
    settings = config(AdapterKind::full);
    settings.max_rows = 1;
    check(settings);
    settings = config(AdapterKind::full);
    settings.max_bytes = (64U << 10) + 1;
    check(settings);
    settings = config(AdapterKind::full);
    settings.versions.output = std::numeric_limits<uint64_t>::max();
    check(settings);
    settings = config(AdapterKind::full);
    settings.minimum_improvement = -1;
    check(settings);
    settings = config(AdapterKind::full);
    settings.kind = static_cast<AdapterKind>(255);
    check(settings);
    rejected([&] {
        AdapterControl invalid(config(AdapterKind::full), base().to(at::kFloat), inputs(),
                               base().t());
    });
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
} // namespace
int main(int argc, char** argv) {
    try {
        const std::span arguments(argv, static_cast<std::size_t>(argc));
        const std::string_view device = arguments.size() == 2 ? arguments[1] : "cpu";
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        divergent_restores_reject_stale_predictions(device);
        different_modes_and_states_have_distinct_identities(device);
        different_selected_states_at_the_same_step_are_incompatible(device);
        serialized_dependency_identities_are_preserved();
        dependencies_follow_the_changed_stage();
        full_and_residual_match_two_analytical_updates(device);
        low_rank_matches_an_independent_scalar_gradient(device);
        parent_selection_and_recovery(device);
        rejects_corruption_and_separates_rng();
        failed_update_rolls_back_parameters_and_momentum();
        rejects_parameter_ranges_before_allocating();
        std::cout << "Adaptadores comprobados en " << device << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
