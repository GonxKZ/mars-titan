"""El grafo de llamadas del inventario detecta los ajustes que eluden el bloqueo.

Cada caso escribe un proyecto mínimo con la misma disposición que el repositorio
(`src/mars_titan` y `scripts`) y comprueba qué sitios de ajuste quedan sin proteger.
"""

import textwrap

from tests.training.learning_hold_graph import CallGraph


def project(tmp_path, files):
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for package in {path.parent for path in tmp_path.joinpath("src").rglob("*.py")}:
        init = package / "__init__.py"
        if not init.exists():
            init.write_text("")
    return CallGraph(tmp_path)


def code(text, *, guarded=False):
    """Módulo con la importación de la guarda delante si la usa."""
    head = "from mars_titan.training.learning_hold import require_learning_allowed\n"
    return (head if guarded else "") + textwrap.dedent(text).lstrip("\n")


HOLD = {
    "src/mars_titan/training/learning_hold.py": "def require_learning_allowed(action):\n    pass\n"
}


def unprotected(graph, **exempt):
    return set(graph.unprotected(**exempt))


def test_an_unguarded_optimizer_step_is_reported(tmp_path):
    graph = project(
        tmp_path,
        {
            **HOLD,
            "src/mars_titan/fit.py": code("""
                import torch

                def fit(model):
                    optimizer = torch.optim.AdamW(model.parameters())
                    optimizer.step()
            """),
        },
    )
    assert unprotected(graph) == {"mars_titan.fit:fit"}
    assert {site.kind for site in graph.sites} == {
        "paso de optimizador",
        "optimizador de torch.optim",
    }


def test_a_guard_in_the_entry_protects_the_helpers_it_calls(tmp_path):
    graph = project(
        tmp_path,
        {
            **HOLD,
            "src/mars_titan/fit.py": code(
                """
                def _step(optimizer):
                    optimizer.step()

                def fit(optimizer):
                    require_learning_allowed("ajuste")
                    _step(optimizer)
            """,
                guarded=True,
            ),
        },
    )
    assert unprotected(graph) == set()
    assert graph.late_guards() == []


def test_a_second_unguarded_caller_of_a_protected_helper_is_reported(tmp_path):
    graph = project(
        tmp_path,
        {
            **HOLD,
            "src/mars_titan/fit.py": code(
                """
                def _step(optimizer):
                    optimizer.step()

                def fit(optimizer):
                    require_learning_allowed("ajuste")
                    _step(optimizer)
            """,
                guarded=True,
            ),
            "src/mars_titan/other.py": code("""
                from mars_titan.fit import _step

                def shortcut(optimizer):
                    _step(optimizer)
            """),
        },
    )
    assert unprotected(graph) == {"mars_titan.fit:_step"}
    chain = graph.unprotected()["mars_titan.fit:_step"][1]
    assert chain == ["mars_titan.fit:_step", "mars_titan.other:shortcut"]


def test_methods_reached_through_an_unknown_receiver_keep_their_callers(tmp_path):
    # El medidor recibe el entrenador de otra función y llama a un método privado.
    graph = project(
        tmp_path,
        {
            **HOLD,
            "src/mars_titan/trainer.py": code(
                """
                class Trainer:
                    def _update(self, optimizer):
                        optimizer.step()

                    def run(self, optimizer):
                        require_learning_allowed("ajuste")
                        self._update(optimizer)
            """,
                guarded=True,
            ),
            "src/mars_titan/measure.py": code("""
                from mars_titan import trainer

                def measure(built, optimizer):
                    built._update(optimizer)
            """),
        },
    )
    assert unprotected(graph) == {"mars_titan.trainer:Trainer._update"}
    assert unprotected(graph, exempt_modules=["mars_titan.measure"]) == set()


def test_overrides_in_subclasses_are_dispatched_from_the_base_class(tmp_path):
    graph = project(
        tmp_path,
        {
            **HOLD,
            "src/mars_titan/base.py": code(
                """
                class Base:
                    def _update(self, optimizer):
                        pass

                    def run(self, optimizer):
                        require_learning_allowed("ajuste")
                        self._update(optimizer)

                    def measure(self, optimizer):
                        self._update(optimizer)
            """,
                guarded=True,
            ),
            "src/mars_titan/child.py": code("""
                from mars_titan.base import Base

                class Child(Base):
                    def _update(self, optimizer):
                        optimizer.step()
            """),
        },
    )
    assert unprotected(graph) == {"mars_titan.child:Child._update"}


def test_estimators_xgboost_and_closed_form_solutions_are_fitting_sites(tmp_path):
    graph = project(
        tmp_path,
        {
            **HOLD,
            "src/mars_titan/tabular.py": code("""
                import numpy as np
                import xgboost as xgb
                from sklearn.linear_model import Ridge

                from mars_titan.grid import ActionGrid

                def sklearn(x, y):
                    model = Ridge()
                    model.fit(x, y)

                def boosting(matrix):
                    return xgb.train({}, matrix)

                def normal(gram, rhs):
                    return np.linalg.solve(gram, rhs)

                def grid(targets):
                    return ActionGrid.fit(targets)
            """),
            "src/mars_titan/grid.py": code("""
                class ActionGrid:
                    @classmethod
                    def fit(cls, targets):
                        return cls()
            """),
        },
    )
    kinds = {site.function: site.kind for site in graph.sites}
    assert kinds == {
        "sklearn": "fit de un estimador",
        "boosting": "xgboost.train",
        "normal": "solución cerrada",
    }


def test_a_guard_after_the_fitting_call_is_reported_as_late(tmp_path):
    graph = project(
        tmp_path,
        {
            **HOLD,
            "src/mars_titan/fit.py": code(
                """
                def _step(optimizer):
                    optimizer.step()

                def fit(optimizer):
                    _step(optimizer)
                    require_learning_allowed("ajuste")
            """,
                guarded=True,
            ),
        },
    )
    assert unprotected(graph) == set()
    assert graph.late_guards() == [("mars_titan.fit:fit", 7, 6, "mars_titan.fit:_step")]


def test_scripts_and_module_level_code_are_roots(tmp_path):
    graph = project(
        tmp_path,
        {
            **HOLD,
            "src/mars_titan/fit.py": code("""
                def fit(optimizer):
                    optimizer.step()
            """),
            "scripts/run_fit.py": code("""
                from mars_titan.fit import fit

                fit(None)
            """),
        },
    )
    chain = graph.unprotected()["mars_titan.fit:fit"][1]
    assert chain == ["mars_titan.fit:fit", "scripts.run_fit:<module>"]


def test_a_hook_that_rejects_steps_only_covers_pytorch_steps(tmp_path):
    files = {
        **HOLD,
        "src/mars_titan/measure.py": code("""
            import xgboost as xgb


            def _forbid():
                from torch.optim.optimizer import register_optimizer_step_pre_hook

                def forbid(optimizer, args, kwargs):
                    raise RuntimeError("sin pasos")

                return register_optimizer_step_pre_hook(forbid)


            def _step(optimizer):
                optimizer.step()


            def measure(optimizer):
                hook = _forbid()
                _step(optimizer)
                hook.remove()


            def boosting(matrix):
                hook = _forbid()
                xgb.train({}, matrix)
                hook.remove()
        """),
    }
    graph = project(tmp_path, files)
    assert graph.hooks == {("mars_titan.measure", "_forbid")}
    assert unprotected(graph) == {"mars_titan.measure:boosting"}


def test_a_hook_that_only_records_steps_does_not_count(tmp_path):
    graph = project(
        tmp_path,
        {
            **HOLD,
            "src/mars_titan/measure.py": code("""
                def _watch(steps):
                    from torch.optim.optimizer import register_optimizer_step_pre_hook

                    def count(optimizer, args, kwargs):
                        steps.append(optimizer)

                    return register_optimizer_step_pre_hook(count)


                def measure(optimizer, steps):
                    _watch(steps)
                    optimizer.step()
            """),
        },
    )
    assert graph.hooks == set()
    assert unprotected(graph) == {"mars_titan.measure:measure"}
