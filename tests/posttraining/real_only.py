"""Bloqueo de los módulos de aumento y del postentrenamiento emparejado anterior.

`real_data_only` sustituye esos módulos por centinelas durante un bloque y hace fallar
cualquier importación o atributo que se les pida. Sirve para comprobar que la etapa de
adaptadores de la campaña aprende solo con datos reales.
"""

import contextlib
import importlib.abc
import sys
import types

# Cola y preparación del diseño de #128, entradas con episodios añadidos, aumento, mundos
# sintéticos y episodios remuestreados.
BLOCKED = (
    "mars_titan.posttraining.queue",
    "mars_titan.posttraining.preparation",
    "mars_titan.posttraining.augmented_inputs",
    "mars_titan.episodes.augmentation",
    "mars_titan.episodes.worlds",
    "mars_titan.episodes.windows",
)


class _Sentinel(types.ModuleType):
    touched = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        _Sentinel.touched.append(f"{self.__name__}.{name}")
        raise AssertionError(f"Solo datos reales: se pidió {self.__name__}.{name}")


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name in BLOCKED:
            _Sentinel.touched.append(name)
            raise ImportError(f"Solo datos reales: se importó {name}")
        return None


@contextlib.contextmanager
def real_data_only():
    """Ejecutar un bloque con los módulos bloqueados y exigir que nadie los pidiera."""
    saved = {name: sys.modules.get(name) for name in BLOCKED}
    packages = {}
    _Sentinel.touched.clear()
    finder = _Finder()
    for name in BLOCKED:
        sentinel = _Sentinel(name)
        sys.modules[name] = sentinel
        package, _, attribute = name.rpartition(".")
        if package in sys.modules:
            packages[name] = getattr(sys.modules[package], attribute, None)
            setattr(sys.modules[package], attribute, sentinel)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        for name, value in packages.items():
            package, _, attribute = name.rpartition(".")
            if value is None:
                delattr(sys.modules[package], attribute)
            else:
                setattr(sys.modules[package], attribute, value)
    assert not _Sentinel.touched, _Sentinel.touched
