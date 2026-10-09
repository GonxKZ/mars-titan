"""Apoyo común de la suite que no depende de ningún dominio del proyecto."""

import sys
from pathlib import Path


def python_shebang(directory, executable=None):
    """Línea #! que arranca el intérprete y el entorno actuales desde una ruta sin espacios.

    Linux corta la línea #! en el primer espacio. Con el entorno en una ruta con espacios,
    como el repositorio principal, un ejecutable falso no arranca y la búsqueda en PATH pasa
    al siguiente con el mismo nombre, por ejemplo el nvidia-smi real. El enlace apunta al
    directorio del entorno, así que Python sigue encontrando `pyvenv.cfg` junto al ejecutable.
    """
    executable = Path(executable or sys.executable)
    link = Path(directory) / "environment"
    link.symlink_to(executable.parent.parent, target_is_directory=True)
    line = f"#!{link / executable.parent.name / executable.name}"
    if any(character.isspace() for character in line):
        raise ValueError(f"La ruta del enlace al intérprete contiene espacios: {line}")
    return line + "\n"
