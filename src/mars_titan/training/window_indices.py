"""Índices de observaciones por fase, propios de un trabajo o compartidos por vista.

Un índice solo depende de la vista, de la fase y del código y las bibliotecas que lo
construyen (`index_identity`), y `prepare_observation_index` comprueba esa identidad al
reutilizarlo. Su manifiesto guarda la ruta de la vista y una ruta relativa a sus eventos, así
que la huella que registran los informes no depende de la carpeta donde se guarda. Con
`MARS_TITAN_SHARED_INDICES`, los trabajos de una campaña guardan sus índices en una raíz común
por vista y cada índice se construye una sola vez. Sin la variable, cada trabajo los guarda en
su propia carpeta.
"""

import fcntl
import hashlib
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from mars_titan.data.cohort_files import safe_destination
from mars_titan.memory.financial_observations import (
    FinancialObservationSource,
    index_identity,
    prepare_observation_index,
)
from mars_titan.models.titans.config import canonical

SHARED_INDICES_ENV = "MARS_TITAN_SHARED_INDICES"


def shared_root():
    """Raíz compartida que declara la campaña, o None si cada trabajo guarda los suyos."""
    raw = os.environ.get(SHARED_INDICES_ENV)
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{SHARED_INDICES_ENV} debe ser una ruta absoluta y regular")
    return path


def _view_folder(root, identity):
    return root / identity[:16]


def index_root(dataset, output):
    """Carpeta de los índices de un trabajo: la de su vista en la raíz común o la suya."""
    root = shared_root()
    return Path(output) / "indices" if root is None else _view_folder(root, dataset.identity)


def shared_file(dataset, name):
    """Ruta de un resultado que solo depende de la vista, junto a sus índices compartidos.

    Sin raíz compartida devuelve None y cada trabajo calcula el suyo.
    """
    root = shared_root()
    return None if root is None else _view_folder(root, dataset.identity) / name


@contextmanager
def locked(path):
    """Cerrojo con espera para preparar `path`. Dos procesos no lo preparan a la vez."""
    path = Path(path)
    safe_destination(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(f"{path.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def phase_sources(dataset, phases, root):
    """Preparar o reutilizar un índice por fase, con la huella de su identidad en el nombre.

    La identidad incluye la vista, la fase y el código y las bibliotecas que construyen el
    índice. Si cambian entre un corte y la reanudación, el índice nuevo va a otra carpeta y el
    anterior se queda sin usar hasta que se libera la vista. El cerrojo junto a cada índice
    impide que dos trabajos simultáneos construyan el mismo: el segundo espera y lo reutiliza.
    Un índice que quedó a medias se completa, porque `prepare_observation_index` lo reanuda con
    la identidad ya escrita.
    """
    root, sources = Path(root), {}
    for name, phase in phases.items():
        key = canonical(index_identity(dataset, phase))
        folder = root / f"{name}-{hashlib.sha256(key.encode()).hexdigest()[:16]}"
        with locked(folder):
            manifest = prepare_observation_index(
                dataset, folder, phase=phase, resume=folder.exists()
            )
        sources[name] = FinancialObservationSource(dataset, manifest)
    return sources


# Prefijo de una carpeta de vista ya retirada que falta por borrar.
_RELEASED = ".released-"


def _remove(folder):
    size = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
    shutil.rmtree(folder)
    return size


def release_view_indices(identity):
    """Borrar los índices compartidos de la vista con esa huella y devolver los bytes.

    Solo se llama cuando ya no queda ningún trabajo que lea la vista. Si después alguno los
    necesitara, los volvería a construir con el mismo resultado. La carpeta se retira primero
    con un renombrado atómico, así que un corte durante el borrado nunca deja un índice a
    medias con su manifiesto. Los restos de un corte anterior se borran antes, también los de
    esta misma vista, para que el renombrado no encuentre su destino ocupado.
    """
    root = shared_root()
    if root is None or not root.is_dir():
        return 0
    safe_destination(root)
    freed = sum(
        _remove(retired)
        for retired in sorted(root.glob(f"{_RELEASED}*"))
        if retired.is_dir() and not retired.is_symlink()
    )
    folder = _view_folder(root, identity)
    if folder.is_dir() and not folder.is_symlink():
        retired = root / f"{_RELEASED}{folder.name}"
        os.replace(folder, retired)
        freed += _remove(retired)
    return freed
