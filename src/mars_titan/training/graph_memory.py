"""Medir los tensores que un grafo de autograd conserva para su backward."""

import torch


def saved_graph_bytes(roots, *, exclude=()):
    """Bytes de los almacenamientos únicos guardados en el grafo alcanzable desde `roots`.

    Recorre `next_functions` y lee los atributos `_saved_*` de cada nodo. Los tensores de
    `exclude`, normalmente los parámetros, no cuentan porque existen sin el grafo. No usa
    `saved_tensors_hooks`: devolver el propio tensor desde el empaquetado crea ciclos con
    las salidas guardadas y retrasa la liberación de grafos ya descartados.
    """
    skip = {tensor.untyped_storage().data_ptr() for tensor in exclude}
    pending = [root.grad_fn for root in roots if root.grad_fn is not None]
    visited, storages = set(), {}

    def add(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            if storage.data_ptr() not in skip:
                storages[storage.data_ptr()] = storage.nbytes()
        elif isinstance(value, (tuple, list)):
            for item in value:
                add(item)

    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        for name in dir(node):
            if name.startswith("_saved_"):
                try:
                    add(getattr(node, name))
                except RuntimeError:
                    # El backward ya liberó ese tensor y no ocupa memoria.
                    continue
        pending.extend(child for child, _ in node.next_functions if child is not None)
    return sum(storages.values())
